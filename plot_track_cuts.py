"""Build distributions of the two cut variables used to identify anode-piercing
tracks and emit a cut_config.json with recommended numerical thresholds.

Two samples are wanted:
  A) Anode-cathode through-going crossers — total drift-time length ≈ 2575
     ticks on the collection plane.
  B) In-time anode crossers — earliest collection-plane hit at the anode
     tick (~412).

Inputs:
  drift_df_full_combined.parquet (per-track, has tpc{0,1}_p2_min/max_time)

Outputs (in --outdir, default ./cut_tuning/):
  hist_t_span_p2.{png,pdf}       — total time-span on the collection plane
  hist_t_min_p2.{png,pdf}        — earliest collection-plane hit time
  hist_2d_tmin_tspan.{png,pdf}   — joint distribution
  cut_config.json                — recommended numerical cuts

Usage:
  python plot_track_cuts.py
  python plot_track_cuts.py --parquet drift_df_full.parquet --outdir cut_tuning
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANODE_TICK = 412
CATHODE_TICK = 2985
FULL_DRIFT_TICKS = CATHODE_TICK - ANODE_TICK


def load_track_features(parquet_path):
    """Read the columns needed for cut tuning from the per-track parquet.

    Returns a DataFrame with one row per track and columns:
      - selected, clear_cosmic_muon (filters)
      - tpc0_p2_min_time, tpc0_p2_max_time, tpc1_p2_min_time, tpc1_p2_max_time
      - t_min_p2, t_max_p2, t_span_p2 (both-TPC envelope on the collection plane)
      - has_tpc0_p2, has_tpc1_p2 (any plane-2 hits in that TPC)
    """
    cols = ["selected", "clear_cosmic_muon",
            "tpc0_p2_min_time", "tpc0_p2_max_time",
            "tpc1_p2_min_time", "tpc1_p2_max_time"]
    df = pd.read_parquet(parquet_path, columns=cols)

    df["has_tpc0_p2"] = df["tpc0_p2_min_time"].notna()
    df["has_tpc1_p2"] = df["tpc1_p2_min_time"].notna()

    df["t_min_p2"] = np.fmin(df["tpc0_p2_min_time"], df["tpc1_p2_min_time"])
    df["t_max_p2"] = np.fmax(df["tpc0_p2_max_time"], df["tpc1_p2_max_time"])
    df["t_span_p2"] = df["t_max_p2"] - df["t_min_p2"]
    return df


def plot_hist(values, outpath, *, title, xlabel,
              bins=200, xrange=None, vlines=None, log=True):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if xrange is None:
        lo, hi = np.nanpercentile(values, [0.1, 99.9])
    else:
        lo, hi = xrange
    h, edges, _ = ax.hist(values.dropna(), bins=bins, range=(lo, hi),
                          histtype="step", linewidth=1.5)
    if log:
        ax.set_yscale("log")
    if vlines:
        for x, label in vlines:
            ax.axvline(x, color="C3", linestyle="--", linewidth=1.0)
            ax.text(x, ax.get_ylim()[1] * 0.5, label,
                    rotation=90, color="C3", fontsize=8,
                    va="center", ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("tracks / bin")
    ax.set_title(title)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{outpath}.{ext}", dpi=150)
    plt.close(fig)
    return h, edges


def _peak_window_above_background(h, edges, search_lo, search_hi,
                                  bg_lo, bg_hi, threshold_frac=0.5):
    """Return (lo, hi) bin centers bounding a peak by going down to a
    fraction of (peak_height − background) from the peak.

    Background level is the median histogram height in [bg_lo, bg_hi]
    bin-center range. Peak location is argmax inside [search_lo,
    search_hi]. The returned window is tight: it follows the peak down
    on both sides until the height drops below
    ``background + threshold_frac × (peak − background)``.
    """
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask_search = (centers >= search_lo) & (centers <= search_hi)
    if not mask_search.any():
        return None, None
    mask_bg = (centers >= bg_lo) & (centers <= bg_hi)
    bg = float(np.median(h[mask_bg])) if mask_bg.any() else 0.0
    peak_idx_local = int(np.argmax(h[mask_search]))
    search_idxs = np.where(mask_search)[0]
    peak_idx = int(search_idxs[peak_idx_local])
    peak_h = float(h[peak_idx])
    cut_h = bg + threshold_frac * (peak_h - bg)
    if cut_h <= 0:
        return None, None
    # Walk left until h < cut_h.
    i = peak_idx
    while i > 0 and h[i] >= cut_h:
        i -= 1
    lo = float(centers[i + 1]) if i + 1 < len(centers) else float(centers[i])
    # Walk right until h < cut_h.
    j = peak_idx
    while j < len(h) - 1 and h[j] >= cut_h:
        j += 1
    hi = float(centers[j - 1]) if j - 1 >= 0 else float(centers[j])
    return lo, hi


def recommend_cuts(df):
    """Pick TIGHT cut windows that directly bound each peak.

    Both samples use a window that tracks the peak shape via FWHM-above-
    background on its own histogram. The cut is intentionally tight — we
    prefer to lose statistics over admitting continuum/transition-region
    tracks into the alignment fit.
    """
    # Use 1-tick bins so the recommended cuts have integer-tick resolution.
    # The two peaks are very narrow (tens of ticks for t_span, single-digit
    # ticks at FWHM for t_min) — coarser binning loses them.
    span = df["t_span_p2"].dropna()
    h_span, edges_span = np.histogram(span, bins=4000, range=(0, 4000))
    span_lo, span_hi = _peak_window_above_background(
        h_span, edges_span,
        search_lo=2520, search_hi=2600,   # peak observed at ~2563 ticks
        bg_lo=1800, bg_hi=2300,           # cosmic-bulk plateau as background
        threshold_frac=0.25,              # 25% above bg — captures the
                                          # main peak without admitting
                                          # the broad shoulders / continuum
    )
    if span_lo is None or span_hi is None:
        span_lo, span_hi = 2549.0, 2575.0  # conservative fallback

    tmin = df["t_min_p2"].dropna()
    h_tmin, edges_tmin = np.histogram(tmin, bins=4000, range=(0, 4000))
    tmin_lo, tmin_hi = _peak_window_above_background(
        h_tmin, edges_tmin,
        search_lo=410, search_hi=425,     # in-time peak observed at ~418
        bg_lo=600, bg_hi=2500,            # flat continuum away from peak
        threshold_frac=0.25,
    )
    if tmin_lo is None or tmin_hi is None:
        tmin_lo, tmin_hi = 414.0, 421.0  # conservative fallback

    return {
        "anode_tick": ANODE_TICK,
        "cathode_tick": CATHODE_TICK,
        "anode_cathode_crosser": {
            "t_span_p2_lo": float(span_lo),
            "t_span_p2_hi": float(span_hi),
            "comment": "Tracks with collection-plane time-span inside "
                       "this tight window around 2575 ticks (FWHM-above-"
                       "background) are flagged anode-cathode crossers.",
        },
        "intime_anode_crosser": {
            "t_min_p2_lo": float(tmin_lo),
            "t_min_p2_hi": float(tmin_hi),
            "comment": "Tracks with earliest collection-plane hit time "
                       "inside this tight window around the anode tick "
                       "(FWHM-above-background) are flagged in-time "
                       "anode crossers.",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet",
                    default="drift_df_full_combined.parquet",
                    help="per-track parquet (output of make_driftV_df_sbnd.py)")
    ap.add_argument("--outdir", default="cut_tuning",
                    help="directory for plots and cut_config.json")
    ap.add_argument("--filter-selected", action="store_true",
                    help="only use tracks with selected != 0 (default: all)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading {args.parquet} ...", flush=True)
    df = load_track_features(args.parquet)
    print(f"  total tracks: {len(df):,}", flush=True)
    if args.filter_selected:
        before = len(df)
        df = df[df["selected"] != 0]
        print(f"  selected != 0: {len(df):,} (was {before:,})", flush=True)

    n_both = int((df["has_tpc0_p2"] & df["has_tpc1_p2"]).sum())
    n_only0 = int((df["has_tpc0_p2"] & ~df["has_tpc1_p2"]).sum())
    n_only1 = int((~df["has_tpc0_p2"] & df["has_tpc1_p2"]).sum())
    print(f"  has hits in both TPCs (cathode-crossing candidates): {n_both:,}", flush=True)
    print(f"  TPC0 only: {n_only0:,}, TPC1 only: {n_only1:,}", flush=True)

    cuts = recommend_cuts(df)

    plot_hist(
        df["t_span_p2"], os.path.join(args.outdir, "hist_t_span_p2"),
        title=("Collection-plane time-span per track\n"
               f"vlines = recommended anode-cathode window "
               f"[{cuts['anode_cathode_crosser']['t_span_p2_lo']:.0f}, "
               f"{cuts['anode_cathode_crosser']['t_span_p2_hi']:.0f}] ticks; "
               f"reference: full drift = {FULL_DRIFT_TICKS} ticks"),
        xlabel="t_max - t_min on plane 2 (ticks)",
        xrange=(0, 4000), bins=400,
        vlines=[(cuts["anode_cathode_crosser"]["t_span_p2_lo"], "lo"),
                (cuts["anode_cathode_crosser"]["t_span_p2_hi"], "hi"),
                (FULL_DRIFT_TICKS, "full drift")],
    )
    plot_hist(
        df["t_min_p2"], os.path.join(args.outdir, "hist_t_min_p2"),
        title=("Earliest collection-plane hit time per track\n"
               f"vlines = recommended in-time window "
               f"[{cuts['intime_anode_crosser']['t_min_p2_lo']:.0f}, "
               f"{cuts['intime_anode_crosser']['t_min_p2_hi']:.0f}] ticks; "
               f"anode tick reference = {ANODE_TICK}"),
        xlabel="min(t_hit) on plane 2 (ticks)",
        xrange=(0, 4000), bins=400,
        vlines=[(cuts["intime_anode_crosser"]["t_min_p2_lo"], "lo"),
                (cuts["intime_anode_crosser"]["t_min_p2_hi"], "hi"),
                (ANODE_TICK, "anode")],
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sub = df.dropna(subset=["t_min_p2", "t_span_p2"]).sample(
        min(2_000_000, len(df.dropna(subset=["t_min_p2", "t_span_p2"]))),
        random_state=0)
    h2, xe, ye, im = ax.hist2d(
        sub["t_min_p2"], sub["t_span_p2"],
        bins=[200, 200], range=[[0, 4000], [0, 4000]],
        norm=matplotlib.colors.LogNorm(),
    )
    ax.set_xlabel("min(t_hit) on plane 2 (ticks)")
    ax.set_ylabel("t_max - t_min on plane 2 (ticks)")
    ax.set_title("Joint distribution (log color, sub-sampled)")
    ax.axvline(cuts["intime_anode_crosser"]["t_min_p2_lo"], color="r", ls="--", lw=0.8)
    ax.axvline(cuts["intime_anode_crosser"]["t_min_p2_hi"], color="r", ls="--", lw=0.8)
    ax.axhline(cuts["anode_cathode_crosser"]["t_span_p2_lo"], color="orange", ls="--", lw=0.8)
    ax.axhline(cuts["anode_cathode_crosser"]["t_span_p2_hi"], color="orange", ls="--", lw=0.8)
    fig.colorbar(im, ax=ax, label="tracks / bin")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.outdir, f"hist_2d_tmin_tspan.{ext}"), dpi=150)
    plt.close(fig)

    n_anode_cathode = int(
        ((df["t_span_p2"] >= cuts["anode_cathode_crosser"]["t_span_p2_lo"]) &
         (df["t_span_p2"] <= cuts["anode_cathode_crosser"]["t_span_p2_hi"])).sum())
    n_intime = int(
        ((df["t_min_p2"] >= cuts["intime_anode_crosser"]["t_min_p2_lo"]) &
         (df["t_min_p2"] <= cuts["intime_anode_crosser"]["t_min_p2_hi"])).sum())
    cuts["yield"] = {
        "n_anode_cathode_crossers": n_anode_cathode,
        "n_intime_anode_crossers": n_intime,
        "n_total_tracks": int(len(df)),
        "n_with_p2_hits": int((df["has_tpc0_p2"] | df["has_tpc1_p2"]).sum()),
    }
    print(f"\n  yield: anode-cathode = {n_anode_cathode:,}", flush=True)
    print(f"         in-time anode    = {n_intime:,}", flush=True)

    cfg_path = os.path.join(args.outdir, "cut_config.json")
    with open(cfg_path, "w") as fp:
        json.dump(cuts, fp, indent=2)
    print(f"\n  wrote {cfg_path}", flush=True)
    for k, v in cuts.items():
        print(f"    {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
