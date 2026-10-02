"""Tune the per-TPC anode-piercing cut.

For each track that's matched to a CRT track (whicht0=1), the
collection-plane hit times in TrackCaloSkim are already CRT-t0-shifted
— so the earliest TPC hit time, expressed in TPC ticks, IS the
"CRT-shifted earliest hit". An anode-piercing track in TPC i has

    tpc_i_t_min ≈ anode_tick

regardless of the actual muon t0 (the shift is built in). We use this
as the unified anode-piercing test, replacing the older
(global-envelope) anode-cathode + in-time cuts.

This script builds the per-TPC `t_min` distribution and picks a
peak-bounded window around the empirical peak (which sits near tick 418,
NOT the nominal 412 quoted in the FCL), then emits
cut_tuning/cut_config_per_tpc.json.

Two data sources:

  parquet (default)  — reads align_df parquet, but that parquet is
                       already filtered by an earlier window cut, so
                       the tails are biased.  Useful for quick replay.

  --filelist FILE    — reads RAW ntuples and applies upstream cuts
                       (CRT-matched, whicht0 != -1, sentinel-finite,
                       has collection-plane hits) but NOT the t_min
                       cut, giving the unbiased peak shape needed for
                       re-tuning.

Usage:
  python plot_track_cuts_per_tpc.py \\
      --filelist filelist_nonempty.txt --maxfiles 400 --nproc 32 \\
      --threshold-frac 0.05
  python plot_track_cuts_per_tpc.py \\
      --parquet /exp/sbnd/data/users/gputnam/efield-study/align_df
"""

import argparse
import glob
import json
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from crtlib.constants import SENTINEL, WHICHT0_NONE
from crtlib.io import load_align_df, pnfs_to_xroot
from crtlib.ntuple import TREE_TPC, TREE_CRT


def stack_per_tpc_tmin(df):
    pieces = []
    for tpc in (0, 1):
        m = df[f"tpc{tpc}_n_p2_hits"] > 0
        sub = df.loc[m]
        pieces.append(pd.DataFrame({
            "tpc": tpc,
            "t_min": sub[f"tpc{tpc}_t_min"].values,
            "t_max": sub[f"tpc{tpc}_t_max"].values,
        }))
    return pd.concat(pieces, ignore_index=True)


def per_track_tmin(df):
    """Return DataFrame with one row per track and columns
    [t_min_tpc0, t_min_tpc1] (NaN if that TPC has no hits)."""
    return pd.DataFrame({
        "t_min_tpc0": np.where(df["tpc0_n_p2_hits"] > 0,
                               df["tpc0_t_min"], np.nan),
        "t_min_tpc1": np.where(df["tpc1_n_p2_hits"] > 0,
                               df["tpc1_t_min"], np.nan),
    })


# ---------- raw-ntuple loader (replicates make_align_df_sbnd.py up to
#            but NOT including the t_min cut) ------------------------

TCS_TRK_BRANCHES = [
    "meta.run", "meta.subrun", "meta.evt", "id", "whicht0",
]
TCS_HIT_BRANCHES = [
    "hits2.h.time", "hits2.h.tpc", "hits2.oncalo",
]
CRT_BRANCHES = [
    "run", "subrun", "event",
    "tpc_tr_matched",
    "tpc_tr_start_x", "tpc_tr_start_y", "tpc_tr_start_z",
    "tpc_tr_end_x",   "tpc_tr_end_y",   "tpc_tr_end_z",
]


def _ntuple_load_one(args):
    """Read one ROOT file; emit per-track (t_min_tpc0, t_min_tpc1).

    Mirrors load_one_file() in make_align_df_sbnd.py up through the
    "≥1 collection-plane hit" cut.  Returns ALL such tracks, including
    those that would later fail the t_min cut.
    """
    import awkward as ak
    import uproot
    path, use_xroot = args
    url = pnfs_to_xroot(path, use_xroot)
    try:
        with uproot.open(url, timeout=120) as f:
            tcs_arr = f[TREE_TPC].arrays(TCS_TRK_BRANCHES + TCS_HIT_BRANCHES, library="ak")
            crt_arr = f[TREE_CRT].arrays(CRT_BRANCHES, library="ak")
    except Exception as e:
        print(f"[skip] {path}: {type(e).__name__}: {e}", flush=True)
        return None
    if len(tcs_arr) == 0 or len(crt_arr) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    crt_run = ak.to_numpy(crt_arr["run"]).astype(np.int64)
    crt_sub = ak.to_numpy(crt_arr["subrun"]).astype(np.int64)
    crt_evt = ak.to_numpy(crt_arr["event"]).astype(np.int64)
    ev_index = {(int(r), int(s), int(e)): i for i, (r, s, e)
                in enumerate(zip(crt_run, crt_sub, crt_evt))}
    cmatched = crt_arr["tpc_tr_matched"]
    csx, csy, csz = crt_arr["tpc_tr_start_x"], crt_arr["tpc_tr_start_y"], crt_arr["tpc_tr_start_z"]
    cex, cey, cez = crt_arr["tpc_tr_end_x"],   crt_arr["tpc_tr_end_y"],   crt_arr["tpc_tr_end_z"]

    run = ak.to_numpy(tcs_arr["meta.run"]).astype(np.int64)
    sub = ak.to_numpy(tcs_arr["meta.subrun"]).astype(np.int64)
    evt = ak.to_numpy(tcs_arr["meta.evt"]).astype(np.int64)
    tid = ak.to_numpy(tcs_arr["id"]).astype(np.int64)
    wt0 = ak.to_numpy(tcs_arr["whicht0"]).astype(np.int32)

    out = []
    for i in range(len(tcs_arr)):
        ev_key = (int(run[i]), int(sub[i]), int(evt[i]))
        ic = ev_index.get(ev_key)
        if ic is None: continue
        k = int(tid[i])
        if k < 0 or k >= len(cmatched[ic]): continue
        if not bool(cmatched[ic][k]): continue
        if int(wt0[i]) == WHICHT0_NONE: continue
        sx = float(csx[ic][k]); sy = float(csy[ic][k]); sz = float(csz[ic][k])
        ex = float(cex[ic][k]); ey = float(cey[ic][k]); ez = float(cez[ic][k])
        if (sx < SENTINEL or sy < SENTINEL or sz < SENTINEL
                or ex < SENTINEL or ey < SENTINEL or ez < SENTINEL):
            continue
        ht = np.asarray(tcs_arr["hits2.h.time"][i], dtype=np.float64)
        htpc = np.asarray(tcs_arr["hits2.h.tpc"][i], dtype=np.int32)
        hon = np.asarray(tcs_arr["hits2.oncalo"][i]).astype(bool)
        ok = hon & np.isfinite(ht)
        if not ok.any(): continue
        ht = ht[ok]; htpc = htpc[ok]
        m0 = (htpc == 0); m1 = (htpc == 1)
        t0 = float(ht[m0].min()) if m0.any() else np.nan
        t1 = float(ht[m1].min()) if m1.any() else np.nan
        out.append((t0, t1))
    return np.asarray(out, dtype=np.float32) if out else np.zeros((0, 2), dtype=np.float32)


def load_tmin_from_ntuples(filelist, maxfiles, nproc, use_xroot=True):
    with open(filelist) as f:
        files = [ln.strip() for ln in f if ln.strip()][:maxfiles]
    print(f"  reading {len(files)} ROOT files (nproc={nproc}) ...", flush=True)
    args = [(p, use_xroot) for p in files]
    nproc = min(nproc, len(files), cpu_count())
    chunks = []
    t0 = time.time()
    with Pool(processes=nproc) as pool:
        for k, arr in enumerate(pool.imap_unordered(_ntuple_load_one, args, chunksize=2)):
            if arr is None: continue
            chunks.append(arr)
            if (k + 1) % max(1, len(files) // 10) == 0:
                rate = (k + 1) / (time.time() - t0)
                print(f"    {k+1}/{len(files)}  ({rate:.1f} files/s)", flush=True)
    if not chunks:
        return pd.DataFrame({"t_min_tpc0": [], "t_min_tpc1": []}, dtype=np.float32)
    arr = np.concatenate(chunks, axis=0)
    return pd.DataFrame({"t_min_tpc0": arr[:, 0], "t_min_tpc1": arr[:, 1]})


def find_peak_window(values, search_lo, search_hi, bg_lo, bg_hi,
                     threshold_frac=0.25):
    """Return (peak_x, lo, hi) where lo, hi bracket the peak above
    threshold_frac × (peak − background)."""
    h, edges = np.histogram(values, bins=4000, range=(0, 4000))
    centers = 0.5 * (edges[:-1] + edges[1:])

    mask_search = (centers >= search_lo) & (centers <= search_hi)
    mask_bg = (centers >= bg_lo) & (centers <= bg_hi)
    bg = float(np.median(h[mask_bg])) if mask_bg.any() else 0.0
    peak_idx_local = int(np.argmax(h[mask_search]))
    search_idxs = np.where(mask_search)[0]
    peak_idx = int(search_idxs[peak_idx_local])
    peak_h = float(h[peak_idx])
    cut_h = bg + threshold_frac * (peak_h - bg)
    if cut_h <= 0:
        return None, None, None

    i = peak_idx
    while i > 0 and h[i] >= cut_h:
        i -= 1
    lo = float(centers[i + 1]) if i + 1 < len(centers) else float(centers[i])
    j = peak_idx
    while j < len(h) - 1 and h[j] >= cut_h:
        j += 1
    hi = float(centers[j - 1]) if j - 1 >= 0 else float(centers[j])
    return float(centers[peak_idx]), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet",
                    default="/exp/sbnd/data/users/gputnam/efield-study/align_df")
    ap.add_argument("--filelist", default=None,
                    help="if set, read raw ntuples (unbiased) instead of parquet")
    ap.add_argument("--maxfiles", type=int, default=400)
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--no-xroot", action="store_true")
    ap.add_argument("--outdir", default="cut_tuning")
    ap.add_argument("--threshold-frac", type=float, default=0.25,
                    help="window goes from peak down to bg + this × (peak − bg)")
    ap.add_argument("--write-config", action="store_true",
                    help="overwrite cut_tuning/cut_config_per_tpc.json with the result")
    ap.add_argument("--config-suffix", default="",
                    help="if set, write to cut_config_per_tpc_<SUFFIX>.json instead")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.filelist:
        per_track = load_tmin_from_ntuples(
            args.filelist, args.maxfiles, args.nproc,
            use_xroot=not args.no_xroot)
        print(f"  CRT-matched whicht0!=-1 tracks (raw ntuples): {len(per_track):,}",
              flush=True)
        # Stack per-TPC for histogramming.
        t0 = per_track["t_min_tpc0"].dropna().values
        t1 = per_track["t_min_tpc1"].dropna().values
        pt_all = np.concatenate([t0, t1])
        n_track_total = len(per_track)
        source = f"raw ntuples ({args.maxfiles} files)"
    else:
        print(f"Loading {args.parquet} ...", flush=True)
        df = load_align_df(args.parquet)
        print(f"  {len(df):,} matched whicht0=1 tracks (parquet)", flush=True)
        pt = stack_per_tpc_tmin(df)
        per_track = per_track_tmin(df)
        pt_all = pt["t_min"].values
        n_track_total = len(df)
        source = "align_df parquet (already filtered)"
    n_tpc_entries = len(pt_all)
    print(f"  per-TPC anode candidates: {n_tpc_entries:,}", flush=True)

    peak, lo, hi = find_peak_window(
        pt_all, search_lo=400, search_hi=440,
        bg_lo=600, bg_hi=1500,
        threshold_frac=args.threshold_frac,
    )
    if peak is None:
        peak, lo, hi = 418.5, 416.5, 420.5

    in_window_per_tpc = int(((pt_all >= lo) & (pt_all <= hi)).sum())
    pass0 = (per_track["t_min_tpc0"].between(lo, hi)).fillna(False)
    pass1 = (per_track["t_min_tpc1"].between(lo, hi)).fillna(False)
    n_track_pass = int((pass0 | pass1).sum())
    frac_track = n_track_pass / max(n_track_total, 1)
    frac_tpc   = in_window_per_tpc / max(n_tpc_entries, 1)

    print(f"\n  source           : {source}")
    print(f"  threshold_frac   : {args.threshold_frac}")
    print(f"  per-TPC peak     : {peak:.1f} ticks")
    print(f"  selected window  : [{lo:.1f}, {hi:.1f}]  (width {hi-lo:.2f})")
    print(f"  TPC entries in window  : {in_window_per_tpc:,} / {n_tpc_entries:,} "
          f"({100*frac_tpc:.2f}%)")
    print(f"  TRACKS passing (either TPC in window): "
          f"{n_track_pass:,} / {n_track_total:,} "
          f"({100*frac_track:.2f}%)", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    h, edges = np.histogram(pt_all, bins=4000, range=(0, 4000))
    c = 0.5 * (edges[:-1] + edges[1:])

    # Left pane: zoom, linear y.
    axes[0].step(c, h, where="mid", linewidth=1.2)
    axes[0].set_xlim(400, 440)
    axes[0].set_xlabel("tpc t_min (ticks), zoom")
    axes[0].set_ylabel("entries / 1-tick bin (linear)")
    axes[0].axvline(peak, color="r", ls="--", label=f"peak {peak:.1f}")
    axes[0].axvspan(lo, hi, color="orange", alpha=0.25,
                    label=f"cut [{lo:.1f}, {hi:.1f}], "
                          f"track-pass {100*frac_track:.1f}%")
    axes[0].set_title(f"per-TPC earliest hit time, n={n_tpc_entries:,}")
    axes[0].legend()

    # Right pane: zoom, log y (unchanged behaviour).
    axes[1].step(c, h, where="mid", linewidth=1.2)
    axes[1].set_xlim(400, 440); axes[1].set_yscale("log")
    axes[1].set_xlabel("tpc t_min (ticks), zoom")
    axes[1].set_ylabel("entries / 1-tick bin (log)")
    axes[1].axvline(peak, color="r", ls="--")
    axes[1].axvspan(lo, hi, color="orange", alpha=0.25)
    axes[1].set_title(f"thresh_frac={args.threshold_frac}, "
                      f"window [{lo:.1f}, {hi:.1f}]")
    fig.tight_layout()
    suffix = args.config_suffix or f"thresh{args.threshold_frac:.2f}"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.outdir, f"per_tpc_tmin_{suffix}.{ext}"), dpi=140)
    plt.close(fig)

    cfg = {
        "comment": ("Per-TPC anode-piercing cut on tpc_{0,1}_t_min. "
                    "Track is kept iff EITHER TPC's t_min lies in this window."),
        "anode_tick_peak": float(peak),
        "tpc_t_min_lo": float(lo),
        "tpc_t_min_hi": float(hi),
        "threshold_frac": float(args.threshold_frac),
        "data_source": source,
        "n_track_total": int(n_track_total),
        "n_track_pass":  int(n_track_pass),
        "track_pass_fraction": float(frac_track),
        "n_tpc_candidates": int(n_tpc_entries),
        "n_tpc_in_window":  int(in_window_per_tpc),
        "tpc_pass_fraction": float(frac_tpc),
    }
    if args.write_config:
        outpath = os.path.join(args.outdir, "cut_config_per_tpc.json")
    else:
        outpath = os.path.join(args.outdir,
                               f"cut_config_per_tpc_{suffix}.json")
    with open(outpath, "w") as fp:
        json.dump(cfg, fp, indent=2)
    print(f"\n  wrote {outpath}", flush=True)


if __name__ == "__main__":
    main()
