#!/usr/bin/env python3
"""Per-TPC, per-plane anode-cathode tick-length histograms from the
anode+cathode crosser sample, restricted to STABLE cathode regions.

For each (track, TPC, plane): dt = max_time - min_time, with
  * an anode anchor: min_time within the anode window (the track segment
    starts at the anode);
  * a cathode-position stability selection: the cathode-side endpoint
    (max_y, max_z) must lie in the STABLE windows -- the regions where the
    per-pixel cathode maps are flat, i.e. AWAY from the support-frame
    features at z ~ 255 and y ~ +-50 / +-150 (CATHODE_DRIFT_MC.md /
    cathode_study heatmaps), and away from the detector edges.

STABLE WINDOWS (reported in the figure):
    y in [-190,-160] U [-140,-60] U [-40,+40] U [+60,+140] U [+160,+190]
    z in [30, 240] U [270, 470]
    (excludes +-10 cm around y = +-50, +-150; 240-270 around z = 255;
     30 cm at the z edges; 10 cm at the y edges)

Plot: 2x3 panels (TPC x plane), LINEAR y, x zoomed to peak +- 30 ticks;
annotated with the peak, and the mean/median within the zoom window.

Usage:
    python plot_ac_ticklength.py --input <drift_df.parquet> --out DIR \
        [--label TAG]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit_driftv_cb import crystal_ball, fit_cb

ANODE_WIN = (410.0, 430.0)       # min_time anchor (anode peak ~418.5)
DT_RANGE = (2400.0, 2700.0)      # search range for the dt peak
ZOOM = 30.0
TICK_US = 0.5
L_TPC = 200.0                    # cm, single-TPC anode-cathode (nominal)

Y_STABLE = [(-190.0, -160.0), (-140.0, -60.0), (-40.0, 40.0),
            (60.0, 140.0), (160.0, 190.0)]
Z_STABLE = [(30.0, 240.0), (270.0, 470.0)]


def in_windows(v, wins):
    m = np.zeros(v.shape, dtype=bool)
    for lo, hi in wins:
        m |= (v >= lo) & (v < hi)
    return m


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="ac_ticklength")
    ap.add_argument("--label", default="")
    ap.add_argument("--truth-v", type=float, default=None,
                    help="true drift velocity [cm/us] (MC): overlays the "
                         "expected tick length L/v as a dashed line")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cols = []
    for t in (0, 1):
        for p in (0, 1, 2):
            pfx = f"tpc{t}_p{p}"
            cols += [f"{pfx}_min_time", f"{pfx}_max_time",
                     f"{pfx}_max_y", f"{pfx}_max_z"]
    df = pd.read_parquet(args.input, columns=cols)

    edges = np.arange(DT_RANGE[0], DT_RANGE[1] + 1.0, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    summary = {}
    for t in (0, 1):
        for p in (0, 1, 2):
            pfx = f"tpc{t}_p{p}"
            tmin = df[f"{pfx}_min_time"].to_numpy()
            tmax = df[f"{pfx}_max_time"].to_numpy()
            ymax = df[f"{pfx}_max_y"].to_numpy()
            zmax = df[f"{pfx}_max_z"].to_numpy()
            sel = (np.isfinite(tmin) & np.isfinite(tmax)
                   & (tmin >= ANODE_WIN[0]) & (tmin <= ANODE_WIN[1])
                   & in_windows(ymax, Y_STABLE) & in_windows(zmax, Z_STABLE))
            dt = (tmax - tmin)[sel]
            dt = dt[(dt >= DT_RANGE[0]) & (dt < DT_RANGE[1])]
            h, _ = np.histogram(dt, bins=edges)
            pk = centers[np.argmax(h)]
            # Crystal Ball fit (the determination)
            mu, mu_err, n_cb, popt = fit_cb(dt)
            v_cb = L_TPC / (mu * TICK_US)
            summary[(t, p)] = (pk, mu, mu_err, v_cb, int(len(dt)))

            ax = axes[t, p]
            m = (centers >= pk - ZOOM) & (centers <= pk + ZOOM)
            ax.step(centers[m], h[m], where="mid", color="navy", lw=1.3)
            if popt is not None:
                xs = np.linspace(mu - ZOOM, mu + ZOOM, 400)
                ax.plot(xs, crystal_ball(xs, *popt), "r-", lw=1.6,
                        label=f"CB fit: $\\mu$={mu:.1f}\nv={v_cb:.5f}")
            ax.axvline(mu, color="red", ls="--", lw=1.0)
            if args.truth_v is not None:
                dt_true = L_TPC / (args.truth_v * TICK_US)
                ax.axvline(dt_true, color="black", ls="--", lw=1.4,
                           label=f"true ({args.truth_v:.5f}):\n"
                                 f"{dt_true:.1f} tk")
            ax.set_xlim(pk - ZOOM, pk + ZOOM)
            ax.set_ylim(0, None)
            ax.set_title(f"TPC{t} plane{p}"
                         + ("  (coll.)" if p == 2 else "  (ind.)")
                         + f"   N={len(dt):,}", fontsize=10)
            ax.grid(alpha=0.3)
            if t == 1:
                ax.set_xlabel(r"$dt = t_{max} - t_{min}$ [ticks]")
            if p == 0:
                ax.set_ylabel("tracks / tick")
            ax.legend(fontsize=7.5)
    fig.suptitle(f"anode-cathode tick length, stable cathode regions only "
                 f"{args.label}\n"
                 f"anchors: $t_{{min}}\\in$[{ANODE_WIN[0]:g},"
                 f"{ANODE_WIN[1]:g}]; stable y: "
                 "[-190,-160]U[-140,-60]U[-40,40]U[60,140]U[160,190]; "
                 "stable z: [30,240]U[270,470] cm", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "ac_ticklength.png"), dpi=120)
    fig.savefig(os.path.join(args.out, "ac_ticklength.pdf"))

    print(f"{'panel':14s} {'peak':>8s} {'CB mu':>9s} {'CB v':>9s} "
          f"{'N':>10s}")
    for (t, p), (pk, mu, mu_err, v_cb, n) in summary.items():
        print(f"TPC{t} plane{p}  {pk:>8.1f} {mu:>9.2f} {v_cb:>9.5f} {n:>10,}")
    if args.truth_v is not None:
        dt_true = L_TPC / (args.truth_v * TICK_US)
        print(f"  true tick length (L={L_TPC:g} cm / "
              f"{args.truth_v:.5f} cm/us / {TICK_US} us): {dt_true:.1f} ticks")
    np.savez_compressed(os.path.join(args.out, "ac_ticklength.npz"),
                        summary=np.array([(t, p, *summary[(t, p)])
                                          for t in (0, 1)
                                          for p in (0, 1, 2)]))
    print(f"wrote {args.out}/", flush=True)


if __name__ == "__main__":
    main()
