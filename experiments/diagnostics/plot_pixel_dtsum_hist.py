#!/usr/bin/env python3
"""Histogram of per-pixel summed TPC0+TPC1 drift-time lengths.

Input: the per-2cm-pixel Crystal Ball cathode-tick fits of the cathode
heatmap workflow (cathode_fit_results_tpc{0,1}.csv: per (y, z) pixel, the
CB peak mu of the dt = max_time - min_time distribution in that TPC).

For each (y, z) pixel present in BOTH TPCs (after quality cuts), one entry:

    T(pixel) = mu_TPC0 + mu_TPC1   [ticks]

— the full anode-to-anode drift-time length at that (y, z). The sum is
independent of the cathode position (a cathode displacement adds to one
TPC's mu what it removes from the other's), so this distribution isolates
the total-span/velocity observable and its (y, z) structure, pixel by
pixel, with the CB peak (mode) as the per-pixel estimator.

Outputs (--out): pixel_dtsum_hist.png (the requested histogram),
pixel_dtsum_map.png, pixel_dtsum.csv, and a printed velocity table
(v = span / <T>) under selectable span assumptions.

Usage:
    python plot_pixel_dtsum_hist.py --fits-dir cathode_study_2cm_v4 \
        --out cathode_study_2cm_v4
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TICK_US = 0.5
SPANS_CM = (400.0, 402.5, 397.1)   # production assumption; GDML wire-plane
                                   # to x=0 (x2); GDML minus ~5.4 cm real
                                   # mesh standoff


def load_fits(path, max_mu_err, min_entries):
    df = pd.read_csv(path)
    ok = (np.isfinite(df.mu) & (df.mu_err < max_mu_err)
          & (df.n_entries >= min_entries))
    df = df[ok].copy()
    df["yc"] = 0.5 * (df.y_lo + df.y_hi)
    df["zc"] = 0.5 * (df.z_lo + df.z_hi)
    return df.set_index(["yc", "zc"])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits-dir", default="cathode_study_2cm_v4")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-mu-err", type=float, default=10.0,
                    help="drop pixels with CB mu_err above this (ticks); "
                         "same cut as cathode_step1")
    ap.add_argument("--min-entries", type=int, default=10)
    ap.add_argument("--hist-bin", type=float, default=2.0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    out = args.out or args.fits_dir
    os.makedirs(out, exist_ok=True)

    f0 = load_fits(os.path.join(args.fits_dir, "cathode_fit_results_tpc0.csv"),
                   args.max_mu_err, args.min_entries)
    f1 = load_fits(os.path.join(args.fits_dir, "cathode_fit_results_tpc1.csv"),
                   args.max_mu_err, args.min_entries)
    j = f0.join(f1, how="inner", lsuffix="_t0", rsuffix="_t1")
    T = (j.mu_t0 + j.mu_t1).to_numpy()
    Terr = np.hypot(j.mu_err_t0.to_numpy(), j.mu_err_t1.to_numpy())
    yc = np.array([i[0] for i in j.index])
    zc = np.array([i[1] for i in j.index])
    print(f"pixels: TPC0 {len(f0):,}, TPC1 {len(f1):,}, "
          f"matched {len(j):,}", flush=True)

    mean = float(np.mean(T))
    med = float(np.median(T))
    wmean = float(np.sum(T / Terr**2) / np.sum(1.0 / Terr**2))
    rms = float(np.std(T))

    # ---- the requested histogram: one entry per pixel -----------------------
    lo = np.floor(np.percentile(T, 0.2)) - 10
    hi = np.ceil(np.percentile(T, 99.8)) + 10
    bins = np.arange(lo, hi + args.hist_bin, args.hist_bin)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.hist(T, bins=bins, color="steelblue", edgecolor="navy", lw=0.3)
    for val, color, name in ((mean, "red", f"mean {mean:.1f}"),
                             (med, "darkorange", f"median {med:.1f}"),
                             (wmean, "green", f"wtd mean {wmean:.1f}")):
        ax.axvline(val, color=color, ls="--", lw=1.2, label=name)
    txt = []
    for S in SPANS_CM:
        txt.append(f"span {S:5.1f} cm: v = {S/(med*TICK_US):.5f} (median) / "
                   f"{S/(mean*TICK_US):.5f} (mean) cm/us")
    ax.text(0.02, 0.97, "\n".join(txt), transform=ax.transAxes, va="top",
            fontsize=8.5, family="monospace")
    ax.set_xlabel(r"per-pixel $\mu_{TPC0} + \mu_{TPC1}$ [ticks]")
    ax.set_ylabel(f"pixels / {args.hist_bin:g} ticks")
    ax.set_title(f"per-pixel summed drift-time length "
                 f"({len(j):,} matched 2x2 cm$^2$ pixels) {args.label}\n"
                 f"mean {mean:.2f}  median {med:.2f}  rms {rms:.2f} ticks")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pixel_dtsum_hist.png"), dpi=120)
    fig.savefig(os.path.join(out, "pixel_dtsum_hist.pdf"))
    plt.close(fig)

    # ---- map of the per-pixel sums ------------------------------------------
    ys = np.unique(yc); zs = np.unique(zc)
    grid = np.full((len(ys), len(zs)), np.nan)
    iy = np.searchsorted(ys, yc); iz = np.searchsorted(zs, zc)
    grid[iy, iz] = T
    fig, ax = plt.subplots(figsize=(11, 6.5))
    im = ax.imshow(grid, origin="lower", aspect="auto",
                   extent=(zs[0] - 1, zs[-1] + 1, ys[0] - 1, ys[-1] + 1),
                   cmap="RdYlBu_r",
                   vmin=np.nanpercentile(grid, 2),
                   vmax=np.nanpercentile(grid, 98), interpolation="nearest")
    fig.colorbar(im, ax=ax, label=r"$\mu_{TPC0}+\mu_{TPC1}$ [ticks]")
    ax.set_xlabel("Z [cm]"); ax.set_ylabel("Y [cm]")
    ax.set_title(f"per-pixel summed drift-time length {args.label}")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pixel_dtsum_map.png"), dpi=120)
    plt.close(fig)

    pd.DataFrame(dict(yc=yc, zc=zc, dtsum=T, dtsum_err=Terr)).to_csv(
        os.path.join(out, "pixel_dtsum.csv"), index=False)

    print(f"per-pixel sum: mean {mean:.2f}  median {med:.2f}  "
          f"wtd-mean {wmean:.2f}  rms {rms:.2f} ticks "
          f"({mean*TICK_US:.1f} us mean)")
    for line in txt:
        print(" ", line)
    print(f"wrote {out}/pixel_dtsum_hist.png, pixel_dtsum_map.png, "
          f"pixel_dtsum.csv", flush=True)


if __name__ == "__main__":
    main()
