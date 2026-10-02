#!/usr/bin/env python3
"""1D expectation-vs-measurement profile plots for EACH iteration of the
ramp calibration.

The voxelizer (--x-ramp-correct hits) stores every iteration's map as
`delta_mu_all_iter{N}` in voxel_data.npz. This script draws, per iteration,
the y/z-averaged dx/dy/dz profiles vs X (measured TPC0/TPC1) against the
truth field (dual-map TrueFwd_*, sampled at voxel centers), plus a
dx-evolution overlay showing the ramp flattening across iterations.

dy/dz are ramp-invariant (the correction only rescales the drift
coordinate), so only dx evolves -- the overlay makes that explicit.

Usage:
    python make_iter_profiles.py --input <maps>/voxel_data.npz \
        --truth-root SCEoffsets_SBND_E500_dualmap_voxelTH3.root \
        --out iter_profiles [--label TAG]
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sce_input as B
import compare_2d_vs_truth as C

COMPONENTS = ("dx", "dy", "dz")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--truth-root",
                    default="SCEoffsets_SBND_E500_dualmap_voxelTH3.root")
    ap.add_argument("--out", default="iter_profiles")
    ap.add_argument("--label", default="")
    ap.add_argument("--min-counts", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.input)
    iters = sorted(int(re.search(r"iter(\d+)", k).group(1))
                   for k in d.files if re.match(r"delta_mu_all_iter\d+$", k))
    if not iters:
        sys.exit("no delta_mu_all_iter* maps in the npz (need a "
                 "--x-ramp-correct run with the per-iteration save)")
    vx = B.voxel_centers(d["x_edges"])
    vy = B.voxel_centers(d["y_edges"])
    vz = B.voxel_centers(d["z_edges"])
    cnt = d["count_all"]
    truth = C.sample_truth_at(vx, vy, vz, args.truth_root, unflip_x=False)
    tpc0 = (vx < 0)

    def prof(arr, keep, comp_i, tsel):
        a = np.where(keep, arr[..., comp_i], np.nan)
        a = a[tsel] if tsel is not None else a
        return np.nanmean(a, axis=(1, 2))

    # per-iteration 3-panel grids
    dx_curves = {}
    for it in iters:
        mu = d[f"delta_mu_all_iter{it}"].astype(np.float64)
        keep = np.isfinite(mu).all(-1) & (cnt >= args.min_counts)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ci, (ax, comp) in enumerate(zip(axes, COMPONENTS)):
            for tname, tsel, col in (("TPC0", tpc0, "tab:blue"),
                                     ("TPC1", ~tpc0, "tab:red")):
                vxx = vx[tsel]
                mp = prof(mu, keep, ci, tsel)
                tp = prof(truth, keep, ci, tsel)
                ax.plot(vxx, mp, "o-", ms=3.5, color=col, label=f"meas {tname}")
                ax.plot(vxx, tp, "--", color=col, lw=1.4, alpha=0.7,
                        label=f"truth {tname}")
            ax.axhline(0, color="0.6", ls=":", lw=0.7)
            ax.axvline(0, color="0.85", lw=0.6)
            ax.set_xlabel("X [cm]")
            ax.set_ylabel(f"$\\Delta_{{{comp[1]}}}$ [cm]")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle(f"ramp iteration {it}  {args.label}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"profiles_iter{it:02d}.png"),
                    dpi=120)
        plt.close(fig)
        # dx curves for the overlay (full detector, both TPCs)
        keepA = np.isfinite(mu).all(-1) & (cnt >= args.min_counts)
        dx_curves[it] = prof(mu, keepA, 0, None)

    # dx-evolution overlay (per TPC)
    keepF = (np.isfinite(d[f"delta_mu_all_iter{iters[-1]}"]).all(-1)
             & (cnt >= args.min_counts))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    cmap = matplotlib.colormaps["viridis"]
    for ax, (tname, tsel) in zip(axes, (("TPC0", tpc0), ("TPC1", ~tpc0))):
        vxx = vx[tsel]
        for i, it in enumerate(iters):
            mu = d[f"delta_mu_all_iter{it}"].astype(np.float64)
            mp = prof(mu, keepF, 0, tsel)
            ax.plot(vxx, mp, "o-", ms=3, color=cmap(i / max(len(iters)-1, 1)),
                    label=f"iter {it}")
        tp = prof(truth, keepF, 0, tsel)
        ax.plot(vxx, tp, "r--", lw=2, label="truth")
        ax.axhline(0, color="0.6", ls=":", lw=0.7)
        ax.set_xlabel("X [cm]")
        ax.set_title(tname)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$\Delta_x$ [cm]")
    fig.suptitle(f"dx ramp-calibration convergence by iteration  {args.label}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "dx_evolution.png"), dpi=120)
    plt.close(fig)
    print(f"wrote {len(iters)} per-iteration profile grids + dx_evolution.png "
          f"to {args.out}/ (iterations {iters})", flush=True)


if __name__ == "__main__":
    main()
