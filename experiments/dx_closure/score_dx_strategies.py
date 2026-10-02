#!/usr/bin/env python3
"""Unified scoring of dx-inversion strategies against the dual truth map.

Extends explore_inversion.py (linear per-voxel strategies recomputed from the
accumulator dump) with externally-produced maps -- the constrained global
spline (fit_global_spline_from_accum.py), the per-voxel median estimator
(diagnose_dx_offsets.py), map-feedback iteration runs
(plot_doubleanode_2d_voxels.py --iter-correct-positions), per-plane-t0
recalibrated runs, etc. All strategies are scored on the SAME fixed voxel
set (baseline cond<=100 & n>=min) with explore_inversion.metrics, and the
decisive 1D dx-vs-x profile overlay is drawn per TPC against the truth dome.

External maps are passed as repeatable  --map NAME=PATH  arguments; PATH is
an npz whose displacement array is auto-detected among keys
(delta_mu_all | spline_mu | delta_median | delta), shape (nx,ny,nz,3) on the
same voxel grid as the accumulator.

Usage:
    python score_dx_strategies.py --accum accumulator_mc_subset.npz \
        --truth-root SCEoffsets_SBND_E500_dualmap_voxelTH3.root \
        --map global_spline=global_spline_mc/spline_fit.npz \
        --map median=diagnose_dx_mc/diagnose_dx.npz \
        --out score_dx_mc
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import compare_2d_vs_truth as C
import explore_inversion as E

COMPONENTS = ("dx", "dy", "dz")
MAP_KEYS = ("delta_mu_all", "spline_mu", "delta_median", "delta")


def load_map(path):
    d = np.load(path)
    for k in MAP_KEYS:
        if k in d:
            return d[k].astype(np.float64)
    raise KeyError(f"none of {MAP_KEYS} in {path} (have {list(d.keys())})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accum", default="accumulator_mc_subset.npz")
    ap.add_argument("--truth-root",
                    default="SCEoffsets_SBND_E500_dualmap_voxelTH3.root")
    ap.add_argument("--apparent-truth", default=None,
                    help="npz from build_apparent_truth_map.py; replaces the "
                         "SCE-map truth with the TRUE APPARENT displacement "
                         "(the correct dx benchmark)")
    ap.add_argument("--out", default="score_dx_mc")
    ap.add_argument("--map", action="append", default=[],
                    metavar="NAME=PATH",
                    help="external strategy map npz (repeatable)")
    ap.add_argument("--min-counts", type=int, default=30)
    ap.add_argument("--cond-max", type=float, default=100.0)
    ap.add_argument("--x-ridge", type=float, default=0.1,
                    help="reference x-ridge strength from the previous study")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    M, v, n, sh, vx, vy, vz = E.build_MV(args.accum)
    if args.apparent_truth:
        truth = np.load(args.apparent_truth)["delta_mu_all"]\
            .astype(np.float64).reshape(-1, 3)
        print(f"benchmark: APPARENT truth ({args.apparent_truth})")
    else:
        truth = C.sample_truth_at(vx, vy, vz, args.truth_root,
                                  unflip_x=False).reshape(-1, 3)

    base = E.solve(M, v, R=1e-6 * n[:, None, None] * np.eye(3))
    cond = np.full(M.shape[0], np.inf)
    valid = n >= args.min_counts
    cond[valid] = np.linalg.cond(M[valid])
    keep = (valid & (cond <= args.cond_max) & np.isfinite(base).all(1)
            & np.isfinite(truth).all(1))
    print(f"voxels scored: {keep.sum()}", flush=True)

    ex = np.zeros((M.shape[0], 3, 3))
    ex[:, 0, 0] = n
    strategies = [
        ("baseline_pervoxel", base),
        (f"x_ridge_{args.x_ridge}", E.solve(M, v, args.x_ridge * ex)),
    ]
    for spec in args.map:
        name, path = spec.split("=", 1)
        strategies.append((name, load_map(path).reshape(-1, 3)))

    # ---- metric table -------------------------------------------------------
    records = []
    for name, delta in strategies:
        mr = E.metrics(delta, truth, keep, vx, sh)
        rec = {"strategy": name}
        for c in COMPONENTS:
            rec[f"{c}_slope"] = mr[(c, "all")][0]
            rec[f"{c}_rms"] = mr[(c, "all")][1]
            rec[f"{c}_slope_T0"] = mr[(c, "T0")][0]
            rec[f"{c}_slope_T1"] = mr[(c, "T1")][0]
            rec[f"{c}_rms_T0"] = mr[(c, "T0")][1]
            rec[f"{c}_rms_T1"] = mr[(c, "T1")][1]
        records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.out, "strategy_metrics.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df[["strategy", "dx_slope", "dx_rms", "dy_slope", "dy_rms",
              "dz_slope", "dz_rms"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"), flush=True)

    # ---- dx-vs-x profile overlay --------------------------------------------
    keep3 = keep.reshape(sh)
    t3 = truth.reshape(*sh, 3)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    cmap = matplotlib.colormaps["tab10"]
    for ci, (ax, comp) in enumerate(zip(axes, COMPONENTS)):
        for si, (name, delta) in enumerate(strategies):
            d3 = delta.reshape(*sh, 3)
            prof = np.nanmean(np.where(keep3[..., None], d3, np.nan)[..., ci],
                              axis=(1, 2))
            ax.plot(vx, prof, "o-", ms=3, lw=1.2, color=cmap(si % 10),
                    alpha=0.85, label=name)
        tprof = np.nanmean(np.where(keep3[..., None], t3, np.nan)[..., ci],
                           axis=(1, 2))
        ax.plot(vx, tprof, "r-", lw=2.5, label="truth (dual map)")
        ax.axhline(0, color="0.6", ls=":", lw=0.7)
        ax.axvline(0, color="0.8", ls="-", lw=0.6)
        ax.set_xlabel("X [cm]")
        ax.set_ylabel(f"$\\Delta_{{{comp[1]}}}$ [cm]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("dx-strategy comparison: y/z-averaged profiles vs X "
                 "(kept voxels only)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "profiles_vs_x.png"), dpi=120)
    fig.savefig(os.path.join(args.out, "profiles_vs_x.pdf"))
    plt.close(fig)
    print(f"wrote {args.out}/", flush=True)


if __name__ == "__main__":
    main()
