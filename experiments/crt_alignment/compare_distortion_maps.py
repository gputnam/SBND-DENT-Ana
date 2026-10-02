"""Compare CRT-displacement voxel maps before vs after alignment.

Reads voxel_data.npz from three directories and produces side-by-side
images of the per-voxel Gaussian-fit mu for each (subset, component).

Usage:
    python compare_distortion_maps.py
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS = [
    ("pre-alignment",  "crtdisp_maps_20cm"),
    ("wall, GDML",     "crtdisp_maps_20cm_wall_gdml"),
    ("module, GDML",   "crtdisp_maps_20cm_module_gdml"),
]

SUBSETS = ("side", "ns", "ew", "topside",
           "topn", "tops", "tope", "topw")
COMPONENTS = ("dx", "dy", "dz")

VRANGE = 5.0          # ±cm color half-range for mu maps
MIN_COUNTS = 30


def load(npz_path):
    return np.load(npz_path) if os.path.exists(npz_path) else None


def voxel_extent_yz(d):
    return (d["z_edges"][0], d["z_edges"][-1],
            d["y_edges"][0], d["y_edges"][-1])


def median_mu_in_volume(d, sub, comp, mask=None):
    g = d[f"mu_{sub}_{comp}"]
    c = d[f"count_{sub}"]
    valid = np.isfinite(g) & (c >= MIN_COUNTS)
    if mask is not None:
        valid &= mask
    if not valid.any():
        return float("nan"), 0
    return float(np.median(g[valid])), int(valid.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="compare_distortion",
                    help="output directory")
    ap.add_argument("--x-slice-idx", type=int, default=10,
                    help="x slice index (default 10 = x in [0, 20] cm)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    datasets = {}
    for label, d in VARIANTS:
        path = os.path.join(d, "voxel_data.npz")
        z = load(path)
        if z is None:
            print(f"  WARNING: {path} not found, skipping")
            continue
        datasets[label] = z

    if len(datasets) < 2:
        raise SystemExit("Need at least 2 variants present.")

    # 1. Per-(subset, component) median |mu| across the volume.
    print("\nMedian |mu| (cm) of fitted voxel residuals — central X slice "
          f"(idx={args.x_slice_idx}, x ∈ [{-200 + args.x_slice_idx*20}, "
          f"{-200 + (args.x_slice_idx+1)*20}] cm):")
    print(f"  {'subset':<8s}  {'comp':<3s}  ", end="")
    for label in datasets.keys():
        print(f"{label:>20s}", end="")
    print()

    summary_rows = []
    for sub in SUBSETS:
        for comp in COMPONENTS:
            print(f"  {sub:<8s}  {comp:<3s}  ", end="")
            row = {"subset": sub, "component": comp}
            for label, d in datasets.items():
                slc = d[f"mu_{sub}_{comp}"][args.x_slice_idx]
                cs = d[f"count_{sub}"][args.x_slice_idx]
                v = slc[np.isfinite(slc) & (cs >= MIN_COUNTS)]
                if v.size == 0:
                    print(f"{'—':>20s}", end="")
                    row[f"{label}_med_mu"] = np.nan
                    row[f"{label}_n_vox"] = 0
                else:
                    print(f"{f'{np.median(v):+.2f} ({v.size})':>20s}", end="")
                    row[f"{label}_med_mu"] = float(np.median(v))
                    row[f"{label}_n_vox"] = int(v.size)
            summary_rows.append(row)
            print()

    # 2. Side-by-side images per (subset, component) for the chosen X slice.
    cmap = matplotlib.colormaps["RdYlBu_r"].copy()
    cmap.set_bad("white")
    for sub in SUBSETS:
        for comp in COMPONENTS:
            fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5),
                                     sharey=True)
            if len(datasets) == 1:
                axes = [axes]
            for ax, (label, d) in zip(axes, datasets.items()):
                grid = d[f"mu_{sub}_{comp}"][args.x_slice_idx].copy()
                cs = d[f"count_{sub}"][args.x_slice_idx]
                grid[cs < MIN_COUNTS] = np.nan
                im = ax.imshow(grid.T, origin="lower", aspect="equal",
                               extent=voxel_extent_yz(d), cmap=cmap,
                               vmin=-VRANGE, vmax=VRANGE,
                               interpolation="nearest")
                ax.set_xlabel("Y [cm]")
                ax.set_title(f"{label}\nmu({comp}) median = "
                             f"{np.nanmedian(grid):+.2f} cm")
            axes[0].set_ylabel("Z [cm]")
            cb = fig.colorbar(im, ax=axes, extend="both", shrink=0.85,
                              label=f"Gauss-fit mu({comp}) [cm]")
            fig.suptitle(f"{sub} subset, X slice {args.x_slice_idx} "
                         f"(x in [{-200+args.x_slice_idx*20}, "
                         f"{-200+(args.x_slice_idx+1)*20}] cm)")
            for ext in ("png",):
                fig.savefig(os.path.join(
                    args.out, f"compare_{sub}_{comp}_x{args.x_slice_idx:02d}.{ext}"),
                    dpi=120)
            plt.close(fig)

    # 3. Save summary.
    import pandas as pd
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.out, "median_mu_summary.csv"), index=False)
    print(f"\n  wrote {args.out}/")


if __name__ == "__main__":
    main()
