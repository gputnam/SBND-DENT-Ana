"""6-pane NS-vs-EW comparison plots for the CRT-displacement voxel maps.

For each displacement component (dx, dy, dz) and each X slice, produce a
3-row x 2-col figure:

    columns: NS-only (left)            | EW-only (right)
    rows:    no alignment       (top)
             wall, GDML         (middle)
             module, GDML       (bottom)

Reads voxel_data.npz from the three crtdisp_maps_20cm* directories.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS = [
    ("no alignment", "crtdisp_maps_20cm"),
    ("wall, GDML",   "crtdisp_maps_20cm_wall_gdml"),
    ("module, GDML", "crtdisp_maps_20cm_module_gdml"),
]

SUBSETS = ("ns", "ew")
COMPONENTS = ("dx", "dy", "dz")

VRANGE = 5.0
MIN_COUNTS = 30


def load(npz_path):
    return np.load(npz_path) if os.path.exists(npz_path) else None


def voxel_extent_yz(d):
    return (d["z_edges"][0], d["z_edges"][-1],
            d["y_edges"][0], d["y_edges"][-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="compare_ns_ew",
                    help="output directory")
    ap.add_argument("--x-slice", type=int, default=None,
                    help="If given, only this X slice index. Otherwise all.")
    ap.add_argument("--vrange", type=float, default=VRANGE,
                    help="Color half-range for mu maps (cm).")
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

    any_d = next(iter(datasets.values()))
    nx = any_d[f"mu_{SUBSETS[0]}_{COMPONENTS[0]}"].shape[0]
    x_edges = any_d["x_edges"]
    if args.x_slice is None:
        x_slices = list(range(nx))
    else:
        x_slices = [args.x_slice]

    cmap = matplotlib.colormaps["RdYlBu_r"].copy()
    cmap.set_bad("white")

    n_rows = len(datasets)
    n_cols = len(SUBSETS)

    print(f"Generating {len(COMPONENTS)} components x {len(x_slices)} "
          f"x-slices = {len(COMPONENTS)*len(x_slices)} figures "
          f"into {args.out}/")

    for comp in COMPONENTS:
        for ix in x_slices:
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(5.5 * n_cols, 4.5 * n_rows),
                                     sharex=True, sharey=True)
            if n_rows == 1:
                axes = np.array([axes])
            if n_cols == 1:
                axes = axes.reshape(-1, 1)
            im = None
            for r, (label, d) in enumerate(datasets.items()):
                for c, sub in enumerate(SUBSETS):
                    ax = axes[r, c]
                    grid = d[f"mu_{sub}_{comp}"][ix].copy()
                    cs = d[f"count_{sub}"][ix]
                    grid[cs < MIN_COUNTS] = np.nan
                    # grid is (ny, nz) — DO NOT transpose: imshow places
                    # rows on the vertical axis, cols on the horizontal,
                    # which already matches extent=(z_lo, z_hi, y_lo, y_hi).
                    im = ax.imshow(grid, origin="lower", aspect="equal",
                                   extent=voxel_extent_yz(d), cmap=cmap,
                                   vmin=-args.vrange, vmax=args.vrange,
                                   interpolation="nearest")
                    med = float(np.nanmedian(grid))
                    n_vox = int(np.isfinite(grid).sum())
                    title = (f"{label} - {sub.upper()} only\n"
                             f"mu({comp}) median = {med:+.2f} cm  "
                             f"(n_vox={n_vox})")
                    ax.set_title(title, fontsize=10)
                    if c == 0:
                        ax.set_ylabel("Y [cm]")
                    if r == n_rows - 1:
                        ax.set_xlabel("Z [cm]")
            cb = fig.colorbar(im, ax=axes, extend="both", shrink=0.85,
                              label=f"Gauss-fit mu({comp}) [cm]")
            x_lo, x_hi = float(x_edges[ix]), float(x_edges[ix + 1])
            fig.suptitle(f"NS vs EW comparison — {comp}, "
                         f"X slice {ix} (x in [{x_lo:.0f}, {x_hi:.0f}] cm)",
                         fontsize=13)
            out_path = os.path.join(
                args.out, f"compare6_{comp}_x{ix:02d}.png")
            fig.savefig(out_path, dpi=120)
            plt.close(fig)

    print(f"  wrote {args.out}/  ({len(COMPONENTS)*len(x_slices)} files)")


if __name__ == "__main__":
    main()
