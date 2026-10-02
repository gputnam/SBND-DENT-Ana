"""3-pane per-pixel comparison: NS-only | EW-only | double-anode.

For each displacement component (dx, dy, dz) and each X slice, produce a
1-row x 3-col figure:

    columns: NS-only (CRT, unaligned)   |
             EW-only (CRT, unaligned)   |
             double-anode (CRT-free)

Reads voxel_data.npz from:
    crtdisp_maps_v3_20cm_unalign        (NS, EW columns)
    doubleanode_maps_20cm                (double-anode column)

Both source maps must be at the same voxel size (default 20 cm cubic).
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Default CRT source: unaligned voxels. Override with --crt-dir.
DEFAULT_CRT_DIR = "crtdisp_maps_v3_20cm_unalign"
DEFAULT_CRT_LABEL = "unaligned"
DEFAULT_DA_DIR = "doubleanode_maps_20cm"
DEFAULT_OUT = "compare_ns_ew_doubleanode_3pane"

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
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output directory (default %(default)s)")
    ap.add_argument("--crt-dir", default=DEFAULT_CRT_DIR,
                    help="CRT-displacement voxel-map directory containing "
                         "voxel_data.npz with mu_ns_*, mu_ew_*, count_ns, "
                         "count_ew (default %(default)s)")
    ap.add_argument("--crt-label", default=DEFAULT_CRT_LABEL,
                    help="short tag inserted into the NS/EW column titles, "
                         "e.g. 'unaligned' or 'module-GDML' (default "
                         "%(default)s)")
    ap.add_argument("--da-dir", default=DEFAULT_DA_DIR,
                    help="double-anode voxel-map directory (default "
                         "%(default)s)")
    ap.add_argument("--x-slice", type=int, default=None,
                    help="if given, only this X-slice index; otherwise all")
    ap.add_argument("--vrange", type=float, default=VRANGE,
                    help="color half-range for mu maps (cm)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sources = [
        (f"CRT NS-only ({args.crt_label})", "ns",  args.crt_dir),
        (f"CRT EW-only ({args.crt_label})", "ew",  args.crt_dir),
        ("Double-anode (CRT-free)",         "all", args.da_dir),
    ]

    # Load each unique NPZ once and resolve column -> dataset.
    npz_cache = {}
    columns = []  # list of (label, subset_key, dataset)
    for label, sub, d in sources:
        if d not in npz_cache:
            path = os.path.join(d, "voxel_data.npz")
            z = load(path)
            if z is None:
                print(f"  WARNING: {path} not found, skipping column {label!r}")
                continue
            npz_cache[d] = z
        ds = npz_cache[d]
        if f"mu_{sub}_{COMPONENTS[0]}" not in ds.files:
            print(f"  WARNING: {d}/voxel_data.npz has no key "
                  f"mu_{sub}_{COMPONENTS[0]}, skipping column {label!r}")
            continue
        columns.append((label, sub, ds))

    if len(columns) < 2:
        raise SystemExit("Need at least 2 columns present.")

    # Sanity-check that the voxel grids match across the source NPZs.
    ref = columns[0][2]
    nx_ref = ref[f"mu_{columns[0][1]}_{COMPONENTS[0]}"].shape[0]
    for label, sub, ds in columns:
        nx = ds[f"mu_{sub}_{COMPONENTS[0]}"].shape[0]
        if nx != nx_ref:
            raise SystemExit(
                f"voxel grids differ: column {label!r} has nx={nx} "
                f"(reference nx={nx_ref}) -- rebuild both maps at the "
                f"same voxel size."
            )

    x_edges = ref["x_edges"]
    if args.x_slice is None:
        x_slices = list(range(nx_ref))
    else:
        x_slices = [args.x_slice]

    cmap = matplotlib.colormaps["RdYlBu_r"].copy()
    cmap.set_bad("white")

    n_cols = len(columns)
    print(f"Generating {len(COMPONENTS)} components x {len(x_slices)} "
          f"x-slices = {len(COMPONENTS)*len(x_slices)} figures into "
          f"{args.out}/")

    for comp in COMPONENTS:
        comp_dir = os.path.join(args.out, comp)
        os.makedirs(comp_dir, exist_ok=True)
        for ix in x_slices:
            fig, axes = plt.subplots(1, n_cols,
                                     figsize=(5.5 * n_cols, 4.5),
                                     sharex=True, sharey=True)
            if n_cols == 1:
                axes = np.array([axes])
            im = None
            for c, (label, sub, d) in enumerate(columns):
                ax = axes[c]
                grid = d[f"mu_{sub}_{comp}"][ix].copy()
                cs = d[f"count_{sub}"][ix]
                grid[cs < MIN_COUNTS] = np.nan
                # grid is (ny, nz) -- DO NOT transpose: imshow places
                # rows on the vertical axis, cols on the horizontal,
                # which already matches extent=(z_lo, z_hi, y_lo, y_hi).
                im = ax.imshow(grid, origin="lower", aspect="equal",
                               extent=voxel_extent_yz(d), cmap=cmap,
                               vmin=-args.vrange, vmax=args.vrange,
                               interpolation="nearest")
                med = float(np.nanmedian(grid))
                n_vox = int(np.isfinite(grid).sum())
                title = (f"{label}\n"
                         f"mu({comp}) median = {med:+.2f} cm  "
                         f"(n_vox={n_vox})")
                ax.set_title(title, fontsize=10)
                if c == 0:
                    ax.set_ylabel("Y [cm]")
                ax.set_xlabel("Z [cm]")
            cb = fig.colorbar(im, ax=axes, extend="both", shrink=0.85,
                              label=f"LSQ-inverted $\\hat{{\\Delta}}_{{{comp[1]}}}$ [cm]")
            x_lo, x_hi = float(x_edges[ix]), float(x_edges[ix + 1])
            fig.suptitle(f"NS vs EW ({args.crt_label}) vs double-anode "
                         f"-- {comp}, X slice {ix} "
                         f"(x in [{x_lo:.0f}, {x_hi:.0f}] cm)",
                         fontsize=13)
            png_path = os.path.join(comp_dir, f"x{ix:02d}.png")
            pdf_path = os.path.join(comp_dir, f"x{ix:02d}.pdf")
            fig.savefig(png_path, dpi=120)
            fig.savefig(pdf_path)
            plt.close(fig)

    print(f"  wrote {args.out}/  "
          f"({len(COMPONENTS)*len(x_slices)*2} files; png + pdf)")


if __name__ == "__main__":
    main()
