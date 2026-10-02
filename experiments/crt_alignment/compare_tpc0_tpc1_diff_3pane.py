"""TPC0 - TPC1 cathode-slice difference plot, 3-pane comparison.

For each displacement component (dx, dy, dz) and each subset
(NS / EW / double-anode), compute per-(Y, Z) pixel:

    diff(Y, Z) = mu_TPC0_cathode_slice - mu_TPC1_cathode_slice

where the TPC0 cathode slice is the X-bin edges [-20, 0] cm and the TPC1
cathode slice is [0, +20] cm — the two voxels straddling the central
cathode at X=0. With the 20 cm cubic voxel binning these correspond to
ix=9 and ix=10 in the saved voxel_data.npz arrays.

A diff that's near zero means the field-distortion (or CRT-projection
systematic) is symmetric across the cathode. A diff that's non-zero
indicates an asymmetric residual between the two TPCs.

Sources (default; overridable via CLI):
    NS, EW       : crtdisp_maps_v3_20cm_unalign_singleg/voxel_data.npz
    double-anode : doubleanode_maps_20cm_singleg/voxel_data.npz

Output: <out>/{dx,dy,dz}.{png,pdf}  -- one 1x3 figure per component.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CRT_DIR = "crtdisp_maps_v3_20cm_unalign_singleg"
DEFAULT_CRT_LABEL = "unaligned (1G)"
DEFAULT_DA_DIR = "doubleanode_maps_20cm_singleg"
DEFAULT_OUT = "compare_tpc0_tpc1_diff_3pane"

# X-slice indices that straddle the cathode (X=0) at 20 cm voxel size.
TPC0_IX = 9   # X in [-20, 0]
TPC1_IX = 10  # X in [0, 20]

COMPONENTS = ("dx", "dy", "dz")

VRANGE = 5.0           # cm; symmetric color half-range for the diff maps
MIN_COUNTS = 30        # both sides must satisfy count >= this


def load(npz_path):
    return np.load(npz_path) if os.path.exists(npz_path) else None


def voxel_extent_yz(d):
    return (d["z_edges"][0], d["z_edges"][-1],
            d["y_edges"][0], d["y_edges"][-1])


def diff_slice(d, sub, comp, min_counts):
    """Return per-(Y, Z) diff = mu_TPC0 - mu_TPC1 with NaN where either
    side has insufficient counts."""
    mu = d[f"mu_{sub}_{comp}"]
    cs = d[f"count_{sub}"]
    mu0, mu1 = mu[TPC0_IX].copy(), mu[TPC1_IX].copy()
    cs0, cs1 = cs[TPC0_IX], cs[TPC1_IX]
    bad = (cs0 < min_counts) | (cs1 < min_counts) \
        | ~np.isfinite(mu0) | ~np.isfinite(mu1)
    diff = mu0 - mu1
    diff[bad] = np.nan
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output directory (default %(default)s)")
    ap.add_argument("--crt-dir", default=DEFAULT_CRT_DIR,
                    help="CRT-displacement voxel-map directory (default "
                         "%(default)s)")
    ap.add_argument("--crt-label", default=DEFAULT_CRT_LABEL,
                    help="short tag for the NS/EW columns (default "
                         "%(default)s)")
    ap.add_argument("--da-dir", default=DEFAULT_DA_DIR,
                    help="double-anode voxel-map directory (default "
                         "%(default)s)")
    ap.add_argument("--vrange", type=float, default=VRANGE,
                    help="color half-range for diff maps (cm)")
    ap.add_argument("--min-counts", type=int, default=MIN_COUNTS,
                    help="minimum hits per voxel on BOTH sides for a "
                         "valid diff pixel (default %(default)s)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sources = [
        (f"CRT NS-only ({args.crt_label})",  "ns",  args.crt_dir),
        (f"CRT EW-only ({args.crt_label})",  "ew",  args.crt_dir),
        ("Double-anode (CRT-free)",          "all", args.da_dir),
    ]

    npz_cache = {}
    columns = []
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

    ref = columns[0][2]
    nx_ref = ref[f"mu_{columns[0][1]}_{COMPONENTS[0]}"].shape[0]
    if max(TPC0_IX, TPC1_IX) >= nx_ref:
        raise SystemExit(
            f"X-slice indices ({TPC0_IX}, {TPC1_IX}) out of range for "
            f"nx={nx_ref}. Are the maps built at voxel-size 20?"
        )
    for label, sub, ds in columns:
        nx = ds[f"mu_{sub}_{COMPONENTS[0]}"].shape[0]
        if nx != nx_ref:
            raise SystemExit(
                f"voxel grids differ: column {label!r} has nx={nx} "
                f"(reference nx={nx_ref})."
            )

    x_edges = ref["x_edges"]
    x0_lo, x0_hi = float(x_edges[TPC0_IX]), float(x_edges[TPC0_IX + 1])
    x1_lo, x1_hi = float(x_edges[TPC1_IX]), float(x_edges[TPC1_IX + 1])
    print(f"TPC0 slice: X in [{x0_lo:+.0f}, {x0_hi:+.0f}] cm  (ix={TPC0_IX})")
    print(f"TPC1 slice: X in [{x1_lo:+.0f}, {x1_hi:+.0f}] cm  (ix={TPC1_IX})")

    cmap = matplotlib.colormaps["RdYlBu_r"].copy()
    cmap.set_bad("white")

    n_cols = len(columns)
    print(f"Generating {len(COMPONENTS)} components -> "
          f"{len(COMPONENTS)} figures into {args.out}/")

    for comp in COMPONENTS:
        fig, axes = plt.subplots(1, n_cols,
                                 figsize=(5.5 * n_cols, 4.5),
                                 sharex=True, sharey=True)
        if n_cols == 1:
            axes = np.array([axes])
        im = None
        for c, (label, sub, d) in enumerate(columns):
            ax = axes[c]
            diff = diff_slice(d, sub, comp, args.min_counts)
            im = ax.imshow(diff, origin="lower", aspect="equal",
                           extent=voxel_extent_yz(d), cmap=cmap,
                           vmin=-args.vrange, vmax=args.vrange,
                           interpolation="nearest")
            med = float(np.nanmedian(diff))
            n_vox = int(np.isfinite(diff).sum())
            title = (f"{label}\n"
                     f"diff median = {med:+.2f} cm  (n_vox={n_vox})")
            ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel("Y [cm]")
            ax.set_xlabel("Z [cm]")
        cb = fig.colorbar(im, ax=axes, extend="both", shrink=0.85,
                          label=f"$\\hat{{\\Delta}}_{{{comp[1]}}}$ "
                                f"TPC0 - $\\hat{{\\Delta}}_{{{comp[1]}}}$ TPC1 [cm]")
        fig.suptitle(f"TPC0 - TPC1 cathode-slice diff, {comp}  "
                     f"(TPC0: X in [{x0_lo:+.0f}, {x0_hi:+.0f}], "
                     f"TPC1: X in [{x1_lo:+.0f}, {x1_hi:+.0f}])",
                     fontsize=13)
        png = os.path.join(args.out, f"{comp}.png")
        pdf = os.path.join(args.out, f"{comp}.pdf")
        fig.savefig(png, dpi=120)
        fig.savefig(pdf)
        plt.close(fig)

    print(f"  wrote {args.out}/  ({len(COMPONENTS)*2} files; png + pdf)")


if __name__ == "__main__":
    main()
