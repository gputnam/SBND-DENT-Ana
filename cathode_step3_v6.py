"""STEP 3 — assemble the v6 ASCII cathode file from the step-2 spline
fits.

Uses the user-selected mapping:
    x_neg(y, z)  =  spline_fit_TPC0(y, z)  -  panel_half_thickness
    x_pos(y, z)  =  spline_fit_TPC1(y, z)  +  panel_half_thickness

Pixels are NMIN-masked: a v6 (y, z) bin needs at least `--nmin` native
2-cm pixels with finite measurement within `--search-cm` of the bin
centre to be filled; otherwise the cell is NaN. The NMIN test is run on
step-1's measured grid (i.e. it counts how many native pixels actually
had a valid CB fit in that neighbourhood, not how many the spline could
extrapolate over).
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


V6_NY, V6_NZ = 160, 200
V6_Y_MIN, V6_Y_MAX = -200.0, +200.0
V6_Z_MIN, V6_Z_MAX =    0.0, +500.0
V6_BIN = 2.5


def v6_nmin_mask(x_measured_native, y_native, z_native, *,
                 nmin, search_cm):
    y_v6 = V6_Y_MIN + (np.arange(V6_NY) + 0.5) * V6_BIN
    z_v6 = V6_Z_MIN + (np.arange(V6_NZ) + 0.5) * V6_BIN
    mask = np.zeros((V6_NY, V6_NZ), dtype=bool)
    finite = np.isfinite(x_measured_native)
    for jy, yc in enumerate(y_v6):
        sel_y = np.abs(y_native - yc) <= search_cm
        if not sel_y.any():
            continue
        slab = finite[sel_y, :]
        for jz, zc in enumerate(z_v6):
            sel_z = np.abs(z_native - zc) <= search_cm
            if not sel_z.any():
                continue
            cell = slab[:, sel_z]
            if cell.sum() >= nmin:
                mask[jy, jz] = True
    return mask, y_v6, z_v6


def write_v6(out_path, x_neg, x_pos, *, header_extra):
    with open(out_path, "w") as f:
        f.write("# SBND cathode v6 volume, exported from drift-time analysis\n")
        f.write("# Units: centimetres.  Coordinates are (x,y,z) with x normal to cathode.\n")
        f.write("# A point (x,y,z) is *inside* the cathode volume when\n")
        f.write("#    Y_MIN <= y <= Y_MAX,  Z_MIN <= z <= Z_MAX,\n")
        f.write("#    x_neg(y,z) <= x <= x_pos(y,z),\n")
        f.write("# where x_neg and x_pos are read/interpolated from the grids below.\n")
        f.write("# A NaN value in either grid means \"no data\" at that (y,z) bin;\n")
        f.write("# the C++ loader treats NaN as \"outside the volume\".\n")
        for line in header_extra.splitlines():
            f.write(f"# {line}\n")
        f.write(f"VERSION  1\nNY       {V6_NY}\nNZ       {V6_NZ}\n")
        f.write(f"Y_MIN    {V6_Y_MIN:.6f}\nY_MAX    {V6_Y_MAX:.6f}\n")
        f.write(f"Z_MIN    {V6_Z_MIN:.6f}\nZ_MAX    {V6_Z_MAX:.6f}\n")
        f.write(f"BIN_CM   {V6_BIN:.6f}\n")
        for tag, grid in (("X_NEG", x_neg), ("X_POS", x_pos)):
            f.write(tag + "\n")
            for row in grid:
                f.write(" ".join("nan" if not np.isfinite(v) else f"{v:.4f}"
                                 for v in row) + "\n")


def plot_grid(grid, title, out_png, *, vmin, vmax,
              cmap="RdYlBu_r", cb_label="x [cm]"):
    cm = matplotlib.colormaps[cmap].copy()
    cm.set_bad("white")
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid, origin="lower", aspect="equal",
                   extent=(V6_Z_MIN, V6_Z_MAX, V6_Y_MIN, V6_Y_MAX),
                   cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.colorbar(im, ax=ax, extend="both", label=cb_label)
    ax.set_xlabel("Z [cm]"); ax.set_ylabel("Y [cm]"); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1-dir",
                    default="/exp/sbnd/data/users/gputnam/efield-study/cathode_pipeline/step1_displacement")
    ap.add_argument("--step2-dir",
                    default="/exp/sbnd/data/users/gputnam/efield-study/cathode_pipeline/step2_spline")
    ap.add_argument("--outdir",
                    default="/exp/sbnd/data/users/gputnam/efield-study/cathode_pipeline/step3_v6")
    ap.add_argument("--panel-half-thickness", type=float, default=1.5,
                    help="half-thickness offset (cm) added to x_pos and "
                         "subtracted from x_neg (default %(default)s)")
    ap.add_argument("--nmin", type=int, default=5,
                    help="min native pixels with finite measurement within "
                         "--search-cm of each v6 bin centre to keep it "
                         "(default %(default)s)")
    ap.add_argument("--search-cm", type=float, default=5.0,
                    help="search radius for the NMIN test (default %(default)s)")
    ap.add_argument("--vrange-cm", type=float, default=2.5)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.outdir, "png"), exist_ok=True)

    grids = {}
    for tpc in (0, 1):
        meas_npz = np.load(os.path.join(args.step1_dir,
                                         f"tpc{tpc}_displacement.npz"))
        fit_npz  = np.load(os.path.join(args.step2_dir,
                                         f"tpc{tpc}_spline.npz"))
        mask, y_v6, z_v6 = v6_nmin_mask(
            meas_npz["x"], meas_npz["y_centers"], meas_npz["z_centers"],
            nmin=args.nmin, search_cm=args.search_cm)
        x_v6 = fit_npz["x_fit_v6"].astype(np.float64)
        x_v6 = np.where(mask, x_v6, np.nan)
        grids[tpc] = x_v6
        n_keep = int(mask.sum())
        print(f"  TPC{tpc}: NMIN mask keeps {n_keep:,d} / {mask.size:,d} v6 pixels "
              f"  fit range [{np.nanmin(x_v6):+.3f}, {np.nanmax(x_v6):+.3f}] cm",
              flush=True)

    h = float(args.panel_half_thickness)
    x_neg = grids[0] - h
    x_pos = grids[1] + h

    both = np.isfinite(x_neg) & np.isfinite(x_pos)
    flip = int(np.sum(both & (x_neg > x_pos)))
    print(f"  pixels with x_neg > x_pos: {flip:,d} / {int(both.sum()):,d}",
          flush=True)

    out_txt = os.path.join(args.outdir, "sbnd_cathode_v6_drift.txt")
    write_v6(out_txt, x_neg, x_pos,
             header_extra=(
                 f"Built from drift-time analysis (stepped pipeline).\n"
                 f"x_neg = spline_fit_TPC0(y,z) - {h:.2f} cm\n"
                 f"x_pos = spline_fit_TPC1(y,z) + {h:.2f} cm"))
    print(f"  wrote {out_txt}", flush=True)

    plot_grid(grids[0], f"TPC0 spline fit on v6 grid (cm)",
              os.path.join(args.outdir, "png", "x_TPC0_fit.png"),
              vmin=-args.vrange_cm, vmax=+args.vrange_cm)
    plot_grid(grids[1], f"TPC1 spline fit on v6 grid (cm)",
              os.path.join(args.outdir, "png", "x_TPC1_fit.png"),
              vmin=-args.vrange_cm, vmax=+args.vrange_cm)
    plot_grid(x_neg, f"x_neg = TPC0 - {h:.1f} cm",
              os.path.join(args.outdir, "png", "x_neg.png"),
              vmin=-args.vrange_cm, vmax=+args.vrange_cm)
    plot_grid(x_pos, f"x_pos = TPC1 + {h:.1f} cm",
              os.path.join(args.outdir, "png", "x_pos.png"),
              vmin=-args.vrange_cm, vmax=+args.vrange_cm)
    plot_grid(x_pos - x_neg, "x_pos - x_neg [cm] (panel thickness)",
              os.path.join(args.outdir, "png", "thickness.png"),
              vmin=0, vmax=2 * h + args.vrange_cm, cmap="viridis")
    plot_grid(grids[0] + grids[1],
              "x_TPC0_fit + x_TPC1_fit [cm] (consistency; ideally near 0)",
              os.path.join(args.outdir, "png", "consistency.png"),
              vmin=-args.vrange_cm, vmax=+args.vrange_cm)

    print(f"\nDone. {args.outdir}/", flush=True)


if __name__ == "__main__":
    main()
