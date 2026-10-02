"""Build the SBND cathode v6 ASCII grid from the hit-based per-pixel
cathode-tick analysis (cathode_fit_results_tpc{0,1}.csv).

Pipeline:
    1. Load tpc0/tpc1 csvs (cathode_study_*) and pivot to 2D mu, mu_err
       grids on the native pixel grid.
    2. Drop outliers per-TPC: pixels with mu_err above a floor are NaN'd;
       remaining pixels checked against the 8-neighbor median MAD.
    3. Fit a 2D cubic LSQBivariateSpline per TPC (1/mu_err^2 weights).
    4. Convert ticks -> cm with the v6 sign convention:
         x_TPC0 = +(2/3) * v_drift * tick_period * (mu0 - <mu0>_global)
         x_TPC1 = -(2/3) * v_drift * tick_period * (mu1 - <mu1>_global)
       The (2/3) is from Taylor-expanding v ~ sqrt(E) = sqrt(V/L) about
       L_nom: a cathode shift delta_x changes L_eff by delta_x AND
       v_eff by -(1/2)delta_x/L_nom, so the drift time changes by
       (3/2) delta_x / v_nom. v_drift defaults to the empirical
       0.15748 cm/us (measured 4m / <dt0+dt1>) per CLAUDE.md.
    5. Evaluate the splines on the v6 grid (NY=160, NZ=200; 2.5 cm bins;
       Y in [-200,+200], Z in [0,+500]).
    6. Apply the v6 NMIN rule: pixels with <NMIN raw measurement pixels
       within +/-5 cm of the bin centre are set to NaN.
    7. Assemble x_neg = x_TPC0 (TPC0-derived), x_pos = x_TPC1
       (TPC1-derived). Flag any pixel where x_neg > x_pos.
    8. Write sbnd_cathode_v6_drift.txt with the v6 header.
    9. Reference plots.

Usage:
    python build_cathode_v6.py                       # uses defaults below
    python build_cathode_v6.py --csv-dir DIR --outdir DIR
        [--v-drift 0.15748] [--ny-knots 8 --nz-knots 12]
        [--mad-k 5] [--err-floor 10]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import LSQBivariateSpline
from scipy.signal import medfilt2d

# v6 ASCII grid parameters - documented in sbnd_cathode_v6_handoff.md and
# baked into the C++ loader. DO NOT CHANGE.
V6_NY, V6_NZ = 160, 200
V6_Y_MIN, V6_Y_MAX = -200.0, +200.0
V6_Z_MIN, V6_Z_MAX =    0.0, +500.0
V6_BIN = 2.5

# Default tick-to-cm conversion. SBND ClockSpeedTPC = 2 MHz so 0.5 us/tick.
# v_drift from the measured 4m / <dt0+dt1> in drift_df_full_combined.parquet
# (CLAUDE.md "v_drift sanity reference" section, 2026-05-12).
TICK_PERIOD_US = 0.5
V_DRIFT_DEFAULT = 0.15748  # cm/us

# (2/3) factor from Taylor expanding v ~ sqrt(E) = sqrt(V/L) — see plan.
DRIFT_TIME_FACTOR = 2.0 / 3.0

# native cathode-pixel grid from plot_cathode_heatmap.py at --pixel-size 2 cm.
NATIVE_Y_LO, NATIVE_Y_HI = -200.0, +200.0
NATIVE_Z_LO, NATIVE_Z_HI =    0.0, +500.0
NATIVE_BIN = 2.0


def csv_to_grids(csv_path):
    """Load one TPC's CSV and pivot to (ny, nz) mu and mu_err grids on
    the native 2-cm pixel grid."""
    df = pd.read_csv(csv_path)
    # use the y_lo/z_lo to find the pixel index
    iy = ((df["y_lo"].to_numpy() - NATIVE_Y_LO) / NATIVE_BIN).astype(int)
    iz = ((df["z_lo"].to_numpy() - NATIVE_Z_LO) / NATIVE_BIN).astype(int)
    ny = int(round((NATIVE_Y_HI - NATIVE_Y_LO) / NATIVE_BIN))
    nz = int(round((NATIVE_Z_HI - NATIVE_Z_LO) / NATIVE_BIN))
    mu = np.full((ny, nz), np.nan, dtype=np.float64)
    mu_err = np.full((ny, nz), np.nan, dtype=np.float64)
    n_ent  = np.zeros((ny, nz), dtype=np.int64)
    in_grid = (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
    mu[iy[in_grid], iz[in_grid]] = df["mu"].to_numpy()[in_grid]
    mu_err[iy[in_grid], iz[in_grid]] = df["mu_err"].to_numpy()[in_grid]
    n_ent[iy[in_grid], iz[in_grid]] = df["n_entries"].to_numpy()[in_grid]
    return mu, mu_err, n_ent


def mad_outlier_mask(grid, *, k_mad, abs_floor):
    """8-neighbor MAD outlier mask. Returns True where the pixel passes."""
    g = grid.copy()
    # neighbor median (3x3) ignoring NaN; treat NaN as missing via medfilt
    # over a NaN-replaced copy is imperfect — use a hand-rolled 3x3 median.
    pad = np.pad(g, 1, constant_values=np.nan)
    stacks = []
    for dy in (-1, 0, 1):
        for dz in (-1, 0, 1):
            if dy == 0 and dz == 0:
                continue
            stacks.append(pad[1+dy:1+dy+g.shape[0], 1+dz:1+dz+g.shape[1]])
    nbhd = np.stack(stacks, axis=0)
    med = np.nanmedian(nbhd, axis=0)
    mad = np.nanmedian(np.abs(nbhd - med[None, ...]), axis=0)
    threshold = np.maximum(k_mad * mad, abs_floor)
    keep = np.abs(g - med) <= threshold
    keep |= ~np.isfinite(med)       # if no neighbors, keep
    keep &= np.isfinite(g)
    return keep


def fit_2d_spline(mu, mu_err, *, ny_knots, nz_knots, err_floor, weight_floor):
    """LSQ cubic bivariate spline. Returns the fitted scipy spline object.

    mu, mu_err shape (ny, nz). Valid pixels have isfinite(mu) and pass
    upstream filters. Knots are uniform interior knots in y and z;
    LSQBivariateSpline expects the interior knots (not the clamped ones).
    """
    ny, nz = mu.shape
    y_centers = NATIVE_Y_LO + (np.arange(ny) + 0.5) * NATIVE_BIN
    z_centers = NATIVE_Z_LO + (np.arange(nz) + 0.5) * NATIVE_BIN
    Y, Z = np.meshgrid(y_centers, z_centers, indexing="ij")

    finite = np.isfinite(mu) & np.isfinite(mu_err)
    y_pts = Y[finite]
    z_pts = Z[finite]
    v_pts = mu[finite]
    e_pts = np.clip(mu_err[finite], err_floor, None)
    w_pts = np.clip(1.0 / e_pts, 0.0, 1.0 / weight_floor)

    y_int = np.linspace(NATIVE_Y_LO, NATIVE_Y_HI, ny_knots + 2)[1:-1]
    z_int = np.linspace(NATIVE_Z_LO, NATIVE_Z_HI, nz_knots + 2)[1:-1]

    spline = LSQBivariateSpline(
        y_pts, z_pts, v_pts, y_int, z_int, w=w_pts,
        bbox=[NATIVE_Y_LO, NATIVE_Y_HI, NATIVE_Z_LO, NATIVE_Z_HI],
        kx=3, ky=3,
    )
    return spline, y_centers, z_centers


def evaluate_on_v6(spline):
    """Evaluate the spline at the v6 bin centres (160x200)."""
    yv = V6_Y_MIN + (np.arange(V6_NY) + 0.5) * V6_BIN
    zv = V6_Z_MIN + (np.arange(V6_NZ) + 0.5) * V6_BIN
    grid = spline(yv, zv)  # shape (NY, NZ)
    return grid, yv, zv


def v6_nmin_mask(mu_native, y_v6, z_v6, *, nmin=5, search_cm=5.0):
    """For each v6 bin centre, count native pixels within +/- search_cm
    that have a finite measurement. Return True where >= nmin."""
    ny_n, nz_n = mu_native.shape
    y_native = NATIVE_Y_LO + (np.arange(ny_n) + 0.5) * NATIVE_BIN
    z_native = NATIVE_Z_LO + (np.arange(nz_n) + 0.5) * NATIVE_BIN
    mask = np.zeros((V6_NY, V6_NZ), dtype=bool)
    for jy, yc in enumerate(y_v6):
        sel_y = np.abs(y_native - yc) <= search_cm
        if not sel_y.any():
            continue
        slab = mu_native[sel_y, :]
        for jz, zc in enumerate(z_v6):
            sel_z = np.abs(z_native - zc) <= search_cm
            if not sel_z.any():
                continue
            cell = slab[:, sel_z]
            n = int(np.isfinite(cell).sum())
            mask[jy, jz] = (n >= nmin)
    return mask


def write_v6(out_path, x_neg, x_pos, *, header_extra=""):
    """Write the v6 ASCII file per sbnd_cathode_v6_handoff.md."""
    with open(out_path, "w") as f:
        f.write("# SBND cathode v6 volume, exported from drift-time analysis\n")
        f.write("# Units: centimetres.  Coordinates are (x,y,z) with x normal to cathode.\n")
        f.write("# A point (x,y,z) is *inside* the cathode volume when\n")
        f.write("#    Y_MIN <= y <= Y_MAX,  Z_MIN <= z <= Z_MAX,\n")
        f.write("#    x_neg(y,z) <= x <= x_pos(y,z),\n")
        f.write("# where x_neg and x_pos are read/interpolated from the grids below.\n")
        f.write("# A NaN value in either grid means \"no data\" at that (y,z) bin;\n")
        f.write("# the C++ loader treats NaN as \"outside the volume\".\n")
        if header_extra:
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


def plot_grid(grid, title, out_png, *, vrange=5.0, cmap_name="RdYlBu_r",
              y_min=V6_Y_MIN, y_max=V6_Y_MAX, z_min=V6_Z_MIN, z_max=V6_Z_MAX):
    cmap = matplotlib.colormaps[cmap_name].copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid,
                   origin="lower", aspect="equal",
                   extent=(z_min, z_max, y_min, y_max),
                   cmap=cmap, vmin=-vrange, vmax=vrange,
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, extend="both")
    ax.set_xlabel("Z [cm]")
    ax.set_ylabel("Y [cm]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="cathode_study_2cm_combined",
                    help="directory containing cathode_fit_results_tpc{0,1}.csv "
                         "(default: %(default)s)")
    ap.add_argument("--outdir",
                    default="/exp/sbnd/data/users/gputnam/efield-study/cathode_v6_doubleanode",
                    help="output dir (default %(default)s)")
    ap.add_argument("--v-drift", type=float, default=V_DRIFT_DEFAULT,
                    help="v_drift in cm/us for the tick-to-cm conversion "
                         "(default %(default)s, the measured 4m/<dt0+dt1>)")
    ap.add_argument("--err-floor", type=float, default=2.0,
                    help="floor on mu_err (ticks) used in the spline weights "
                         "(default %(default)s; pixels with err above the "
                         "--err-drop ceiling are dropped entirely)")
    ap.add_argument("--err-drop", type=float, default=10.0,
                    help="drop pixels with mu_err > this many ticks "
                         "(default %(default)s)")
    ap.add_argument("--mad-k", type=float, default=5.0,
                    help="MAD multiplier for the 8-neighbor outlier filter "
                         "(default %(default)s)")
    ap.add_argument("--mad-abs", type=float, default=20.0,
                    help="absolute floor on the 8-neighbor outlier filter "
                         "threshold, ticks (default %(default)s)")
    ap.add_argument("--ny-knots", type=int, default=8)
    ap.add_argument("--nz-knots", type=int, default=12)
    ap.add_argument("--nmin", type=int, default=5,
                    help="v6 NMIN: per v6 bin, need at least this many native "
                         "pixels within +/-5 cm to keep the value "
                         "(default %(default)s)")
    ap.add_argument("--vrange", type=float, default=2.0,
                    help="diagnostic-plot colour scale half-range in cm "
                         "(default %(default)s)")
    ap.add_argument("--centring", choices=("per_tpc", "global"),
                    default="global",
                    help="how to centre the per-TPC mu before applying the "
                         "(2/3) conversion. 'global' subtracts the universal "
                         "200cm/v_drift nominal (gives absolute cathode "
                         "displacement, preserves per-TPC systematic offset). "
                         "'per_tpc' subtracts the per-TPC global median "
                         "(centers each view at 0). (default %(default)s)")
    ap.add_argument("--panel-half-thickness", type=float, default=0.0,
                    help="optional additive shift: x_neg -= h, x_pos += h. "
                         "h is the half-thickness of the cathode panel in cm "
                         "(default %(default)s = no shift; try 1.5 cm to "
                         "give the panel a ~3 cm thickness like the v5/v6 "
                         "endpoint-based reference).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    png_dir = os.path.join(args.outdir, "png")
    os.makedirs(png_dir, exist_ok=True)

    # ---- load both TPCs
    tpc_csv = {
        0: os.path.join(args.csv_dir, "cathode_fit_results_tpc0.csv"),
        1: os.path.join(args.csv_dir, "cathode_fit_results_tpc1.csv"),
    }
    mu_t = {}
    err_t = {}
    nent_t = {}
    for t, path in tpc_csv.items():
        if not os.path.exists(path):
            print(f"ERROR: {path} not found", file=sys.stderr)
            sys.exit(1)
        mu, err, nent = csv_to_grids(path)
        # drop pixels with very large mu_err (failed fits)
        bad_err = ~np.isfinite(err) | (err > args.err_drop)
        mu[bad_err] = np.nan
        err[bad_err] = np.nan
        keep = mad_outlier_mask(mu, k_mad=args.mad_k, abs_floor=args.mad_abs)
        mu_t[t] = np.where(keep, mu, np.nan)
        err_t[t] = err
        nent_t[t] = nent
        print(f"TPC{t}: {np.isfinite(mu_t[t]).sum():,} pixels kept "
              f"(/{mu_t[t].size:,} grid) after err+MAD filter", flush=True)

    # ---- fit splines
    spline_t = {}
    for t in (0, 1):
        spline, _, _ = fit_2d_spline(mu_t[t], err_t[t],
                                     ny_knots=args.ny_knots,
                                     nz_knots=args.nz_knots,
                                     err_floor=args.err_floor,
                                     weight_floor=1.0)
        spline_t[t] = spline
        # quick residual report
        finite = np.isfinite(mu_t[t])
        y_c = NATIVE_Y_LO + (np.arange(mu_t[t].shape[0]) + 0.5) * NATIVE_BIN
        z_c = NATIVE_Z_LO + (np.arange(mu_t[t].shape[1]) + 0.5) * NATIVE_BIN
        fitted = spline(y_c, z_c)
        resid = (mu_t[t] - fitted)
        rms = float(np.sqrt(np.nanmean(resid[finite] ** 2)))
        print(f"  TPC{t} spline: fit RMS = {rms:.3f} ticks", flush=True)

    # ---- evaluate on v6 grid
    grids_v6 = {}
    for t in (0, 1):
        grid_ticks, yv, zv = evaluate_on_v6(spline_t[t])
        grids_v6[t] = grid_ticks
        print(f"  TPC{t} v6 grid: range {grid_ticks.min():.1f}..{grid_ticks.max():.1f} ticks",
              flush=True)

    # ---- absolute tick -> cm conversion
    # Reference: mu at the nominal flat cathode (x_c = 0). For a 200 cm
    # drift, mu_nominal = 200 / (v_drift * tick_period).
    cm_factor = DRIFT_TIME_FACTOR * args.v_drift * TICK_PERIOD_US
    mu_nom = 200.0 / (args.v_drift * TICK_PERIOD_US)
    print(f"\nv_drift = {args.v_drift:.5f} cm/us, tick period = {TICK_PERIOD_US} us, "
          f"factor (2/3)*v*dt = {cm_factor:.6f} cm/tick", flush=True)
    print(f"global nominal mu (cathode at x=0, flat) = "
          f"{mu_nom:.2f} ticks (independent of TPC)", flush=True)

    x_t = {}
    for t in (0, 1):
        med = float(np.nanmedian(grids_v6[t]))
        if args.centring == "per_tpc":
            mu_ref = med
        else:
            mu_ref = mu_nom
        sign = +1.0 if t == 0 else -1.0
        x_t[t] = sign * cm_factor * (grids_v6[t] - mu_ref)
        print(f"  TPC{t}: median mu = {med:.2f} ticks, mu_ref = {mu_ref:.2f}, "
              f"x range [{x_t[t].min():+.3f}, {x_t[t].max():+.3f}] cm",
              flush=True)

    # ---- apply NMIN mask
    nmin_mask = {}
    for t in (0, 1):
        m = v6_nmin_mask(mu_t[t], yv, zv,
                         nmin=args.nmin, search_cm=5.0)
        nmin_mask[t] = m
        n_keep = int(m.sum())
        print(f"  TPC{t} NMIN mask: keep {n_keep:,} / {m.size:,} v6 pixels",
              flush=True)

    x_neg = np.where(nmin_mask[0], x_t[0], np.nan) - args.panel_half_thickness
    x_pos = np.where(nmin_mask[1], x_t[1], np.nan) + args.panel_half_thickness

    # flag x_neg > x_pos cases (where the hard-coded TPC0/TPC1 mapping
    # produces an inconsistent panel ordering at that pixel)
    both = np.isfinite(x_neg) & np.isfinite(x_pos)
    flip = both & (x_neg > x_pos)
    print(f"  pixels with x_neg > x_pos (mapping flip): {int(flip.sum()):,} / "
          f"{int(both.sum()):,} pairs", flush=True)

    # Use the centring-shift approach (option (a) in the plan): nothing
    # to shift if the global median was the same on both TPCs (it should
    # be the same physical cathode); if the flip fraction is large we'd
    # need a per-TPC additive shift. Just report and move on.
    write_v6(os.path.join(args.outdir, "sbnd_cathode_v6_drift.txt"),
             x_neg, x_pos,
             header_extra=(
                 f"Built from drift-time analysis: "
                 f"x_neg=TPC0 estimate, x_pos=TPC1 estimate. "
                 f"v_drift = {args.v_drift:.5f} cm/us. "
                 f"(2/3) factor from v ~ sqrt(E) = sqrt(V/L)."))
    print(f"wrote {os.path.join(args.outdir, 'sbnd_cathode_v6_drift.txt')}",
          flush=True)

    # ---- plots
    plot_grid(grids_v6[0], "TPC0 cathode tick (mu) on v6 grid",
              os.path.join(png_dir, "mu_TPC0.png"),
              vrange=80.0, cmap_name="RdYlBu_r")
    plot_grid(grids_v6[1], "TPC1 cathode tick (mu) on v6 grid",
              os.path.join(png_dir, "mu_TPC1.png"),
              vrange=80.0, cmap_name="RdYlBu_r")
    plot_grid(x_neg, "x_neg (TPC0-derived cathode position, cm)",
              os.path.join(png_dir, "x_neg.png"),
              vrange=args.vrange)
    plot_grid(x_pos, "x_pos (TPC1-derived cathode position, cm)",
              os.path.join(png_dir, "x_pos.png"),
              vrange=args.vrange)
    # thickness
    thickness = x_pos - x_neg
    plot_grid(thickness, "x_pos - x_neg [cm]  (panel thickness)",
              os.path.join(png_dir, "thickness.png"),
              vrange=args.vrange, cmap_name="viridis")

    # cross-check: x_TPC0 + (-x_TPC1) should be small if both views agree.
    # x_t[1] already has the sign flip; x_t[0] + x_t[1] is the symmetric
    # quantity. Plot to inspect.
    consistency = x_t[0] + x_t[1]
    plot_grid(consistency, "x_TPC0 + x_TPC1 [cm]  (consistency; ideally near 0)",
              os.path.join(png_dir, "consistency.png"),
              vrange=args.vrange)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
