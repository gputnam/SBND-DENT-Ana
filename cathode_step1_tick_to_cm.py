"""STEP 1 — turn the per-pixel cathode-tick map into a per-TPC spatial
displacement map x_cathode(y, z) in cm.

Reads `cathode_fit_results_tpc{0,1}.csv` from a plot_cathode_heatmap.py
output directory. Drops pixels whose CB fit reported `mu_err > --err-drop`
(default 10 ticks). Converts the surviving `mu` (ticks) to a geometric
x-cathode displacement in cm using the (2/3) factor derived from
``v ~ sqrt(E) = sqrt(V/L)``:

    delta_x_c  =  (2/3) * v_drift * tick_period * (mu - mu_nominal)

The reference `mu_nominal = 200 / (v_drift * tick_period)` corresponds to
a flat cathode at the geometric x = 0. The sign convention is per-TPC:
TPC0 keeps the +sign (mu_0 increasing means cathode shifted toward +x =
TPC1), TPC1 gets the −sign (mu_1 increasing means cathode shifted toward
−x = TPC0).

No spatial smoothing, outlier filtering, or spline fitting at this stage —
that's step 2.

Outputs (under --outdir):
    tpc0_displacement.npz, tpc1_displacement.npz
        keys: x, mu, mu_err, n_entries, y_centers, z_centers, v_drift, ...
    png/tpc0_displacement.png, png/tpc1_displacement.png
        heatmaps of x in cm
    png/tpc0_mu.png, png/tpc1_mu.png
        heatmaps of mu in ticks (the input to the conversion)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TICK_PERIOD_US     = 0.5
DRIFT_TIME_FACTOR  = 2.0 / 3.0
V_DRIFT_DEFAULT    = 0.15542            # cm/us, from heatmap_v3

NATIVE_Y_LO, NATIVE_Y_HI = -200.0, +200.0
NATIVE_Z_LO, NATIVE_Z_HI =    0.0, +500.0
NATIVE_BIN = 2.0


def csv_to_grids(csv_path):
    df = pd.read_csv(csv_path)
    iy = ((df["y_lo"].to_numpy() - NATIVE_Y_LO) / NATIVE_BIN).astype(int)
    iz = ((df["z_lo"].to_numpy() - NATIVE_Z_LO) / NATIVE_BIN).astype(int)
    ny = int(round((NATIVE_Y_HI - NATIVE_Y_LO) / NATIVE_BIN))
    nz = int(round((NATIVE_Z_HI - NATIVE_Z_LO) / NATIVE_BIN))
    mu      = np.full((ny, nz), np.nan, dtype=np.float64)
    mu_err  = np.full((ny, nz), np.nan, dtype=np.float64)
    n_ent   = np.zeros((ny, nz), dtype=np.int64)
    in_grid = (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
    mu     [iy[in_grid], iz[in_grid]] = df["mu"].to_numpy()[in_grid]
    mu_err [iy[in_grid], iz[in_grid]] = df["mu_err"].to_numpy()[in_grid]
    n_ent  [iy[in_grid], iz[in_grid]] = df["n_entries"].to_numpy()[in_grid]
    y_centers = NATIVE_Y_LO + (np.arange(ny) + 0.5) * NATIVE_BIN
    z_centers = NATIVE_Z_LO + (np.arange(nz) + 0.5) * NATIVE_BIN
    return mu, mu_err, n_ent, y_centers, z_centers


def plot_grid(grid, title, out_png, *, vmin, vmax, cmap="RdYlBu_r",
              cb_label="x_cathode [cm]"):
    cm = matplotlib.colormaps[cmap].copy()
    cm.set_bad("white")
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid, origin="lower", aspect="equal",
                   extent=(NATIVE_Z_LO, NATIVE_Z_HI, NATIVE_Y_LO, NATIVE_Y_HI),
                   cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.colorbar(im, ax=ax, extend="both", label=cb_label)
    ax.set_xlabel("Z [cm]"); ax.set_ylabel("Y [cm]"); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir",
                    default="/exp/sbnd/data/users/gputnam/efield-study/cathode_study_2cm_v3",
                    help="dir with cathode_fit_results_tpc{0,1}.csv "
                         "(default: %(default)s)")
    ap.add_argument("--outdir",
                    default="/exp/sbnd/data/users/gputnam/efield-study/cathode_pipeline/step1_displacement")
    ap.add_argument("--v-drift", type=float, default=V_DRIFT_DEFAULT,
                    help="v_drift in cm/us (default %(default)s)")
    ap.add_argument("--err-drop", type=float, default=10.0,
                    help="drop pixels with mu_err > this many ticks "
                         "(default %(default)s)")
    ap.add_argument("--vrange-cm", type=float, default=2.5,
                    help="heatmap colour scale half-range in cm "
                         "(default %(default)s)")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.outdir, "png"), exist_ok=True)
    cm_factor = DRIFT_TIME_FACTOR * args.v_drift * TICK_PERIOD_US
    mu_nom    = 200.0 / (args.v_drift * TICK_PERIOD_US)
    print(f"v_drift = {args.v_drift:.5f} cm/us, tick_period = {TICK_PERIOD_US} us,",
          flush=True)
    print(f"  (2/3)*v*dt = {cm_factor:.6f} cm/tick,  "
          f"mu_nominal = {mu_nom:.2f} ticks", flush=True)

    for tpc in (0, 1):
        csv = os.path.join(args.csv_dir, f"cathode_fit_results_tpc{tpc}.csv")
        if not os.path.exists(csv):
            print(f"ERROR: {csv} not found", file=sys.stderr); sys.exit(1)
        mu, mu_err, n_ent, y_c, z_c = csv_to_grids(csv)

        # drop unreliable fits
        bad = ~np.isfinite(mu_err) | (mu_err > args.err_drop)
        mu_drop = mu.copy(); mu_drop[bad] = np.nan
        n_keep_pix = int(np.isfinite(mu_drop).sum())
        n_total_pix = mu.size
        n_dropped  = int((~bad & np.isfinite(mu)).sum() - n_keep_pix) if False else int(np.isfinite(mu).sum() - n_keep_pix)
        print(f"  TPC{tpc}: {n_keep_pix:,d} / {n_total_pix:,d} pixels kept "
              f"(dropped {n_dropped:,d} via mu_err > {args.err_drop:g})",
              flush=True)

        # tick -> cm
        sign = +1.0 if tpc == 0 else -1.0
        x = sign * cm_factor * (mu_drop - mu_nom)

        # save
        out_npz = os.path.join(args.outdir, f"tpc{tpc}_displacement.npz")
        np.savez(out_npz,
                 x=x.astype(np.float32),
                 mu=mu_drop.astype(np.float32),
                 mu_err=mu_err.astype(np.float32),
                 n_entries=n_ent,
                 y_centers=y_c, z_centers=z_c,
                 v_drift=np.float32(args.v_drift),
                 tick_period_us=np.float32(TICK_PERIOD_US),
                 mu_nominal=np.float32(mu_nom),
                 cm_per_tick=np.float32(cm_factor),
                 sign=np.float32(sign),
                 err_drop=np.float32(args.err_drop),
                 tpc=np.int32(tpc))
        print(f"  wrote {out_npz}", flush=True)

        # plots
        plot_grid(x, f"TPC{tpc} spatial cathode displacement (cm)",
                  os.path.join(args.outdir, "png", f"tpc{tpc}_displacement.png"),
                  vmin=-args.vrange_cm, vmax=+args.vrange_cm)
        # mu plot — colour scale centred on per-TPC median
        med = float(np.nanmedian(mu_drop))
        plot_grid(mu_drop, f"TPC{tpc} CB-fit mu (ticks)  median = {med:.2f}",
                  os.path.join(args.outdir, "png", f"tpc{tpc}_mu.png"),
                  vmin=med - 50, vmax=med + 50, cb_label="mu [ticks]")

    print(f"\nDone. {args.outdir}/", flush=True)


if __name__ == "__main__":
    main()
