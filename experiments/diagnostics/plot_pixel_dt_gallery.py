"""Per-pixel dt = (max_time - min_time) distributions, full-range view.

Writes one PNG/PDF per pixel. Two sub-directories under
/exp/sbnd/data/users/gputnam/efield-study/cathode_v6_doubleanode/png/:

    pixel_dt_circle/ – ten pixels in the cathode "white circle" regions
                       (near z=250 cm; y ≈ ±55 and ±150).
    pixel_dt_healthy/ – five typical pixels (clean cathode peak).

Same binning/fit logic as plot_cathode_heatmap.py (combined mode, planes
0/1/2 pooled).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_cathode_heatmap
# Match the v3 cathode-heatmap production settings (run_cathode_heatmap_v3):
#   DT_WINDOW shifted 10 ticks earlier
#   DT_BIN doubled to 4 ticks
#   MIN_COUNTS dropped to 10
plot_cathode_heatmap.DT_WINDOW  = (2440.0, 2640.0)
plot_cathode_heatmap.DT_BIN     = 4.0
plot_cathode_heatmap.MIN_COUNTS = 10
from plot_cathode_heatmap import (crystal_ball, fit_pixel)
DT_WINDOW  = plot_cathode_heatmap.DT_WINDOW
DT_BIN     = plot_cathode_heatmap.DT_BIN
MIN_COUNTS = plot_cathode_heatmap.MIN_COUNTS

PARQUET = "/exp/sbnd/data/users/gputnam/efield-study/drift_df/drift_df_full_v2.parquet"
OUT_BASE = "/exp/sbnd/data/users/gputnam/efield-study/cathode_pipeline/step3_v6/png"

HIST_LO, HIST_HI = 2400.0, 2700.0  # cropped around the v3 fit window
                                   # (the heatmap_v3 window is [2440, 2640])


CIRCLE_PIXELS = [
    # south "circle" near y=-150
    (-156, 250),
    (-154, 250),
    (-152, 250),
    (-150, 250),
    # north "circle" near y=+150
    (+148, 250),
    (+150, 250),
    (+152, 250),
    # smaller secondary clusters at y=±55
    ( -58, 250),
    ( +54, 250),
    ( +56, 250),
]

HEALTHY_PIXELS = [
    (   0, 124),
    ( 100, 250),
    (-100, 400),
    (  50, 200),
    ( -50, 300),
]


def pool_planes(df, tpc):
    ys, zs, dts = [], [], []
    for p in (0, 1, 2):
        pfx = f"tpc{tpc}_p{p}"
        have = df[f"{pfx}_min_time"].notna() & df[f"{pfx}_max_time"].notna()
        ys.append(df.loc[have, f"{pfx}_max_y"].to_numpy())
        zs.append(df.loc[have, f"{pfx}_max_z"].to_numpy())
        dts.append((df.loc[have, f"{pfx}_max_time"]
                    - df.loc[have, f"{pfx}_min_time"]).to_numpy())
    return np.concatenate(ys), np.concatenate(zs), np.concatenate(dts)


def pick_pixel(y, z, dt, *, y_lo, y_hi, z_lo, z_hi):
    sel = (y >= y_lo) & (y < y_hi) & (z >= z_lo) & (z < z_hi)
    return dt[sel]


def plot_one(dts, y_lo, y_hi, z_lo, z_hi, *, kind, idx, out_dir):
    edges = np.arange(HIST_LO, HIST_HI + DT_BIN, DT_BIN)
    counts, _ = np.histogram(dts, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    err = np.sqrt(np.maximum(counts.astype(float), 1.0))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axvspan(DT_WINDOW[0], DT_WINDOW[1], color="lightyellow", alpha=0.55,
               zorder=0, label=f"fit window [{DT_WINDOW[0]:.0f}, {DT_WINDOW[1]:.0f}]")
    ax.errorbar(centers, counts, yerr=err, fmt="o", ms=2.5, lw=0.6,
                color="black", label=f"data (N total = {len(dts):,})")

    dts_for_fit = dts[(dts >= DT_WINDOW[0]) & (dts <= DT_WINDOW[1])]
    n_in_fit = len(dts_for_fit)
    fit = fit_pixel(dts_for_fit) if n_in_fit >= MIN_COUNTS else None

    if fit is not None:
        xs = np.linspace(DT_WINDOW[0], DT_WINDOW[1], 800)
        ys_curve = crystal_ball(xs, fit["alpha"], fit["n"], fit["mu"],
                                fit["sigma"], fit["N"])
        ax.plot(xs, ys_curve, color="red", lw=1.6, label="CB fit (window)")
        ax.axvline(fit["mu"], color="red", ls="--", lw=0.8)
        txt = (f"N in fit window = {n_in_fit:,d}\n"
               f"$\\mu$ = {fit['mu']:.2f} $\\pm$ {fit['mu_err']:.2f}\n"
               f"$\\sigma$ = {fit['sigma']:.2f}\n"
               f"$\\alpha$ = {fit['alpha']:.2f},  $n$ = {fit['n']:.2f}\n"
               f"$\\chi^2$ / ndf = {fit['chi2']:.1f} / {fit['ndf']}\n"
               f"keep? {'YES (mu_err <= 10)' if fit['mu_err'] <= 10 else 'NO  (mu_err > 10)'}")
    else:
        txt = (f"N in fit window = {n_in_fit:,d}\n"
               f"(< MIN_COUNTS = {MIN_COUNTS}, fit not attempted)")

    ax.set_xlim(HIST_LO, HIST_HI)
    ax.set_xlabel("dt = max_time - min_time [ticks]")
    ax.set_ylabel(f"entries / {DT_BIN:g} ticks")
    ax.set_title(f"({kind} #{idx:02d}) TPC1, "
                 f"Y in [{y_lo:g}, {y_hi:g}], Z in [{z_lo:g}, {z_hi:g}] (cm), "
                 f"planes 0/1/2 pooled")
    ax.grid(alpha=0.3)

    # In the cropped [2400, 2700] view the cathode peak (mu ~ 2520, sigma
    # ~ 2 ticks) is narrow and centred, so both upper corners are above
    # near-zero data. Stack legend + parameter textbox in the upper-right
    # away from the peak's rising edge on the left.
    leg = ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    fig.canvas.draw()
    # Place parameter textbox just below the legend (in axes coords).
    leg_bbox = leg.get_window_extent().transformed(ax.transAxes.inverted())
    txt_top_y = leg_bbox.y0 - 0.02   # 2 % padding below the legend box
    ax.text(0.98, max(txt_top_y, 0.30), txt,
            transform=ax.transAxes, va="top", ha="right",
            family="monospace", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.92))

    tag = f"{kind}_{idx:02d}_y{int(round(y_lo)):+04d}_z{int(round(z_lo)):+04d}"
    png_path = os.path.join(out_dir, tag + ".png")
    pdf_path = os.path.join(out_dir, tag + ".pdf")
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, n_in_fit, fit


def main():
    print(f"loading {PARQUET}", flush=True)
    cols = []
    for p in (0, 1, 2):
        cols += [f"tpc1_p{p}_min_time", f"tpc1_p{p}_max_time",
                 f"tpc1_p{p}_max_y", f"tpc1_p{p}_max_z"]
    df = pd.read_parquet(PARQUET, columns=cols)
    y, z, dt = pool_planes(df, 1)
    print(f"pooled TPC1 plane-rows: {len(y):,d}", flush=True)

    circle_dir  = os.path.join(OUT_BASE, "pixel_dt_circle")
    healthy_dir = os.path.join(OUT_BASE, "pixel_dt_healthy")
    os.makedirs(circle_dir,  exist_ok=True)
    os.makedirs(healthy_dir, exist_ok=True)

    print("\n--- white-circle pixels (z ~ 250 cm) ---")
    for i, (yc, zc) in enumerate(CIRCLE_PIXELS):
        y_lo, y_hi = yc, yc + 2
        z_lo, z_hi = zc, zc + 2
        dts = pick_pixel(y, z, dt, y_lo=y_lo, y_hi=y_hi, z_lo=z_lo, z_hi=z_hi)
        path, n_in_fit, fit = plot_one(dts, y_lo, y_hi, z_lo, z_hi,
                                       kind="circle", idx=i, out_dir=circle_dir)
        flag = "FIT FAILED" if fit is None else (
            "ok" if fit['mu_err'] <= 10 else "DROP (mu_err > 10)")
        print(f"  #{i:02d} y=[{y_lo:+4.0f},{y_hi:+4.0f}] z=[{z_lo:4.0f},{z_hi:4.0f}]  "
              f"N_total={len(dts):4d}  N_fit_win={n_in_fit:4d}  {flag}")

    print("\n--- healthy pixels ---")
    for i, (yc, zc) in enumerate(HEALTHY_PIXELS):
        y_lo, y_hi = yc, yc + 2
        z_lo, z_hi = zc, zc + 2
        dts = pick_pixel(y, z, dt, y_lo=y_lo, y_hi=y_hi, z_lo=z_lo, z_hi=z_hi)
        path, n_in_fit, fit = plot_one(dts, y_lo, y_hi, z_lo, z_hi,
                                       kind="healthy", idx=i, out_dir=healthy_dir)
        flag = "FIT FAILED" if fit is None else (
            "ok" if fit['mu_err'] <= 10 else "DROP (mu_err > 10)")
        print(f"  #{i:02d} y=[{y_lo:+4.0f},{y_hi:+4.0f}] z=[{z_lo:4.0f},{z_hi:4.0f}]  "
              f"N_total={len(dts):4d}  N_fit_win={n_in_fit:4d}  {flag}")

    print(f"\nwrote {circle_dir}/")
    print(f"wrote {healthy_dir}/")


if __name__ == "__main__":
    main()
