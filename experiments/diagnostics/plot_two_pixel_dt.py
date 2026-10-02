"""Produce a side-by-side dt = (max_time - min_time) distribution + CB fit
for one typical TPC1 pixel and one pixel in the big white circle near
(y=-150, z=250) cm.

Same binning, fit window, and CB definition as
``plot_cathode_heatmap.py`` (combined mode, planes 0/1/2 pooled).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_cathode_heatmap import (DT_WINDOW, DT_BIN, MIN_COUNTS,
                                   crystal_ball, fit_pixel)

PARQUET = "/exp/sbnd/data/users/gputnam/efield-study/drift_df/drift_df_full_v2.parquet"
OUT_PNG = "/exp/sbnd/data/users/gputnam/efield-study/cathode_v6_doubleanode/png/pixel_dt_examples.png"
OUT_PDF = OUT_PNG.replace(".png", ".pdf")

# Wider histogram extent than DT_WINDOW so the non-cathode background is
# also visible at the bad pixel.
HIST_LO, HIST_HI = 0.0, 2700.0


def pool_planes(df, tpc):
    ys, zs, dts = [], [], []
    for p in (0, 1, 2):
        pfx = f"tpc{tpc}_p{p}"
        have = df[f"{pfx}_min_time"].notna() & df[f"{pfx}_max_time"].notna()
        ys.append(df.loc[have, f"{pfx}_max_y"].to_numpy())
        zs.append(df.loc[have, f"{pfx}_max_z"].to_numpy())
        dts.append((df.loc[have, f"{pfx}_max_time"]
                    - df.loc[have, f"{pfx}_min_time"]).to_numpy())
    return (np.concatenate(ys), np.concatenate(zs), np.concatenate(dts))


def pick_pixel(y, z, dt, *, y_lo, y_hi, z_lo, z_hi):
    sel = (y >= y_lo) & (y < y_hi) & (z >= z_lo) & (z < z_hi)
    return dt[sel]


def plot_one(ax, dts, y_lo, y_hi, z_lo, z_hi, *, title_prefix):
    edges = np.arange(HIST_LO, HIST_HI + DT_BIN, DT_BIN)
    counts, _ = np.histogram(dts, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    err = np.sqrt(np.maximum(counts.astype(float), 1.0))

    ax.axvspan(DT_WINDOW[0], DT_WINDOW[1], color="lightyellow", alpha=0.55,
               zorder=0, label=f"fit window [{DT_WINDOW[0]:.0f}, {DT_WINDOW[1]:.0f}]")
    ax.errorbar(centers, counts, yerr=err, fmt="o", ms=2.5, lw=0.6,
                color="black", label=f"data (N total = {len(dts):,})")

    # Try the fit (same as plot_cathode_heatmap.fit_pixel)
    dts_for_fit = dts[(dts >= DT_WINDOW[0]) & (dts <= DT_WINDOW[1])]
    n_in_fit = len(dts_for_fit)
    fit = fit_pixel(dts_for_fit) if n_in_fit >= MIN_COUNTS else None

    if fit is not None:
        xs = np.linspace(DT_WINDOW[0], DT_WINDOW[1], 800)
        ys = crystal_ball(xs, fit["alpha"], fit["n"], fit["mu"],
                          fit["sigma"], fit["N"])
        ax.plot(xs, ys, color="red", lw=1.6, label="CB fit (window)")
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
    ax.set_title(f"{title_prefix} — TPC1, Y in [{y_lo:g}, {y_hi:g}], "
                 f"Z in [{z_lo:g}, {z_hi:g}] (cm), planes 0/1/2 pooled")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5)
    # parameters textbox below the legend on the right
    ax.text(0.98, 0.62, txt, transform=ax.transAxes, va="top", ha="right",
            family="monospace", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))


def main():
    print(f"loading {PARQUET}", flush=True)
    cols = ["meta.run", "meta.evt"]
    for p in (0, 1, 2):
        cols += [f"tpc1_p{p}_min_time", f"tpc1_p{p}_max_time",
                 f"tpc1_p{p}_max_y", f"tpc1_p{p}_max_z"]
    df = pd.read_parquet(PARQUET, columns=cols)
    print(f"loaded {len(df):,d} tracks", flush=True)
    y, z, dt = pool_planes(df, 1)
    print(f"pooled TPC1 plane-rows: {len(y):,d}", flush=True)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    # (a) typical pixel: well away from any structures (y=0, z=125), 2 cm box
    dts_a = pick_pixel(y, z, dt, y_lo= -1, y_hi= +1, z_lo=124, z_hi=126)
    plot_one(axs[0], dts_a,  -1.0, +1.0, 124.0, 126.0,
             title_prefix="(a) typical pixel")

    # (b) inside the big white circle south of centre: y in [-152, -150),
    # z in [250, 252). Single 2x2 cm native pixel.
    dts_b = pick_pixel(y, z, dt, y_lo=-152, y_hi=-150, z_lo=250, z_hi=252)
    plot_one(axs[1], dts_b, -152.0, -150.0, 250.0, 252.0,
             title_prefix="(b) inside white circle (south)")

    fig.suptitle("SBND TPC1 cathode-tick distribution per pixel "
                 "(full-sample drift_df_full_v2)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"wrote {OUT_PNG}", flush=True)
    print(f"wrote {OUT_PDF}", flush=True)


if __name__ == "__main__":
    main()
