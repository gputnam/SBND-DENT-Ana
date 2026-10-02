"""Map the empirical cathode tick across the (Y, Z) face of each SBND TPC.

For each (Y, Z) pixel of the readout face, fit the per-track drift duration
dt = |max_time - min_time| with a Crystal Ball function. The CB peak (mu) is
the empirical cathode position in tick units for that pixel. Two heatmaps
(one per TPC) and one histogram-with-fit per pixel are written to disk in
both PDF and PNG.

Pixel binning is keyed on the cathode-side hit (the latest hit, i.e. the
hit at max_time -> closest to the cathode in drift), since that's the hit
whose Y, Z corresponds to where the track crossed the cathode.

Outputs (under ./cathode_study/):
    png/cathode_tick_tpc{0,1}.png                  heatmap (png)
    pdf/cathode_tick_tpc{0,1}.pdf                  heatmap (pdf)
    pixels/tpc{0,1}/png/y{Y}_z{Z}.png              per-pixel hist+fit (png)
    pixels/tpc{0,1}/pdf/y{Y}_z{Z}.pdf              per-pixel hist+fit (pdf)
    cathode_fit_results_tpc{0,1}.csv               numerical results
"""

import argparse
import os
import sys
import warnings
from multiprocessing import Pool, cpu_count

# Run at low priority by default so we don't fight other jobs on the node.
try:
    os.nice(10)
except (AttributeError, PermissionError):
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Diverging colormap with a yellow center (not white), so NaN pixels (rendered
# white) are visually distinct from values near the colorbar center.
HEATMAP_CMAP = matplotlib.colormaps["RdYlBu_r"].copy()
HEATMAP_CMAP.set_bad("white")
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

INPUT = "drift_df_full_combined.parquet"
OUTDIR = "cathode_study_5cm_combined"

PIXEL_SIZE = 5.0            # cm
Y_RANGE = (-200.0, 200.0)   # cm; SBND Y active half-height
Z_RANGE = (0.0, 500.0)      # cm; SBND Z active length

DT_WINDOW = (2450.0, 2650.0)  # ticks; fit range, drift ~2575 ticks
DT_BIN = 2.0                  # ticks/bin
MIN_COUNTS = 30               # minimum entries to attempt a fit

DRIFT_LENGTH_CM = 400.0  # 2 TPCs x 200 cm anode-cathode
TICK_PERIOD_US = 0.5     # ClockSpeedTPC = 2 MHz from detectorclocks_sbnd.fcl
# LArSoft DetectorPropertiesStandard::DriftVelocity at SBND nominal
# (E=0.5 kV/cm, T=88.4 K) -> 0.15627 cm/us. See print_drift_velocity.py.
SBND_NOMINAL_VDRIFT = 0.15627


def crystal_ball(x, alpha, n, mu, sigma, N):
    """Crystal Ball with Gaussian core (z > -alpha) and power-law left tail."""
    x = np.asarray(x, dtype=float)
    z = (x - mu) / sigma
    abs_a = max(abs(alpha), 1e-6)
    A = (n / abs_a) ** n * np.exp(-(alpha ** 2) / 2)
    B = n / abs_a - abs_a

    out = np.empty_like(x)
    in_gauss = z > -alpha
    out[in_gauss] = N * np.exp(-z[in_gauss] ** 2 / 2)
    arg = np.maximum(B - z[~in_gauss], 1e-300)
    out[~in_gauss] = N * A * arg ** (-n)
    return out


def fit_pixel(dts):
    edges = np.arange(DT_WINDOW[0], DT_WINDOW[1] + DT_BIN, DT_BIN)
    counts, _ = np.histogram(dts, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    err = np.sqrt(np.maximum(counts.astype(float), 1.0))

    peak_idx = int(np.argmax(counts))
    mu0 = centers[peak_idx]
    N0 = max(counts[peak_idx], 1)

    p0 = [1.0, 2.0, mu0, 6.0, float(N0)]
    bounds = ([0.1, 1.01, DT_WINDOW[0], 0.5, 1.0],
              [10.0, 100.0, DT_WINDOW[1], 50.0, 1e8])
    try:
        popt, pcov = curve_fit(crystal_ball, centers, counts.astype(float),
                               p0=p0, sigma=err, absolute_sigma=True,
                               bounds=bounds, maxfev=8000)
    except (RuntimeError, ValueError):
        return None

    pred = crystal_ball(centers, *popt)
    chi2 = float(np.sum(((counts - pred) / err) ** 2))
    ndf = int(len(centers) - len(popt))

    perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf))
    return {
        "centers": centers, "counts": counts, "err": err, "pred": pred,
        "alpha": popt[0], "n": popt[1], "mu": popt[2], "sigma": popt[3], "N": popt[4],
        "alpha_err": perr[0], "n_err": perr[1], "mu_err": perr[2],
        "sigma_err": perr[3], "N_err": perr[4],
        "chi2": chi2, "ndf": ndf, "n_entries": int(counts.sum()),
    }


def plot_pixel(fit, ylo, yhi, zlo, zhi, tpc, plane_label, outpath_png, outpath_pdf):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(fit["centers"], fit["counts"], yerr=fit["err"],
                fmt="o", ms=3, lw=0.8, color="black", label="data")
    xs = np.linspace(DT_WINDOW[0], DT_WINDOW[1], 800)
    ax.plot(xs, crystal_ball(xs, fit["alpha"], fit["n"], fit["mu"],
                             fit["sigma"], fit["N"]),
            color="red", lw=1.4, label="CB fit")
    ax.axvline(fit["mu"], color="red", ls="--", lw=0.8)
    ax.set_xlabel("dt = max_time - min_time [ticks]")
    ax.set_ylabel(f"tracks / {DT_BIN:g} ticks")
    ax.set_title(f"SBND TPC{tpc}  {plane_label}  "
                 f"Y in [{ylo:g}, {yhi:g}] cm,  Z in [{zlo:g}, {zhi:g}] cm")
    txt = (f"$\\mu$ = {fit['mu']:.2f} $\\pm$ {fit['mu_err']:.2f}\n"
           f"$\\sigma$ = {fit['sigma']:.2f}\n"
           f"$\\alpha$ = {fit['alpha']:.2f},  $n$ = {fit['n']:.2f}\n"
           f"$\\chi^2$ / ndf = {fit['chi2']:.1f} / {fit['ndf']}\n"
           f"N entries = {fit['n_entries']}")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.set_xlim(DT_WINDOW)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath_png, dpi=110)
    fig.savefig(outpath_pdf)
    plt.close(fig)


PLANES = (0, 1, 2)


def _fit_and_plot_pixel(args):
    """Worker: fit a single pixel; write its histogram if png/pdf paths set."""
    (tpc, plane_label, jy, jz, ylo, yhi, zlo, zhi, dts, png_path, pdf_path) = args
    fit = fit_pixel(dts)
    if fit is None:
        return None
    if png_path is not None:
        plot_pixel(fit, ylo, yhi, zlo, zhi, tpc, plane_label, png_path, pdf_path)
    return {
        "tpc": tpc, "plane": plane_label, "jy": jy, "jz": jz,
        "y_lo": ylo, "y_hi": yhi, "z_lo": zlo, "z_hi": zhi,
        "n_entries": fit["n_entries"],
        "mu": fit["mu"], "mu_err": fit["mu_err"],
        "sigma": fit["sigma"], "alpha": fit["alpha"], "n": fit["n"],
        "chi2": fit["chi2"], "ndf": fit["ndf"],
    }


def process_tpc(df, tpc, planes, suffix, plane_label, nproc,
                save_pixel_hists=False):
    """Pool dt = max-min across the given planes for one TPC and fit per pixel.

    For each plane in ``planes``, take (max_y, max_z, max_time-min_time) and
    concatenate. Bin by (Y, Z) into pixels and fit the pooled dt distribution
    per pixel. ``suffix`` is appended to filenames (e.g. "" for the combined
    map, "_p2" when fitting plane 2 alone). ``plane_label`` shows up in plot
    titles.
    """
    ys, zs, dts_list = [], [], []
    for plane in planes:
        pfx = f"tpc{tpc}_p{plane}"
        if f"{pfx}_min_time" not in df.columns:
            continue
        have = df[f"{pfx}_min_time"].notna() & df[f"{pfx}_max_time"].notna()
        ys.append(df.loc[have, f"{pfx}_max_y"].to_numpy())
        zs.append(df.loc[have, f"{pfx}_max_z"].to_numpy())
        dts_list.append((df.loc[have, f"{pfx}_max_time"]
                         - df.loc[have, f"{pfx}_min_time"]).to_numpy())
    if not ys:
        raise RuntimeError(f"no plane columns found for TPC{tpc} in {list(planes)}")
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    dt = np.concatenate(dts_list)

    in_window = (dt >= DT_WINDOW[0]) & (dt <= DT_WINDOW[1]) \
                & (y >= Y_RANGE[0]) & (y < Y_RANGE[1]) \
                & (z >= Z_RANGE[0]) & (z < Z_RANGE[1])
    y, z, dt = y[in_window], z[in_window], dt[in_window]

    y_edges = np.arange(Y_RANGE[0], Y_RANGE[1] + PIXEL_SIZE, PIXEL_SIZE)
    z_edges = np.arange(Z_RANGE[0], Z_RANGE[1] + PIXEL_SIZE, PIXEL_SIZE)
    iy = np.digitize(y, y_edges) - 1
    iz = np.digitize(z, z_edges) - 1

    ny = len(y_edges) - 1
    nz = len(z_edges) - 1
    mu_grid = np.full((ny, nz), np.nan)
    mu_err_grid = np.full((ny, nz), np.nan)
    nfit_grid = np.zeros((ny, nz), dtype=int)
    chi2_grid = np.full((ny, nz), np.nan)

    pix_png_dir = pix_pdf_dir = None
    if save_pixel_hists:
        pix_png_dir = os.path.join(OUTDIR, "pixels", f"tpc{tpc}{suffix}", "png")
        pix_pdf_dir = os.path.join(OUTDIR, "pixels", f"tpc{tpc}{suffix}", "pdf")
        os.makedirs(pix_png_dir, exist_ok=True)
        os.makedirs(pix_pdf_dir, exist_ok=True)

    flat_idx = iy * nz + iz
    order = np.argsort(flat_idx, kind="stable")
    flat_sorted = flat_idx[order]
    dt_sorted = dt[order]
    splits = np.searchsorted(flat_sorted, np.arange(ny * nz + 1))

    tasks = []
    for jy in range(ny):
        for jz in range(nz):
            k = jy * nz + jz
            lo, hi = splits[k], splits[k + 1]
            if hi - lo < MIN_COUNTS:
                continue
            ylo, yhi = y_edges[jy], y_edges[jy + 1]
            zlo, zhi = z_edges[jz], z_edges[jz + 1]
            if save_pixel_hists:
                tag = f"y{int(ylo):+05d}_z{int(zlo):+05d}"
                png_path = os.path.join(pix_png_dir, tag + ".png")
                pdf_path = os.path.join(pix_pdf_dir, tag + ".pdf")
            else:
                png_path = pdf_path = None
            tasks.append((tpc, plane_label, jy, jz, ylo, yhi, zlo, zhi,
                          dt_sorted[lo:hi].copy(), png_path, pdf_path))

    print(f"  TPC{tpc} {plane_label}: {len(tasks)} pixels >= {MIN_COUNTS} entries to fit "
          f"(of {ny*nz} pixel grid, {(splits[1:]-splits[:-1] > 0).sum()} populated)",
          flush=True)

    rows = []
    n_ok = 0
    desc = f"TPC{tpc}{suffix or '_all'}"
    if nproc <= 1:
        it = (_fit_and_plot_pixel(t) for t in tasks)
        for r in tqdm(it, total=len(tasks), unit="pix",
                      desc=desc, smoothing=0.6, file=sys.stderr):
            if r is not None:
                rows.append(r)
                n_ok += 1
                mu_grid[r["jy"], r["jz"]] = r["mu"]
                mu_err_grid[r["jy"], r["jz"]] = r["mu_err"]
                nfit_grid[r["jy"], r["jz"]] = r["n_entries"]
                chi2_grid[r["jy"], r["jz"]] = r["chi2"] / max(r["ndf"], 1)
    else:
        with Pool(processes=nproc) as pool:
            for r in tqdm(pool.imap_unordered(_fit_and_plot_pixel, tasks,
                                              chunksize=4),
                          total=len(tasks), unit="pix",
                          desc=desc, smoothing=0.6, file=sys.stderr):
                if r is not None:
                    rows.append(r)
                    n_ok += 1
                    mu_grid[r["jy"], r["jz"]] = r["mu"]
                    mu_err_grid[r["jy"], r["jz"]] = r["mu_err"]
                    nfit_grid[r["jy"], r["jz"]] = r["n_entries"]
                    chi2_grid[r["jy"], r["jz"]] = r["chi2"] / max(r["ndf"], 1)
    print(f"  TPC{tpc} {plane_label}: {n_ok}/{len(tasks)} fits succeeded", flush=True)

    csv = pd.DataFrame(rows).drop(columns=["jy", "jz"], errors="ignore")
    csv.to_csv(os.path.join(OUTDIR, f"cathode_fit_results_tpc{tpc}{suffix}.csv"),
               index=False)
    return y_edges, z_edges, mu_grid, mu_err_grid, nfit_grid, chi2_grid


HEATMAP_HALF_RANGE = 50.0  # ticks; bwr scale spans mean +/- HEATMAP_HALF_RANGE


def plot_heatmap(y_edges, z_edges, mu_grid, tpc, plane_label, outpath_png, outpath_pdf):
    valid = mu_grid[np.isfinite(mu_grid)]
    if valid.size == 0:
        print(f"  TPC{tpc}: no successful fits, skipping heatmap")
        return
    center = float(np.mean(valid))
    vmin = center - HEATMAP_HALF_RANGE
    vmax = center + HEATMAP_HALF_RANGE
    n_below = int(np.sum(valid < vmin))
    n_above = int(np.sum(valid > vmax))

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(mu_grid,
                   origin="lower", aspect="equal",
                   extent=(z_edges[0], z_edges[-1], y_edges[0], y_edges[-1]),
                   cmap=HEATMAP_CMAP,
                   vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, extend="both",
                      label="empirical cathode tick (CB peak $\\mu$)")
    cb.ax.axhline(center, color="black", lw=0.8)
    ax.set_xlabel("Z [cm]")
    ax.set_ylabel("Y [cm]")
    ax.set_title(f"SBND TPC{tpc}  {plane_label}  cathode tick over (Z, Y)\n"
                 f"mean $\\mu$ = {center:.2f} ticks  "
                 f"(scale $\\pm${HEATMAP_HALF_RANGE:g}, "
                 f"clipped {n_below} low / {n_above} high,  "
                 f"pixel = {PIXEL_SIZE:g} x {PIXEL_SIZE:g} cm)")
    fig.tight_layout()
    fig.savefig(outpath_png, dpi=130)
    fig.savefig(outpath_pdf)
    plt.close(fig)


def plot_sum_heatmap(y_edges, z_edges, sum_grid, plane_label, outpath_png, outpath_pdf):
    """Heatmap of mu_TPC0(y,z) + mu_TPC1(y,z): the two-side drift duration
    across the cathode at each (Y, Z). NaN where either TPC has no fit."""
    valid = sum_grid[np.isfinite(sum_grid)]
    if valid.size == 0:
        print("  cathode tick SUM: no overlap pixels, skipping")
        return
    center = float(np.mean(valid))
    vmin = center - HEATMAP_HALF_RANGE
    vmax = center + HEATMAP_HALF_RANGE
    n_below = int(np.sum(valid < vmin))
    n_above = int(np.sum(valid > vmax))

    # Per-pixel local v_drift estimate (4 m / sum in ticks * 0.5 us/tick)
    v_local = DRIFT_LENGTH_CM / (center * TICK_PERIOD_US)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(sum_grid,
                   origin="lower", aspect="equal",
                   extent=(z_edges[0], z_edges[-1], y_edges[0], y_edges[-1]),
                   cmap=HEATMAP_CMAP,
                   vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, extend="both",
                      label="$\\mu_{TPC0} + \\mu_{TPC1}$ [ticks]  "
                            "(two-sided cathode drift duration)")
    cb.ax.axhline(center, color="black", lw=0.8)
    ax.set_xlabel("Z [cm]")
    ax.set_ylabel("Y [cm]")
    ax.set_title(f"SBND {plane_label}  $\\mu_{{TPC0}} + \\mu_{{TPC1}}$ over (Z, Y)\n"
                 f"mean = {center:.2f} ticks  "
                 f"$\\rightarrow$ $\\langle v_\\mathrm{{drift}}\\rangle$ = "
                 f"{v_local:.5f} cm/$\\mu$s   "
                 f"(scale $\\pm${HEATMAP_HALF_RANGE:g}, "
                 f"clipped {n_below}/{n_above},  "
                 f"pixel = {PIXEL_SIZE:g}x{PIXEL_SIZE:g} cm)")
    fig.tight_layout()
    fig.savefig(outpath_png, dpi=130)
    fig.savefig(outpath_pdf)
    plt.close(fig)


def plot_diff_heatmap(y_edges, z_edges, diff_grid, plane_label, outpath_png, outpath_pdf):
    """Heatmap of mu_TPC0(y,z) - mu_TPC1(y,z): the per-pixel difference of
    the two cathode-side drift durations. NaN where either TPC has no fit.
    Centered at zero; colorbar half-range = HEATMAP_HALF_RANGE."""
    valid = diff_grid[np.isfinite(diff_grid)]
    if valid.size == 0:
        print("  cathode tick DIFF: no overlap pixels, skipping")
        return
    mean = float(np.mean(valid))
    std = float(np.std(valid))
    vmin, vmax = -HEATMAP_HALF_RANGE, HEATMAP_HALF_RANGE
    n_below = int(np.sum(valid < vmin))
    n_above = int(np.sum(valid > vmax))

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(diff_grid,
                   origin="lower", aspect="equal",
                   extent=(z_edges[0], z_edges[-1], y_edges[0], y_edges[-1]),
                   cmap=HEATMAP_CMAP,
                   vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, extend="both",
                      label="$\\mu_{TPC0} - \\mu_{TPC1}$ [ticks]")
    cb.ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Z [cm]")
    ax.set_ylabel("Y [cm]")
    ax.set_title(f"SBND {plane_label}  $\\mu_{{TPC0}} - \\mu_{{TPC1}}$ over (Z, Y)\n"
                 f"mean = {mean:+.2f} ticks  ($\\sigma$ = {std:.2f}),  "
                 f"scale $\\pm${HEATMAP_HALF_RANGE:g}, "
                 f"clipped {n_below}/{n_above},  "
                 f"pixel = {PIXEL_SIZE:g}x{PIXEL_SIZE:g} cm")
    fig.tight_layout()
    fig.savefig(outpath_png, dpi=130)
    fig.savefig(outpath_pdf)
    plt.close(fig)


def report_drift_velocity(df, plane):
    """Estimate the mean drift velocity from anode-cathode through-going
    tracks that span both TPCs on plane `plane`:
        v_drift = (4 m) / (dt_TPC0 + dt_TPC1)."""
    needed = [f"tpc0_p{plane}_min_time", f"tpc0_p{plane}_max_time",
              f"tpc1_p{plane}_min_time", f"tpc1_p{plane}_max_time"]
    print(f"\n--- mean drift velocity (plane {plane}) ---")
    if not all(c in df.columns for c in needed):
        print(f"  skipped: missing columns for plane {plane}")
        return
    both = df.dropna(subset=needed)
    dt0 = (both[f"tpc0_p{plane}_max_time"] - both[f"tpc0_p{plane}_min_time"]).to_numpy()
    dt1 = (both[f"tpc1_p{plane}_max_time"] - both[f"tpc1_p{plane}_min_time"]).to_numpy()
    full = (dt0 > 2400) & (dt1 > 2400)
    if full.sum() < 100:
        print(f"  only {full.sum()} anode-cathode through-going tracks; skipping")
        return
    total_us = (dt0[full] + dt1[full]) * TICK_PERIOD_US
    v_per_track = DRIFT_LENGTH_CM / total_us
    v_global = DRIFT_LENGTH_CM / total_us.mean()
    print(f"  N anode-cathode through-going tracks (dt0>2400 & dt1>2400 ticks): "
          f"{full.sum():,}")
    print(f"  <dt0+dt1>          = {total_us.mean():.3f} us "
          f"({total_us.mean()/TICK_PERIOD_US:.2f} ticks)")
    print(f"  v_drift = 4 m / <dt0+dt1>      = {v_global:.5f} cm/us")
    print(f"  per-track <v_drift>            = {v_per_track.mean():.5f} cm/us "
          f"(median {np.median(v_per_track):.5f}, std {v_per_track.std():.5f})")
    print(f"  sbndcode sim nominal           = {SBND_NOMINAL_VDRIFT:.5f} cm/us")
    print(f"  measured / nominal             = "
          f"{v_global/SBND_NOMINAL_VDRIFT:.4f}")


def main():
    global OUTDIR, PIXEL_SIZE, MIN_COUNTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--nproc", default="auto",
                    help="number of worker processes (default: auto = cpu_count)")
    ap.add_argument("--planes", default="0,1,2",
                    help="comma list of planes to use (default 0,1,2)")
    ap.add_argument("--pixel-size", type=float, default=PIXEL_SIZE,
                    help=f"(Y, Z) pixel size in cm (default {PIXEL_SIZE:g})")
    ap.add_argument("--min-counts", type=int, default=MIN_COUNTS,
                    help=f"min entries to attempt a CB fit (default {MIN_COUNTS})")
    ap.add_argument("--mode", default="combined",
                    choices=("combined", "per-plane"),
                    help="combined: pool dt across the listed planes into a "
                         "single per-TPC map (default). "
                         "per-plane: one map per plane per TPC (old behavior).")
    ap.add_argument("--save-pixel-hists", action="store_true",
                    help="also write per-pixel histogram+fit plots "
                         "(thousands of files per plane)")
    args = ap.parse_args()
    OUTDIR = args.outdir
    PIXEL_SIZE = args.pixel_size
    MIN_COUNTS = args.min_counts
    os.makedirs(OUTDIR, exist_ok=True)

    planes = tuple(int(p) for p in args.planes.split(","))
    nproc = cpu_count() if args.nproc == "auto" else int(args.nproc)

    print(f"loading {args.input}", flush=True)
    df = pd.read_parquet(args.input)
    print(f"loaded {len(df):,} tracks", flush=True)
    for p in planes:
        report_drift_velocity(df, p)

    png_dir = os.path.join(OUTDIR, "png")
    pdf_dir = os.path.join(OUTDIR, "pdf")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    if args.mode == "combined":
        plist = ",".join(str(p) for p in planes)
        groups = [(planes, "", f"planes {{{plist}}} pooled")]
    else:
        groups = [((p,), f"_p{p}", f"plane {p}") for p in planes]

    for plane_group, suffix, plane_label in groups:
        print(f"\n========== {plane_label} ==========", flush=True)
        mu_grids = {}
        y_edges = z_edges = None
        for tpc in (0, 1):
            print(f"\n--- TPC{tpc} {plane_label} (nproc={nproc}) ---", flush=True)
            y_edges, z_edges, mu_grid, _, _, _ = process_tpc(
                df, tpc, plane_group, suffix, plane_label, nproc,
                save_pixel_hists=args.save_pixel_hists)
            mu_grids[tpc] = mu_grid
            png = os.path.join(png_dir, f"cathode_tick_tpc{tpc}{suffix}.png")
            pdf = os.path.join(pdf_dir, f"cathode_tick_tpc{tpc}{suffix}.pdf")
            plot_heatmap(y_edges, z_edges, mu_grid, tpc, plane_label, png, pdf)
            print(f"  wrote {png} and {pdf}", flush=True)

        sum_grid = mu_grids[0] + mu_grids[1]
        sum_png = os.path.join(png_dir, f"cathode_tick_sum{suffix}.png")
        sum_pdf = os.path.join(pdf_dir, f"cathode_tick_sum{suffix}.pdf")
        plot_sum_heatmap(y_edges, z_edges, sum_grid, plane_label, sum_png, sum_pdf)
        print(f"  wrote {sum_png} and {sum_pdf}", flush=True)

        diff_grid = mu_grids[0] - mu_grids[1]
        diff_png = os.path.join(png_dir, f"cathode_tick_diff{suffix}.png")
        diff_pdf = os.path.join(pdf_dir, f"cathode_tick_diff{suffix}.pdf")
        plot_diff_heatmap(y_edges, z_edges, diff_grid, plane_label, diff_png, diff_pdf)
        print(f"  wrote {diff_png} and {diff_pdf}", flush=True)


if __name__ == "__main__":
    main()
