"""Voxel-binned displacement maps for the double-anode (CRT-free) sample.

CRT-free analog of `plot_crtdisp_voxels.py`. The input parquet is
produced by `make_doubleanode_df_sbnd.py`: per-hit (x, y, z) plus the
residual (dx, dy, dz) from the line through the track's two
anode-piercing endpoints (no CRT taggers, so no per-tagger subsets).

The per-voxel mu is recovered by the **MicroBooNE-style LSQ inversion**
of the perpendicular projection (see crtlib.voxel_invert), so the
reported Δ̂_v is the unbiased 3-D distortion vector, not the projected
component bias. The CAVEAT for the double-anode sample is that
anode-piercing tracks have a restricted angular distribution — they're
constrained to traverse from one anode (X = -200) to the other (X = +200),
so the x-component of t̂ is always large. Per-voxel angular diversity
is therefore worse than for CRT crossers, and `cond_<subset>` should be
checked carefully: voxels with `cond > --cond-max` get NaN'd.

A single subset "all" covers every hit. The output schema
(`voxel_data.npz` keys) mirrors `plot_crtdisp_voxels.py` exactly so
downstream comparison scripts can load both maps with the same code:
``mu_<subset>_<comp>`` now holds the LSQ-inverted Δ̂ component (NOT the
projected-component arithmetic mean). New fields ``delta_mu_<subset>``,
``delta_cov_<subset>``, ``cond_<subset>`` carry the full LSQ output.

Output layout (under --outdir):
    all/{dx,dy,dz}/{png,pdf}/x{+/-NNN}.{png,pdf}        Δ̂ maps
    sigma/all/{dx,dy,dz}/{png,pdf}/x{NNN}.{png,pdf}     LSQ σ̂ maps
    counts/all/{png,pdf}/x{+/-NNN}.{png,pdf}            hit counts
    voxel_data.npz                                      raw + LSQ data
"""

import argparse
import glob
import os
import sys
import warnings
from multiprocessing import Pool, cpu_count

try:
    os.nice(10)
except (AttributeError, PermissionError):
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from tqdm.auto import tqdm

from crtlib import voxel_invert

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Half-length of the anode-to-anode line in X. Used to build the track
# direction from the (anode0_y, anode0_z) and (anode1_y, anode1_z)
# endpoints stored in the parquet. Matches the convention in
# make_doubleanode_df_sbnd.py.
ANODE_X_HALF = 200.0

INPUT_DIR = "/exp/sbnd/data/users/gputnam/efield-study/doubleanode_df"
OUTDIR = "doubleanode_maps"

VOXEL_SIZE = 20.0
X_RANGE = (-200.0, 200.0)
Y_RANGE = (-200.0, 200.0)
Z_RANGE = (0.0, 500.0)

HIST_HALF_RANGE = 100.0
HIST_BIN = 1.0

# Single-Gaussian fit window (cm around the seeded peak). Bins outside
# the window are dropped from the fit so any tails don't drag the
# fitted sigma wide.
FIT_WINDOW_HALF = 25.0

# Single subset; no CRT taggers in the double-anode sample.
SUBSETS = ("all",)
COMPONENTS = ("dx", "dy", "dz")

DEFAULT_VRANGE = 5.0
DEFAULT_MIN_COUNTS = 30
DEFAULT_SIGMA_VRANGE = 20.0

HEATMAP_CMAP = matplotlib.colormaps["RdYlBu_r"].copy()
HEATMAP_CMAP.set_bad("white")
SIGMA_CMAP = matplotlib.colormaps["viridis"].copy()
SIGMA_CMAP.set_bad("white")


def grid_shape(size):
    nx = int(round((X_RANGE[1] - X_RANGE[0]) / size))
    ny = int(round((Y_RANGE[1] - Y_RANGE[0]) / size))
    nz = int(round((Z_RANGE[1] - Z_RANGE[0]) / size))
    return nx, ny, nz


def hist_setup():
    n_bins = int(round(2 * HIST_HALF_RANGE / HIST_BIN))
    edges = np.linspace(-HIST_HALF_RANGE, +HIST_HALF_RANGE, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return n_bins, edges, centers


def make_accumulators(size):
    nx, ny, nz = grid_shape(size)
    n_bins, _, _ = hist_setup()
    accs = {}
    for s in SUBSETS:
        inv = voxel_invert.make_accumulator(nx, ny, nz)
        accs[s] = {
            "hist_dx": np.zeros((nx, ny, nz, n_bins), dtype=np.int32),
            "hist_dy": np.zeros((nx, ny, nz, n_bins), dtype=np.int32),
            "hist_dz": np.zeros((nx, ny, nz, n_bins), dtype=np.int32),
            "invert":  inv,
            "count":   inv["n"],
        }
    return accs


def gauss(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


def _seed_priors(counts, centers):
    """Per-voxel adaptive priors for a single-Gaussian fit.

    Returns (A0, mu0, sigma0, fit_lo, fit_hi). ``A0`` and ``mu0`` come from
    the histogram peak. ``sigma0`` is FWHM-based (FWHM/2.355) when the
    half-max width is well-defined, otherwise from a weighted variance over
    the immediate +/-5-bin neighborhood of the peak.
    """
    peak_idx = int(np.argmax(counts))
    A0 = float(counts[peak_idx])
    mu0 = float(centers[peak_idx])

    half = 0.5 * A0
    above = counts >= half
    sigma0 = float("nan")
    if above[peak_idx]:
        lo = peak_idx
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = peak_idx
        while hi < len(counts) - 1 and above[hi + 1]:
            hi += 1
        if hi > lo:
            fwhm = float(centers[hi] - centers[lo]) + HIST_BIN
            sigma0 = fwhm / 2.3548

    if not np.isfinite(sigma0) or sigma0 <= 0:
        lo = max(0, peak_idx - 5)
        hi = min(len(counts), peak_idx + 6)
        local_w = counts[lo:hi]
        local_c = centers[lo:hi]
        wsum = float(local_w.sum())
        if wsum > 0:
            local_mu = float((local_c * local_w).sum() / wsum)
            local_var = float(((local_c - local_mu) ** 2 * local_w).sum() / wsum)
            sigma0 = max(np.sqrt(max(local_var, 0.0)), 1.0)
        else:
            sigma0 = 2.0
    sigma0 = max(sigma0, 0.5)

    fit_half = max(FIT_WINDOW_HALF, 4.0 * sigma0)
    fit_lo = mu0 - fit_half
    fit_hi = mu0 + fit_half
    return A0, mu0, sigma0, fit_lo, fit_hi


def fit_voxel_gauss(hist, centers, min_counts):
    """Single-Gaussian fit with per-voxel adaptive priors. Returns
    (mu, mu_err, sigma, n)."""
    n = int(hist.sum())
    if n < min_counts:
        return float("nan"), float("nan"), float("nan"), n
    counts = hist.astype(float)
    if counts.max() <= 0:
        return float("nan"), float("nan"), float("nan"), n

    A0, mu0, sigma0, fit_lo, fit_hi = _seed_priors(counts, centers)
    in_win = (centers >= fit_lo) & (centers <= fit_hi)
    if in_win.sum() < 5:
        return float("nan"), float("nan"), float("nan"), n
    fc = counts[in_win]
    fx = centers[in_win]
    if fc.sum() < min_counts:
        return float("nan"), float("nan"), float("nan"), n
    err = np.sqrt(np.maximum(fc, 1.0))

    p0 = [A0, mu0, sigma0]
    bounds = (
        [1.0,             -HIST_HALF_RANGE, 0.1],
        [10.0 * A0 + 1e3, +HIST_HALF_RANGE, 200.0],
    )
    try:
        popt, pcov = curve_fit(
            gauss, fx, fc, p0=p0, sigma=err, absolute_sigma=True,
            bounds=bounds, maxfev=10000,
        )
        A_fit, mu_fit, sigma_fit = popt
        mu_err = float(np.sqrt(max(pcov[1, 1], 0.0)))
        return float(mu_fit), mu_err, float(sigma_fit), n
    except (RuntimeError, ValueError):
        return float("nan"), float("nan"), float("nan"), n


def accumulate_shard(path, accs, size, max_r_cm=25.0):
    cols = ["hit_x", "hit_y", "hit_z", "dx", "dy", "dz",
            "anode0_y", "anode0_z", "anode1_y", "anode1_z"]
    df = pd.read_parquet(path, columns=cols)
    x = df["hit_x"].to_numpy()
    y = df["hit_y"].to_numpy()
    z = df["hit_z"].to_numpy()
    dx = df["dx"].to_numpy(dtype=np.float64)
    dy = df["dy"].to_numpy(dtype=np.float64)
    dz = df["dz"].to_numpy(dtype=np.float64)
    # Track direction t̂ from the anode-to-anode line:
    # P0 = (-ANODE_X_HALF, anode0_y, anode0_z), P1 = (+ANODE_X_HALF, ...).
    a0y = df["anode0_y"].to_numpy(dtype=np.float64)
    a0z = df["anode0_z"].to_numpy(dtype=np.float64)
    a1y = df["anode1_y"].to_numpy(dtype=np.float64)
    a1z = df["anode1_z"].to_numpy(dtype=np.float64)
    del df

    track_vec = np.stack([
        np.full_like(a0y, 2.0 * ANODE_X_HALF),
        a1y - a0y,
        a1z - a0z,
    ], axis=1)
    track_len = np.linalg.norm(track_vec, axis=1)
    safe = track_len > 1e-3
    track_len_safe = np.where(safe, track_len, 1.0)
    t_hat = track_vec / track_len_safe[:, None]

    nx, ny, nz = grid_shape(size)
    n_bins, _, _ = hist_setup()
    ix = np.floor((x - X_RANGE[0]) / size).astype(np.int64)
    iy = np.floor((y - Y_RANGE[0]) / size).astype(np.int64)
    iz = np.floor((z - Z_RANGE[0]) / size).astype(np.int64)
    in_vol = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        & np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dz)
        & safe
        & (ix >= 0) & (ix < nx)
        & (iy >= 0) & (iy < ny)
        & (iz >= 0) & (iz < nz)
    )
    ix = ix[in_vol]; iy = iy[in_vol]; iz = iz[in_vol]
    dx = dx[in_vol]; dy = dy[in_vol]; dz = dz[in_vol]
    t_hat = t_hat[in_vol]
    r_vec = np.stack([dx, dy, dz], axis=1)

    N = nx * ny * nz
    flat = ix * (ny * nz) + iy * nz + iz

    sub = "all"
    # Per-hit outlier filter for the LSQ inverter (same convention as
    # plot_crtdisp_voxels.py: drop |r| > max_r_cm before accumulation).
    r_mag = np.linalg.norm(r_vec, axis=1)
    inv_ok = r_mag <= max_r_cm
    voxel_invert.add_hits(accs[sub]["invert"],
                          ix[inv_ok], iy[inv_ok], iz[inv_ok],
                          r_vec[inv_ok], t_hat[inv_ok])

    # Histograms (legacy diagnostic).
    for hist_key, vals in (("hist_dx", dx),
                           ("hist_dy", dy),
                           ("hist_dz", dz)):
        ib = np.floor((vals - (-HIST_HALF_RANGE)) / HIST_BIN).astype(np.int64)
        in_range = (ib >= 0) & (ib < n_bins)
        ff = flat[in_range]
        ibb = ib[in_range]
        comb = ff * n_bins + ibb
        cnt = np.bincount(comb, minlength=N * n_bins)
        accs[sub][hist_key] += cnt.reshape((nx, ny, nz, n_bins)).astype(np.int32)


SUB_LABEL = {"all": "all double-anode hits"}


def x_label(ix, size):
    lo = X_RANGE[0] + ix * size
    return f"x{int(round(lo)):+04d}"


def _plot_one(args):
    (subset, comp, ix, slice_grid, slice_count, vrange, min_counts,
     png_path, pdf_path, x_lo, x_hi, size) = args
    grid = slice_grid.copy()
    grid[slice_count < min_counts] = np.nan
    valid = grid[np.isfinite(grid)]
    if valid.size == 0:
        return (subset, comp, ix, 0)
    n_below = int(np.sum(valid < -vrange))
    n_above = int(np.sum(valid > vrange))
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid,
                   origin="lower", aspect="equal",
                   extent=(Z_RANGE[0], Z_RANGE[1], Y_RANGE[0], Y_RANGE[1]),
                   cmap=HEATMAP_CMAP,
                   vmin=-vrange, vmax=vrange,
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, extend="both",
                      label=f"LSQ-inverted $\\hat{{\\Delta}}_{{{comp[1]}}}$ [cm]")
    cb.ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Z [cm]")
    ax.set_ylabel("Y [cm]")
    ax.set_title(f"SBND double-anode LSQ-inverted $\\hat{{\\Delta}}_{{{comp[1]}}}$ "
                 f"({SUB_LABEL[subset]})\n"
                 f"X in [{x_lo:+.0f}, {x_hi:+.0f}] cm,  "
                 f"valid voxels = {valid.size}, "
                 f"clipped {n_below}/{n_above},  "
                 f"voxel = {size:g}$^3$ cm$^3$")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    fig.savefig(pdf_path)
    plt.close(fig)
    return (subset, comp, ix, valid.size)


def _plot_sigma(args):
    (subset, comp, ix, slice_grid, slice_count, vmax, min_counts,
     png_path, pdf_path, x_lo, x_hi, size) = args
    grid = slice_grid.copy()
    grid[slice_count < min_counts] = np.nan
    valid = grid[np.isfinite(grid)]
    if valid.size == 0:
        return (subset, comp, ix, 0)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid,
                   origin="lower", aspect="equal",
                   extent=(Z_RANGE[0], Z_RANGE[1], Y_RANGE[0], Y_RANGE[1]),
                   cmap=SIGMA_CMAP,
                   vmin=0, vmax=vmax,
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, extend="max",
                 label=f"LSQ $\\hat{{\\sigma}}_{{{comp[1]}}}$ [cm]")
    ax.set_xlabel("Z [cm]")
    ax.set_ylabel("Y [cm]")
    ax.set_title(f"SBND double-anode LSQ $\\hat{{\\sigma}}_{{{comp[1]}}}$ "
                 f"({SUB_LABEL[subset]})\n"
                 f"X in [{x_lo:+.0f}, {x_hi:+.0f}] cm,  "
                 f"valid voxels = {valid.size},  voxel = {size:g}$^3$ cm$^3$")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    fig.savefig(pdf_path)
    plt.close(fig)
    return (subset, comp, ix, valid.size)


def _plot_count(args):
    (subset, ix, slice_count, png_path, pdf_path, x_lo, x_hi, size) = args
    grid = slice_count.astype(float).copy()
    grid[grid == 0] = np.nan
    valid = grid[np.isfinite(grid)]
    if valid.size == 0:
        return (subset, ix, 0)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid,
                   origin="lower", aspect="equal",
                   extent=(Z_RANGE[0], Z_RANGE[1], Y_RANGE[0], Y_RANGE[1]),
                   cmap="viridis",
                   norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(1, valid.max())),
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="hits per voxel")
    ax.set_xlabel("Z [cm]")
    ax.set_ylabel("Y [cm]")
    ax.set_title(f"SBND double-anode voxel hit counts ({SUB_LABEL[subset]})\n"
                 f"X in [{x_lo:+.0f}, {x_hi:+.0f}] cm,  "
                 f"populated voxels = {valid.size},  "
                 f"voxel = {size:g}$^3$ cm$^3$")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    fig.savefig(pdf_path)
    plt.close(fig)
    return (subset, ix, valid.size)


def _fit_voxel_task(args):
    sub, comp, ix, iy, iz, hist, centers, min_counts = args
    mu, mu_err, sigma, n = fit_voxel_gauss(hist, centers, min_counts)
    return sub, comp, ix, iy, iz, mu, mu_err, sigma, n


def _plot_voxel_fit(args):
    (hist, centers, mu, mu_err, sigma, sub, comp,
     cx, cy, cz, vx_size, png_path) = args
    n = int(hist.sum())
    if n == 0:
        return None
    counts = hist.astype(float)
    err = np.sqrt(np.maximum(counts, 1.0))

    A0_seed, mu0_seed, sigma0_seed, fit_lo, fit_hi = _seed_priors(counts, centers)
    in_win = (centers >= fit_lo) & (centers <= fit_hi)
    fx = centers[in_win]; fc = counts[in_win]
    fe = np.sqrt(np.maximum(fc, 1.0))
    p0 = [A0_seed, mu0_seed, sigma0_seed]
    bounds = (
        [1.0,             -HIST_HALF_RANGE, 0.1],
        [10.0 * A0_seed + 1e3, +HIST_HALF_RANGE, 200.0],
    )
    try:
        popt, _ = curve_fit(gauss, fx, fc, p0=p0, sigma=fe,
                            absolute_sigma=True, bounds=bounds, maxfev=10000)
        A_fit, mu_fit, sigma_fit = popt
    except (RuntimeError, ValueError):
        A_fit, mu_fit, sigma_fit = A0_seed, mu0_seed, sigma0_seed

    arith_mean = (centers * counts).sum() / max(counts.sum(), 1.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(centers, counts, yerr=err, fmt="o", ms=3, lw=0.5,
                color="black", label=f"data (N={n:,})")
    xs = np.linspace(fit_lo, fit_hi, 400)
    ax.plot(xs, gauss(xs, A_fit, mu_fit, sigma_fit), color="darkgreen", lw=1.5,
            label=f"1G: $\\mu$={mu_fit:+.2f}$\\pm${mu_err if np.isfinite(mu_err) else 0:.2f}, "
                  f"$\\sigma$={sigma_fit:.2f}")
    ax.axvline(mu_fit, color="darkgreen", ls=":", lw=0.6)
    ax.axvline(0, color="grey", ls=":", lw=0.5)
    ax.axvline(arith_mean, color="orange", ls="-.", lw=0.6,
               label=f"arith mean = {arith_mean:+.2f}")
    ax.axvspan(fit_lo, fit_hi, color="lightyellow", alpha=0.4, zorder=0,
               label=f"fit window [{fit_lo:+.0f}, {fit_hi:+.0f}]")
    ax.set_xlabel(f"{comp} [cm]")
    ax.set_ylabel(f"hits / {HIST_BIN:g} cm")
    ax.set_title(f"{SUB_LABEL[sub]}, voxel center = "
                 f"({cx:+.0f}, {cy:+.0f}, {cz:+.0f}) cm  ({comp}, "
                 f"voxel = {vx_size:g}^3 cm^3)")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def stratified_voxel_sample(mu_grid, count_grid, min_counts, max_n, rng):
    valid = np.isfinite(mu_grid) & (count_grid >= min_counts)
    idxs = np.argwhere(valid)
    if len(idxs) == 0:
        return idxs
    if len(idxs) > max_n:
        sel = rng.choice(len(idxs), size=max_n, replace=False)
        idxs = idxs[sel]
    return idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT_DIR,
                    help="parquet directory or glob (default %(default)s)")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--voxel-size", type=float, default=VOXEL_SIZE)
    ap.add_argument("--min-counts", type=int, default=DEFAULT_MIN_COUNTS,
                    help="minimum hits per voxel for LSQ inversion to be "
                         "attempted (default %(default)s)")
    ap.add_argument("--max-r-cm", type=float, default=25.0,
                    help="reject hits with |r| > this BEFORE the LSQ "
                         "accumulation (default 25 cm). Diagnostic "
                         "histograms still include all hits.")
    ap.add_argument("--cond-max", type=float, default=voxel_invert.DEFAULT_COND_MAX,
                    help="reject voxels whose 3x3 LSQ matrix has condition "
                         "number greater than this (default %(default).0f). "
                         "Double-anode tracks have a restricted angular range "
                         "(piercing both anodes), so per-voxel angular "
                         "diversity is worse than for CRT crossers — this "
                         "threshold gates the worst voxels.")
    ap.add_argument("--vrange", type=float, default=DEFAULT_VRANGE)
    ap.add_argument("--sigma-vrange", type=float, default=DEFAULT_SIGMA_VRANGE)
    ap.add_argument("--nproc", default="auto")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--no-counts-maps", action="store_true")
    ap.add_argument("--no-sigma-maps", action="store_true")
    ap.add_argument("--no-save-npz", action="store_true")
    ap.add_argument("--num-sample-fits", type=int, default=200)
    args = ap.parse_args()

    if os.path.isdir(args.input):
        shards = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    else:
        shards = sorted(glob.glob(args.input))
    if not shards:
        print(f"No parquet shards found at {args.input}", file=sys.stderr)
        sys.exit(1)
    if args.max_shards:
        shards = shards[:args.max_shards]
    print(f"reading {len(shards)} shards from {args.input}", flush=True)

    nproc = cpu_count() if args.nproc == "auto" else int(args.nproc)
    size = args.voxel_size
    nx, ny, nz = grid_shape(size)
    print(f"voxel grid: {nx} x {ny} x {nz} = {nx*ny*nz:,} voxels", flush=True)

    accs = make_accumulators(size)
    for path in tqdm(shards, unit="shard", file=sys.stderr):
        accumulate_shard(path, accs, size, max_r_cm=args.max_r_cm)
    total_rows = sum(int(accs[s]["count"].sum()) for s in SUBSETS)
    print(f"\nrows accumulated: {total_rows:,}", flush=True)
    for sub in SUBSETS:
        n = int(accs[sub]["count"].sum())
        print(f"  {sub:5s}: {n:,} hits, "
              f"{(accs[sub]['count'] >= args.min_counts).sum():,} voxels >= "
              f"{args.min_counts} entries", flush=True)

    _, _, centers = hist_setup()
    mu_grids    = {s: {c: np.full((nx, ny, nz), np.nan, dtype=np.float32)
                       for c in COMPONENTS} for s in SUBSETS}
    sigma_grids = {s: {c: np.full((nx, ny, nz), np.nan, dtype=np.float32)
                       for c in COMPONENTS} for s in SUBSETS}
    cond_grids  = {s: np.full((nx, ny, nz), np.nan, dtype=np.float32)
                   for s in SUBSETS}
    delta_mu_grids  = {s: np.full((nx, ny, nz, 3), np.nan, dtype=np.float32)
                       for s in SUBSETS}
    delta_cov_grids = {s: np.full((nx, ny, nz, 6), np.nan, dtype=np.float32)
                       for s in SUBSETS}

    print(f"\nLSQ-inverting per-voxel projection for {len(SUBSETS)} subsets "
          f"(min_n={args.min_counts}, cond_max={args.cond_max}) ...", flush=True)
    for sub in SUBSETS:
        res = voxel_invert.invert_accumulator(
            accs[sub]["invert"],
            min_n=args.min_counts,
            cond_max=args.cond_max,
        )
        delta_mu_grids[sub]  = res["delta_mu"]
        delta_cov_grids[sub] = res["delta_cov"]
        cond_grids[sub]      = res["cond"]
        for ci, c in enumerate(COMPONENTS):
            mu_grids[sub][c]    = res["delta_mu"][..., ci]
            sigma_grids[sub][c] = np.sqrt(np.maximum(0.0,
                                                     res["delta_cov"][..., ci]))
        n_ok = int(np.isfinite(res["delta_mu"][..., 0]).sum())
        n_total = int((res["n"] >= args.min_counts).sum())
        print(f"  {sub:>5s}: {n_ok:,} voxels inverted "
              f"({n_total - n_ok:,} rejected by cond>{args.cond_max:.0f}; "
              f"{int(res['n'].sum()):,} total hits)", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    tasks = []
    for sub in SUBSETS:
        for comp in COMPONENTS:
            png_dir = os.path.join(args.outdir, sub, comp, "png")
            pdf_dir = os.path.join(args.outdir, sub, comp, "pdf")
            os.makedirs(png_dir, exist_ok=True)
            os.makedirs(pdf_dir, exist_ok=True)
            for ix_ in range(nx):
                x_lo = X_RANGE[0] + ix_ * size
                x_hi = x_lo + size
                tag = x_label(ix_, size)
                tasks.append((sub, comp, ix_,
                              mu_grids[sub][comp][ix_],
                              accs[sub]["count"][ix_],
                              args.vrange, args.min_counts,
                              os.path.join(png_dir, f"{tag}.png"),
                              os.path.join(pdf_dir, f"{tag}.pdf"),
                              x_lo, x_hi, size))

    print(f"\nplotting {len(tasks)} mu slices on {nproc} workers", flush=True)
    if nproc > 1:
        with Pool(processes=nproc) as pool:
            for _ in tqdm(pool.imap_unordered(_plot_one, tasks, chunksize=4),
                          total=len(tasks), unit="map", file=sys.stderr):
                pass
    else:
        for t in tqdm(tasks, unit="map", file=sys.stderr):
            _plot_one(t)

    if not args.no_sigma_maps:
        sig_tasks = []
        for sub in SUBSETS:
            for comp in COMPONENTS:
                png_dir = os.path.join(args.outdir, "sigma", sub, comp, "png")
                pdf_dir = os.path.join(args.outdir, "sigma", sub, comp, "pdf")
                os.makedirs(png_dir, exist_ok=True)
                os.makedirs(pdf_dir, exist_ok=True)
                for ix_ in range(nx):
                    x_lo = X_RANGE[0] + ix_ * size
                    x_hi = x_lo + size
                    tag = x_label(ix_, size)
                    sig_tasks.append((sub, comp, ix_,
                                      sigma_grids[sub][comp][ix_],
                                      accs[sub]["count"][ix_],
                                      args.sigma_vrange, args.min_counts,
                                      os.path.join(png_dir, f"{tag}.png"),
                                      os.path.join(pdf_dir, f"{tag}.pdf"),
                                      x_lo, x_hi, size))
        print(f"\nplotting {len(sig_tasks)} sigma slices", flush=True)
        if nproc > 1:
            with Pool(processes=nproc) as pool:
                for _ in tqdm(pool.imap_unordered(_plot_sigma, sig_tasks,
                                                  chunksize=4),
                              total=len(sig_tasks), unit="map", file=sys.stderr):
                    pass
        else:
            for t in tqdm(sig_tasks, unit="map", file=sys.stderr):
                _plot_sigma(t)

    if not args.no_save_npz:
        npz_path = os.path.join(args.outdir, "voxel_data.npz")
        print(f"\nsaving {npz_path}", flush=True)
        x_edges = np.linspace(X_RANGE[0], X_RANGE[1], nx + 1)
        y_edges = np.linspace(Y_RANGE[0], Y_RANGE[1], ny + 1)
        z_edges = np.linspace(Z_RANGE[0], Z_RANGE[1], nz + 1)
        n_bins, hist_edges, _ = hist_setup()
        save_args = {
            "voxel_size": np.float32(size),
            "x_edges": x_edges, "y_edges": y_edges, "z_edges": z_edges,
            "hist_edges": hist_edges,
            "min_counts": np.int32(args.min_counts),
        }
        for sub in SUBSETS:
            save_args[f"count_{sub}"]     = accs[sub]["count"].astype(np.int64)
            save_args[f"delta_mu_{sub}"]  = delta_mu_grids[sub]
            save_args[f"delta_cov_{sub}"] = delta_cov_grids[sub]
            save_args[f"cond_{sub}"]      = cond_grids[sub]
            for comp in COMPONENTS:
                save_args[f"hist_{sub}_{comp}"]  = accs[sub][f"hist_{comp}"]
                save_args[f"mu_{sub}_{comp}"]    = mu_grids[sub][comp]
                save_args[f"sigma_{sub}_{comp}"] = sigma_grids[sub][comp]
        np.savez_compressed(npz_path, **save_args)
        print(f"  -> {npz_path} "
              f"({os.path.getsize(npz_path)/1e9:.2f} GB)", flush=True)

    if args.num_sample_fits > 0:
        rng = np.random.default_rng(42)
        fit_plot_tasks = []
        for sub in SUBSETS:
            for comp in COMPONENTS:
                idxs = stratified_voxel_sample(
                    mu_grids[sub][comp], accs[sub]["count"],
                    args.min_counts, args.num_sample_fits, rng)
                if len(idxs) == 0:
                    continue
                out_subdir = os.path.join(args.outdir, "fits", sub, comp)
                os.makedirs(out_subdir, exist_ok=True)
                for ix_, iy_, iz_ in idxs:
                    cx_ = X_RANGE[0] + (ix_ + 0.5) * size
                    cy_ = Y_RANGE[0] + (iy_ + 0.5) * size
                    cz_ = Z_RANGE[0] + (iz_ + 0.5) * size
                    h = accs[sub][f"hist_{comp}"][ix_, iy_, iz_].copy()
                    mu_v = float(mu_grids[sub][comp][ix_, iy_, iz_])
                    sigma_v = float(sigma_grids[sub][comp][ix_, iy_, iz_])
                    tag = (f"x{int(round(cx_)):+04d}"
                           f"_y{int(round(cy_)):+04d}"
                           f"_z{int(round(cz_)):+04d}")
                    png_path = os.path.join(out_subdir, f"{tag}.png")
                    fit_plot_tasks.append((h, centers, mu_v, float("nan"),
                                           sigma_v, sub, comp, cx_, cy_, cz_,
                                           size, png_path))
        if fit_plot_tasks:
            print(f"\nplotting {len(fit_plot_tasks):,} sample voxel fits "
                  f"on {nproc} workers", flush=True)
            if nproc > 1:
                with Pool(processes=nproc) as pool:
                    for _ in tqdm(pool.imap_unordered(_plot_voxel_fit,
                                                       fit_plot_tasks,
                                                       chunksize=8),
                                  total=len(fit_plot_tasks), unit="fit",
                                  file=sys.stderr):
                        pass
            else:
                for t in tqdm(fit_plot_tasks, unit="fit", file=sys.stderr):
                    _plot_voxel_fit(t)

    if not args.no_counts_maps:
        cnt_tasks = []
        for sub in SUBSETS:
            png_dir = os.path.join(args.outdir, "counts", sub, "png")
            pdf_dir = os.path.join(args.outdir, "counts", sub, "pdf")
            os.makedirs(png_dir, exist_ok=True)
            os.makedirs(pdf_dir, exist_ok=True)
            for ix in range(nx):
                x_lo = X_RANGE[0] + ix * size
                x_hi = x_lo + size
                tag = x_label(ix, size)
                cnt_tasks.append((sub, ix, accs[sub]["count"][ix],
                                  os.path.join(png_dir, f"{tag}.png"),
                                  os.path.join(pdf_dir, f"{tag}.pdf"),
                                  x_lo, x_hi, size))
        print(f"\nplotting {len(cnt_tasks)} count slices", flush=True)
        if nproc > 1:
            with Pool(processes=nproc) as pool:
                for _ in tqdm(pool.imap_unordered(_plot_count, cnt_tasks, chunksize=4),
                              total=len(cnt_tasks), unit="map", file=sys.stderr):
                    pass
        else:
            for t in tqdm(cnt_tasks, unit="map", file=sys.stderr):
                _plot_count(t)

    print(f"\nwrote {args.outdir}/", flush=True)


if __name__ == "__main__":
    main()
