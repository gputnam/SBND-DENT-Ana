"""Voxel-binned CRT-displacement maps from the CRT-disp parquet shards.

Per voxel + per subset, accumulate the moments
    Σ t̂_i t̂_iᵀ,  Σ r_i,  Σ t̂_i (t̂_i·r_i),  Σ |r_i|², n
where r_i = (dx, dy, dz) is the perpendicular residual from the CRT
straight line to a TPC-reconstructed hit and t̂_i is the unit track
direction (from `crt_end_xyz - crt_start_xyz`). The LSQ solution to

    (Σ_i (I - t̂_i t̂_iᵀ)) Δ̂_v  =  Σ_i (I - t̂_i t̂_iᵀ) r_i

recovers the UNBIASED 3-D distortion vector Δ̂_v in each voxel,
inverting the projection-perpendicular operator that the raw r_i
measurements implement (MicroBooNE technique, 2008.09765).
``crtlib.voxel_invert`` implements the streaming accumulator + batched
inverter; this script just wires it up + emits per-x-slice heatmaps.

For backwards-compat with the existing comparison plotters, the
**inverted Δ̂_v components are saved as ``mu_<subset>_<comp>``** in
voxel_data.npz (so compare_ns_ew_6pane_v*.py etc. read it transparently).
The LSQ per-axis uncertainty is saved as ``sigma_<subset>_<comp>``. New
fields ``delta_mu_<subset>`` (3-vec), ``delta_cov_<subset>`` (packed
6-entry symmetric covariance), and ``cond_<subset>`` (3×3 condition
number per voxel) are added for full-detail downstream analysis.

Per-voxel residual histograms (the legacy ``hist_<subset>_<comp>``
4-D arrays) are still accumulated and saved in the npz for offline
diagnostic — they are NOT used to compute mu anymore.

Streaming reader: shards are processed one at a time so peak resident
memory stays small (one shard's columns plus the 4D histogram
accumulators).

Output layout (under --outdir):
    {subset}/{dx,dy,dz}/{png,pdf}/x{+/-NNN}.{png,pdf}     Δ̂ maps
    sigma/{subset}/{dx,dy,dz}/{png,pdf}/x{NNN}.{png,pdf}  LSQ σ̂ maps
    cond/{subset}/{png,pdf}/x{NNN}.{png,pdf}              cond-number maps
    counts/{subset}/{png,pdf}/x{+/-NNN}.{png,pdf}         hit counts
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

INPUT_DIR = "/exp/sbnd/data/users/gputnam/efield-study"
OUTDIR = "crtdisp_maps"

VOXEL_SIZE = 20.0
X_RANGE = (-200.0, 200.0)   # cm; both TPCs span -200..+200 across cathode at 0
Y_RANGE = (-200.0, 200.0)
Z_RANGE = (0.0, 500.0)

# Per-voxel residual histogram (used for the Gaussian fit).
HIST_HALF_RANGE = 100.0     # cm; histogram covers (-HIST_HALF_RANGE, +HIST_HALF_RANGE)
HIST_BIN = 1.0              # cm/bin

# Single-Gaussian fit window (cm around the seeded peak). Bins outside
# the window are dropped from the fit so the long, flat bad-CRT-match
# tail doesn't drag the fitted sigma wide.
FIT_WINDOW_HALF = 25.0

# Subsets defined by which CRT taggers a track touched.
# Using parquet columns: crt_has_top, crt_has_S, crt_has_N, crt_has_W, crt_has_E.
#   side    : crt_side_only            (any side wall, no top)
#   ns      : touches only N or S walls (no E/W, no top)
#   ew      : touches only E or W walls (no N/S, no top)
#   topside : at least one top AND at least one side wall (any side)
#   topn    : at least one top AND N, no other side walls
#   tops    : at least one top AND S, no other side walls
#   tope    : at least one top AND E, no other side walls
#   topw    : at least one top AND W, no other side walls
_ALL_SUBSETS = ("side", "ns", "ew", "topside", "topn", "tops", "tope", "topw")
# SUBSETS is the active set; main() may shrink it via --subsets.
SUBSETS = _ALL_SUBSETS
COMPONENTS = ("dx", "dy", "dz")

DEFAULT_VRANGE = 5.0        # cm; tighter default since Gauss-fit mu sits near 0
DEFAULT_MIN_COUNTS = 30
DEFAULT_SIGMA_VRANGE = 20.0 # cm; sigma maps colour scale upper end

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
    """Per-subset accumulators.

    Each subset gets:
      - 4D histogram accumulators (nx, ny, nz, n_bins) for (dx, dy, dz)
        kept as diagnostic
      - voxel_invert moment accumulator (Σ t̂t̂ᵀ, Σ r, Σ t̂(t̂·r), Σ|r|², n)
        used for the unbiased LSQ inversion
      - a count grid (for the per-voxel hit-count plots; ALIAS to the
        invert accumulator's "n" — it's the same number)
    """
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
            # "count" is exposed as an alias to invert["n"] for back-compat
            # with the existing plotting helpers.
            "count":   inv["n"],
        }
    return accs


def gauss(x, A, mu, sigma):
    """Single Gaussian: A * exp(-(x - mu)^2 / (2 sigma^2))."""
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


def _seed_priors(counts, centers):
    """Per-voxel adaptive priors for a single-Gaussian fit.

    Returns (A0, mu0, sigma0, fit_lo, fit_hi). ``A0`` and ``mu0`` come from
    the histogram peak. ``sigma0`` is FWHM-based (FWHM/2.355) when the
    half-max width is well-defined, otherwise from a weighted variance over
    the immediate ±5-bin neighborhood of the peak. ``[fit_lo, fit_hi]`` is
    the symmetric window around the peak used to fit (defaults to
    ``FIT_WINDOW_HALF`` cm; widened if it would clip the half-max width).
    """
    peak_idx = int(np.argmax(counts))
    A0 = float(counts[peak_idx])
    mu0 = float(centers[peak_idx])

    # FWHM via the run of bins with counts >= A0/2 around the peak.
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

    # Fall back to a local weighted variance in a ±5-bin window.
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
    """Fit a single Gaussian to a voxel residual histogram.

    Per-voxel adaptive priors: A0 = peak count, mu0 = peak bin center,
    sigma0 = FWHM/2.355 (with a local-variance fallback). The fit is
    restricted to a ±max(FIT_WINDOW_HALF, 4*sigma0) cm window around the
    seeded peak so the long, flat tail of bad CRT-track matches doesn't
    drag the fitted sigma. Returns (mu, mu_err, sigma, n).
    """
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
        [1.0,                 -HIST_HALF_RANGE, 0.1],
        [10.0 * A0 + 1e3,     +HIST_HALF_RANGE, 200.0],
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


def accumulate_shard(path, accs, size, alignment=None,
                     apply_anode_to_tpc1=True, max_r_cm=25.0):
    """Read one parquet shard, bin (voxel x dx-bin), accumulate histograms.

    Per (subset, component) we maintain a 4-D int32 histogram of shape
    (nx, ny, nz, n_bins). We compute a flat index = voxel_index * n_bins +
    dx_bin and use ``np.bincount`` to fill all voxels' histograms in one
    pass per (subset, component).

    If ``alignment`` is provided (dict from
    ``apply_alignment_crtdisp.load_alignment``), the residual components
    (dx, dy, dz) are recomputed under the supplied per-wall or per-module
    correction; voxel position uses the ORIGINAL hit_xyz unchanged.
    """
    # Always read the CRT endpoints — needed for the track-direction t̂
    # that feeds the LSQ inverter (and for the optional alignment-applied
    # residual recomputation).
    cols = ["hit_x", "hit_y", "hit_z", "dx", "dy", "dz",
            "crt_start_x", "crt_start_y", "crt_start_z",
            "crt_end_x",   "crt_end_y",   "crt_end_z",
            "crt_side_only", "crt_has_top",
            "crt_has_S", "crt_has_N", "crt_has_W", "crt_has_E"]
    df = pd.read_parquet(path, columns=cols)
    x = df["hit_x"].to_numpy()
    y = df["hit_y"].to_numpy()
    z = df["hit_z"].to_numpy()
    start_xyz = np.stack([df["crt_start_x"].to_numpy(),
                          df["crt_start_y"].to_numpy(),
                          df["crt_start_z"].to_numpy()], axis=1)
    end_xyz = np.stack([df["crt_end_x"].to_numpy(),
                        df["crt_end_y"].to_numpy(),
                        df["crt_end_z"].to_numpy()], axis=1)
    if alignment is not None:
        from apply_alignment_crtdisp import apply_corrections
        hit_xyz = np.stack([x, y, z], axis=1)
        dx, dy, dz = apply_corrections(start_xyz, end_xyz, hit_xyz, alignment,
                                       apply_anode_to_tpc1=apply_anode_to_tpc1)
    else:
        dx = df["dx"].to_numpy(dtype=np.float64)
        dy = df["dy"].to_numpy(dtype=np.float64)
        dz = df["dz"].to_numpy(dtype=np.float64)
    side_only = df["crt_side_only"].to_numpy()
    has_top = df["crt_has_top"].to_numpy()
    has_S = df["crt_has_S"].to_numpy()
    has_N = df["crt_has_N"].to_numpy()
    has_W = df["crt_has_W"].to_numpy()
    has_E = df["crt_has_E"].to_numpy()
    del df

    # Build the per-row subset masks.
    has_any_side = has_S | has_N | has_W | has_E
    masks = {
        "side":    side_only,
        "ns":      (has_N | has_S) & ~has_W & ~has_E & ~has_top,
        "ew":      (has_E | has_W) & ~has_N & ~has_S & ~has_top,
        "topside": has_top & has_any_side,
        "topn":    has_top & has_N & ~has_S & ~has_W & ~has_E,
        "tops":    has_top & has_S & ~has_N & ~has_W & ~has_E,
        "tope":    has_top & has_E & ~has_N & ~has_S & ~has_W,
        "topw":    has_top & has_W & ~has_N & ~has_S & ~has_E,
    }

    nx, ny, nz = grid_shape(size)
    n_bins, _, _ = hist_setup()
    ix = np.floor((x - X_RANGE[0]) / size).astype(np.int64)
    iy = np.floor((y - Y_RANGE[0]) / size).astype(np.int64)
    iz = np.floor((z - Z_RANGE[0]) / size).astype(np.int64)
    in_vol = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        & np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dz)
        & (ix >= 0) & (ix < nx)
        & (iy >= 0) & (iy < ny)
        & (iz >= 0) & (iz < nz)
    )
    # Track-direction unit vector t̂ per row (constant per track but every
    # hit on the same track will have the same crt_start/end → same t̂).
    track_vec = end_xyz - start_xyz
    track_len = np.linalg.norm(track_vec, axis=1)
    safe = track_len > 1e-3
    in_vol = in_vol & safe
    # Avoid divide-by-zero
    track_len_safe = np.where(safe, track_len, 1.0)
    t_hat = track_vec / track_len_safe[:, None]

    ix = ix[in_vol]; iy = iy[in_vol]; iz = iz[in_vol]
    dx = dx[in_vol]; dy = dy[in_vol]; dz = dz[in_vol]
    t_hat = t_hat[in_vol]
    masks_in = {k: m[in_vol] for k, m in masks.items()}

    N = nx * ny * nz
    flat = ix * (ny * nz) + iy * nz + iz
    r_vec = np.stack([dx, dy, dz], axis=1)

    # Per-hit outlier filter for the LSQ inverter. The expected |r_i|
    # for a true distortion |Δ| ≤ ~10 cm is bounded by |Δ|·sin(θ_td) ≤
    # |Δ|. Hits with |r| > max_r_cm are almost certainly bad-quality
    # CRT matches (the same long tail the legacy Gauss fit excluded
    # with FIT_WINDOW_HALF=25 cm). They are EXCLUDED from the inverter
    # but still feed the diagnostic histograms.
    r_mag = np.linalg.norm(r_vec, axis=1)
    inv_ok = r_mag <= max_r_cm

    for sub in SUBSETS:
        m = masks_in[sub]
        f = flat[m]; sx = dx[m]; sy = dy[m]; sz = dz[m]

        # voxel_invert.add_hits accumulates Σ t̂t̂ᵀ, Σ r, Σ t̂(t̂·r),
        # Σ |r|², n directly into the per-subset moment accumulator
        # (with the outlier cut applied).
        m_inv = m & inv_ok
        voxel_invert.add_hits(accs[sub]["invert"],
                              ix[m_inv], iy[m_inv], iz[m_inv],
                              r_vec[m_inv], t_hat[m_inv])

        # Histograms (legacy; kept as diagnostic — not used to compute mu).
        for hist_key, vals in (("hist_dx", sx),
                               ("hist_dy", sy),
                               ("hist_dz", sz)):
            ib = np.floor((vals - (-HIST_HALF_RANGE)) / HIST_BIN).astype(np.int64)
            in_range = (ib >= 0) & (ib < n_bins)
            ff = f[in_range]
            ibb = ib[in_range]
            comb = ff * n_bins + ibb
            cnt = np.bincount(comb, minlength=N * n_bins)
            accs[sub][hist_key] += cnt.reshape((nx, ny, nz, n_bins)).astype(np.int32)


SUB_LABEL = {
    "side":    "side-CRT only",
    "ns":      "N+S walls only",
    "ew":      "E+W walls only",
    "topside": "top + any side wall",
    "topn":    "top + N only",
    "tops":    "top + S only",
    "tope":    "top + E only",
    "topw":    "top + W only",
}


def x_label(ix, size):
    """Filename label like 'x-200', 'x+090'."""
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
    ax.set_title(f"SBND CRT-TPC inverted $\\hat{{\\Delta}}_{{{comp[1]}}}$ "
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
    ax.set_title(f"SBND CRT-TPC LSQ $\\hat{{\\sigma}}_{{{comp[1]}}}$ "
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
    ax.set_title(f"SBND CRT-TPC voxel hit counts ({SUB_LABEL[subset]})\n"
                 f"X in [{x_lo:+.0f}, {x_hi:+.0f}] cm,  "
                 f"populated voxels = {valid.size},  "
                 f"voxel = {size:g}$^3$ cm$^3$")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    fig.savefig(pdf_path)
    plt.close(fig)
    return (subset, ix, valid.size)


def _fit_voxel_task(args):
    """Worker: fit one voxel's histogram. Returns flattened result."""
    sub, comp, ix, iy, iz, hist, centers, min_counts = args
    mu, mu_err, sigma, n = fit_voxel_gauss(hist, centers, min_counts)
    return sub, comp, ix, iy, iz, mu, mu_err, sigma, n


def _plot_voxel_fit(args):
    """Worker: render histogram + single-Gauss fit + arith mean for one voxel."""
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
    """Pick up to max_n (ix, iy, iz) tuples uniformly at random from voxels
    with successful fits. Returns an int array of shape (k, 3)."""
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
                    help="minimum hits per voxel for the LSQ inversion to "
                         "be attempted (default %(default)s)")
    ap.add_argument("--max-r-cm", type=float, default=25.0,
                    help="reject hits with |r| > this BEFORE the LSQ "
                         "accumulation (default 25 cm; matches the old "
                         "Gauss-fit window). The legacy diagnostic "
                         "histograms still include all hits.")
    ap.add_argument("--cond-max", type=float, default=voxel_invert.DEFAULT_COND_MAX,
                    help="reject voxels whose 3x3 LSQ matrix has condition "
                         "number greater than this (default %(default).0f). "
                         "Larger = include more ill-conditioned voxels with "
                         "less reliable Δ̂.")
    ap.add_argument("--vrange", type=float, default=DEFAULT_VRANGE,
                    help="symmetric mu-map color half-range, cm (default 5)")
    ap.add_argument("--sigma-vrange", type=float, default=DEFAULT_SIGMA_VRANGE,
                    help="sigma-map color upper limit, cm (default 20)")
    ap.add_argument("--nproc", default="auto")
    ap.add_argument("--max-shards", type=int, default=None,
                    help="for testing: only read this many shards")
    ap.add_argument("--no-counts-maps", action="store_true",
                    help="skip per-subset hit-count slice plots")
    ap.add_argument("--no-sigma-maps", action="store_true",
                    help="skip per-subset sigma slice plots")
    ap.add_argument("--no-save-npz", action="store_true",
                    help="skip writing voxel_data.npz with all hists/fits")
    ap.add_argument("--num-sample-fits", type=int, default=200,
                    help="random voxel-fit histograms to plot per "
                         "(subset, component) (default 200; 0 disables)")
    ap.add_argument("--alignment-results", default=None,
                    help="path to align_crt.py CSV (with centroid_x/y/z "
                         "columns). When supplied, the per-row residual "
                         "(dx, dy, dz) is recomputed by applying the "
                         "alignment to crt_start_xyz, crt_end_xyz, and "
                         "(for TPC1 hits) the hit position. Voxel binning "
                         "still uses the original hit_xyz.")
    ap.add_argument("--no-anode-shift-tpc1", action="store_true",
                    help="When --alignment-results is set: don't apply the "
                         "TPC1 anode correction to TPC1 hits. Useful as a "
                         "control to see CRT-only realignment effects.")
    ap.add_argument("--subsets", default=",".join(_ALL_SUBSETS),
                    help="Comma-separated list of subsets to produce. "
                         f"Choose from {_ALL_SUBSETS}. Default: all of them. "
                         "Example: --subsets side,ns,ew skips the top* "
                         "subsets and runs faster with smaller histograms.")
    args = ap.parse_args()

    # Apply subset filter early so all module-level helpers see it.
    requested = tuple(s.strip() for s in args.subsets.split(",") if s.strip())
    bad = [s for s in requested if s not in _ALL_SUBSETS]
    if bad:
        ap.error(f"unknown subsets {bad}; valid are {_ALL_SUBSETS}")
    global SUBSETS
    SUBSETS = requested
    print(f"subsets to produce: {SUBSETS}", flush=True)

    alignment = None
    if args.alignment_results:
        from apply_alignment_crtdisp import load_alignment
        alignment = load_alignment(args.alignment_results)
        print(f"loaded alignment: mode={alignment['mode']}, "
              f"n_units={len(alignment['ids'])}, "
              f"anode TPC1 dT = {alignment['anode_dT'][1]}, "
              f"dR(mrad) = {alignment['anode_dR'][1] * 1e3}",
              flush=True)
        print(f"  anode shift to TPC1 hits: "
              f"{'OFF' if args.no_anode_shift_tpc1 else 'ON'}", flush=True)

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

    # 1. Accumulate sum/count over shards (single-process; reading is I/O bound).
    accs = make_accumulators(size)
    total_rows = 0
    for path in tqdm(shards, unit="shard", file=sys.stderr):
        accumulate_shard(path, accs, size, alignment=alignment,
                         apply_anode_to_tpc1=not args.no_anode_shift_tpc1,
                         max_r_cm=args.max_r_cm)
        n = sum(int(accs[s]["count"].sum()) for s in SUBSETS)
        total_rows = n
    print(f"\nrows accumulated: {total_rows:,}", flush=True)
    for sub in SUBSETS:
        n = int(accs[sub]["count"].sum())
        print(f"  {sub:5s}: {n:,} hits, "
              f"{(accs[sub]['count'] >= args.min_counts).sum():,} voxels >= "
              f"{args.min_counts} entries", flush=True)

    # 2. Per-voxel LSQ inversion: solve M_v Δ̂_v = v_v in each voxel to
    # recover the UNBIASED 3-D distortion vector (MicroBooNE technique,
    # 2008.09765). Replaces the legacy Gaussian-fit-per-projection-component
    # step that was biased by the perpendicular projection geometry.
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
        # Back-compat: populate mu_<subset>_<comp> and sigma_<subset>_<comp>
        # with the inverted Δ̂ components + per-axis LSQ σ̂ so downstream
        # comparison plotters (compare_ns_ew_6pane_v4.py etc.) keep working.
        for ci, c in enumerate(COMPONENTS):
            mu_grids[sub][c]    = res["delta_mu"][..., ci]
            sigma_grids[sub][c] = np.sqrt(np.maximum(0.0,
                                                     res["delta_cov"][..., ci]))
        n_ok = int(np.isfinite(res["delta_mu"][..., 0]).sum())
        n_total = int((res["n"] >= args.min_counts).sum())
        print(f"  {sub:>7s}: {n_ok:,} voxels inverted "
              f"({n_total - n_ok:,} rejected by cond>{args.cond_max:.0f}; "
              f"{int(res['n'].sum()):,} total hits)", flush=True)

    # 3. Set up output directories and dispatch mu-map plotting tasks.
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

    # 3b. Sigma maps (one per subset/component/x-slice).
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

    # 4. Save the full per-voxel histogram + fit data to a single compressed
    # .npz so any voxel can be re-plotted offline without rerunning.
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

    # 5. Sample voxel-fit plots: pick a random subset per (subset, component)
    # and render histogram + Gauss+const fit. Saved to fits/{subset}/{comp}/.
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

    # 4. Counts maps (one per subset).
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
