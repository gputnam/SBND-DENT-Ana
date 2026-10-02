"""Fit the doubleanode_maps_v4 voxel grid with two tensor-product cubic
B-spline meshes (one per TPC half), evaluate on the SBND SCE service's
TH3F binning, and write a ROOT file consumable by SpaceChargeServiceSBND
in the "Voxelized_TH3" representation.

Two splines:
    TPC0 spline: x in [-200, 0]
    TPC1 spline: x in [  0, +200]
No continuity is enforced across the cathode at x=0. The X-component of
each spline is **clamped to zero at its anode plane** (x=-200 or x=+200)
by dropping the corresponding boundary basis-function column from the
design matrix (clamped cubic B-spline endpoint property: only the first
basis function equals 1 at x=a; the last equals 1 at x=b; all others
vanish there). The same zero-anode constraint applies to all three
components (dx, dy, dz).

X-neighbor outlier filter: a voxel is marked NaN per-component if
|Delta_v[c] - median(Delta_{v +/- x-neighbor}[c])| exceeds
K * MAD_local (across the X-neighborhood) or exceeds an absolute floor.

Sign convention for the X output: SpaceChargeServiceSBND.cxx flips the
sign of TrueFwd_Displacement_X when x<0 (TPC0). To make the loader
reproduce our measured Delta_x, we store -Delta_x for x<0 and +Delta_x
for x>0. Y and Z components stored as-is.

Output: see --outdir (default
/exp/sbnd/data/users/gputnam/efield-study/sce_input_doubleanode/):
    sbnd_sce_fwd_doubleanode.root      # the 9 TH3F histograms
    fit_diagnostics/{dx,dy,dz}/png/x{NNN}.png  # 49 per-component slices
    fit_diagnostics/{dx,dy,dz}/{dx,dy,dz}.gif  # animated summary
    outlier_report.csv                 # voxels dropped at step 1.2
    spline_knots.json                  # the chosen knot vectors
    fit_summary.csv                    # per-component chi2/dof, residual rms

Usage:
    python build_sce_input.py
    python build_sce_input.py --nx-knots 4 --ny-knots 5 --nz-knots 7
    python build_sce_input.py --diagnostic-knots   # scan a few choices

The --diagnostic-knots mode prints chi2/dof and the worst extrapolated
value for several (nx,ny,nz) settings without writing the ROOT file.
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from PIL import Image
from scipy.interpolate import BSpline
from scipy.sparse.linalg import lsmr
from tqdm.auto import tqdm

# Shared spline + plotting primitives (consolidated into crtlib). Re-exported
# here under their historical names so the scripts that do
# `import build_sce_input as B; B.univariate_basis(...)` keep working.
from crtlib.spline import (
    clamped_knots, univariate_basis,
    khatri_rao_sparse as _khatri_rao_sparse,
    deriv_basis as _deriv_basis,
    eval_spline_grid, eval_spline_deriv, invert_displacement_grid,
)
from crtlib.plotting import make_gif, quantize_rgb
from crtlib.constants import DATA_DIR, SCE_REFERENCE_ROOT

NPZ_DEFAULT = "doubleanode_maps_v4_20cm_singleg/voxel_data.npz"
OUTDIR_DEFAULT = os.path.join(DATA_DIR, "sce_input_doubleanode")

# SCE TH3F binning verified from the cvmfs reference file
# (SCEoffsets_SBND_E500_voxelTH3.root):
#   X: 49 bins, [-204.17, +204.17] cm, bin width 8.333... cm,
#      bin centers -200.00, -191.67, ..., +200.00.
#   Y: same as X.
#   Z: 61 bins, [-4.17, +504.17] cm, bin centers 0, 8.33, ..., 500.
SCE_NX, SCE_NY, SCE_NZ = 49, 49, 61
SCE_X_EDGE = (-204.166667, 204.166667)
SCE_Y_EDGE = (-204.166667, 204.166667)
SCE_Z_EDGE = (-4.166667,   504.166667)

ANODE_X_TPC0 = -200.0
ANODE_X_TPC1 = +200.0
CATHODE_X    = 0.0

COMPONENTS = ("dx", "dy", "dz")  # axis index 0, 1, 2 in delta_mu_all


def sce_bin_centers():
    """SCE grid bin centers in (x, y, z), shape (49,), (49,), (61,)."""
    xc = np.linspace(SCE_X_EDGE[0], SCE_X_EDGE[1], SCE_NX + 1)
    yc = np.linspace(SCE_Y_EDGE[0], SCE_Y_EDGE[1], SCE_NY + 1)
    zc = np.linspace(SCE_Z_EDGE[0], SCE_Z_EDGE[1], SCE_NZ + 1)
    return (0.5 * (xc[:-1] + xc[1:]),
            0.5 * (yc[:-1] + yc[1:]),
            0.5 * (zc[:-1] + zc[1:]))


def voxel_centers(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def iterative_neighbor_fill(mu, sigma, *, n_iter, min_neighbors=1,
                            sigma_inflate=1.0):
    """Fill NaN voxels with the 26-neighbor (3-D) mean over n_iter passes.

    `mu`, `sigma` shape (nx, ny, nz, 3). Each pass:
      - find currently-empty voxels (`isnan` in component 0; the LSQ
        inversion fails for all three components together, so component
        0 is a reliable indicator).
      - average their 26 immediate 3-D neighbors over the finite values.
      - replace the NaN with that average; replace sigma with
        `sigma_inflate * mean(neighbor sigmas)`.
      - voxels with fewer than `min_neighbors` finite neighbors are NOT
        filled this pass (so they may still get filled in a later pass
        once their neighborhood grows).

    Returns `(mu_filled, sigma_filled, filled_mask, per_iter_counts)`.
    `filled_mask` is True for any voxel that was filled in any pass.
    """
    mu_out    = mu.copy()
    sigma_out = sigma.copy()
    filled_total = np.zeros(mu.shape[:3], dtype=bool)
    per_iter = []

    for it in range(int(n_iter)):
        empty = ~np.isfinite(mu_out[..., 0])
        if not empty.any():
            per_iter.append(0); break

        pad_mu    = np.pad(mu_out,    ((1,1),(1,1),(1,1),(0,0)),
                           constant_values=np.nan)
        pad_sigma = np.pad(sigma_out, ((1,1),(1,1),(1,1),(0,0)),
                           constant_values=np.nan)
        nx, ny, nz, _ = mu_out.shape

        nbr_mu, nbr_sig = [], []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nbr_mu .append(pad_mu   [1+dx:1+dx+nx, 1+dy:1+dy+ny, 1+dz:1+dz+nz, :])
                    nbr_sig.append(pad_sigma[1+dx:1+dx+nx, 1+dy:1+dy+ny, 1+dz:1+dz+nz, :])
        stk_mu  = np.stack(nbr_mu,  axis=0)        # (26, nx, ny, nz, 3)
        stk_sig = np.stack(nbr_sig, axis=0)

        n_finite_per_voxel = np.sum(np.isfinite(stk_mu[..., 0]), axis=0)
        mean_mu  = np.nanmean(stk_mu,  axis=0)     # (nx, ny, nz, 3)
        mean_sig = np.nanmean(stk_sig, axis=0)

        eligible = (empty
                    & (n_finite_per_voxel >= min_neighbors)
                    & np.isfinite(mean_mu[..., 0]))
        mu_out   [eligible] = mean_mu[eligible]
        sigma_out[eligible] = sigma_inflate * mean_sig[eligible]
        filled_total |= eligible
        per_iter.append(int(eligible.sum()))

    return mu_out, sigma_out, filled_total, per_iter


def x_neighbor_outlier_mask(mu, count, cond, *, min_count, cond_max,
                            k_mad, abs_floor):
    """Return mask of voxels to KEEP after the X-neighbor outlier cut.

    `mu` shape (nx, ny, nz, 3). Returns a bool mask of shape (nx, ny, nz)
    where True = keep all 3 components. A single bad component drops the
    whole voxel for that fit pass.
    """
    nx, ny, nz, _ = mu.shape
    keep = np.isfinite(mu).all(axis=-1)
    keep &= (count >= min_count)
    keep &= np.isfinite(cond) & (cond <= cond_max)

    # X-neighbor MAD test per component.
    drop_reason = np.zeros((nx, ny, nz), dtype=np.int8)
    for c in range(3):
        v = np.where(keep, mu[..., c], np.nan)  # (nx, ny, nz)
        for ix in range(nx):
            lo = ix - 1
            hi = ix + 1
            neighbors = []
            if lo >= 0:
                neighbors.append(v[lo])
            if hi < nx:
                neighbors.append(v[hi])
            stk = np.stack(neighbors, axis=0)  # (1 or 2, ny, nz)
            med = np.nanmedian(stk, axis=0)
            mad = np.nanmedian(np.abs(stk - med[None, ...]), axis=0)
            threshold = np.maximum(k_mad * mad, abs_floor)
            bad = (np.abs(v[ix] - med) > threshold) & np.isfinite(med) & np.isfinite(v[ix])
            keep[ix] &= ~bad
            drop_reason[ix][bad & (drop_reason[ix] == 0)] = c + 1
    return keep, drop_reason


# clamped_knots, univariate_basis, _khatri_rao_sparse now live in
# crtlib.spline (imported above) -- shared with cathode_step2_spline.


def fit_spline_half(x_v, y_v, z_v, vals, sigmas,
                    *, xa, xb, anode_side,
                    nx_int, ny_int, nz_int,
                    sigma_floor, ridge_lambda=1.0):
    """Fit one TPC half. Returns (knots_x, knots_y, knots_z, coeffs_3d).

    `vals` shape (N,3) with NaNs allowed (rows with any NaN dropped per
    component); we fit each component independently.

    `anode_side` in {"low", "high", "none"}: "low" drops the i=0 X-basis
    column (zeros the value at x=xa); "high" drops the last X-basis column;
    "none" keeps every column (no anode clamp -- for quantities that are
    nonzero at the anode, e.g. the E-field distortion, as opposed to the
    spatial displacement which vanishes there).
    Y and Z are unconstrained at their boundaries.

    Tikhonov regularization with ridge_lambda penalizes ||c||^2 to damp
    spurious oscillation in the under-constrained interior.
    """
    kx = clamped_knots(xa, xb, nx_int)
    ky = clamped_knots(-200.0, +200.0, ny_int)
    kz = clamped_knots(   0.0, +500.0, nz_int)

    Ax = univariate_basis(x_v, kx, sparse=True)  # (N, nx_basis) CSR
    Ay = univariate_basis(y_v, ky, sparse=True)
    Az = univariate_basis(z_v, kz, sparse=True)
    N   = Ax.shape[0]
    nx_b = Ax.shape[1]; ny_b = Ay.shape[1]; nz_b = Az.shape[1]

    # Apply boundary constraint: drop the X-basis column we want pinned to 0.
    # anode_side="none" keeps all columns (no anode clamp).
    if anode_side == "low":
        drop_x_idx = 0
    elif anode_side == "high":
        drop_x_idx = nx_b - 1
    elif anode_side == "none":
        drop_x_idx = None
    else:
        raise ValueError(anode_side)

    coeffs_per_comp = []
    chi2_per_comp = []
    for c in range(3):
        m = np.isfinite(vals[:, c])
        if m.sum() < 50:
            coeffs_per_comp.append(np.zeros((nx_b, ny_b, nz_b)))
            chi2_per_comp.append((np.nan, 0))
            continue
        Axm = Ax[m]; Aym = Ay[m]; Azm = Az[m]
        v = vals[m, c]
        s = np.clip(sigmas[m, c], sigma_floor, None)
        w = 1.0 / s

        # Sparse tensor-product design matrix.
        # outer-A_yz: (N, ny_b * nz_b) with 16 nonzeros/row.
        Ayz = _khatri_rao_sparse(Aym, Azm)
        # full A_3D: (N, nx_b * ny_b * nz_b) with 64 nonzeros/row.
        DM = _khatri_rao_sparse(Axm, Ayz).tocsr()
        # Boundary mask on the flat column index: keep columns whose
        # x-basis-index != drop_x_idx.
        if drop_x_idx is None:
            keep_cols = np.ones(nx_b * ny_b * nz_b, dtype=bool)
        else:
            all_idx = np.arange(nx_b * ny_b * nz_b)
            bad_cols = (all_idx // (ny_b * nz_b)) == drop_x_idx
            keep_cols = ~bad_cols
        # column slicing on sparse CSR
        DM_red = DM[:, keep_cols]

        # Weighted ridge LSQ: minimize ||W(DM_red c - v)||^2 + lambda^2 |c|^2
        WDM = sp.diags(w) @ DM_red
        Wv = v * w
        n_red = int(keep_cols.sum())
        if ridge_lambda > 0:
            ridge = ridge_lambda * sp.eye(n_red, format="csr")
            stacked = sp.vstack([WDM, ridge]).tocsr()
            rhs = np.r_[Wv, np.zeros(n_red)]
        else:
            stacked = WDM.tocsr(); rhs = Wv
        res = lsmr(stacked, rhs, atol=1e-8, btol=1e-8, maxiter=10000)
        c_red = res[0]

        full_c = np.zeros(nx_b * ny_b * nz_b)
        full_c[keep_cols] = c_red
        coeffs = full_c.reshape(nx_b, ny_b, nz_b)

        pred = DM @ full_c
        resid = v - pred
        chi2 = float(np.sum((resid * w) ** 2))
        dof = max(int(m.sum()) - n_red, 1)
        chi2_per_comp.append((chi2, dof))
        coeffs_per_comp.append(coeffs)

    return kx, ky, kz, coeffs_per_comp, chi2_per_comp


# eval_spline_grid, _deriv_basis, eval_spline_deriv now live in crtlib.spline
# (imported above).


def _build_th3f(arr_inner, x_edges, y_edges, z_edges, name):
    """Construct a TH3F (float32) model. arr_inner shape = (nx, ny, nz).

    uproot writes ROOT histograms with underflow/overflow bins included
    in the storage array, so we pad arr_inner to (nx+2, ny+2, nz+2)
    along each axis with zeros. The float32 dtype causes
    uproot.writing.identify.to_TH3x to select the TH3F subclass.
    """
    import uproot.writing.identify as wi
    nx, ny, nz = arr_inner.shape
    # ROOT TH3 layout: linear index = ix + (nx+2)*(iy + (ny+2)*iz),
    # i.e. X varies fastest, Z slowest. We pad with one underflow + one
    # overflow bin per axis and flatten in that order.
    full3d = np.zeros((nx + 2, ny + 2, nz + 2), dtype=np.float32)
    full3d[1:-1, 1:-1, 1:-1] = arr_inner.astype(np.float32)
    full = np.transpose(full3d, axes=(2, 1, 0)).ravel(order="C").astype(np.float32)
    x_ax = wi.to_TAxis("xaxis", "", nx, float(x_edges[0]), float(x_edges[-1]))
    y_ax = wi.to_TAxis("yaxis", "", ny, float(y_edges[0]), float(y_edges[-1]))
    z_ax = wi.to_TAxis("zaxis", "", nz, float(z_edges[0]), float(z_edges[-1]))
    return wi.to_TH3x(
        fName=name, fTitle=name, data=full,
        fEntries=float(np.count_nonzero(arr_inner)),
        fTsumw=float(arr_inner.sum()), fTsumw2=float((arr_inner ** 2).sum()),
        fTsumwx=0.0, fTsumwx2=0.0, fTsumwy=0.0, fTsumwy2=0.0,
        fTsumwxy=0.0, fTsumwz=0.0, fTsumwz2=0.0,
        fTsumwxz=0.0, fTsumwyz=0.0, fSumw2=None,
        fXaxis=x_ax, fYaxis=y_ax, fZaxis=z_ax,
    )


# Gen1 reference SCE ROOT (env-overridable via SBND_SCE_REFERENCE; see
# crtlib.constants). Kept as a module name for the many scripts that read
# `B.REFERENCE_SCE_ROOT_DEFAULT` as their --reference-root default.
REFERENCE_SCE_ROOT_DEFAULT = SCE_REFERENCE_ROOT


# invert_displacement_grid now lives in crtlib.spline (imported above).


def write_th3f_root(out_path, fx_hist, fy_hist, fz_hist,
                    x_edges, y_edges, z_edges,
                    *, reference_root=REFERENCE_SCE_ROOT_DEFAULT,
                    include_efield_mag=True, efield=None, efield_mag=None,
                    backward=None):
    """Write the SBND SCE TH3F file: forward displacement = our fit;
    backward displacement and E-field = our maps if supplied, else copied
    verbatim from the reference file the SBND Gen1 chain points at.

    `reference_root` is the path to that template file (default: the
    sbnd_data v01_03_00 release, which is bit-identical to v01_02_00 and
    v01_04_00 — the same SBND SCE input has shipped since Gen1).

    `include_efield_mag` also copies True_ElecField_Mag (the loader in
    SpaceChargeSBND.cxx doesn't read it, but it's present in the
    reference file and may be consumed by analysis code).

    `efield`, if given, is a (ex, ey, ez) tuple of (nx,ny,nz) arrays written
    as True_ElecField_{X,Y,Z} in place of the reference copy — used to ship
    a *measured* E-field map. `efield_mag` likewise overrides
    True_ElecField_Mag (else the magnitude is recomputed from `efield`, or,
    if `efield` is None too, copied from the reference).

    `backward`, if given, is a (bx, by, bz) tuple of (nx,ny,nz) arrays
    written as TrueBkwd_Displacement_{X,Y,Z} in place of the reference
    copy — used to ship a *measured* (numerically inverted) backward map.
    The arrays must already be in the file's storage convention (the same
    per-TPC X sign flip as the forward histogram).
    """
    import uproot
    src = uproot.open(reference_root)
    ref_keys = {k.split(";")[0] for k in src.keys()}

    def from_ref(name):
        if name not in ref_keys:
            raise KeyError(f"{name} missing from {reference_root}")
        a = src[name].values().astype(np.float32)
        return a

    out_arrays = {
        "TrueFwd_Displacement_X":  fx_hist.astype(np.float32),
        "TrueFwd_Displacement_Y":  fy_hist.astype(np.float32),
        "TrueFwd_Displacement_Z":  fz_hist.astype(np.float32),
    }
    if backward is not None:
        bx, by, bz = (np.asarray(a, np.float32) for a in backward)
        out_arrays["TrueBkwd_Displacement_X"] = bx
        out_arrays["TrueBkwd_Displacement_Y"] = by
        out_arrays["TrueBkwd_Displacement_Z"] = bz
        bkwd_src = "measured backward displacement"
    else:
        out_arrays["TrueBkwd_Displacement_X"] = from_ref("TrueBkwd_Displacement_X")
        out_arrays["TrueBkwd_Displacement_Y"] = from_ref("TrueBkwd_Displacement_Y")
        out_arrays["TrueBkwd_Displacement_Z"] = from_ref("TrueBkwd_Displacement_Z")
        bkwd_src = f"backward displacement copied from {reference_root}"
    if efield is not None:
        ex, ey, ez = (np.asarray(a, np.float32) for a in efield)
        out_arrays["True_ElecField_X"] = ex
        out_arrays["True_ElecField_Y"] = ey
        out_arrays["True_ElecField_Z"] = ez
        efield_src = "measured E-field"
    else:
        out_arrays["True_ElecField_X"] = from_ref("True_ElecField_X")
        out_arrays["True_ElecField_Y"] = from_ref("True_ElecField_Y")
        out_arrays["True_ElecField_Z"] = from_ref("True_ElecField_Z")
        efield_src = "reference E-field"
    if include_efield_mag:
        if efield_mag is not None:
            out_arrays["True_ElecField_Mag"] = np.asarray(efield_mag,
                                                          np.float32)
        elif efield is not None:
            out_arrays["True_ElecField_Mag"] = np.sqrt(
                ex.astype(np.float64) ** 2 + ey.astype(np.float64) ** 2
                + ez.astype(np.float64) ** 2).astype(np.float32)
        elif "True_ElecField_Mag" in ref_keys:
            out_arrays["True_ElecField_Mag"] = from_ref("True_ElecField_Mag")

    # Sanity-check that the reference file's binning matches ours.
    ref_x_ax = src["TrueFwd_Displacement_X"].axes[0]
    if not (np.allclose(ref_x_ax.edges()[0], x_edges[0], atol=1e-3)
            and np.allclose(ref_x_ax.edges()[-1], x_edges[-1], atol=1e-3)
            and len(ref_x_ax.edges()) == len(x_edges)):
        raise RuntimeError(
            f"binning mismatch between reference ({len(ref_x_ax.edges())-1} bins, "
            f"[{ref_x_ax.edges()[0]}, {ref_x_ax.edges()[-1]}]) and ours.")

    with uproot.recreate(out_path) as f:
        for name, arr in out_arrays.items():
            f[name] = _build_th3f(arr, x_edges, y_edges, z_edges, name)
    print(f"  wrote {out_path} "
          f"(forward displacement = fit; {efield_src}; {bkwd_src})",
          flush=True)


def make_diagnostic_plots(measured_x_slices, fitted_x_slices, x_centers,
                          y_centers, z_centers, outdir, *, vrange):
    """One 3-pane figure per (component, X-slice). Returns list of paths
    grouped by component."""
    cmap = matplotlib.colormaps["RdYlBu_r"].copy()
    cmap.set_bad("white")
    by_comp = {c: [] for c in COMPONENTS}

    for ci, comp in enumerate(COMPONENTS):
        sub = os.path.join(outdir, comp, "png")
        os.makedirs(sub, exist_ok=True)
        for ix, xc in enumerate(x_centers):
            meas = measured_x_slices[ci][ix]    # (nye_meas, nze_meas) or padded
            fit  = fitted_x_slices[ci][ix]      # (nye_eval, nze_eval)
            resid = fit - meas if meas.shape == fit.shape else None
            fig, axs = plt.subplots(1, 3, figsize=(18, 5),
                                    gridspec_kw={"width_ratios": [1, 1, 1]})
            for ax, grid, title in zip(
                axs,
                [meas, fit, resid if resid is not None else np.full_like(fit, np.nan)],
                [f"measured (voxel) {comp}",
                 f"fit on SCE grid {comp}",
                 f"fit - measured {comp}" if resid is not None else f"(no measurement)"]):
                if grid is None or not np.any(np.isfinite(grid)):
                    ax.set_axis_off()
                    ax.set_title(title)
                    continue
                im = ax.imshow(grid,
                               origin="lower", aspect="equal",
                               extent=(0, 500, -200, 200),
                               cmap=cmap, vmin=-vrange, vmax=vrange,
                               interpolation="nearest")
                ax.set_xlabel("Z [cm]")
                ax.set_ylabel("Y [cm]")
                ax.set_title(title)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(f"X = {xc:+.1f} cm", fontsize=12)
            fig.tight_layout()
            fname = os.path.join(sub, f"x{int(round(xc)):+04d}.png")
            fig.savefig(fname, dpi=110)
            plt.close(fig)
            by_comp[comp].append(fname)

    return by_comp


# make_gif now lives in crtlib.plotting (imported above).


def voxel_to_xslice_grid(mu_3d, vy_centers, vz_centers, vrange):
    """Pad measured voxel slice to display on the same extent (Y, Z).

    Returns shape (ny, nz) at the voxel resolution. Plot extents follow."""
    return mu_3d  # voxel grid already covers (-200,200)x(0,500) at 20 cm.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=NPZ_DEFAULT)
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    ap.add_argument("--min-count", type=int, default=200,
                    help="minimum hits per voxel to use in the fit "
                         "(default %(default)s; matches the LSQ-inversion floor)")
    ap.add_argument("--cond-max", type=float, default=300.0,
                    help="reject voxels with LSQ matrix cond > this "
                         "(default %(default)s)")
    ap.add_argument("--k-mad", type=float, default=5.0,
                    help="X-neighbor MAD multiplier for outlier flag "
                         "(default %(default)s)")
    ap.add_argument("--abs-floor", type=float, default=2.0,
                    help="absolute floor on the X-neighbor difference "
                         "(default %(default)s cm)")
    ap.add_argument("--sigma-floor", type=float, default=0.5,
                    help="floor on per-voxel sigma in the LSQ weight "
                         "(default %(default)s cm)")
    ap.add_argument("--ridge-lambda", type=float, default=2.0,
                    help="Tikhonov ridge multiplier on the spline "
                         "coefficient norm. Higher = smoother / more "
                         "damped. (default %(default)s)")
    ap.add_argument("--nx-knots", type=int, default=4,
                    help="interior knots in X per TPC half (default %(default)s)")
    ap.add_argument("--ny-knots", type=int, default=5,
                    help="interior knots in Y (default %(default)s)")
    ap.add_argument("--nz-knots", type=int, default=7,
                    help="interior knots in Z (default %(default)s)")
    ap.add_argument("--vrange", type=float, default=5.0,
                    help="color scale half-range for diagnostic plots "
                         "(default %(default)s cm)")
    ap.add_argument("--diagnostic-knots", action="store_true",
                    help="run several knot configurations and print "
                         "chi2/dof + extrapolation summary; do not write "
                         "the ROOT file")
    ap.add_argument("--no-gif", action="store_true",
                    help="skip GIF assembly (faster, smaller)")
    ap.add_argument("--fill-iters", type=int, default=2,
                    help="iterations of 26-neighbor 3-D average-fill on "
                         "empty voxels BEFORE the spline fit. 0 = disabled "
                         "(original behaviour). 2 fills voxels up to two "
                         "units from a measured value. (default %(default)s)")
    ap.add_argument("--fill-min-neighbors", type=int, default=1,
                    help="require this many finite 26-neighbors to fill a "
                         "voxel in a given iteration (default %(default)s)")
    ap.add_argument("--fill-sigma-inflate", type=float, default=2.0,
                    help="multiply the neighbor-mean sigma by this factor "
                         "when assigning sigma to a filled voxel "
                         "(higher => filled voxels get less weight in the "
                         "LSQ) (default %(default)s)")
    ap.add_argument("--reference-root", default=REFERENCE_SCE_ROOT_DEFAULT,
                    help="reference SCE root file from which the backward "
                         "displacement and E-field maps are copied "
                         "(default %(default)s, the Gen1 SBND sbnd_data "
                         "release)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- load
    d = np.load(args.npz)
    mu = d["delta_mu_all"].astype(np.float64)         # (20,20,25,3)
    cov = d["delta_cov_all"].astype(np.float64)       # (20,20,25,6) xx,yy,zz,xy,xz,yz
    count = d["count_all"]                            # (20,20,25)
    cond = d["cond_all"]                              # (20,20,25)
    x_edges = d["x_edges"]; y_edges = d["y_edges"]; z_edges = d["z_edges"]
    vx_c = voxel_centers(x_edges)
    vy_c = voxel_centers(y_edges)
    vz_c = voxel_centers(z_edges)

    # per-voxel per-component sigma = sqrt(diag(cov))
    sigma_v = np.sqrt(np.maximum(cov[..., :3], 0.0))  # (nx,ny,nz,3)

    # ---- outlier filter
    keep_mask, drop_reason = x_neighbor_outlier_mask(
        mu, count, cond,
        min_count=args.min_count, cond_max=args.cond_max,
        k_mad=args.k_mad, abs_floor=args.abs_floor)
    n_keep = int(keep_mask.sum())
    print(f"voxels kept after outlier filter: {n_keep:,} / {mu.size//3:,}",
          flush=True)

    # outlier report
    drop_rows = []
    for ix in range(mu.shape[0]):
        for iy in range(mu.shape[1]):
            for iz in range(mu.shape[2]):
                if drop_reason[ix, iy, iz] > 0:
                    drop_rows.append({
                        "ix": ix, "iy": iy, "iz": iz,
                        "x": vx_c[ix], "y": vy_c[iy], "z": vz_c[iz],
                        "mu_x": float(mu[ix, iy, iz, 0]),
                        "mu_y": float(mu[ix, iy, iz, 1]),
                        "mu_z": float(mu[ix, iy, iz, 2]),
                        "drop_component": int(drop_reason[ix, iy, iz] - 1),
                        "count": int(count[ix, iy, iz]),
                        "cond": float(cond[ix, iy, iz]),
                    })
    pd.DataFrame(drop_rows).to_csv(os.path.join(args.outdir, "outlier_report.csv"),
                                   index=False)
    print(f"wrote outlier_report.csv ({len(drop_rows)} rows)", flush=True)

    # apply mask: zero out the values we drop by setting to NaN
    mu_clean = np.where(keep_mask[..., None], mu, np.nan)
    sigma_clean = np.where(keep_mask[..., None], sigma_v, np.nan)

    # Iterative 26-neighbor fill of empty voxels.
    if args.fill_iters > 0:
        n_empty_before = int((~np.isfinite(mu_clean[..., 0])).sum())
        mu_clean, sigma_clean, filled_mask, per_iter = iterative_neighbor_fill(
            mu_clean, sigma_clean,
            n_iter=args.fill_iters,
            min_neighbors=args.fill_min_neighbors,
            sigma_inflate=args.fill_sigma_inflate)
        n_empty_after  = int((~np.isfinite(mu_clean[..., 0])).sum())
        n_filled_total = int(filled_mask.sum())
        print(f"neighbor-fill: {args.fill_iters} pass(es), "
              f"min_neighbors={args.fill_min_neighbors}, "
              f"sigma_inflate={args.fill_sigma_inflate}", flush=True)
        print(f"  per-iteration filled counts: {per_iter}", flush=True)
        print(f"  empty voxels before -> after: "
              f"{n_empty_before:,d} -> {n_empty_after:,d}  "
              f"(filled {n_filled_total:,d})", flush=True)

    # ---- split into two halves at x=0
    # Voxel center indices: TPC0 = ix in [0, 9] (x_c in [-190, -10]),
    #                      TPC1 = ix in [10, 19] (x_c in [+10, +190]).
    tpc0_mask = vx_c < 0
    tpc1_mask = vx_c > 0
    print(f"TPC0 voxel x range: {vx_c[tpc0_mask].min():.0f}..{vx_c[tpc0_mask].max():.0f} "
          f"(n_x={tpc0_mask.sum()})", flush=True)
    print(f"TPC1 voxel x range: {vx_c[tpc1_mask].min():.0f}..{vx_c[tpc1_mask].max():.0f} "
          f"(n_x={tpc1_mask.sum()})", flush=True)

    def collect(half_mask, x_axis_lo, x_axis_hi):
        """Return per-voxel (x, y, z, value, sigma) flat arrays for the half."""
        idx_x = np.where(half_mask)[0]
        n_per = len(idx_x) * len(vy_c) * len(vz_c)
        x_v = np.repeat(vx_c[idx_x], len(vy_c) * len(vz_c))
        y_v = np.tile(np.repeat(vy_c, len(vz_c)), len(idx_x))
        z_v = np.tile(vz_c, len(idx_x) * len(vy_c))
        sub_mu = mu_clean[idx_x].reshape(-1, 3)
        sub_sig = sigma_clean[idx_x].reshape(-1, 3)
        return x_v, y_v, z_v, sub_mu, sub_sig

    fit_summary = []
    halves = []
    for tag, half_mask, xa, xb, anode_side in [
        ("TPC0", tpc0_mask, ANODE_X_TPC0, CATHODE_X, "low"),
        ("TPC1", tpc1_mask, CATHODE_X, ANODE_X_TPC1, "high"),
    ]:
        x_v, y_v, z_v, vals, sigs = collect(half_mask, xa, xb)
        kx, ky, kz, coeffs, chi2 = fit_spline_half(
            x_v, y_v, z_v, vals, sigs,
            xa=xa, xb=xb, anode_side=anode_side,
            nx_int=args.nx_knots, ny_int=args.ny_knots, nz_int=args.nz_knots,
            sigma_floor=args.sigma_floor, ridge_lambda=args.ridge_lambda)
        print(f"  [{tag}] spline knots: nx={len(kx)-4-2} interior, "
              f"ny={len(ky)-4-2} interior, nz={len(kz)-4-2} interior", flush=True)
        for c_idx, c_name in enumerate(COMPONENTS):
            chi2_v, dof = chi2[c_idx]
            print(f"    {c_name}: chi2/dof = {chi2_v:.1f}/{dof}", flush=True)
            fit_summary.append({"half": tag, "component": c_name,
                                "chi2": chi2_v, "dof": dof,
                                "nx_knots": args.nx_knots,
                                "ny_knots": args.ny_knots,
                                "nz_knots": args.nz_knots})
        halves.append((tag, kx, ky, kz, coeffs))

    pd.DataFrame(fit_summary).to_csv(os.path.join(args.outdir, "fit_summary.csv"),
                                     index=False)

    # save knots
    knots_json = {}
    for tag, kx, ky, kz, _ in halves:
        knots_json[tag] = {"x": list(map(float, kx)),
                           "y": list(map(float, ky)),
                           "z": list(map(float, kz))}
    with open(os.path.join(args.outdir, "spline_knots.json"), "w") as fh:
        json.dump(knots_json, fh, indent=2)

    if args.diagnostic_knots:
        # extrapolation summary: evaluate at the SCE grid corners.
        sx, sy, sz = sce_bin_centers()
        for tag, kx, ky, kz, coeffs in halves:
            x_use = sx[sx < 0] if tag == "TPC0" else sx[sx > 0]
            for ci, comp in enumerate(COMPONENTS):
                grid = eval_spline_grid(x_use, sy, sz, kx, ky, kz, coeffs[ci])
                print(f"  [{tag} {comp}] eval range "
                      f"[{grid.min():+.2f}, {grid.max():+.2f}] cm",
                      flush=True)
        print("\ndiagnostic-knots run complete (no ROOT file written).",
              flush=True)
        return

    # ---- evaluate on SCE grid
    sx, sy, sz = sce_bin_centers()
    fwd = np.zeros((SCE_NX, SCE_NY, SCE_NZ, 3), dtype=np.float64)

    for tag, kx, ky, kz, coeffs in halves:
        if tag == "TPC0":
            x_idx = np.where(sx < 0)[0]
        else:
            x_idx = np.where(sx > 0)[0]
        x_use = sx[x_idx]
        for c_idx in range(3):
            grid = eval_spline_grid(x_use, sy, sz, kx, ky, kz, coeffs[c_idx])
            fwd[x_idx, :, :, c_idx] = grid

    # apply TPC0 sign flip on the X component to match loader's corr=-1
    flip_x = sx < 0
    fwd[flip_x, :, :, 0] *= -1.0

    # ---- write ROOT
    x_edges_full = np.linspace(SCE_X_EDGE[0], SCE_X_EDGE[1], SCE_NX + 1)
    y_edges_full = np.linspace(SCE_Y_EDGE[0], SCE_Y_EDGE[1], SCE_NY + 1)
    z_edges_full = np.linspace(SCE_Z_EDGE[0], SCE_Z_EDGE[1], SCE_NZ + 1)
    root_path = os.path.join(args.outdir, "sbnd_sce_fwd_doubleanode.root")
    write_th3f_root(root_path,
                    fwd[..., 0], fwd[..., 1], fwd[..., 2],
                    x_edges_full, y_edges_full, z_edges_full,
                    reference_root=args.reference_root)

    # ---- diagnostic plots: one per X slice (49 X slices per component)
    diag_dir = os.path.join(args.outdir, "fit_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    # measured slices: pad to (49, ny_meas, nz_meas) by nearest-X mapping
    # (the voxel grid has 20 X bins; SCE has 49; assign each SCE-X to the
    # voxel-X containing it, NaN if outside).
    measured_x_slices = [[None]*SCE_NX for _ in range(3)]
    fitted_x_slices   = [[None]*SCE_NX for _ in range(3)]
    # We display fits at voxel-Y-Z resolution to keep "measured" and "fit"
    # on identical grids; resample the fit onto vy_c / vz_c.
    for ci, comp in enumerate(COMPONENTS):
        for ix_sce, xc in enumerate(sx):
            # find matching voxel X bin
            iv = int(np.floor((xc - x_edges[0]) / 20.0))
            iv = max(0, min(iv, len(vx_c) - 1))
            measured = mu_clean[iv, :, :, ci].copy()  # (ny_vox, nz_vox)
            measured_x_slices[ci][ix_sce] = measured
        # evaluate fit on voxel grid for each SCE X-center
        for tag, kx, ky, kz, coeffs in halves:
            x_mask = (sx < 0) if tag == "TPC0" else (sx > 0)
            x_use = sx[x_mask]
            grid = eval_spline_grid(x_use, vy_c, vz_c, kx, ky, kz, coeffs[ci])
            for j, ix_sce in enumerate(np.where(x_mask)[0]):
                fitted_x_slices[ci][ix_sce] = grid[j]

    by_comp = make_diagnostic_plots(measured_x_slices, fitted_x_slices,
                                    sx, vy_c, vz_c, diag_dir,
                                    vrange=args.vrange)

    if not args.no_gif:
        for comp, files in by_comp.items():
            gif = os.path.join(diag_dir, comp, f"{comp}.gif")
            make_gif(files, gif, duration_ms=180)
            print(f"  GIF: {gif}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
