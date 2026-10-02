"""Tensor-product cubic B-spline machinery shared by the field-distortion
and cathode pipelines.

Both ``build_sce_input.fit_spline_half`` (3-D, per-TPC SCE displacement) and
``cathode_step2_spline.fit_spline`` (2-D, cathode face) previously carried
byte-identical copies of the clamped-knot / design-matrix / row-wise
Kronecker (Khatri-Rao) / LSMR-ridge machinery.  It lives here once.

Core primitives
---------------
``clamped_knots``        clamped cubic knot vector
``univariate_basis``     cubic B-spline design matrix (dense or sparse CSR)
``khatri_rao_sparse``    row-wise Kronecker product of two CSR matrices
``deriv_basis``          d^nu/dx of the design matrix (column-by-column)

High level
----------
``fit_tensor_spline``    weighted ridge LSQ fit of an N-D (2-D or 3-D)
                         tensor-product spline, optional boundary-column drop
``eval_spline_grid``     evaluate a fitted 3-D spline on a tensor grid
``eval_spline_deriv``    evaluate a mixed partial derivative on a tensor grid
``invert_displacement_grid``  numerically invert a forward displacement map
"""

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import BSpline
from scipy.sparse.linalg import lsmr

__all__ = [
    "clamped_knots", "univariate_basis", "khatri_rao_sparse", "deriv_basis",
    "fit_tensor_spline", "eval_spline_grid", "eval_spline_deriv",
    "invert_displacement_grid",
]


def clamped_knots(a, b, n_interior):
    """Clamped cubic knot vector with ``n_interior`` interior knots
    uniformly spaced in ``(a, b)``."""
    k = 3
    interior = np.linspace(a, b, n_interior + 2)[1:-1]
    return np.r_[np.full(k + 1, a), interior, np.full(k + 1, b)]


def univariate_basis(x, knots, *, sparse=False):
    """Design matrix from ``BSpline.design_matrix``.

    Returns shape ``(len(x), n_basis)`` dense ndarray (or sparse CSR if
    ``sparse=True``). ``n_basis = len(knots) - 4`` for cubic. Inputs are
    clipped to ``[knots[0], knots[-1]]`` for slight out-of-range queries.
    """
    k = 3
    a, b = float(knots[0]), float(knots[-1])
    x_clipped = np.clip(np.asarray(x, dtype=float), a, b)
    M = BSpline.design_matrix(x_clipped, knots, k, extrapolate=False)
    return M.tocsr() if sparse else np.asarray(M.todense())


def khatri_rao_sparse(A, B):
    """Row-wise Kronecker product of CSR matrices (same ``n_rows``).
    Result shape: ``(N, n_cols_A * n_cols_B)``."""
    A = A.tocsr(); B = B.tocsr()
    N = A.shape[0]; nB = B.shape[1]
    rows_o, cols_o, vals_o = [], [], []
    for n in range(N):
        ia = A.indices[A.indptr[n]:A.indptr[n+1]]
        va = A.data   [A.indptr[n]:A.indptr[n+1]]
        ib = B.indices[B.indptr[n]:B.indptr[n+1]]
        vb = B.data   [B.indptr[n]:B.indptr[n+1]]
        if ia.size == 0 or ib.size == 0:
            continue
        mi, mj = np.meshgrid(ia, ib, indexing="ij")
        rows_o.append(np.full(mi.size, n, dtype=np.int64))
        cols_o.append((mi * nB + mj).ravel())
        vals_o.append((va[:, None] * vb[None, :]).ravel())
    rows = np.concatenate(rows_o)
    cols = np.concatenate(cols_o)
    vals = np.concatenate(vals_o)
    return sp.csr_matrix((vals, (rows, cols)),
                         shape=(N, A.shape[1] * B.shape[1]))


def deriv_basis(x, knots, nu=1, k=3):
    """``d^nu/dx`` of the cubic B-spline design matrix, shape
    ``(len(x), n_basis)``.

    ``BSpline.design_matrix`` has no derivative option in older scipy, so the
    derivative is assembled column-by-column from each per-basis-function
    ``BSpline.derivative()``. ``nu=0`` reproduces the value basis.
    """
    n_basis = len(knots) - k - 1
    a, b = float(knots[0]), float(knots[-1])
    xc = np.clip(np.asarray(x, dtype=float), a, b)
    if nu == 0:
        return univariate_basis(xc, knots)
    cols = np.empty((xc.size, n_basis), dtype=float)
    for j in range(n_basis):
        e = np.zeros(n_basis)
        e[j] = 1.0
        spl = BSpline(knots, e, k, extrapolate=False).derivative(nu)
        cols[:, j] = np.nan_to_num(spl(xc), nan=0.0)
    return cols


def _solve_ridge(design, vals, weights, *, ridge_lambda):
    """Weighted ridge LSQ via LSMR: minimise ``||W(design c - vals)||^2 +
    lambda^2 ||c||^2``. ``design`` is a sparse CSR matrix ``(N, ncoef)``.
    Returns the coefficient vector ``c`` of length ``ncoef``."""
    w = np.asarray(weights, dtype=float)
    WDM = sp.diags(w) @ design
    Wv = np.asarray(vals, dtype=float) * w
    ncoef = design.shape[1]
    if ridge_lambda and ridge_lambda > 0:
        ridge = ridge_lambda * sp.eye(ncoef, format="csr")
        stacked = sp.vstack([WDM, ridge]).tocsr()
        rhs = np.r_[Wv, np.zeros(ncoef)]
    else:
        stacked = WDM.tocsr(); rhs = Wv
    return lsmr(stacked, rhs, atol=1e-8, btol=1e-8, maxiter=10000)[0]


def fit_tensor_spline(coords, knots, vals, weights, *,
                      ridge_lambda=0.0, drop_axis0_index=None):
    """Weighted ridge LSQ fit of a tensor-product cubic B-spline in 2-D or 3-D.

    Parameters
    ----------
    coords   : sequence of ``d`` arrays, each length ``N`` -- the sample
               coordinates along each axis (``d`` in {2, 3}).
    knots    : sequence of ``d`` clamped knot vectors.
    vals     : length-``N`` array of sampled values.
    weights  : length-``N`` array of LSQ weights (e.g. ``1/sigma``).
    ridge_lambda : Tikhonov coefficient-norm penalty.
    drop_axis0_index : if not None, drop the basis column with this index
               along axis 0 (clamps the fit to zero at that axis-0 knot end --
               used for the anode zero-displacement constraint).

    Returns
    -------
    coeffs : ndarray with shape ``(n_basis_0, ..., n_basis_{d-1})``.
    """
    coords = [np.asarray(c, dtype=float) for c in coords]
    d = len(coords)
    if d not in (2, 3):
        raise ValueError("fit_tensor_spline supports 2-D or 3-D only")
    bases = [univariate_basis(c, k, sparse=True) for c, k in zip(coords, knots)]
    nbasis = [b.shape[1] for b in bases]

    design = bases[0]
    for b in bases[1:]:
        design = khatri_rao_sparse(design, b)
    design = design.tocsr()
    ncoef = int(np.prod(nbasis))

    if drop_axis0_index is None:
        keep = np.ones(ncoef, dtype=bool)
    else:
        stride = int(np.prod(nbasis[1:]))
        all_idx = np.arange(ncoef)
        keep = (all_idx // stride) != drop_axis0_index
    design_red = design[:, keep]

    c_red = _solve_ridge(design_red, vals, weights, ridge_lambda=ridge_lambda)
    full = np.zeros(ncoef)
    full[keep] = c_red
    return full.reshape(*nbasis)


def eval_spline_grid(x_eval, y_eval, z_eval, kx, ky, kz, coeffs):
    """Evaluate the 3-D tensor-product spline at the eval-grid centers.

    Returns array of shape ``(len(x_eval), len(y_eval), len(z_eval))``.
    """
    Ax = univariate_basis(x_eval, kx)
    Ay = univariate_basis(y_eval, ky)
    Az = univariate_basis(z_eval, kz)
    t1 = np.einsum("ia,abc->ibc", Ax, coeffs)
    t2 = np.einsum("jb,ibc->ijc", Ay, t1)
    t3 = np.einsum("kc,ijc->ijk", Az, t2)
    return t3


def eval_spline_deriv(x_eval, y_eval, z_eval, kx, ky, kz, coeffs,
                      *, dx=0, dy=0, dz=0):
    """Evaluate a mixed partial derivative of the 3-D tensor-product spline.

    Same triple contraction as :func:`eval_spline_grid` but each axis design
    matrix is replaced by its derivative of the requested order (0 = value
    basis). Returns shape ``(len(x_eval), len(y_eval), len(z_eval))``.
    """
    Ax = deriv_basis(x_eval, kx, nu=dx)
    Ay = deriv_basis(y_eval, ky, nu=dy)
    Az = deriv_basis(z_eval, kz, nu=dz)
    t1 = np.einsum("ia,abc->ibc", Ax, coeffs)
    t2 = np.einsum("jb,ibc->ijc", Ay, t1)
    t3 = np.einsum("kc,ijc->ijk", Az, t2)
    return t3


def invert_displacement_grid(fwd_phys, x_centers, y_centers, z_centers,
                             *, n_iter=6, verbose=True):
    """Numerically invert a forward spatial-distortion map.

    ``fwd_phys`` is the *physical* forward displacement F on the SCE grid,
    shape ``(nx, ny, nz, 3)``: a true point ``r`` is reconstructed at
    ``r + F(r)``. Returns the physical backward map B of the same shape, with
    ``B(r) = -F(r + B(r))`` solved per grid point by fixed-point iteration
    with trilinear interpolation of F (matching the loader's
    ``TH3F::Interpolate``). Converges in ~4-5 passes.

    The map is inverted independently in each TPC half (x<0 incl. the cathode
    bin, and x>0); query points are clamped to their own half so the inversion
    never blends a forward value across the cathode.
    """
    from scipy.interpolate import RegularGridInterpolator as _RGI
    bkwd = np.zeros_like(fwd_phys, dtype=np.float64)
    for tag, ix in (("TPC0", np.where(x_centers < 0.0)[0]),
                    ("TPC1", np.where(x_centers > 0.0)[0])):
        if ix.size == 0:
            continue
        xc = x_centers[ix]
        F = [_RGI((xc, y_centers, z_centers),
                  np.asarray(fwd_phys[ix, :, :, c], dtype=np.float64),
                  method="linear", bounds_error=False, fill_value=None)
             for c in range(3)]
        X, Y, Z = np.meshgrid(xc, y_centers, z_centers, indexing="ij")
        R = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        lo = np.array([xc[0], y_centers[0], z_centers[0]])
        hi = np.array([xc[-1], y_centers[-1], z_centers[-1]])
        b = np.zeros_like(R)
        for it in range(1, n_iter + 1):
            P = np.clip(R + b, lo, hi)
            b_new = -np.stack([F[c](P) for c in range(3)], axis=1)
            dmax = float(np.abs(b_new - b).max())
            b = b_new
            if verbose:
                print(f"    {tag} invert iter {it}/{n_iter}: "
                      f"max|dB| = {dmax:.5f} cm", flush=True)
        bkwd[ix, :, :, :] = b.reshape(xc.size, y_centers.size,
                                      z_centers.size, 3)
    return bkwd
