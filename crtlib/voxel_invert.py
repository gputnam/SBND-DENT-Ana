"""Per-voxel least-squares inversion of the perpendicular-to-track
distortion projection.

Each measurement at a hit position is
    r_i  =  (I - t̂_i t̂_iᵀ) Δ
where r_i is the perpendicular vector from the CRT-derived straight line
to the TPC-reconstructed hit (already in the parquet as ``dx, dy, dz``),
t̂_i is the unit track direction, and Δ is the true 3-D distortion at
that voxel. The track-parallel component (Δ · t̂_i) is unobservable on
a single track — but across many tracks at different angles through
the same voxel, ordinary least squares recovers Δ.

Stack N hits in a voxel:
    M = Σ_i (I - t̂_i t̂_iᵀ)  =  N·I - Σ_i t̂_i t̂_iᵀ           (3×3)
    v = Σ_i (I - t̂_i t̂_iᵀ) r_i                                  (3-vec)
    Δ̂ = M⁻¹ v
The matrix M is positive semi-definite, and is full rank iff the
{t̂_i} span 3D. Its condition number is a reliable proxy for
inversion quality — masking on `cond > cond_max` rejects voxels
where tracks are nearly coplanar.

Streaming accumulator (so we don't hold per-voxel lists of hits):
per voxel we accumulate

    T_v = Σ_i t̂_i t̂_iᵀ                  (3×3 symmetric — 6 entries)
    s_v = Σ_i r_i                          (3-vec)
    p_v = Σ_i t̂_i (t̂_i · r_i)            (3-vec; correction term)
    q_v = Σ_i |r_i|²                       (scalar; for σ estimate)
    n_v = number of contributing hits      (int64)

Then v_v = s_v - p_v (= Σ_i P_i r_i, exploiting P_i symmetry), and we
solve M_v Δ̂ = v_v. Memory: 6 + 3 + 3 + 1 + 1 = 14 floats per voxel
per subset (SBND 20-cm grid × 8 subsets ≈ 11 MB, negligible).

Compatibility note: this module operates on PERPENDICULAR residuals r_i
which the existing builders (``make_crtdisp_df_sbnd.py``,
``make_doubleanode_df_sbnd.py``) already emit as the ``dx, dy, dz``
columns. The voxel binning still uses the *original* hit position
(``hit_x/y/z``), unchanged.
"""

import numpy as np

# Default thresholds — voxels failing either get NaN'd in invert_accumulator.
DEFAULT_MIN_HITS = 30
DEFAULT_COND_MAX = 1.0e3
DEFAULT_RIDGE = 1.0e-6


def make_accumulator(nx, ny, nz):
    """Return a per-voxel accumulator dict (numpy-array-backed, dense)."""
    return {
        # T_v = Σ t̂t̂ᵀ flattened as (xx, yy, zz, xy, xz, yz)
        "ttT":   np.zeros((nx, ny, nz, 6), dtype=np.float64),
        # s_v = Σ r_i (3-vec)
        "sum_r": np.zeros((nx, ny, nz, 3), dtype=np.float64),
        # p_v = Σ t̂_i (t̂_i · r_i) — used to assemble Σ P_i r_i
        "sum_tdr_t": np.zeros((nx, ny, nz, 3), dtype=np.float64),
        # q_v = Σ |r_i|² (for the per-voxel σ estimate)
        "sum_r2": np.zeros((nx, ny, nz), dtype=np.float64),
        # n_v
        "n":     np.zeros((nx, ny, nz), dtype=np.int64),
    }


def add_hits(acc, ix, iy, iz, r, t):
    """Add a batch of hits to the per-voxel accumulator.

    Parameters
    ----------
    acc : dict from :func:`make_accumulator`
    ix, iy, iz : (N,) int arrays — voxel indices (no in-bounds check;
        caller is responsible for filtering invalid voxel coordinates).
    r : (N, 3) — perpendicular residuals (cm).
    t : (N, 3) — UNIT track directions. If non-unit, the math still
        works but the M matrix is rescaled — keep t unit-normalized
        for clean cond numbers.
    """
    nx, ny, nz, _ = acc["ttT"].shape
    N = nx * ny * nz
    flat = (ix.astype(np.int64) * (ny * nz)
            + iy.astype(np.int64) * nz
            + iz.astype(np.int64))

    # 6 unique entries of t̂t̂ᵀ
    txx = t[:, 0] * t[:, 0]; tyy = t[:, 1] * t[:, 1]; tzz = t[:, 2] * t[:, 2]
    txy = t[:, 0] * t[:, 1]; txz = t[:, 0] * t[:, 2]; tyz = t[:, 1] * t[:, 2]
    for k, w in enumerate((txx, tyy, tzz, txy, txz, tyz)):
        acc["ttT"][..., k] += (
            np.bincount(flat, weights=w, minlength=N).reshape(nx, ny, nz)
        )

    for k in range(3):
        acc["sum_r"][..., k] += (
            np.bincount(flat, weights=r[:, k], minlength=N).reshape(nx, ny, nz)
        )

    # p_v = Σ t̂_i (t̂_i · r_i)
    tdr = t[:, 0] * r[:, 0] + t[:, 1] * r[:, 1] + t[:, 2] * r[:, 2]
    for k in range(3):
        acc["sum_tdr_t"][..., k] += (
            np.bincount(flat, weights=tdr * t[:, k], minlength=N)
            .reshape(nx, ny, nz)
        )

    acc["sum_r2"] += (
        np.bincount(flat, weights=(r * r).sum(axis=1), minlength=N)
        .reshape(nx, ny, nz)
    )
    acc["n"] += (
        np.bincount(flat, minlength=N).reshape(nx, ny, nz).astype(np.int64)
    )


def invert_accumulator(acc, min_n=DEFAULT_MIN_HITS, cond_max=DEFAULT_COND_MAX,
                        ridge=DEFAULT_RIDGE):
    """Solve M_v Δ̂_v = v_v in each voxel from the accumulated moments.

    Returns dict with keys:
      delta_mu  (nx, ny, nz, 3)  best-fit Δ (float32; NaN where invalid)
      delta_cov (nx, ny, nz, 6)  symmetric covariance, packed as
                                 (xx, yy, zz, xy, xz, yz) (float32)
      cond      (nx, ny, nz)     condition number of M_v
      n         (nx, ny, nz)     contributing hit count
    """
    nx, ny, nz, _ = acc["ttT"].shape
    nvox = nx * ny * nz

    n_flat   = acc["n"].reshape(nvox).astype(np.float64)
    T_flat   = acc["ttT"].reshape(nvox, 6)
    sumr_flat = acc["sum_r"].reshape(nvox, 3)
    sumtdrt_flat = acc["sum_tdr_t"].reshape(nvox, 3)
    sumr2_flat = acc["sum_r2"].reshape(nvox)

    # Build M = n I - T for every voxel.
    M = np.zeros((nvox, 3, 3), dtype=np.float64)
    M[:, 0, 0] = n_flat - T_flat[:, 0]
    M[:, 1, 1] = n_flat - T_flat[:, 1]
    M[:, 2, 2] = n_flat - T_flat[:, 2]
    M[:, 0, 1] = -T_flat[:, 3];  M[:, 1, 0] = -T_flat[:, 3]
    M[:, 0, 2] = -T_flat[:, 4];  M[:, 2, 0] = -T_flat[:, 4]
    M[:, 1, 2] = -T_flat[:, 5];  M[:, 2, 1] = -T_flat[:, 5]

    # Tikhonov ridge ∝ n (keeps relative scaling clean).
    diag = ridge * n_flat
    M[:, 0, 0] += diag; M[:, 1, 1] += diag; M[:, 2, 2] += diag

    # v = Σ P_i r_i = sum_r - sum_tdr_t
    v_flat = sumr_flat - sumtdrt_flat

    valid = n_flat >= min_n
    delta = np.full((nvox, 3), np.nan, dtype=np.float64)
    cond  = np.full(nvox, np.nan, dtype=np.float64)
    M_inv = np.full((nvox, 3, 3), np.nan, dtype=np.float64)

    if valid.any():
        Mv  = M[valid]
        # Batched cond + solve
        condv = np.linalg.cond(Mv)
        keep_local = condv <= cond_max
        condv_keep = np.where(keep_local, condv, np.nan)
        cond[valid] = condv_keep

        if keep_local.any():
            Mkeep = Mv[keep_local]
            vkeep = v_flat[valid][keep_local]
            try:
                # np.linalg.solve treats the last dim of b as "columns of B",
                # so a 3-vec right-hand side must be passed as (..., 3, 1)
                # and squeezed afterwards.
                solx = np.linalg.solve(Mkeep, vkeep[..., None])[..., 0]
            except np.linalg.LinAlgError:
                solx = np.full((Mkeep.shape[0], 3), np.nan)
            try:
                invx = np.linalg.inv(Mkeep)
            except np.linalg.LinAlgError:
                invx = np.full(Mkeep.shape, np.nan)
            keep_idx = np.where(valid)[0][keep_local]
            delta[keep_idx] = solx
            M_inv[keep_idx] = invx

    # σ² estimate per voxel (single scalar):
    # If we believe r_i = M Δ + ε with isotropic ε, then
    #     RSS = Σ |r_i - P_i Δ̂|² ≈ Σ|r|² - Δ̂ᵀ v
    # and σ² ≈ RSS / max(3n - 3, 1). Negative truncated to 0.
    rss_per_vox = sumr2_flat - np.einsum("ij,ij->i", delta, v_flat)
    rss_per_vox = np.where(np.isfinite(rss_per_vox), rss_per_vox, 0.0)
    sigma2 = np.maximum(0.0, rss_per_vox) / np.maximum(3.0 * n_flat - 3.0, 1.0)
    cov3 = sigma2[:, None, None] * M_inv

    cov_packed = np.full((nvox, 6), np.nan, dtype=np.float32)
    finite_inv = np.isfinite(M_inv[:, 0, 0])
    cov_packed[finite_inv, 0] = cov3[finite_inv, 0, 0]
    cov_packed[finite_inv, 1] = cov3[finite_inv, 1, 1]
    cov_packed[finite_inv, 2] = cov3[finite_inv, 2, 2]
    cov_packed[finite_inv, 3] = cov3[finite_inv, 0, 1]
    cov_packed[finite_inv, 4] = cov3[finite_inv, 0, 2]
    cov_packed[finite_inv, 5] = cov3[finite_inv, 1, 2]

    return {
        "delta_mu":  delta.astype(np.float32).reshape(nx, ny, nz, 3),
        "delta_cov": cov_packed.reshape(nx, ny, nz, 6),
        "cond":      cond.astype(np.float32).reshape(nx, ny, nz),
        "n":         acc["n"].copy(),
    }


def combine_accumulators(a, b):
    """Sum two accumulators (e.g. from different shards) in-place into a."""
    for k in ("ttT", "sum_r", "sum_tdr_t", "sum_r2"):
        a[k] += b[k]
    a["n"] += b["n"]
    return a


# ---------------------------------------------------------------------------
# Rank-1 (scalar-measurement) variant — used by the 2D double-anode method.
#
# The 2D wire-plane method measures, per hit and per wire plane, a single
# SCALAR perpendicular residual m_i of the hit from that plane's 2D
# (drift-x, wire) track line. That scalar is the projection of the 3-D
# distortion onto a known unit direction n̂_i:
#
#     m_i  =  n̂_i · Δ
#
# where n̂_i = (-slope_i, ŵ_y, ŵ_z) / sqrt(1 + slope_i²), slope_i is the
# 2D line's d(s_view)/dx, and ŵ is the plane's wire-measurement direction
# in (y, z). Stacking N such measurements in a voxel:
#
#     M = Σ_i n̂_i n̂_iᵀ        (3×3, PSD; full rank iff the {n̂_i} span 3D)
#     v = Σ_i n̂_i m_i          (3-vec)
#     Δ̂ = M⁻¹ v
#
# This is the rank-1 analogue of the rank-2 `(I - t̂t̂ᵀ)` projector above:
# three wire planes give three distinct n̂ directions per voxel, and the
# spread of track slopes supplies the x-sensitivity.
# ---------------------------------------------------------------------------

def make_scalar_accumulator(nx, ny, nz):
    """Per-voxel accumulator for scalar projected measurements m = n̂·Δ."""
    return {
        # Σ n̂n̂ᵀ flattened as (xx, yy, zz, xy, xz, yz)
        "nnT":    np.zeros((nx, ny, nz, 6), dtype=np.float64),
        # Σ n̂_i m_i  (3-vec) — the RHS vector v
        "sum_nm": np.zeros((nx, ny, nz, 3), dtype=np.float64),
        # Σ m_i²  (for the per-voxel σ estimate)
        "sum_m2": np.zeros((nx, ny, nz), dtype=np.float64),
        "n":      np.zeros((nx, ny, nz), dtype=np.int64),
    }


def add_scalar_hits(acc, ix, iy, iz, m, nhat):
    """Add a batch of scalar measurements to the accumulator.

    Parameters
    ----------
    acc : dict from :func:`make_scalar_accumulator`
    ix, iy, iz : (N,) int voxel indices (caller filters out-of-bounds).
    m : (N,) scalar measurements m_i = n̂_i·Δ (cm).
    nhat : (N, 3) UNIT measurement directions n̂_i.
    """
    nx, ny, nz, _ = acc["nnT"].shape
    N = nx * ny * nz
    flat = (ix.astype(np.int64) * (ny * nz)
            + iy.astype(np.int64) * nz
            + iz.astype(np.int64))
    nxc, nyc, nzc = nhat[:, 0], nhat[:, 1], nhat[:, 2]
    comps = (nxc * nxc, nyc * nyc, nzc * nzc,
             nxc * nyc, nxc * nzc, nyc * nzc)
    for k, w in enumerate(comps):
        acc["nnT"][..., k] += (
            np.bincount(flat, weights=w, minlength=N).reshape(nx, ny, nz))
    for k in range(3):
        acc["sum_nm"][..., k] += (
            np.bincount(flat, weights=nhat[:, k] * m,
                        minlength=N).reshape(nx, ny, nz))
    acc["sum_m2"] += (
        np.bincount(flat, weights=m * m, minlength=N).reshape(nx, ny, nz))
    acc["n"] += (
        np.bincount(flat, minlength=N).reshape(nx, ny, nz).astype(np.int64))


def invert_scalar_accumulator(acc, min_n=DEFAULT_MIN_HITS,
                              cond_max=DEFAULT_COND_MAX,
                              ridge=DEFAULT_RIDGE):
    """Solve M_v Δ̂_v = v_v per voxel for the scalar-measurement accumulator.

    Returns the same dict shape as :func:`invert_accumulator`:
      delta_mu (nx,ny,nz,3), delta_cov (nx,ny,nz,6), cond (nx,ny,nz),
      n (nx,ny,nz).
    """
    nx, ny, nz, _ = acc["nnT"].shape
    nvox = nx * ny * nz
    n_flat = acc["n"].reshape(nvox).astype(np.float64)
    T = acc["nnT"].reshape(nvox, 6)
    v_flat = acc["sum_nm"].reshape(nvox, 3)
    sum_m2 = acc["sum_m2"].reshape(nvox)

    M = np.zeros((nvox, 3, 3), dtype=np.float64)
    M[:, 0, 0] = T[:, 0]; M[:, 1, 1] = T[:, 1]; M[:, 2, 2] = T[:, 2]
    M[:, 0, 1] = T[:, 3]; M[:, 1, 0] = T[:, 3]
    M[:, 0, 2] = T[:, 4]; M[:, 2, 0] = T[:, 4]
    M[:, 1, 2] = T[:, 5]; M[:, 2, 1] = T[:, 5]
    diag = ridge * n_flat
    M[:, 0, 0] += diag; M[:, 1, 1] += diag; M[:, 2, 2] += diag

    valid = n_flat >= min_n
    delta = np.full((nvox, 3), np.nan, dtype=np.float64)
    cond = np.full(nvox, np.nan, dtype=np.float64)
    M_inv = np.full((nvox, 3, 3), np.nan, dtype=np.float64)

    if valid.any():
        Mv = M[valid]
        condv = np.linalg.cond(Mv)
        keep_local = condv <= cond_max
        cond[valid] = np.where(keep_local, condv, np.nan)
        if keep_local.any():
            Mkeep = Mv[keep_local]
            vkeep = v_flat[valid][keep_local]
            try:
                solx = np.linalg.solve(Mkeep, vkeep[..., None])[..., 0]
            except np.linalg.LinAlgError:
                solx = np.full((Mkeep.shape[0], 3), np.nan)
            try:
                invx = np.linalg.inv(Mkeep)
            except np.linalg.LinAlgError:
                invx = np.full(Mkeep.shape, np.nan)
            keep_idx = np.where(valid)[0][keep_local]
            delta[keep_idx] = solx
            M_inv[keep_idx] = invx

    # σ² per voxel: RSS = Σ(m_i - n̂_i·Δ̂)² = Σm² - Δ̂ᵀv ; dof = n - 3.
    rss = sum_m2 - np.einsum("ij,ij->i", delta, v_flat)
    rss = np.where(np.isfinite(rss), rss, 0.0)
    sigma2 = np.maximum(0.0, rss) / np.maximum(n_flat - 3.0, 1.0)
    cov3 = sigma2[:, None, None] * M_inv

    cov_packed = np.full((nvox, 6), np.nan, dtype=np.float32)
    fin = np.isfinite(M_inv[:, 0, 0])
    for k, (a, b) in enumerate(((0, 0), (1, 1), (2, 2),
                                (0, 1), (0, 2), (1, 2))):
        cov_packed[fin, k] = cov3[fin, a, b]

    return {
        "delta_mu":  delta.astype(np.float32).reshape(nx, ny, nz, 3),
        "delta_cov": cov_packed.reshape(nx, ny, nz, 6),
        "cond":      cond.astype(np.float32).reshape(nx, ny, nz),
        "n":         acc["n"].copy(),
    }


def combine_scalar_accumulators(a, b):
    """Sum two scalar accumulators in-place into a."""
    for k in ("nnT", "sum_nm", "sum_m2"):
        a[k] += b[k]
    a["n"] += b["n"]
    return a
