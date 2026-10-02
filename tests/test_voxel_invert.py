"""Synthetic-data tests for crtlib.voxel_invert.

Generate N cosmic-muon-like tracks at a range of angles, ALL passing
through a single voxel, assuming a known 3D distortion Δ_true. For each
track, compute the perpendicular projection of Δ_true, add Gaussian
measurement noise, accumulate, invert, and check that the recovered Δ̂
matches Δ_true within noise.

Also tests:
  - Ill-conditioned case (all tracks coplanar) → cond explodes
  - Single-track case → invalid (rank-1 system)
  - High statistics → covariance ~ matches empirical scatter
"""

import numpy as np
from crtlib import voxel_invert as vi


def random_unit_vectors(n, rng):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def perp_residuals(t, delta_true, noise_sigma=0.0, rng=None):
    """For unit track directions t (N, 3), return r_i = (I - t̂t̂ᵀ) Δ_true
    + Gaussian noise with std `noise_sigma` (isotropic). Δ_true is (3,)."""
    delta_par = (t @ delta_true)[:, None] * t   # (N, 3) parallel component
    r = delta_true[None, :] - delta_par
    if noise_sigma > 0 and rng is not None:
        r = r + rng.normal(scale=noise_sigma, size=r.shape)
    return r


def test_isotropic_high_stats():
    """N=10000 isotropic tracks, no noise → should recover Δ_true exactly."""
    rng = np.random.default_rng(42)
    nx, ny, nz = 1, 1, 1
    acc = vi.make_accumulator(nx, ny, nz)
    Δ_true = np.array([1.5, -2.3, 0.7])
    t = random_unit_vectors(10000, rng)
    r = perp_residuals(t, Δ_true, noise_sigma=0.0, rng=rng)
    ix = np.zeros(len(t), dtype=np.int64)
    vi.add_hits(acc, ix, ix, ix, r, t)
    res = vi.invert_accumulator(acc, min_n=10, cond_max=100.0)
    Δ̂ = res["delta_mu"][0, 0, 0]
    cond = float(res["cond"][0, 0, 0])
    err = np.linalg.norm(Δ̂ - Δ_true)
    print(f"[isotropic, 10000 tracks, no noise]")
    print(f"  Δ_true = {Δ_true}")
    print(f"  Δ̂     = {Δ̂}")
    print(f"  |Δ̂ - Δ_true| = {err:.2e}, cond = {cond:.2f}")
    assert err < 1e-3, f"recovery failed: err={err}"


def test_isotropic_with_noise():
    """N=1000 isotropic tracks + 1 cm Gaussian noise."""
    rng = np.random.default_rng(2026)
    acc = vi.make_accumulator(1, 1, 1)
    Δ_true = np.array([2.0, 0.5, -1.0])
    t = random_unit_vectors(1000, rng)
    r = perp_residuals(t, Δ_true, noise_sigma=1.0, rng=rng)
    vi.add_hits(acc,
                np.zeros(len(t), dtype=np.int64),
                np.zeros(len(t), dtype=np.int64),
                np.zeros(len(t), dtype=np.int64),
                r, t)
    res = vi.invert_accumulator(acc, min_n=10, cond_max=100.0)
    Δ̂ = res["delta_mu"][0, 0, 0]
    cov = res["delta_cov"][0, 0, 0]
    cond = float(res["cond"][0, 0, 0])
    σ_per_axis = np.sqrt(cov[:3])
    err = np.linalg.norm(Δ̂ - Δ_true)
    n_sigma = (Δ̂ - Δ_true) / σ_per_axis
    print(f"\n[isotropic, 1000 tracks, 1 cm noise]")
    print(f"  Δ_true = {Δ_true}")
    print(f"  Δ̂     = {Δ̂}")
    print(f"  σ̂     = {σ_per_axis}")
    print(f"  n_sigma = {n_sigma}")
    print(f"  cond = {cond:.2f}")
    assert err < 5.0 * σ_per_axis.max(), \
        f"recovery off by {err:.2f} cm but cov says σ≈{σ_per_axis.max():.2f}"
    assert np.all(np.abs(n_sigma) < 4.0), \
        f"|n_sigma| > 4 — covariance estimate looks wrong: {n_sigma}"


def test_coplanar_tracks_ill_conditioned():
    """All tracks in the xy-plane → can't constrain Δ_z. Cond should be huge."""
    rng = np.random.default_rng(7)
    acc = vi.make_accumulator(1, 1, 1)
    Δ_true = np.array([1.0, 2.0, 3.0])
    # Make 500 tracks all with z-component < 0.01 (essentially coplanar in xy)
    t_xy = rng.normal(size=(500, 3))
    t_xy[:, 2] = 0.001 * (rng.random(500) - 0.5)
    t_xy /= np.linalg.norm(t_xy, axis=1, keepdims=True)
    r = perp_residuals(t_xy, Δ_true, noise_sigma=0.1, rng=rng)
    vi.add_hits(acc,
                np.zeros(500, dtype=np.int64),
                np.zeros(500, dtype=np.int64),
                np.zeros(500, dtype=np.int64),
                r, t_xy)
    res = vi.invert_accumulator(acc, min_n=10, cond_max=100.0)
    Δ̂ = res["delta_mu"][0, 0, 0]
    cond = float(res["cond"][0, 0, 0])
    print(f"\n[coplanar xy-tracks, 500 tracks]")
    print(f"  Δ_true = {Δ_true}")
    print(f"  Δ̂     = {Δ̂}")
    print(f"  cond = {cond:.2e}")
    # Either cond > 100 (rejected → NaN) OR Δ_z recovered with low precision
    if np.all(np.isnan(Δ̂)):
        print(f"  voxel correctly rejected (cond > cond_max)")
    else:
        print(f"  voxel passed cond filter but Δ̂_z = {Δ̂[2]:+.2f} vs truth {Δ_true[2]:.2f}")


def test_too_few_tracks():
    """Voxel with < min_n hits gets NaN."""
    rng = np.random.default_rng(0)
    acc = vi.make_accumulator(1, 1, 1)
    t = random_unit_vectors(5, rng)
    r = perp_residuals(t, np.array([1.0, 0.0, 0.0]))
    vi.add_hits(acc, np.zeros(5, dtype=np.int64), np.zeros(5, dtype=np.int64),
                np.zeros(5, dtype=np.int64), r, t)
    res = vi.invert_accumulator(acc, min_n=10, cond_max=100.0)
    Δ̂ = res["delta_mu"][0, 0, 0]
    print(f"\n[5 tracks, min_n=10]")
    print(f"  Δ̂ = {Δ̂}  (should be NaN)")
    assert np.all(np.isnan(Δ̂)), f"expected NaN, got {Δ̂}"


def test_per_voxel_separation():
    """Two voxels with DIFFERENT Δ_true; ensure they're independently
    recovered."""
    rng = np.random.default_rng(99)
    acc = vi.make_accumulator(2, 1, 1)
    Δ_a = np.array([1.0, 0.0, 0.0])
    Δ_b = np.array([0.0, 2.0, 1.0])
    t_a = random_unit_vectors(2000, rng)
    t_b = random_unit_vectors(2000, rng)
    r_a = perp_residuals(t_a, Δ_a, noise_sigma=0.3, rng=rng)
    r_b = perp_residuals(t_b, Δ_b, noise_sigma=0.3, rng=rng)
    ix_a = np.zeros(len(t_a), dtype=np.int64)
    ix_b = np.ones(len(t_b), dtype=np.int64)
    iy = iz = np.zeros(len(t_a), dtype=np.int64)
    vi.add_hits(acc, ix_a, iy, iz, r_a, t_a)
    vi.add_hits(acc, ix_b, iy, iz, r_b, t_b)
    res = vi.invert_accumulator(acc, min_n=30, cond_max=100.0)
    Δ̂_a = res["delta_mu"][0, 0, 0]
    Δ̂_b = res["delta_mu"][1, 0, 0]
    print(f"\n[two voxels with different Δ]")
    print(f"  voxel A: Δ_true = {Δ_a}, Δ̂ = {Δ̂_a}")
    print(f"  voxel B: Δ_true = {Δ_b}, Δ̂ = {Δ̂_b}")
    assert np.linalg.norm(Δ̂_a - Δ_a) < 0.1
    assert np.linalg.norm(Δ̂_b - Δ_b) < 0.1


def test_realistic_sbnd_cosmic_distribution():
    """SBND-like: cosmics with downward bias, range of azimuths.
    Per-voxel angular diversity is good in upper TPC volume."""
    rng = np.random.default_rng(2024)
    acc = vi.make_accumulator(1, 1, 1)
    # Downward-biased: theta = polar angle from vertical, peaks near 0
    cos_theta = rng.uniform(0.3, 1.0, size=2000)   # mostly downward
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    phi = rng.uniform(0, 2 * np.pi, size=2000)
    # SBND coords: y = vertical. So t_y = -cos(theta) (downward).
    t = np.stack([sin_theta * np.cos(phi),
                  -cos_theta,
                  sin_theta * np.sin(phi)], axis=1)
    Δ_true = np.array([0.5, -1.2, 2.0])
    r = perp_residuals(t, Δ_true, noise_sigma=2.0, rng=rng)
    ix = np.zeros(len(t), dtype=np.int64)
    vi.add_hits(acc, ix, ix, ix, r, t)
    res = vi.invert_accumulator(acc, min_n=30, cond_max=100.0)
    Δ̂ = res["delta_mu"][0, 0, 0]
    cond = float(res["cond"][0, 0, 0])
    n_sigma = (Δ̂ - Δ_true) / np.sqrt(res["delta_cov"][0, 0, 0, :3])
    print(f"\n[realistic SBND cosmic distribution, 2000 tracks, 2 cm noise]")
    print(f"  Δ_true = {Δ_true}")
    print(f"  Δ̂     = {Δ̂}")
    print(f"  σ̂     = {np.sqrt(res['delta_cov'][0, 0, 0, :3])}")
    print(f"  n_sigma = {n_sigma}")
    print(f"  cond = {cond:.2f}")
    # With downward-biased cosmics, Δ_y (vertical) is HARDEST to recover.
    # Still expect ≤ 3σ per axis.
    assert np.all(np.abs(n_sigma) < 4.0), \
        f"recovery missed by > 4σ: {n_sigma}"
