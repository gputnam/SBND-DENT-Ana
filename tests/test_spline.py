"""Unit tests for the consolidated crtlib.spline machinery and the fact that
build_sce_input / cathode_step2_spline now delegate to it."""
import numpy as np
import pytest

from crtlib import spline as S


def test_clamped_knots_shape():
    k = S.clamped_knots(-200.0, 0.0, 4)
    # cubic clamped: 4 repeats each end + 4 interior = 12 knots, 8 basis fns
    assert len(k) == 4 + 4 + 4
    assert k[0] == -200.0 and k[-1] == 0.0
    assert np.all(np.diff(k) >= 0)


def test_univariate_basis_partition_of_unity():
    k = S.clamped_knots(0.0, 500.0, 6)
    x = np.linspace(0, 500, 97)
    A = S.univariate_basis(x, k)
    # cubic B-spline basis sums to 1 everywhere inside the support
    assert np.allclose(A.sum(axis=1), 1.0, atol=1e-10)


def test_fit_tensor_spline_2d_roundtrip():
    rng = np.random.default_rng(0)
    f = lambda y, z: 0.5 * np.sin(y / 80.0) + 0.3 * np.cos(z / 120.0)
    N = 4000
    y = rng.uniform(-200, 200, N); z = rng.uniform(0, 500, N)
    v = f(y, z)
    ky = S.clamped_knots(-200, 200, 6); kz = S.clamped_knots(0, 500, 9)
    coeffs = S.fit_tensor_spline([y, z], [ky, kz], v, np.ones(N),
                                 ridge_lambda=0.1)
    yg = np.linspace(-150, 150, 13); zg = np.linspace(50, 450, 13)
    Ay = S.univariate_basis(yg, ky); Az = S.univariate_basis(zg, kz)
    pred = Ay @ coeffs @ Az.T
    assert np.abs(pred - f(yg[:, None], zg[None, :])).max() < 2e-3


def test_fit_tensor_spline_3d_and_drop_axis0():
    rng = np.random.default_rng(1)
    N = 6000
    x = rng.uniform(-200, 0, N); y = rng.uniform(-200, 200, N)
    z = rng.uniform(0, 500, N)
    # linear field vanishing at the anode x=-200 (so drop_axis0_index=0 is exact)
    v = (x + 200.0) * 0.01
    kx = S.clamped_knots(-200, 0, 4)
    ky = S.clamped_knots(-200, 200, 4); kz = S.clamped_knots(0, 500, 5)
    coeffs = S.fit_tensor_spline([x, y, z], [kx, ky, kz], v, np.ones(N),
                                 ridge_lambda=0.01, drop_axis0_index=0)
    # evaluate at the anode plane -> must be ~0
    anode = S.eval_spline_grid([-200.0], [0.0], [250.0], kx, ky, kz, coeffs)
    assert abs(float(anode)) < 1e-6


def test_eval_spline_deriv_vs_finite_difference():
    rng = np.random.default_rng(2)
    kx = S.clamped_knots(-200, 0, 4)
    ky = S.clamped_knots(-200, 200, 5); kz = S.clamped_knots(0, 500, 6)
    c = rng.standard_normal((len(kx) - 4, len(ky) - 4, len(kz) - 4))
    xg = np.linspace(-180, -20, 7); yg = np.linspace(-150, 150, 6)
    zg = np.linspace(50, 450, 6)
    ana = S.eval_spline_deriv(xg, yg, zg, kx, ky, kz, c, dx=1)
    h = 1e-3
    num = (S.eval_spline_grid(xg + h, yg, zg, kx, ky, kz, c)
           - S.eval_spline_grid(xg - h, yg, zg, kx, ky, kz, c)) / (2 * h)
    assert np.abs(ana - num).max() < 1e-7


def test_build_sce_input_delegates():
    """The field builder re-exports the shared primitives (no private copy)."""
    import build_sce_input as B
    assert B.univariate_basis is S.univariate_basis
    assert B.eval_spline_grid is S.eval_spline_grid
    assert B.invert_displacement_grid is S.invert_displacement_grid


def test_cathode_step2_uses_shared_fit():
    """cathode_step2_spline.fit_spline reproduces a known smooth map."""
    import cathode_step2_spline as C
    y_c = np.arange(-200, 200, 2.0) + 1.0
    z_c = np.arange(0, 500, 2.0) + 1.0
    Y, Z = np.meshgrid(y_c, z_c, indexing="ij")
    truth = 0.4 * np.sin(Y / 90.0) + 0.2 * np.cos(Z / 140.0)
    err = np.full(truth.shape, 0.1)
    spl = C.fit_spline(truth, err, y_c, z_c, ny_knots=8, nz_knots=10,
                       weight_floor=0.05, ridge_lambda=1.0)
    fit = spl(y_c, z_c)
    assert np.nanmax(np.abs(fit - truth)) < 0.1
