"""Unit test for the numerical forward->backward displacement inversion."""
import numpy as np

from crtlib.spline import invert_displacement_grid


def test_forward_backward_closure():
    # SCE-like grid, per-TPC smooth forward displacement vanishing at anodes.
    xc = np.linspace(-200, 200, 25)
    yc = np.linspace(-200, 200, 13)
    zc = np.linspace(0, 500, 15)
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing="ij")
    fwd = np.zeros((*X.shape, 3))
    # drift-distance from each TPC's anode
    d = np.where(X < 0, X + 200.0, 200.0 - X)
    s = np.sin(np.pi * d / 200.0)
    fwd[..., 0] = 3.0 * s * np.cos(Y / 120.0)
    fwd[..., 1] = 2.0 * s * np.sin(Z / 160.0)
    fwd[..., 2] = 1.5 * s

    bkwd = invert_displacement_grid(fwd, xc, yc, zc, n_iter=12, verbose=False)

    # forward o backward closure well inside each half (avoid the cathode seam
    # and the outer edges where the interpolant clamps)
    from scipy.interpolate import RegularGridInterpolator as RGI
    interior = (np.abs(xc) > 30) & (np.abs(xc) < 180)
    ix = np.where(interior)[0]
    F = [RGI((xc, yc, zc), fwd[..., c], bounds_error=False, fill_value=None)
         for c in range(3)]
    worst = 0.0
    for i in ix:
        for j in range(2, len(yc) - 2):
            for k in range(2, len(zc) - 2):
                r = np.array([xc[i], yc[j], zc[k]])
                b = bkwd[i, j, k]
                # closure: r should map to ~r under B then F
                p = r + b
                fp = np.array([F[c](p)[0] for c in range(3)])
                worst = max(worst, np.linalg.norm(b + fp))
    assert worst < 0.05, f"closure RMS too large: {worst:.4f} cm"
