"""Unit tests for the spatial-gradient -> E-field inversion
(build_efield_2d_voxels.invert_to_efield)."""
import numpy as np

import build_efield_2d_voxels as E
import drift_model as DM


def test_zero_gradient_gives_nominal_field():
    # No distortion gradient -> |E| == E0 == 500 V/cm, zero dE.
    vx = np.array([-150.0, 150.0])  # one voxel per TPC half
    g = np.zeros((2, 1, 1, 3))
    dE, Emag, speed = E.invert_to_efield(g, vx, DM.v_of_E(0.5))
    assert np.allclose(Emag, 500.0, atol=1e-3)
    assert np.allclose(dE, 0.0, atol=1e-3)
    assert np.allclose(speed, DM.v_of_E(0.5), atol=1e-9)


def test_small_distortion_linear_limit():
    # For small gradients, dEx/E0 ~ -gx/alpha and dE{y,z}/E0 ~ -g{y,z}
    # in the TPC-local frame (per build_efield's dE_lin_frac).
    alpha = DM.ve_log_slope()
    vx = np.array([-150.0])          # TPC0 (sign +1)
    g = np.zeros((1, 1, 1, 3))
    g[0, 0, 0] = [0.002, 0.003, -0.004]
    dE, Emag, _ = E.invert_to_efield(g, vx, DM.v_of_E(0.5))
    frac = dE[0, 0, 0] / 500.0       # lab-frame fractional offset, sign=+1 here
    lin = np.array([-g[0, 0, 0, 0] / alpha, -g[0, 0, 0, 1], -g[0, 0, 0, 2]])
    assert np.allclose(frac, lin, atol=2e-3)


def test_tpc_sign_antisymmetry():
    # Same gradient in both halves -> drift-field component flips sign across
    # the cathode (sign = +1 for x<0, -1 for x>0).
    vx = np.array([-150.0, 150.0])
    g = np.zeros((2, 1, 1, 3))
    g[:, 0, 0] = [0.01, 0.0, 0.0]
    dE, _, _ = E.invert_to_efield(g, vx, DM.v_of_E(0.5))
    assert np.sign(dE[0, 0, 0, 0]) == -np.sign(dE[1, 0, 0, 0])
