"""Unit tests for the SBND-adapted drift-velocity model."""
import numpy as np

import drift_model as D
from crtlib.constants import VDRIFT_MEASURED


def test_anchor_at_nominal_field():
    # v_of_E(0.5 kV/cm) is pinned to the SBND measurement by FUDGE.
    assert abs(D.v_of_E(0.5) - VDRIFT_MEASURED) < 1e-6
    assert abs(D.v_of_E(0.5) - D.SBND_V0_CMUS) < 1e-12


def test_inverse_roundtrip():
    for E in np.arange(0.2, 0.95, 0.1):
        assert abs(D.E_of_v(D.v_of_E(E)) - E) < 1e-6


def test_monotonic_increasing():
    Es = np.linspace(D.E_LO_KVCM, D.E_HI_KVCM, 400)
    assert np.all(np.diff(D.v_of_E(Es)) > 0)


def test_log_slope_small_distortion_limit():
    # alpha ~ (E0/v0) dv/dE at nominal; documented ~0.4636.
    alpha = D.ve_log_slope()
    assert 0.40 < alpha < 0.52
    # Small fractional field change -> fractional speed change ~ alpha * dE/E.
    E0 = 0.5
    dE = 1e-3
    frac_v = (D.v_of_E(E0 + dE) - D.v_of_E(E0)) / D.v_of_E(E0)
    frac_E = dE / E0
    assert abs(frac_v / frac_E - alpha) < 1e-2


def test_array_inverse_nan_out_of_range():
    v = np.array([D.v_of_E(0.5), 1e3, -1.0, np.nan])
    out = D.E_of_v_array(v)
    assert abs(out[0] - 0.5) < 1e-6
    assert np.all(np.isnan(out[1:]))
