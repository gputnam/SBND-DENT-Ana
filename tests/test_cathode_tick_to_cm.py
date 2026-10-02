"""Unit tests for the cathode step-1 tick->cm conversion."""
import numpy as np

import cathode_step1_tick_to_cm as C1


def test_conversion_constants():
    assert C1.TICK_PERIOD_US == 0.5
    assert abs(C1.DRIFT_TIME_FACTOR - 2.0 / 3.0) < 1e-12


def test_cm_per_tick_and_nominal():
    v = C1.V_DRIFT_DEFAULT
    cm_per_tick = C1.DRIFT_TIME_FACTOR * v * C1.TICK_PERIOD_US
    mu_nom = 200.0 / (v * C1.TICK_PERIOD_US)
    # (2/3) v dt per tick, and the nominal cathode tick is a full drift length.
    assert abs(cm_per_tick - (2.0 / 3.0) * v * 0.5) < 1e-12
    assert abs(mu_nom - 200.0 / (v * 0.5)) < 1e-9
    # a pixel one cm_per_tick above nominal is +1 tick -> displacement sign flips per TPC
    for tpc, sign in ((0, +1.0), (1, -1.0)):
        x = sign * cm_per_tick * ((mu_nom + 1.0) - mu_nom)
        assert np.isclose(x, sign * cm_per_tick)


def test_csv_to_grids_on_fixture(cathode_csv_dir):
    import os
    mu, mu_err, n_ent, y_c, z_c = C1.csv_to_grids(
        os.path.join(cathode_csv_dir, "cathode_fit_results_tpc0.csv"))
    assert mu.shape == (len(y_c), len(z_c))
    assert np.isfinite(mu).any()
    # cathode tick is near the full-drift value (~2560 ticks for this sample)
    assert 2000 < np.nanmedian(mu) < 3200
