"""Shared pytest configuration.

Puts the repo root on sys.path so the flat top-level pipeline scripts
(build_sce_input, fit_dx_ramp, plot_doubleanode_2d_voxels, ...) import even
without `pip install -e .`, and exposes the committed fixture paths.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FIXTURES = os.path.join(ROOT, "tests", "fixtures")


@pytest.fixture(scope="session")
def repo_root():
    return ROOT


@pytest.fixture(scope="session")
def fixtures_dir():
    return FIXTURES


@pytest.fixture(scope="session")
def mini_parquet():
    p = os.path.join(FIXTURES, "mini_doubleanode_2d_df.parquet")
    if not os.path.exists(p):
        pytest.skip("mini parquet fixture absent (run tests/fixtures/_build_fixtures.py)")
    return p


@pytest.fixture(scope="session")
def ref_root():
    p = os.path.join(FIXTURES, "ref_sce_small.root")
    if not os.path.exists(p):
        pytest.skip("reference SCE ROOT fixture absent")
    return p


@pytest.fixture(scope="session")
def cathode_csv_dir():
    d = os.path.join(FIXTURES, "cathode")
    if not os.path.exists(os.path.join(d, "cathode_fit_results_tpc0.csv")):
        pytest.skip("cathode CSV fixtures absent")
    return d
