"""Round-trip test for the SCE TH3F writer (build_sce_input.write_th3f_root)."""
import numpy as np
import pytest

import build_sce_input as B


def _edges():
    xe = np.linspace(B.SCE_X_EDGE[0], B.SCE_X_EDGE[1], B.SCE_NX + 1)
    ye = np.linspace(B.SCE_Y_EDGE[0], B.SCE_Y_EDGE[1], B.SCE_NY + 1)
    ze = np.linspace(B.SCE_Z_EDGE[0], B.SCE_Z_EDGE[1], B.SCE_NZ + 1)
    return xe, ye, ze


def test_write_read_roundtrip(tmp_path, ref_root):
    uproot = pytest.importorskip("uproot")
    xe, ye, ze = _edges()
    shape = (B.SCE_NX, B.SCE_NY, B.SCE_NZ)
    rng = np.random.default_rng(0)
    fx = rng.standard_normal(shape).astype(np.float32)
    fy = rng.standard_normal(shape).astype(np.float32)
    fz = rng.standard_normal(shape).astype(np.float32)

    out = str(tmp_path / "sce.root")
    B.write_th3f_root(out, fx, fy, fz, xe, ye, ze, reference_root=ref_root)

    with uproot.open(out) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        for name in ("TrueFwd_Displacement_X", "TrueFwd_Displacement_Y",
                     "TrueFwd_Displacement_Z", "TrueBkwd_Displacement_X",
                     "True_ElecField_X"):
            assert name in keys
        back = f["TrueFwd_Displacement_X"].values().astype(np.float32)
    assert back.shape == shape
    assert np.allclose(back, fx, atol=1e-5)


def test_measured_backward_and_efield_written(tmp_path, ref_root):
    uproot = pytest.importorskip("uproot")
    xe, ye, ze = _edges()
    shape = (B.SCE_NX, B.SCE_NY, B.SCE_NZ)
    z = np.zeros(shape, np.float32)
    bx = np.full(shape, 0.5, np.float32)
    ex = np.full(shape, 0.1, np.float32)
    out = str(tmp_path / "sce2.root")
    B.write_th3f_root(out, z, z, z, xe, ye, ze, reference_root=ref_root,
                      backward=(bx, z, z), efield=(ex, z, z))
    with uproot.open(out) as f:
        assert np.allclose(f["TrueBkwd_Displacement_X"].values(), 0.5, atol=1e-5)
        assert np.allclose(f["True_ElecField_X"].values(), 0.1, atol=1e-5)
