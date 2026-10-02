"""End-to-end mini-run of the FIELD-distortion pipeline on the committed
downsampled parquet fixture: voxels -> spline -> forward SCE ROOT ->
E-field voxels -> E-field spline -> combined SCE ROOT.

Marked `integration` (a minute or two). Run with: pytest -m integration
"""
import os
import subprocess
import sys

import numpy as np
import pytest


def _run(repo_root, ref_root, *args):
    env = dict(os.environ, SBND_SCE_REFERENCE=ref_root)
    r = subprocess.run([sys.executable, *args], cwd=repo_root, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError("command failed: " + " ".join(args)
                             + "\nSTDOUT:\n" + r.stdout[-3000:]
                             + "\nSTDERR:\n" + r.stderr[-3000:])
    return r


@pytest.mark.integration
def test_field_pipeline_end_to_end(tmp_path, repo_root, mini_parquet, ref_root):
    uproot = pytest.importorskip("uproot")
    maps = str(tmp_path / "maps")
    spline = str(tmp_path / "spline")
    sce = str(tmp_path / "sce")
    efmaps = str(tmp_path / "efmaps")
    efspline = str(tmp_path / "efspline")

    _run(repo_root, ref_root, "plot_doubleanode_2d_voxels.py",
         "--input", mini_parquet, "--outdir", maps,
         "--voxel-size", "50", "--min-counts", "50", "--nproc", "4",
         "--no-slice-maps", "--no-counts-maps")
    npz = np.load(os.path.join(maps, "voxel_data.npz"))
    assert int(np.isfinite(npz["delta_mu_all"][..., 0]).sum()) > 100

    _run(repo_root, ref_root, "fit_doubleanode_2d_spline.py",
         "--input", os.path.join(maps, "voxel_data.npz"), "--outdir", spline,
         "--nx-knots", "3", "--ny-knots", "3", "--nz-knots", "4",
         "--ridge-lambda", "1", "--no-gif",
         "--write-sce-root", "--sce-outdir", sce, "--sce-root-name", "fwd.root")
    assert os.path.exists(os.path.join(sce, "fwd.root"))
    assert os.path.exists(os.path.join(spline, "spline_knots.json"))

    _run(repo_root, ref_root, "build_efield_2d_voxels.py",
         "--input", os.path.join(maps, "voxel_data.npz"),
         "--spline-knots-json", os.path.join(spline, "spline_knots.json"),
         "--outdir", efmaps)
    _run(repo_root, ref_root, "fit_efield_2d_spline.py",
         "--input", os.path.join(efmaps, "voxel_data.npz"), "--outdir", efspline,
         "--nx-knots", "3", "--ny-knots", "3", "--nz-knots", "4",
         "--ridge-lambda", "0.1", "--no-gif", "--write-sce-root",
         "--displacement-root", os.path.join(sce, "fwd.root"),
         "--sce-outdir", sce, "--sce-root-name", "efield.root")

    combined = os.path.join(sce, "combined.root")
    _run(repo_root, ref_root, "build_sce_root.py",
         "--forward-root", os.path.join(sce, "fwd.root"),
         "--efield-root", os.path.join(sce, "efield.root"),
         "--out", combined, "--reference-root", ref_root, "--bkwd-iterations", "6")
    assert os.path.exists(combined)

    with uproot.open(combined) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        for name in ("TrueFwd_Displacement_X", "TrueBkwd_Displacement_X",
                     "True_ElecField_X"):
            assert name in keys
        fwd = f["TrueFwd_Displacement_X"].values()
        bkwd = f["TrueBkwd_Displacement_X"].values()
    assert np.isfinite(fwd).all() and np.isfinite(bkwd).all()
    # backward is the numerical inverse of forward -> strongly anti-correlated.
    xe = np.linspace(-204.1667, 204.1667, fwd.shape[0] + 1)
    xc = 0.5 * (xe[:-1] + xe[1:])
    for mask in (xc < 0, xc > 0):
        a = fwd[mask].ravel(); b = bkwd[mask].ravel()
        assert np.corrcoef(a, b)[0, 1] < -0.5
