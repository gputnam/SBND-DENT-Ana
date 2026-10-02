"""End-to-end mini-run of the CATHODE pipeline on the committed CB-fit CSV
fixtures: step1 (tick->cm) -> step2 (spline) -> step3 (v6 ASCII).

Marked `integration`. Run with: pytest -m integration
"""
import os
import subprocess
import sys

import numpy as np
import pytest


def _run(repo_root, *args):
    r = subprocess.run([sys.executable, *args], cwd=repo_root,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError("command failed: " + " ".join(args)
                             + "\nSTDOUT:\n" + r.stdout[-3000:]
                             + "\nSTDERR:\n" + r.stderr[-3000:])
    return r


@pytest.mark.integration
def test_cathode_pipeline_end_to_end(tmp_path, repo_root, cathode_csv_dir):
    s1 = str(tmp_path / "s1")
    s2 = str(tmp_path / "s2")
    s3 = str(tmp_path / "s3")

    _run(repo_root, "cathode_step1_tick_to_cm.py",
         "--csv-dir", cathode_csv_dir, "--outdir", s1)
    for tpc in (0, 1):
        d = np.load(os.path.join(s1, f"tpc{tpc}_displacement.npz"))
        assert np.isfinite(d["x"]).any()

    _run(repo_root, "cathode_step2_spline.py", "--step1-dir", s1, "--outdir", s2,
         "--ny-knots", "6", "--nz-knots", "6", "--ridge-lambda", "5")
    for tpc in (0, 1):
        assert os.path.exists(os.path.join(s2, f"tpc{tpc}_spline.npz"))

    _run(repo_root, "cathode_step3_v6.py", "--step1-dir", s1, "--step2-dir", s2,
         "--outdir", s3, "--nmin", "1")
    ascii_out = os.path.join(s3, "sbnd_cathode_v6_drift.txt")
    assert os.path.exists(ascii_out)
    # the ASCII grid should carry finite x_neg / x_pos displacement values
    with open(ascii_out) as f:
        body = [ln for ln in f if ln.strip() and not ln.startswith("#")]
    assert len(body) > 100
