#!/usr/bin/env python3
"""Unit tests of the dx ramp estimator (fit_dx_ramp.py).

1. Exact separation: a profile built from (ramp eps*d) + (dome in the
   estimator's own basis, zero at both ends) must return eps exactly.
2. Approximate dome: a sin(pi d/L) dome (not in the spline space) must
   still recover eps to a few 1e-4 absolute (cm/cm).
3. Null test: pure dome, no ramp -> eps consistent with 0.
"""

import numpy as np

import fit_dx_ramp as F


def synth_profile(eps, dome_amp, dome_kind, n_int=4, noise=0.02, seed=3):
    rng = np.random.default_rng(seed)
    vx = np.arange(-190.0, 200.0, 20.0)   # 20 columns at voxel centers
    prof = np.full(vx.shape, np.nan)
    for t, (xa, s_t) in enumerate(((-200.0, 1.0), (200.0, -1.0))):
        inh = (vx < 0) if t == 0 else (vx > 0)
        d = np.abs(vx[inh] - xa)
        if dome_kind == "spline":
            A = F.dome_basis(d, n_int)
            rngc = np.random.default_rng(seed + t)
            c = dome_amp * rngc.normal(size=A.shape[1])
            dome = A @ c
        else:
            dome = dome_amp * np.sin(np.pi * d / F.L_DRIFT)
        prof[inh] = eps * s_t * d + dome
    prof = prof + rng.normal(0, noise, prof.shape)
    err = np.full(vx.shape, noise)
    return vx, prof, err


def test_exact_separation():
    eps = 0.0065
    vx, prof, err = synth_profile(eps, 0.6, "spline", noise=1e-4)
    fits = F.fit_ramp_profile(vx, prof, err)
    for t in (0, 1):
        assert abs(fits[t]["eps"] - eps) < 5e-5, \
            f"TPC{t}: {fits[t]['eps']} vs {eps}"
    print(f"[ramp] exact separation OK: eps rec "
          f"{fits[0]['eps']:.5f}/{fits[1]['eps']:.5f} vs {eps}")


def test_sin_dome():
    """Dome NOT in the spline space: leakage must stay <= ~0.03 cm/m
    (the basis-mismatch systematic at n_int=4); statistical scatter
    covered by a 4-sigma allowance."""
    eps = 0.0065
    vx, prof, err = synth_profile(eps, 0.6, "sin", noise=0.005, seed=11)
    fits = F.fit_ramp_profile(vx, prof, err)
    for t in (0, 1):
        tol = 4e-4 + 4 * fits[t]["eps_err"]
        assert abs(fits[t]["eps"] - eps) < tol, \
            f"TPC{t}: {fits[t]['eps']} vs {eps} (tol {tol})"
    print(f"[ramp] sin-dome OK: eps rec "
          f"{fits[0]['eps']:.5f}/{fits[1]['eps']:.5f} vs {eps} "
          f"(leakage {(fits[0]['eps']-eps)*100:+.3f}/"
          f"{(fits[1]['eps']-eps)*100:+.3f} cm/m)")


def test_null():
    vx, prof, err = synth_profile(0.0, 0.6, "sin", noise=0.005, seed=11)
    fits = F.fit_ramp_profile(vx, prof, err)
    for t in (0, 1):
        assert abs(fits[t]["eps"]) < 4e-4 + 4 * fits[t]["eps_err"], \
            f"TPC{t} null violated: {fits[t]['eps']} ± {fits[t]['eps_err']}"
    print(f"[ramp] null OK: eps rec {fits[0]['eps']:.5f}/"
          f"{fits[1]['eps']:.5f} (consistent with 0)")


def test_joint_tpc_average():
    """Joint (shared-eps) fit recovers the detector-COMMON artifact while a
    per-TPC fit absorbs a genuine per-TPC drift-velocity difference."""
    eps_art, delta = 0.006, 0.002
    vx = np.arange(-190.0, 200.0, 20.0)
    prof = np.full(vx.shape, np.nan)
    for t, (xa, s_t) in enumerate(((-200.0, 1.0), (200.0, -1.0))):
        inh = (vx < 0) if t == 0 else (vx > 0)
        d = np.abs(vx[inh] - xa)
        eps_t = eps_art + (delta if t == 0 else -delta)
        prof[inh] = eps_t * s_t * d + 0.4 * np.sin(np.pi * d / 200.0)
    err = np.full(vx.shape, 0.01)
    fj = F.fit_ramp_profile(vx, prof, err, joint=True)
    fp = F.fit_ramp_profile(vx, prof, err, joint=False)
    assert abs(fj[0]["eps"] - fj[1]["eps"]) < 1e-9
    assert abs(fj[0]["eps"] - eps_art) < 5e-4
    assert abs(fp[0]["eps"] - 0.008) < 5e-4 and abs(fp[1]["eps"] - 0.004) < 5e-4
    print(f"[ramp] joint-TPC OK: joint eps {fj[0]['eps']:.5f} = artifact "
          f"(per-TPC {fp[0]['eps']:.5f}/{fp[1]['eps']:.5f} absorb the diff)")
