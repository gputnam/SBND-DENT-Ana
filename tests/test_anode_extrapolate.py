#!/usr/bin/env python3
"""Unit test of the --anode-extrapolate chord fix in
plot_doubleanode_2d_voxels.reconstruct_anodes.

A synthetic straight track crosses both anodes; the per-plane "anode hits"
are placed INSET from the anode by a known drift distance (as in reality:
the first wire crossing / the dt-window induction selection). Without the
fix, assigning their view coords to x = +/-200 biases the fitted anode
(y, z) by (inset x track slope); with per-hit x supplied, the iterated
extrapolation must recover the true anode (y, z) to high accuracy.
"""

import numpy as np

import plot_doubleanode_2d_voxels as P2

GEOM = P2.load_wire_geom(P2.WIRE_GEOM)
SLOPE, OFFSET, WHATY, WHATZ = GEOM
V = 0.15627
T0 = 418.5


def synth_track(y0, z0, y1, z1, inset):
    """Anode hits only (one per plane per TPC), inset cm inside each anode."""
    rows = dict(tkey=[], tpc=[], plane=[], time=[], s=[], wy=[], wz=[], x=[])
    for tpc, (xa, sgn) in enumerate((( -200.0, 1.0), (200.0, -1.0))):
        x_hit = xa + sgn * inset
        u = (x_hit + 200.0) / 400.0
        y = y0 + u * (y1 - y0)
        z = z0 + u * (z1 - z0)
        for p in (0, 1, 2):
            rows["tkey"].append(0)
            rows["tpc"].append(tpc)
            rows["plane"].append(p)
            rows["time"].append(T0 + inset / (0.5 * V))
            rows["s"].append(WHATY[tpc, p] * y + WHATZ[tpc, p] * z)
            rows["wy"].append(WHATY[tpc, p])
            rows["wz"].append(WHATZ[tpc, p])
            rows["x"].append(x_hit)
    return {k: np.asarray(v, dtype=(np.int64 if k in ("tkey", "tpc", "plane")
                                    else np.float64))
            for k, v in rows.items()}


def fit(rows, with_x):
    YA, ZA, fom, _ = P2.reconstruct_anodes(
        rows["tkey"], rows["tpc"], rows["plane"], rows["time"], rows["s"],
        rows["wy"], rows["wz"], 1, anode_dt_window=200.0,
        x=(rows["x"] if with_x else None))
    return YA[0], ZA[0]


def test_inset_bias_and_fix():
    y0, z0, y1, z1 = -120.0, 80.0, 150.0, 430.0   # steep track
    inset = 3.0                                    # ~40-tick window scale
    rows = synth_track(y0, z0, y1, z1, inset)

    ya_b, za_b = fit(rows, with_x=False)
    ya_f, za_f = fit(rows, with_x=True)

    bias_b = np.hypot(ya_b - [y0, y1], za_b - [z0, z1]).max()
    bias_f = np.hypot(ya_f - [y0, y1], za_f - [z0, z1]).max()
    exp_b = inset * np.hypot(y1 - y0, z1 - z0) / 400.0
    print(f"[anode-extrapolate] inset={inset} cm: biased fit err "
          f"{bias_b:.3f} cm (expected ~{exp_b:.3f}), fixed fit err "
          f"{bias_f:.5f} cm")
    assert bias_b > 0.5 * exp_b, "expected a visible inset bias without x"
    assert bias_f < 0.01, f"extrapolated fit should be exact: {bias_f}"


def test_zero_inset_noop():
    rows = synth_track(-50.0, 200.0, 30.0, 300.0, 0.0)
    ya_b, za_b = fit(rows, with_x=False)
    ya_f, za_f = fit(rows, with_x=True)
    assert np.allclose([ya_b, za_b], [ya_f, za_f], atol=1e-9), \
        "with zero inset the fix must be a no-op"
    print("[anode-extrapolate] zero-inset no-op OK")
