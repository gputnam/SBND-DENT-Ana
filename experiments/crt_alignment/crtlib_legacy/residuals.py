"""align_df → straightness-residual row builder."""

import numpy as np
import pandas as pd

from crtlib.constants import ANODE_X_NOMINAL


def explode_to_residual_rows(df, tmin_lo, tmin_hi, min_crt_len_cm=400.0):
    """Emit one residual row per (track, TPC) where the track actually
    pierces that TPC's anode AND the CRT-track lever arm is long enough
    that the line direction is well-defined.

    Two purity cuts:
      1) Per-TPC anode-piercing: a TPC contributes an anode point only
         if its earliest collection-plane hit time is in the in-time
         window [tmin_lo, tmin_hi]. (An anode-cathode crosser may pierce
         only ONE anode and stop inside the other TPC; its other-TPC
         t_min will then be far from the anode tick.)
      2) Minimum CRT-track length (>= min_crt_len_cm): tracks with both
         CRT hits in the same wall pair have only ~150 cm of lever arm;
         small CRT spatial errors blow up to >100 cm of lateral error
         when extrapolated 7× the lever-arm distance to the anode plane.
         A typical good wall-to-wall CRT track has > 800 cm of lever arm.
    """
    rows = []
    track_uid = 0
    for _, r in df.iterrows():
        sp1 = np.array([r["crt_sp1_x"], r["crt_sp1_y"], r["crt_sp1_z"]])
        sp2 = np.array([r["crt_sp2_x"], r["crt_sp2_y"], r["crt_sp2_z"]])
        m1 = int(r["crt_sp1_module_id"])
        m2 = int(r["crt_sp2_module_id"])
        if m1 < 0 or m2 < 0:
            continue
        crt_len = float(np.linalg.norm(sp2 - sp1))
        if crt_len < min_crt_len_cm:
            continue
        for tpc in (0, 1):
            n_hits = r[f"tpc{tpc}_n_p2_hits"]
            ay = r[f"tpc{tpc}_anode_y"]
            az = r[f"tpc{tpc}_anode_z"]
            tmin = r[f"tpc{tpc}_t_min"]
            if n_hits <= 0 or not np.isfinite(ay) or not np.isfinite(az):
                continue
            if not (np.isfinite(tmin) and tmin_lo <= tmin <= tmin_hi):
                continue
            rows.append({
                "crt_sp1_x": float(sp1[0]), "crt_sp1_y": float(sp1[1]), "crt_sp1_z": float(sp1[2]),
                "crt_sp2_x": float(sp2[0]), "crt_sp2_y": float(sp2[1]), "crt_sp2_z": float(sp2[2]),
                "mod1_id": m1, "mod2_id": m2,
                "tag1": int(r.get("crt_sp1_tagger", -1)),
                "tag2": int(r.get("crt_sp2_tagger", -1)),
                "anode_x_nom": ANODE_X_NOMINAL[tpc],
                "anode_y": float(ay), "anode_z": float(az),
                "anode_tpc": tpc,
                "crt_len": crt_len,
                "track_uid": track_uid,
            })
        track_uid += 1
    return pd.DataFrame(rows)


def perp_residual(sp1, sp2, anode_pt):
    """Perpendicular distance of `anode_pt` from the line through sp1, sp2.

    All inputs (N, 3) numpy arrays. Returns (N,)-array of distances in cm.
    Used both as the alignment loss target and as a diagnostic for cuts.
    """
    sp1 = np.asarray(sp1); sp2 = np.asarray(sp2); anode_pt = np.asarray(anode_pt)
    line = sp2 - sp1
    L = np.linalg.norm(line, axis=1, keepdims=True)
    u = line / np.maximum(L, 1e-9)
    delta = anode_pt - sp1
    proj = np.sum(delta * u, axis=1, keepdims=True) * u
    return np.linalg.norm(delta - proj, axis=1)
