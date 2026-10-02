"""Run1 (v10_06_03) CRT-TPC re-matching shim.

The v10_06_03 calibration ntuples predate the LArSoft producer that
stamps per-track CRT-track matches onto ``crtana/tree`` (the
``tpc_tr_*`` branch family added in v10_14_02). They DO carry the raw
CRT cluster space-points (``cl_*``) and the TPC reconstructed tracks
(``caloskim/TrackCaloSkim``), plus event-key correlation. This module
re-derives the missing ``tpc_tr_*`` arrays at analysis time so the
existing builders (``make_align_df_sbnd``, ``make_crtdisp_df_sbnd``)
can run unchanged on Run1 inputs.

Matching algorithm (purely geometric):
  - For each TPC track in a given (run, subrun, event), iterate
    over CRT clusters with ``cl_has_sp == True`` in the same event.
  - Compute perpendicular distance from each cluster space-point
    to the infinite line through the TPC track's start and end.
  - Project clusters onto the track direction → signed parameter t.
  - Keep clusters whose perpendicular distance is < PERP_TOL_CM.
  - Pick the two with most extreme t (one ``start side``, one
    ``end side``) provided they are separated by ≥ MIN_ENDPOINT_SEP_CM
    along the track. These become the synthesized
    ``tpc_tr_start_*`` / ``tpc_tr_end_*``.

Schema additions injected into ``crt_arr`` (per-event, jagged by
TPC track ID = caloskim ``id``):
  tpc_tr_matched, tpc_tr_score,
  tpc_tr_start_x/y/z, tpc_tr_end_x/y/z,
  tpc_tr_taggers, tpc_tr_ts0, tpc_tr_ts1

In addition, ``xShiftCRT`` (missing from Run1 caloskim) is injected
as zeros into ``tcs_arr`` for audit-column compatibility.

Detection: call :func:`needs_shim` on the loaded ``crt_arr``; True
iff ``tpc_tr_matched`` is absent.
"""

import awkward as ak
import numpy as np

from crtlib.constants import SENTINEL


PERP_TOL_CM = 30.0
MIN_ENDPOINT_SEP_CM = 100.0
# Two cluster space-points from the SAME muon share a CRT timestamp to
# within a few ns (CRT module resolution is ~1 ns). Clusters from
# different muons in the same event are typically separated by ≥ µs.
# We use 50 ns as a comfortable upper bound that survives drift-time
# spread within a single muon's CRT crossing.
TS0_PAIR_TOL_NS = 50.0


def needs_shim(crt_arr):
    """True if the crtana awkward array lacks pre-computed tpc_tr_* fields."""
    return "tpc_tr_matched" not in ak.fields(crt_arr)


def _match_one_track(track_start, track_end,
                     cl_sp, cl_tagger, cl_ts0, cl_ts1, cl_has_sp,
                     perp_tol=PERP_TOL_CM, min_sep=MIN_ENDPOINT_SEP_CM,
                     ts0_tol=TS0_PAIR_TOL_NS):
    """Geometric match: return (matched, sx, sy, sz, ex, ey, ez, taggers,
    ts0_avg, ts1_avg, score). All sentinels if no pair found."""
    sent = SENTINEL
    if cl_sp is None or len(cl_sp) == 0:
        return False, sent, sent, sent, sent, sent, sent, [], sent, sent, sent

    has_sp = np.asarray(cl_has_sp, dtype=bool)
    if not has_sp.any():
        return False, sent, sent, sent, sent, sent, sent, [], sent, sent, sent

    cl = cl_sp[has_sp]
    tags = cl_tagger[has_sp]
    ts0 = cl_ts0[has_sp]
    ts1 = cl_ts1[has_sp]

    s = np.asarray(track_start, dtype=np.float64)
    e = np.asarray(track_end, dtype=np.float64)
    d = e - s
    dnorm = float(np.linalg.norm(d))
    if dnorm < 1e-6:
        return False, sent, sent, sent, sent, sent, sent, [], sent, sent, sent
    dhat = d / dnorm

    diff = cl - s[None, :]
    t = diff @ dhat
    proj = s[None, :] + np.outer(t, dhat)
    perp = np.linalg.norm(cl - proj, axis=1)

    ok = perp < perp_tol
    if int(ok.sum()) < 2:
        return False, sent, sent, sent, sent, sent, sent, [], sent, sent, sent

    idx_ok = np.where(ok)[0]
    t_ok = t[idx_ok]
    p_ok = perp[idx_ok]
    s_ok = ts0[idx_ok]
    tag_ok = tags[idx_ok]

    # Time-gated pair selection: scan all cluster pairs and pick the
    # one with (a) closest cl_ts0 agreement, (b) different taggers, and
    # (c) largest track-parameter separation. With hundreds of clusters
    # per event from neighbouring muons, time agreement is the critical
    # discriminator — same-muon clusters share cl_ts0 within ~ns; cross-
    # muon clusters are typically µs apart.
    best = None  # (score_metric, j_lo, j_hi)
    n = len(idx_ok)
    for a in range(n):
        for b in range(a + 1, n):
            if tag_ok[a] == tag_ok[b]:
                continue
            dt = abs(s_ok[a] - s_ok[b])
            if dt > ts0_tol:
                continue
            sep = abs(t_ok[a] - t_ok[b])
            if sep < min_sep:
                continue
            # Lower perp_sum + lower dt + higher sep → better.
            metric = (float(p_ok[a]) + float(p_ok[b])) + dt * 0.1 - sep * 0.001
            if best is None or metric < best[0]:
                # order so that j_lo has smaller t (start side)
                if t_ok[a] < t_ok[b]:
                    best = (metric, idx_ok[a], idx_ok[b])
                else:
                    best = (metric, idx_ok[b], idx_ok[a])

    if best is None:
        return False, sent, sent, sent, sent, sent, sent, [], sent, sent, sent
    _, j_lo, j_hi = best

    score = float(1.0 / (1.0 + float(perp[j_lo]) + float(perp[j_hi])))
    taggers = [int(tags[j_lo]), int(tags[j_hi])]
    return (True,
            float(cl[j_lo, 0]), float(cl[j_lo, 1]), float(cl[j_lo, 2]),
            float(cl[j_hi, 0]), float(cl[j_hi, 1]), float(cl[j_hi, 2]),
            taggers,
            float(0.5 * (ts0[j_lo] + ts0[j_hi])),
            float(0.5 * (ts1[j_lo] + ts1[j_hi])),
            score)


def synthesize_run1_schema(crt_arr, tcs_arr):
    """Inject synthesized tpc_tr_* fields into ``crt_arr`` and a synthetic
    ``xShiftCRT`` (zeros) into ``tcs_arr``. Returns ``(crt_arr, tcs_arr)``
    with the additional fields. No-op if ``crt_arr`` already has
    ``tpc_tr_matched``.
    """
    if not needs_shim(crt_arr):
        return crt_arr, tcs_arr

    n_events = len(crt_arr)
    crt_run = ak.to_numpy(crt_arr["run"]).astype(np.int64)
    crt_sub = ak.to_numpy(crt_arr["subrun"]).astype(np.int64)
    crt_evt = ak.to_numpy(crt_arr["event"]).astype(np.int64)
    ev_index = {(int(r), int(s), int(e)): i
                for i, (r, s, e) in enumerate(zip(crt_run, crt_sub, crt_evt))}

    tcs_run = ak.to_numpy(tcs_arr["meta.run"]).astype(np.int64)
    tcs_sub = ak.to_numpy(tcs_arr["meta.subrun"]).astype(np.int64)
    tcs_evt = ak.to_numpy(tcs_arr["meta.evt"]).astype(np.int64)
    tcs_id  = ak.to_numpy(tcs_arr["id"]).astype(np.int64)
    tcs_sx  = ak.to_numpy(tcs_arr["start.x"]).astype(np.float64)
    tcs_sy  = ak.to_numpy(tcs_arr["start.y"]).astype(np.float64)
    tcs_sz  = ak.to_numpy(tcs_arr["start.z"]).astype(np.float64)
    tcs_ex  = ak.to_numpy(tcs_arr["end.x"]).astype(np.float64)
    tcs_ey  = ak.to_numpy(tcs_arr["end.y"]).astype(np.float64)
    tcs_ez  = ak.to_numpy(tcs_arr["end.z"]).astype(np.float64)

    max_id = np.zeros(n_events, dtype=np.int64)
    for j in range(len(tcs_run)):
        key = (int(tcs_run[j]), int(tcs_sub[j]), int(tcs_evt[j]))
        ie = ev_index.get(key)
        if ie is None:
            continue
        if tcs_id[j] + 1 > max_id[ie]:
            max_id[ie] = int(tcs_id[j]) + 1

    out_matched = [np.zeros(int(max_id[i]), dtype=bool) for i in range(n_events)]
    out_score   = [np.full(int(max_id[i]), SENTINEL, dtype=np.float64) for i in range(n_events)]
    out_sx      = [np.full(int(max_id[i]), SENTINEL, dtype=np.float32) for i in range(n_events)]
    out_sy      = [np.full(int(max_id[i]), SENTINEL, dtype=np.float32) for i in range(n_events)]
    out_sz      = [np.full(int(max_id[i]), SENTINEL, dtype=np.float32) for i in range(n_events)]
    out_ex      = [np.full(int(max_id[i]), SENTINEL, dtype=np.float32) for i in range(n_events)]
    out_ey      = [np.full(int(max_id[i]), SENTINEL, dtype=np.float32) for i in range(n_events)]
    out_ez      = [np.full(int(max_id[i]), SENTINEL, dtype=np.float32) for i in range(n_events)]
    out_tag     = [[[] for _ in range(int(max_id[i]))] for i in range(n_events)]
    out_ts0     = [np.full(int(max_id[i]), SENTINEL, dtype=np.float64) for i in range(n_events)]
    out_ts1     = [np.full(int(max_id[i]), SENTINEL, dtype=np.float64) for i in range(n_events)]

    cl_has_sp_all = crt_arr["cl_has_sp"]
    cl_sp_x_all   = crt_arr["cl_sp_x"]
    cl_sp_y_all   = crt_arr["cl_sp_y"]
    cl_sp_z_all   = crt_arr["cl_sp_z"]
    cl_tagger_all = crt_arr["cl_tagger"]
    cl_ts0_all    = crt_arr["cl_ts0"]
    cl_ts1_all    = crt_arr["cl_ts1"]

    # Cache numpy arrays per-event to avoid recomputing for multi-track events
    cache = {}

    for j in range(len(tcs_run)):
        key = (int(tcs_run[j]), int(tcs_sub[j]), int(tcs_evt[j]))
        ie = ev_index.get(key)
        if ie is None:
            continue
        k = int(tcs_id[j])
        if k < 0 or k >= int(max_id[ie]):
            continue

        if ie not in cache:
            cache[ie] = (
                np.stack([
                    ak.to_numpy(cl_sp_x_all[ie]).astype(np.float64),
                    ak.to_numpy(cl_sp_y_all[ie]).astype(np.float64),
                    ak.to_numpy(cl_sp_z_all[ie]).astype(np.float64),
                ], axis=1),
                ak.to_numpy(cl_tagger_all[ie]).astype(np.int32),
                ak.to_numpy(cl_ts0_all[ie]).astype(np.float64),
                ak.to_numpy(cl_ts1_all[ie]).astype(np.float64),
                ak.to_numpy(cl_has_sp_all[ie]).astype(bool),
            )
        cl_sp, ctag, cts0, cts1, chas = cache[ie]

        matched, sx, sy, sz, ex, ey, ez, tags, ts0v, ts1v, score = _match_one_track(
            (tcs_sx[j], tcs_sy[j], tcs_sz[j]),
            (tcs_ex[j], tcs_ey[j], tcs_ez[j]),
            cl_sp, ctag, cts0, cts1, chas,
        )
        if matched:
            out_matched[ie][k] = True
            out_score[ie][k]   = score
            out_sx[ie][k] = sx; out_sy[ie][k] = sy; out_sz[ie][k] = sz
            out_ex[ie][k] = ex; out_ey[ie][k] = ey; out_ez[ie][k] = ez
            out_tag[ie][k] = tags
            out_ts0[ie][k] = ts0v
            out_ts1[ie][k] = ts1v

    out = crt_arr
    out = ak.with_field(out, ak.Array(out_matched), "tpc_tr_matched")
    out = ak.with_field(out, ak.Array(out_score),   "tpc_tr_score")
    out = ak.with_field(out, ak.Array(out_sx),      "tpc_tr_start_x")
    out = ak.with_field(out, ak.Array(out_sy),      "tpc_tr_start_y")
    out = ak.with_field(out, ak.Array(out_sz),      "tpc_tr_start_z")
    out = ak.with_field(out, ak.Array(out_ex),      "tpc_tr_end_x")
    out = ak.with_field(out, ak.Array(out_ey),      "tpc_tr_end_y")
    out = ak.with_field(out, ak.Array(out_ez),      "tpc_tr_end_z")
    out = ak.with_field(out, ak.Array(out_tag),     "tpc_tr_taggers")
    out = ak.with_field(out, ak.Array(out_ts0),     "tpc_tr_ts0")
    out = ak.with_field(out, ak.Array(out_ts1),     "tpc_tr_ts1")

    if "xShiftCRT" not in ak.fields(tcs_arr):
        n_tcs = len(tcs_arr)
        tcs_arr = ak.with_field(tcs_arr,
                                ak.Array(np.zeros(n_tcs, dtype=np.float32)),
                                "xShiftCRT")
    return out, tcs_arr
