"""Caloskim / crtana ROOT branch lists and per-cluster helpers.

The branch lists here are the UNION of branches consumed across the
make_align_df / make_crtdisp / make_driftV / cutflow pipelines. Scripts
may pass a smaller subset to ``uproot.arrays(...)``; this module just
documents the canonical names.
"""

import numpy as np

TREE_TPC = "caloskim/TrackCaloSkim"
TREE_CRT = "crtana/tree"

# ----- caloskim track-level branches used across builders -----
TCS_TRK_BRANCHES_FULL = [
    # identity / event keys
    "meta.run", "meta.subrun", "meta.evt", "id",
    # T0 + provenance
    "t0PFP", "xShiftCRT", "whicht0",
    # selection flags
    "selected", "clear_cosmic_muon", "cryostat",
    # endpoints / direction / length (lab frame)
    "start.x", "start.y", "start.z",
    "end.x", "end.y", "end.z",
    "dir.x", "dir.y", "dir.z", "length",
]

# Collection-plane hit branches. The (Y, Z) of the earliest hit in each
# TPC is the anode-piercing point; X is fit by the alignment optimizer.
TCS_HIT_BRANCHES_FULL = [
    "hits2.h.time", "hits2.h.sp.x", "hits2.h.sp.y", "hits2.h.sp.z",
    "hits2.h.tpc", "hits2.oncalo",
]

# ----- crtana branches needed to build per-track CRT info -----
CRT_BRANCHES_FULL = [
    "run", "subrun", "event",
    "tpc_tr_matched", "tpc_tr_score",
    "tpc_tr_start_x", "tpc_tr_start_y", "tpc_tr_start_z",
    "tpc_tr_end_x", "tpc_tr_end_y", "tpc_tr_end_z",
    "tpc_tr_taggers", "tpc_tr_ts0", "tpc_tr_ts1",
    "cl_has_sp", "cl_sp_x", "cl_sp_y", "cl_sp_z",
    "cl_tagger", "cl_sh_feb_mac5_set",
    # cl_ts0/cl_ts1 are only needed by the Run1 rematch shim
    # (crtlib.rematch_run1) — for v10_14_02 reads they are unused but
    # negligibly small.
    "cl_ts0", "cl_ts1",
]


def dominant_feb_mac(macs):
    """Pick the dominant FEB MAC5 from a strip-hit MAC5 list.

    Most clusters have either 1 unique MAC (single-FEB modules like the
    BERN/MINOS ones) or 2 (X-strip + Y-strip layers on the same physical
    module). We pick the most frequent value; ties broken by smallest
    MAC for determinism. Returns -1 if `macs` is empty.
    """
    if len(macs) == 0:
        return -1
    arr = np.asarray(macs, dtype=np.int64)
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def match_cluster_index(target_xyz, cl_xyz, tol):
    """Return the index of the cluster whose space-point matches
    `target_xyz` within `tol` cm (Euclidean). Returns -1 if none does."""
    if cl_xyz.shape[0] == 0:
        return -1
    d = np.linalg.norm(cl_xyz - target_xyz[None, :], axis=1)
    j = int(np.argmin(d))
    return j if d[j] <= tol else -1
