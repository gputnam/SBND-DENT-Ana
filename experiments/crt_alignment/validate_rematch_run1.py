"""Validate the Run1 rematch shim by stripping the tpc_tr_* fields from
a v10_14_02 InTime file, running the shim, and comparing back to the
ground truth that was stripped."""

import argparse
import sys

import awkward as ak
import numpy as np
import uproot

from crtlib.io import pnfs_to_xroot
from crtlib.ntuple import (TREE_TPC, TREE_CRT,
                           TCS_TRK_BRANCHES_FULL, TCS_HIT_BRANCHES_FULL,
                           CRT_BRANCHES_FULL)
from crtlib.constants import SENTINEL
from crtlib import rematch_run1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?",
                    help="path to a v10_14_02 ntuple (defaults to first of "
                         "filelist_nonempty_combined.txt)")
    ap.add_argument("--max-tracks", type=int, default=None,
                    help="limit number of tracks to inspect (default: all)")
    args = ap.parse_args()

    if args.file is None:
        with open("filelist_nonempty_combined.txt") as f:
            args.file = f.readline().strip()

    url = pnfs_to_xroot(args.file)
    print(f"file: {url[:120]}")
    with uproot.open(url, timeout=60) as f:
        tcs_arr = f[TREE_TPC].arrays(TCS_TRK_BRANCHES_FULL + TCS_HIT_BRANCHES_FULL,
                                     library="ak")
        crt_arr_full = f[TREE_CRT].arrays(CRT_BRANCHES_FULL, library="ak")

    print(f"  caloskim tracks: {len(tcs_arr)}")
    print(f"  crtana events  : {len(crt_arr_full)}")

    # Strip the pre-computed tpc_tr_* fields and pretend we're Run1.
    raw_fields = [f for f in ak.fields(crt_arr_full) if not f.startswith("tpc_tr_")]
    crt_arr_stripped = crt_arr_full[raw_fields]
    assert rematch_run1.needs_shim(crt_arr_stripped)
    print("\n  stripping tpc_tr_* fields from crt_arr; running shim ...")

    crt_arr_shim, tcs_arr_shim = rematch_run1.synthesize_run1_schema(
        crt_arr_stripped, tcs_arr)

    # Compare matched count and matched cluster positions
    truth_run = ak.to_numpy(crt_arr_full["run"]).astype(np.int64)
    truth_sub = ak.to_numpy(crt_arr_full["subrun"]).astype(np.int64)
    truth_evt = ak.to_numpy(crt_arr_full["event"]).astype(np.int64)
    truth_idx = {(int(r), int(s), int(e)): i
                 for i, (r, s, e) in enumerate(zip(truth_run, truth_sub, truth_evt))}

    tcs_run = ak.to_numpy(tcs_arr["meta.run"]).astype(np.int64)
    tcs_sub = ak.to_numpy(tcs_arr["meta.subrun"]).astype(np.int64)
    tcs_evt = ak.to_numpy(tcs_arr["meta.evt"]).astype(np.int64)
    tcs_id  = ak.to_numpy(tcs_arr["id"]).astype(np.int64)

    tt_matched = crt_arr_full["tpc_tr_matched"]
    tt_sx, tt_sy, tt_sz = crt_arr_full["tpc_tr_start_x"], crt_arr_full["tpc_tr_start_y"], crt_arr_full["tpc_tr_start_z"]
    tt_ex, tt_ey, tt_ez = crt_arr_full["tpc_tr_end_x"], crt_arr_full["tpc_tr_end_y"], crt_arr_full["tpc_tr_end_z"]

    sh_matched = crt_arr_shim["tpc_tr_matched"]
    sh_sx, sh_sy, sh_sz = crt_arr_shim["tpc_tr_start_x"], crt_arr_shim["tpc_tr_start_y"], crt_arr_shim["tpc_tr_start_z"]
    sh_ex, sh_ey, sh_ez = crt_arr_shim["tpc_tr_end_x"], crt_arr_shim["tpc_tr_end_y"], crt_arr_shim["tpc_tr_end_z"]

    # Tally
    truth_n_matched = 0
    shim_n_matched = 0
    both_matched = 0
    neither_matched = 0
    only_truth = 0
    only_shim = 0
    pos_resid = []  # |Δsp1| + |Δsp2| for tracks matched by both

    n_iter = len(tcs_run) if args.max_tracks is None else min(len(tcs_run), args.max_tracks)
    for j in range(n_iter):
        key = (int(tcs_run[j]), int(tcs_sub[j]), int(tcs_evt[j]))
        ie = truth_idx.get(key)
        if ie is None:
            continue
        k = int(tcs_id[j])
        if k < 0 or k >= len(tt_matched[ie]) or k >= len(sh_matched[ie]):
            continue
        tm = bool(tt_matched[ie][k])
        sm = bool(sh_matched[ie][k])
        truth_n_matched += int(tm)
        shim_n_matched += int(sm)
        if tm and sm:
            both_matched += 1
            # Compute symmetric residual (truth start might be shim end or shim start)
            t_sp1 = np.array([float(tt_sx[ie][k]), float(tt_sy[ie][k]), float(tt_sz[ie][k])])
            t_sp2 = np.array([float(tt_ex[ie][k]), float(tt_ey[ie][k]), float(tt_ez[ie][k])])
            s_sp1 = np.array([float(sh_sx[ie][k]), float(sh_sy[ie][k]), float(sh_sz[ie][k])])
            s_sp2 = np.array([float(sh_ex[ie][k]), float(sh_ey[ie][k]), float(sh_ez[ie][k])])
            if t_sp1[0] < SENTINEL or t_sp2[0] < SENTINEL or s_sp1[0] < SENTINEL or s_sp2[0] < SENTINEL:
                continue
            d_aligned = np.linalg.norm(t_sp1 - s_sp1) + np.linalg.norm(t_sp2 - s_sp2)
            d_swapped = np.linalg.norm(t_sp1 - s_sp2) + np.linalg.norm(t_sp2 - s_sp1)
            pos_resid.append(min(d_aligned, d_swapped))
        elif tm:
            only_truth += 1
        elif sm:
            only_shim += 1
        else:
            neither_matched += 1

    n_evaluated = truth_n_matched + shim_n_matched - both_matched + neither_matched
    print(f"\n  evaluated {n_iter} tracks ({truth_n_matched + shim_n_matched - both_matched + neither_matched} reach the matched-or-not branch)")
    print(f"    truth matched : {truth_n_matched}")
    print(f"    shim  matched : {shim_n_matched}")
    print(f"    both matched  : {both_matched}")
    print(f"    only truth    : {only_truth}")
    print(f"    only shim     : {only_shim}")
    print(f"    neither       : {neither_matched}")

    if pos_resid:
        pr = np.asarray(pos_resid)
        print(f"\n  cluster-position residual (truth vs shim, both-matched tracks)")
        print(f"    median : {np.median(pr):.2f} cm")
        print(f"    p90    : {np.percentile(pr, 90):.2f} cm")
        print(f"    max    : {pr.max():.2f} cm")
        print(f"    n      : {len(pr)}")

    recall = both_matched / max(1, truth_n_matched)
    print(f"\n  recall (both / truth_matched) = {recall:.2%}")
    purity = both_matched / max(1, shim_n_matched)
    print(f"  purity (both / shim_matched)  = {purity:.2%}")


if __name__ == "__main__":
    main()
