"""Instrumented cutflow for the per-track filters in make_align_df_sbnd.py.

Mirrors the cut sequence in load_one_file() but counts failures at every
``continue`` instead of building rows.  Run on a sub-sample of the input
filelist:

    python cutflow_makedf.py --filelist filelist_nonempty.txt \\
                             --maxfiles 400 --nproc 32

The cuts here MATCH those in make_align_df_sbnd.py exactly; if the
upstream script changes, this file must be updated.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import awkward as ak
import numpy as np
import uproot

from crtlib.constants import SENTINEL, SP_MATCH_TOL_CM, WHICHT0_NONE
from crtlib.io import pnfs_to_xroot, set_use_xroot
from crtlib.ntuple import (TREE_TPC, TREE_CRT,
                            TCS_TRK_BRANCHES_FULL as TCS_TRK_BRANCHES,
                            CRT_BRANCHES_FULL as CRT_BRANCHES,
                            dominant_feb_mac as _dominant_feb_mac,
                            match_cluster_index as _match_cluster_index)
TCS_HIT_BRANCHES = ["hits2.h.time", "hits2.h.sp.x", "hits2.h.sp.y",
                    "hits2.h.sp.z", "hits2.h.tpc", "hits2.oncalo"]

# Cut order — labels MUST stay in sync with the count-vector indices below.
CUT_LABELS = [
    "0. all caloskim tracks in sample",
    "1. event present in crtana",
    "2. trk_id valid in crtana arrays",
    "3. tpc_tr_matched[k] == True",
    "4. whicht0 != -1 (some t0 applied)",
    "5. CRT-track endpoints both finite (not sentinel)",
    "6. >=1 calorimetry-eligible collection-plane hit",
    "7. >=1 TPC's earliest hit in anode tick window",
    "8. event has >=1 cluster with a space-point",
    "9. both CRT endpoints match a cluster (<= 0.5 cm)",
    "10. both endpoint clusters have a non-empty FEB MAC list",
    "11. EMITTED to align_df.parquet",
]
N_CUTS = len(CUT_LABELS)


_CUTS = None


def _worker_init(cuts):
    global _CUTS
    _CUTS = cuts


def count_one_file(args):
    path, idx = args
    cuts = _CUTS
    counts = np.zeros(N_CUTS, dtype=np.int64)
    url = pnfs_to_xroot(path)
    try:
        with uproot.open(url, timeout=120) as f:
            tcs = f[TREE_TPC]
            crt = f[TREE_CRT]
            tcs_arr = tcs.arrays(TCS_TRK_BRANCHES + TCS_HIT_BRANCHES, library="ak")
            crt_arr = crt.arrays(CRT_BRANCHES, library="ak")
    except Exception as e:
        print(f"[skip] {path}: {type(e).__name__}: {e}", flush=True)
        return counts, 0  # n_files_used = 0 on error

    n_tracks_total = int(len(tcs_arr))
    if n_tracks_total == 0:
        return counts, 1
    counts[0] = n_tracks_total

    if len(crt_arr) == 0:
        return counts, 1  # all C0 tracks fail at C1

    crt_run = ak.to_numpy(crt_arr["run"]).astype(np.int64)
    crt_sub = ak.to_numpy(crt_arr["subrun"]).astype(np.int64)
    crt_evt = ak.to_numpy(crt_arr["event"]).astype(np.int64)
    ev_index = {(int(r), int(s), int(e)): i for i, (r, s, e)
                in enumerate(zip(crt_run, crt_sub, crt_evt))}

    cmatched = crt_arr["tpc_tr_matched"]
    csx, csy, csz = (crt_arr["tpc_tr_start_x"], crt_arr["tpc_tr_start_y"], crt_arr["tpc_tr_start_z"])
    cex, cey, cez = (crt_arr["tpc_tr_end_x"], crt_arr["tpc_tr_end_y"], crt_arr["tpc_tr_end_z"])
    cl_has_sp = crt_arr["cl_has_sp"]
    cl_sp_x   = crt_arr["cl_sp_x"]
    cl_sp_y   = crt_arr["cl_sp_y"]
    cl_sp_z   = crt_arr["cl_sp_z"]
    cl_macs   = crt_arr["cl_sh_feb_mac5_set"]

    run = ak.to_numpy(tcs_arr["meta.run"]).astype(np.int64)
    sub = ak.to_numpy(tcs_arr["meta.subrun"]).astype(np.int64)
    evt = ak.to_numpy(tcs_arr["meta.evt"]).astype(np.int64)
    tid = ak.to_numpy(tcs_arr["id"]).astype(np.int64)
    wt0 = ak.to_numpy(tcs_arr["whicht0"]).astype(np.int32)

    for i in range(n_tracks_total):
        ev_key = (int(run[i]), int(sub[i]), int(evt[i]))
        ic = ev_index.get(ev_key)
        if ic is None:
            continue
        counts[1] += 1

        k = int(tid[i])
        if k < 0 or k >= len(cmatched[ic]):
            continue
        counts[2] += 1

        if not bool(cmatched[ic][k]):
            continue
        counts[3] += 1

        if int(wt0[i]) == WHICHT0_NONE:
            continue
        counts[4] += 1

        sx = float(csx[ic][k]); sy = float(csy[ic][k]); sz = float(csz[ic][k])
        ex = float(cex[ic][k]); ey = float(cey[ic][k]); ez = float(cez[ic][k])
        if (sx < SENTINEL or sy < SENTINEL or sz < SENTINEL
                or ex < SENTINEL or ey < SENTINEL or ez < SENTINEL):
            continue
        counts[5] += 1

        ht = np.asarray(tcs_arr["hits2.h.time"][i], dtype=np.float64)
        hy = np.asarray(tcs_arr["hits2.h.sp.y"][i], dtype=np.float64)
        hz = np.asarray(tcs_arr["hits2.h.sp.z"][i], dtype=np.float64)
        htpc = np.asarray(tcs_arr["hits2.h.tpc"][i], dtype=np.int32)
        hon = np.asarray(tcs_arr["hits2.oncalo"][i]).astype(bool)
        ok = hon & np.isfinite(ht) & np.isfinite(hy) & np.isfinite(hz)
        if not ok.any():
            continue
        counts[6] += 1
        ht = ht[ok]; htpc = htpc[ok]

        m0 = (htpc == 0); m1 = (htpc == 1)
        n0, n1 = int(m0.sum()), int(m1.sum())

        t_min0 = float(ht[m0].min()) if n0 > 0 else np.nan
        t_min1 = float(ht[m1].min()) if n1 > 0 else np.nan
        p0 = np.isfinite(t_min0) and cuts["tmin_lo"] <= t_min0 <= cuts["tmin_hi"]
        p1 = np.isfinite(t_min1) and cuts["tmin_lo"] <= t_min1 <= cuts["tmin_hi"]
        if not (p0 or p1):
            continue
        counts[7] += 1

        has_sp = ak.to_numpy(cl_has_sp[ic]).astype(bool)
        if not has_sp.any():
            continue
        counts[8] += 1

        cx = ak.to_numpy(cl_sp_x[ic])[has_sp]
        cy = ak.to_numpy(cl_sp_y[ic])[has_sp]
        cz = ak.to_numpy(cl_sp_z[ic])[has_sp]
        cl_xyz = np.stack([cx, cy, cz], axis=1)
        cluster_macs_all = list(cl_macs[ic])
        cluster_macs_sp = [cluster_macs_all[ii]
                           for ii, okk in enumerate(has_sp) if okk]

        sp1 = _match_cluster_index(np.array([sx, sy, sz]), cl_xyz, SP_MATCH_TOL_CM)
        sp2 = _match_cluster_index(np.array([ex, ey, ez]), cl_xyz, SP_MATCH_TOL_CM)
        if sp1 < 0 or sp2 < 0:
            continue
        counts[9] += 1

        mac1 = _dominant_feb_mac(np.asarray(cluster_macs_sp[sp1]))
        mac2 = _dominant_feb_mac(np.asarray(cluster_macs_sp[sp2]))
        if mac1 < 0 or mac2 < 0:
            continue
        counts[10] += 1
        counts[11] += 1

    return counts, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist", default="filelist_nonempty.txt")
    ap.add_argument("--maxfiles", type=int, default=400,
                    help="default 400 ≈ 1/20 of the 8000-file production")
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--cut-config", default="cut_tuning/cut_config_per_tpc.json")
    ap.add_argument("--no-xroot", action="store_true")
    ap.add_argument("--out-csv", default="cutflow_makedf.csv")
    args = ap.parse_args()

    set_use_xroot(not args.no_xroot)

    from crtlib.io import load_cut_config
    cuts = load_cut_config(args.cut_config)

    with open(args.filelist) as f:
        files = [line.strip() for line in f if line.strip()]
    files = files[:args.maxfiles]
    print(f"  filelist     : {args.filelist}")
    print(f"  N files      : {len(files)} (cap = {args.maxfiles})")
    print(f"  tmin window  : [{cuts['tmin_lo']:.1f}, {cuts['tmin_hi']:.1f}] ticks")
    print(f"  nproc        : {args.nproc}")

    work = [(p, i) for i, p in enumerate(files)]
    nproc = min(args.nproc, len(files), cpu_count())
    t0 = time.time()
    total = np.zeros(N_CUTS, dtype=np.int64)
    n_files_ok = 0
    with Pool(processes=nproc, initializer=_worker_init, initargs=(cuts,)) as pool:
        for k, (counts, used) in enumerate(pool.imap_unordered(count_one_file, work, chunksize=2)):
            total += counts
            n_files_ok += used
            if (k + 1) % max(1, len(files) // 20) == 0:
                rate = (k + 1) / (time.time() - t0)
                eta = (len(files) - (k + 1)) / max(rate, 1e-9)
                print(f"  {k+1}/{len(files)}  ({rate:.1f} files/s, eta {eta:.0f}s)",
                      flush=True)
    elapsed = time.time() - t0
    print(f"\n  done in {elapsed:.1f}s  ({n_files_ok} files contributed)")

    # Print cutflow.
    n_init = total[0]
    print(f"\nmake_align_df_sbnd.py per-track cutflow")
    print(f"  files used   : {n_files_ok} / {len(files)}")
    print(f"  expected ~1/20 of the 8000-file production "
          f"(~78,372 tracks total → ~3,900 here)\n")
    head = f"{'step':<60s} {'survivors':>12s}  {'pass%':>7s}  {'cum%':>7s}"
    print(head); print("-" * len(head))
    prev = None
    for label, n in zip(CUT_LABELS, total):
        if prev is None:
            pct_step, pct_cum = "", "100.00%"
        else:
            pct_step = f"{100.0 * n / max(prev, 1):6.2f}%"
            pct_cum  = f"{100.0 * n / max(n_init, 1):6.2f}%"
        print(f"{label:<60s} {n:>12,d}  {pct_step:>7s}  {pct_cum:>7s}")
        prev = n

    # Persist for re-plotting.
    import csv
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step_idx", "label", "n_pass", "step_pct", "cum_pct"])
        for i, (label, n) in enumerate(zip(CUT_LABELS, total)):
            sp = "" if i == 0 else f"{100*n/max(int(total[i-1]),1):.2f}"
            cp = f"{100*n/max(int(n_init),1):.2f}"
            w.writerow([i, label, int(n), sp, cp])
    print(f"\n  wrote {args.out_csv}")


if __name__ == "__main__":
    main()
