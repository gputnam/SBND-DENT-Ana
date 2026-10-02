"""Count tracks that pierce both anodes AND cross the cathode in both TPCs
(``double anode-cathode crossers``), as a CRT-independent selection for
field-distortion studies.

Per-track criteria (no CRT requirement):
  1. whicht0 != -1 — some T0 was applied so tick numbers are meaningful
  2. tpc0_n_p2_hits > 0 AND tpc1_n_p2_hits > 0
  3. tpc0_t_min in [414.5, 427.5] AND tpc1_t_min in [414.5, 427.5]
  4. (tpc0_t_max - tpc0_t_min) >= 0.95 * 2575 ticks AND same for TPC1

Also reports a "single anode-cathode crosser" count (either TPC alone) for
comparison.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool, cpu_count

import awkward as ak
import numpy as np
import uproot

from crtlib.constants import WHICHT0_NONE, FULL_DRIFT_TICKS
from crtlib.io import pnfs_to_xroot, set_use_xroot, load_cut_config
from crtlib.ntuple import (TREE_TPC,
                            TCS_TRK_BRANCHES_FULL as TCS_TRK_BRANCHES)

TCS_HIT_BRANCHES = ["hits2.h.time", "hits2.h.tpc", "hits2.oncalo"]
DRIFT_FRAC = 0.95


def _scan_one_file(args):
    """Return (n_total, n_single_ac, n_double_ac) for one ntuple file."""
    path, cuts = args
    url = pnfs_to_xroot(path)
    try:
        with uproot.open(url, timeout=120) as f:
            arr = f[TREE_TPC].arrays(TCS_TRK_BRANCHES + TCS_HIT_BRANCHES,
                                      library="ak")
    except Exception as e:
        print(f"[skip] {path}: {type(e).__name__}: {e}", flush=True)
        return (0, 0, 0)

    n_total = int(len(arr))
    if n_total == 0:
        return (0, 0, 0)

    wt0 = ak.to_numpy(arr["whicht0"]).astype(np.int32)

    n_single = 0
    n_double = 0
    drift_thresh = DRIFT_FRAC * FULL_DRIFT_TICKS
    for i in range(n_total):
        if int(wt0[i]) == WHICHT0_NONE:
            continue
        ht = np.asarray(arr["hits2.h.time"][i], dtype=np.float64)
        htpc = np.asarray(arr["hits2.h.tpc"][i], dtype=np.int32)
        hon = np.asarray(arr["hits2.oncalo"][i]).astype(bool)
        ok = hon & np.isfinite(ht)
        if not ok.any():
            continue
        ht = ht[ok]; htpc = htpc[ok]
        m0 = (htpc == 0); m1 = (htpc == 1)
        n0, n1 = int(m0.sum()), int(m1.sum())

        ac0 = ac1 = False
        if n0 > 0:
            t0_min = float(ht[m0].min()); t0_max = float(ht[m0].max())
            if (cuts["tmin_lo"] <= t0_min <= cuts["tmin_hi"]
                    and (t0_max - t0_min) >= drift_thresh):
                ac0 = True
        if n1 > 0:
            t1_min = float(ht[m1].min()); t1_max = float(ht[m1].max())
            if (cuts["tmin_lo"] <= t1_min <= cuts["tmin_hi"]
                    and (t1_max - t1_min) >= drift_thresh):
                ac1 = True
        if ac0 or ac1:
            n_single += 1
        if ac0 and ac1:
            n_double += 1
    return (n_total, n_single, n_double)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filelist",
                    default="filelist_nonempty_combined.txt")
    ap.add_argument("--maxfiles", type=int, default=400,
                    help="default 400 ≈ 1/343 of the 137,445-file production")
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--cut-config",
                    default="cut_tuning/cut_config_per_tpc.json")
    ap.add_argument("--no-xroot", action="store_true")
    ap.add_argument("--full-files", type=int, default=137445,
                    help="full-set file count for scaling")
    ap.add_argument("--align-df-tracks", type=int, default=151141,
                    help="number of tracks in the current align_df parquet "
                         "(the CRT-matched anode-piercing sample)")
    args = ap.parse_args()

    set_use_xroot(not args.no_xroot)
    cuts = load_cut_config(args.cut_config)

    with open(args.filelist) as f:
        files = [ln.strip() for ln in f if ln.strip()][:args.maxfiles]
    print(f"  filelist     : {args.filelist}")
    print(f"  N files      : {len(files)} of {args.full_files}")
    print(f"  tmin window  : [{cuts['tmin_lo']:.1f}, {cuts['tmin_hi']:.1f}] ticks")
    print(f"  drift thresh : >= {DRIFT_FRAC*FULL_DRIFT_TICKS:.1f} ticks")

    work = [(p, cuts) for p in files]
    nproc = min(args.nproc, len(files), cpu_count())
    t0 = time.time()
    n_total = n_single = n_double = 0
    n_files_ok = 0
    with Pool(processes=nproc) as pool:
        for k, (nt, ns, nd) in enumerate(pool.imap_unordered(_scan_one_file, work,
                                                              chunksize=2)):
            n_total += nt
            n_single += ns
            n_double += nd
            if nt > 0:
                n_files_ok += 1
            if (k + 1) % max(1, len(files) // 20) == 0:
                rate = (k + 1) / (time.time() - t0)
                print(f"  {k+1}/{len(files)}  ({rate:.1f} files/s)",
                      flush=True)
    elapsed = time.time() - t0
    print(f"\n  done in {elapsed:.1f}s  ({n_files_ok} files contributed)")

    print(f"\n=== count summary ===")
    print(f"  N tracks (whicht0 != -1)        : {n_total:,}")
    print(f"  single anode-cathode crossers   : {n_single:,}  "
          f"({100*n_single/max(n_total,1):.2f} % of N tracks)")
    print(f"  double anode-cathode crossers   : {n_double:,}  "
          f"({100*n_double/max(n_total,1):.4f} % of N tracks)")
    print(f"  double / single ratio           : "
          f"{n_double/max(n_single,1):.4f}")

    scale = args.full_files / max(n_files_ok, 1)
    print(f"\n=== scaled to full {args.full_files}-file set "
          f"(× {scale:.1f}) ===")
    print(f"  expected double anode crossers  : "
          f"{int(round(n_double * scale)):,}")
    print(f"  expected single anode crossers  : "
          f"{int(round(n_single * scale)):,}")

    print(f"\n=== comparison to CRT-matched anode-piercing sample ===")
    print(f"  align_df.parquet (full set, CRT-matched): {args.align_df_tracks:,} tracks")
    ratio = (n_double * scale) / max(args.align_df_tracks, 1)
    print(f"  double-anode / CRT-crosser              : {ratio:.4f}")
    print(f"    (this is the relative statistical reach of a CRT-FREE "
          f"field-distortion analysis vs. the existing CRT-based one)")


if __name__ == "__main__":
    main()
