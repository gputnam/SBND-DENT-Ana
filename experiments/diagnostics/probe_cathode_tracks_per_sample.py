"""Per-sample track-count probe for the cathode measurement.

Reads a small random sample of files from each of the four filelists
(offbeam, onbeam, Run1 crossingmuon, Run2 crossingmuon), runs the same
caloskim/TrackCaloSkim → per-track (tpcN_pP_min_time, tpcN_pP_max_time)
extraction as make_driftV_df_sbnd.py, and counts how many tracks
satisfy the cathode-measurement selection (dt0 > 2400 AND dt1 > 2400
ticks, plane 2 = collection plane). Scales to the full filelist size.

Writes a summary table.
"""

import os
import random
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from crtlib.io import collect_filelist, pnfs_to_xroot, set_use_xroot

import awkward as ak
import uproot

set_use_xroot(True)

TREE = "caloskim/TrackCaloSkim"
PLANE = 2  # collection plane (plot_cathode_heatmap.report_drift_velocity uses this style)


def count_one_file(path):
    """Returns (n_tracks_total, n_with_both_times, n_AC_through_going)."""
    try:
        with uproot.open(pnfs_to_xroot(path), timeout=120) as f:
            arr = f[TREE].arrays(
                ["hits2.h.time", "hits2.h.tpc"], library="ak",
            )
    except Exception:
        return None
    if len(arr) == 0:
        return (0, 0, 0)

    n_tracks = len(arr)
    n_have_both = 0
    n_ac = 0
    for i in range(n_tracks):
        ht = np.asarray(arr["hits2.h.time"][i], dtype=np.float64)
        ht_tpc = np.asarray(arr["hits2.h.tpc"][i], dtype=np.int32)
        if ht.size == 0:
            continue
        m0 = ht_tpc == 0; m1 = ht_tpc == 1
        if not (m0.any() and m1.any()):
            continue
        dt0 = ht[m0].max() - ht[m0].min()
        dt1 = ht[m1].max() - ht[m1].min()
        if not (np.isfinite(dt0) and np.isfinite(dt1)):
            continue
        n_have_both += 1
        if dt0 > 2400 and dt1 > 2400:
            n_ac += 1
    return (n_tracks, n_have_both, n_ac)


def probe(filelist, *, n_sample, seed=0, nproc=24):
    rng = random.Random(seed)
    all_files = open(filelist).read().split()
    sampled = rng.sample(all_files, min(n_sample, len(all_files)))
    with Pool(processes=nproc) as pool:
        results = pool.map(count_one_file, sampled)
    results = [r for r in results if r is not None]
    n_files = len(results)
    tot_tracks = sum(r[0] for r in results)
    tot_both   = sum(r[1] for r in results)
    tot_ac     = sum(r[2] for r in results)
    return {
        "filelist": filelist,
        "n_files_total": len(all_files),
        "n_files_sampled_OK": n_files,
        "n_files_sampled_total": len(sampled),
        "avg_tracks_per_file":         tot_tracks / max(1, n_files),
        "avg_dt_both_per_file":        tot_both   / max(1, n_files),
        "avg_AC_per_file":             tot_ac     / max(1, n_files),
        "extrapolated_total_AC":       int(round(tot_ac / max(1, n_files) * len(all_files))),
    }


def main():
    samples = [
        ("offbeam",          "filelist_nonempty.txt"),
        # onbeam = combined - offbeam; build it transiently
        ("onbeam",           "_onbeam_only.txt"),
        ("Run1_crossingmu",  "Run1_crossingmuon_filelist.txt"),
        ("Run2_crossingmu",  "Run2_crossingmuon_filelist.txt"),
    ]

    # Build the onbeam-only filelist
    combined = set(open("filelist_nonempty_combined.txt").read().split())
    offbeam  = set(open("filelist_nonempty.txt").read().split())
    onbeam = sorted(combined - offbeam)
    with open("_onbeam_only.txt", "w") as f:
        for p in onbeam: f.write(p + "\n")
    print(f"onbeam = combined - offbeam = {len(onbeam)} files", flush=True)

    n_sample = 50
    print(f"sampling {n_sample} random files per filelist with nproc=24", flush=True)
    rows = []
    for tag, path in samples:
        print(f"\n--- {tag} ({path}) ---", flush=True)
        r = probe(path, n_sample=n_sample, seed=42, nproc=24)
        r["sample"] = tag
        rows.append(r)
        print(f"  files sampled: {r['n_files_sampled_OK']}/{r['n_files_sampled_total']} "
              f"(of {r['n_files_total']:,d} total)", flush=True)
        print(f"  avg tracks/file: {r['avg_tracks_per_file']:.1f}", flush=True)
        print(f"  avg AC-through-going/file: {r['avg_AC_per_file']:.2f}", flush=True)
        print(f"  extrapolated total AC: {r['extrapolated_total_AC']:,d}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    df = pd.DataFrame(rows)
    df = df[["sample", "n_files_total", "n_files_sampled_OK",
             "avg_tracks_per_file", "avg_AC_per_file",
             "extrapolated_total_AC"]]
    print(df.to_string(index=False), flush=True)
    df.to_csv("cathode_tracks_per_sample.csv", index=False)


if __name__ == "__main__":
    main()
