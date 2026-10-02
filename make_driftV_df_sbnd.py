"""
Build a per-track DataFrame of hit start/end times for SBND TrackCaloSkim
ntuples. SBND port of make_driftV_df.py from gputnam/ICARUS-calibrations.

Input:  ROOT files containing a `caloskim/TrackCaloSkim` TTree.
Output: Parquet file with one row per track and the columns documented at
        the bottom of this file.

Usage:
    python make_driftV_df_sbnd.py OUTPUT.parquet (FILELIST.txt | FILE.root [...])
                                  [--nproc N | auto] [--maxfiles N] [--no-xroot]

Files under /pnfs are opened over xrootd (root://fndca1.fnal.gov:1094/...)
unless --no-xroot is given. xrootd avoids dCache POSIX stalls on busy nodes.
"""

import argparse
import os
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count

# Run at low priority by default so this doesn't fight other interactive jobs.
try:
    os.nice(10)
except (AttributeError, PermissionError):
    pass

warnings.simplefilter(action="ignore", category=FutureWarning)

import awkward as ak
import numpy as np
import pandas as pd
import uproot
from tqdm.auto import tqdm

TREE_PATH = "caloskim/TrackCaloSkim"

TRK_BRANCHES = [
    "meta.run", "meta.evt", "meta.subrun", "meta.iproc", "meta.time",
    "t0PFP", "id", "clear_cosmic_muon", "cryostat",
    "start.x", "start.y", "start.z",
    "end.x", "end.y", "end.z",
    "dir.x", "dir.y", "dir.z",
    "length", "selected",
]

PLANES = (0, 1, 2)


def _hit_branches(plane):
    return [f"hits{plane}.h.time", f"hits{plane}.h.sp.y", f"hits{plane}.h.sp.z",
            f"hits{plane}.h.tpc", f"hits{plane}.oncalo"]


from crtlib.io import pnfs_to_xroot, set_use_xroot, collect_filelist  # noqa: F401


def _per_tpc_block(hits_df, label, time_col, y_col, z_col):
    """Per-track min/max block for one (TPC, plane) slice of hits.

    `label` is e.g. ``tpc0_p2`` and forms the column-name prefix.
    Output columns: {label}_min_time, _min_y, _min_z, _min_dt,
                    {label}_max_time, _max_y, _max_z, _max_dt.
    """
    cols_out = [f"{label}_{k}" for k in
                ("min_time", "min_y", "min_z", "min_dt",
                 "max_time", "max_y", "max_z", "max_dt")]
    if hits_df.empty:
        return pd.DataFrame(columns=cols_out)

    time_by_track = hits_df[time_col].groupby(level=0)
    min_idx = time_by_track.idxmin()
    max_idx = time_by_track.idxmax()

    min_rows = hits_df.loc[min_idx, [time_col, y_col, z_col]].droplevel(1)
    max_rows = hits_df.loc[max_idx, [time_col, y_col, z_col]].droplevel(1)
    min_rows.columns = [f"{label}_min_time", f"{label}_min_y", f"{label}_min_z"]
    max_rows.columns = [f"{label}_max_time", f"{label}_max_y", f"{label}_max_z"]

    min_dt = (time_by_track.nsmallest(10, keep="all")
              .groupby(level=0).diff()
              .groupby(level=0).mean())
    min_dt.name = f"{label}_min_dt"
    max_dt = (time_by_track.nlargest(10, keep="all")
              .groupby(level=0).diff(-1)
              .groupby(level=0).mean())
    max_dt.name = f"{label}_max_dt"

    return (min_rows
            .join(min_dt, how="outer")
            .join(max_rows, how="outer")
            .join(max_dt, how="outer"))


def load_one_file(args):
    path, idx = args
    url = pnfs_to_xroot(path)
    all_branches = TRK_BRANCHES + sum((_hit_branches(p) for p in PLANES), [])
    try:
        with uproot.open(url, timeout=120) as f:
            t = f[TREE_PATH]
            arr = t.arrays(all_branches, library="ak")
    except Exception as e:
        print(f"[skip] {path}: {type(e).__name__}: {e}", flush=True)
        return None

    try:
        # Per-track scalars: zip them and convert to a flat (entry,) df.
        trk_zip = ak.zip({c: arr[c] for c in TRK_BRANCHES}, depth_limit=1)
        outdf = ak.to_dataframe(trk_zip)
        if outdf.index.nlevels > 1:
            outdf = outdf.groupby(level=0).first()

        # Per-plane jagged blocks: zip just that plane's fields into a df with
        # a (entry, subentry) MultiIndex, reduce, and join.
        for plane in PLANES:
            plane_zip = ak.zip({
                "time":   arr[f"hits{plane}.h.time"],
                "y":      arr[f"hits{plane}.h.sp.y"],
                "z":      arr[f"hits{plane}.h.sp.z"],
                "tpc":    arr[f"hits{plane}.h.tpc"],
                "oncalo": arr[f"hits{plane}.oncalo"],
            })
            pdf = ak.to_dataframe(plane_zip)
            for tpc in (0, 1):
                sub = pdf[pdf["oncalo"].astype(bool) & (pdf["tpc"] == tpc)]
                block = _per_tpc_block(sub, f"tpc{tpc}_p{plane}", "time", "y", "z")
                outdf = outdf.join(block, how="left")
            del plane_zip, pdf
    except Exception as e:
        print(f"[reduce-skip] {path}: {type(e).__name__}: {e}", flush=True)
        return None

    outdf = outdf.assign(__file=idx).set_index("__file", append=True)
    outdf = outdf.reorder_levels(["__file"] + list(outdf.index.names[:-1]))
    return outdf




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("inputs", nargs="*",
                    default=["filelist_nonempty_combined.txt"],
                    help="ROOT files, glob patterns, or .txt/.list of paths. "
                         "Defaults to filelist_nonempty_combined.txt — the "
                         "FULL set of available calibration ntuples "
                         "(offbeamlight + bnblight, ~137k files).")
    ap.add_argument("--nproc", default="1")
    ap.add_argument("--maxfiles", type=int, default=None)
    ap.add_argument("--no-xroot", action="store_true",
                    help="read /pnfs files directly via NFS instead of xrootd")
    args = ap.parse_args()
    set_use_xroot(not args.no_xroot)

    paths = collect_filelist(args.inputs, args.maxfiles)
    if args.nproc == "auto":
        nproc = min(cpu_count(), len(paths))
    else:
        nproc = int(args.nproc)
    print(f"Processing {len(paths)} files with nproc={nproc}", flush=True)

    pool_args = [(p, i) for i, p in enumerate(paths)]
    t0 = time.time()
    parts = []
    if nproc <= 1:
        for a in tqdm(pool_args, unit="file"):
            r = load_one_file(a)
            if r is not None and not r.empty:
                parts.append(r)
    else:
        with Pool(processes=nproc) as pool:
            for r in tqdm(pool.imap_unordered(load_one_file, pool_args),
                          total=len(pool_args), unit="file", smoothing=0.6):
                if r is not None and not r.empty:
                    parts.append(r)
    elapsed = time.time() - t0
    print(f"Read+reduce wall time: {elapsed:.1f}s ({elapsed/max(1,len(paths)):.2f}s/file)", flush=True)

    if not parts:
        print("No data collected.", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(parts, axis=0, ignore_index=False).sort_index()
    print(f"Total tracks: {len(df):,}", flush=True)
    print(f"Unique runs : {df['meta.run'].nunique() if 'meta.run' in df.columns else '?'}", flush=True)

    df.to_parquet(args.output, compression="zstd", index=True)
    out_size = os.path.getsize(args.output)
    print(f"Wrote {args.output} ({out_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
