"""Build a per-hit DataFrame of raw (wire, tick) hit positions for the
2D double-anode field-distortion method.

CRT-INDEPENDENT and SPACE-POINT-INDEPENDENT. For every TPC track that
pierces BOTH anodes (x = +/- 200 cm) and crosses the cathode in both
TPCs, emit one row per oncalo hit -- ALL THREE wire planes -- carrying
only the raw detector quantities:

    plane (0/1/2), tpc (0/1), wire (integer), time (tick), integral.

Unlike make_doubleanode_df_sbnd.py this reads NO `hits.h.sp.*`
space-point branch. The downstream 2D analysis turns (wire, tick) into
(s_view, x) using the GDML/LArSoft wire geometry + the empirical anode
t0 and the cathode-stage drift velocity, builds one 2D line per plane
from the anode-piercing hits, and inverts per voxel.

Selection (per track, no CRT requirement) -- identical to the original
double-anode builder, evaluated on COLLECTION-plane (plane 2) hit times:
  1. whicht0 != WHICHT0_NONE -- some t0 was applied so tick numbers are
     meaningful in the 2 MHz drift clock.
  2. >= 1 oncalo plane-2 hit with finite time in BOTH TPCs.
  3. Per TPC i in {0, 1}: t_min in [tmin_lo, tmin_hi] AND
     (t_max - t_min) >= DRIFT_FRAC * FULL_DRIFT_TICKS.

A passing track emits every oncalo hit on all 3 planes with a finite
time and a valid (>= 0) wire number. No hasSP requirement -- the 2D
method does not need the 2D->3D match to have succeeded.

Inputs:  SBND TrackCaloSkim ntuples (caloskim/TrackCaloSkim only).
Outputs: parquet shards, one row per oncalo hit on a selected
         double-AC track.

Usage:
    python make_doubleanode_2d_df_sbnd.py OUTDIR (filelist.txt | *.root ...)
        [--nproc N | auto] [--maxfiles N] [--no-xroot]
        [--cut-config cut_tuning/cut_config_per_tpc.json]
        [--drift-frac F] [--rows-per-shard N]
"""

import argparse
import os
import time
import warnings
from multiprocessing import Pool, cpu_count

try:
    os.nice(10)
except (AttributeError, PermissionError):
    pass

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot
from tqdm.auto import tqdm

from crtlib.constants import FULL_DRIFT_TICKS, WHICHT0_NONE
from crtlib.io import (collect_filelist, load_cut_config, pnfs_to_xroot,
                       set_use_xroot)
from crtlib.ntuple import TREE_TPC

DRIFT_FRAC_DEFAULT = 0.95

TCS_TRK_BRANCHES = ["meta.run", "meta.subrun", "meta.evt", "id", "whicht0"]
# per-hit branches, all three planes
TCS_HIT_BRANCHES = []
for _p in (0, 1, 2):
    TCS_HIT_BRANCHES += [f"hits{_p}.h.time", f"hits{_p}.h.wire",
                         f"hits{_p}.h.tpc", f"hits{_p}.h.integral",
                         f"hits{_p}.oncalo"]

_CUTS = None
_DRIFT_THRESH = None


def _worker_init(cuts, drift_thresh):
    global _CUTS, _DRIFT_THRESH
    _CUTS = cuts
    _DRIFT_THRESH = drift_thresh


def load_one_file(args):
    path, idx = args
    cuts = _CUTS
    drift_thresh = _DRIFT_THRESH
    url = pnfs_to_xroot(path)
    try:
        with uproot.open(url, timeout=120) as f:
            arr = f[TREE_TPC].arrays(TCS_TRK_BRANCHES + TCS_HIT_BRANCHES,
                                     library="ak")
    except Exception as e:
        print(f"[skip] {path}: {type(e).__name__}: {e}", flush=True)
        return None

    if len(arr) == 0:
        return None

    try:
        import awkward as ak
        run = ak.to_numpy(arr["meta.run"]).astype(np.int64)
        sub = ak.to_numpy(arr["meta.subrun"]).astype(np.int64)
        evt = ak.to_numpy(arr["meta.evt"]).astype(np.int64)
        tid = ak.to_numpy(arr["id"]).astype(np.int64)
        wt0 = ak.to_numpy(arr["whicht0"]).astype(np.int32)

        rows = []
        for i in range(len(arr)):
            if int(wt0[i]) == WHICHT0_NONE:
                continue

            # --- selection on collection-plane (plane 2) oncalo hits ---
            ht2 = np.asarray(arr["hits2.h.time"][i], dtype=np.float64)
            tp2 = np.asarray(arr["hits2.h.tpc"][i], dtype=np.int32)
            on2 = np.asarray(arr["hits2.oncalo"][i]).astype(bool)
            ok2 = on2 & np.isfinite(ht2)
            if not ok2.any():
                continue
            t2 = ht2[ok2]
            tp2 = tp2[ok2]
            m0 = tp2 == 0
            m1 = tp2 == 1
            if not (m0.any() and m1.any()):
                continue
            t0 = t2[m0]
            t1 = t2[m1]
            t0min, t0max = float(t0.min()), float(t0.max())
            t1min, t1max = float(t1.min()), float(t1.max())
            ac0 = (cuts["tmin_lo"] <= t0min <= cuts["tmin_hi"]
                   and (t0max - t0min) >= drift_thresh)
            ac1 = (cuts["tmin_lo"] <= t1min <= cuts["tmin_hi"]
                   and (t1max - t1min) >= drift_thresh)
            if not (ac0 and ac1):
                continue

            # --- emit every oncalo hit on all 3 planes ---
            planes, tpcs, wires, times, integs = [], [], [], [], []
            for p in (0, 1, 2):
                htp = np.asarray(arr[f"hits{p}.h.time"][i], dtype=np.float64)
                hwp = np.asarray(arr[f"hits{p}.h.wire"][i], dtype=np.int64)
                htc = np.asarray(arr[f"hits{p}.h.tpc"][i], dtype=np.int32)
                hin = np.asarray(arr[f"hits{p}.h.integral"][i],
                                 dtype=np.float64)
                onp = np.asarray(arr[f"hits{p}.oncalo"][i]).astype(bool)
                ok = onp & np.isfinite(htp) & (hwp >= 0)
                if not ok.any():
                    continue
                n = int(ok.sum())
                planes.append(np.full(n, p, dtype=np.int8))
                tpcs.append(htc[ok].astype(np.int8))
                wires.append(hwp[ok].astype(np.int16))
                times.append(htp[ok].astype(np.float32))
                integs.append(hin[ok].astype(np.float32))
            if not planes:
                continue
            n = sum(len(x) for x in planes)
            rows.append(pd.DataFrame({
                "run":      np.full(n, run[i], dtype=np.int32),
                "subrun":   np.full(n, sub[i], dtype=np.int32),
                "event":    np.full(n, evt[i], dtype=np.int32),
                "trk_id":   np.full(n, tid[i], dtype=np.int32),
                "plane":    np.concatenate(planes),
                "tpc":      np.concatenate(tpcs),
                "wire":     np.concatenate(wires),
                "time":     np.concatenate(times),
                "integral": np.concatenate(integs),
                "whicht0":  np.full(n, int(wt0[i]), dtype=np.uint8),
            }))
    except Exception as e:
        print(f"[reduce-skip] {path}: {type(e).__name__}: {e}", flush=True)
        return None

    if not rows:
        return None
    out = pd.concat(rows, axis=0, ignore_index=True)
    out["__file"] = np.int32(idx)
    return out


class ShardWriter:
    """Stream-write pandas frames into ParquetWriter shards capped by rows."""

    def __init__(self, out_dir, rows_per_shard,
                 prefix="doubleanode_2d_df_part", compression="zstd"):
        self.out_dir = out_dir
        self.rows_per_shard = rows_per_shard
        self.prefix = prefix
        self.compression = compression
        self.idx = 0
        self.writer = None
        self.path = None
        self.rows_in_shard = 0
        self.total_rows = 0

    def _open(self, schema):
        self.path = os.path.join(
            self.out_dir, f"{self.prefix}{self.idx:03d}.parquet")
        self.writer = pq.ParquetWriter(self.path, schema,
                                       compression=self.compression)
        self.rows_in_shard = 0

    def _close(self):
        if self.writer is None:
            return
        self.writer.close()
        sz = os.path.getsize(self.path)
        print(f"  -> wrote {self.path} ({self.rows_in_shard:,} rows, "
              f"{sz/1e9:.2f} GB)", flush=True)
        self.idx += 1
        self.writer = None
        self.path = None
        self.rows_in_shard = 0

    def write(self, df):
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self._open(table.schema)
        self.writer.write_table(table)
        self.rows_in_shard += len(df)
        self.total_rows += len(df)
        if self.rows_in_shard >= self.rows_per_shard:
            self._close()

    def close(self):
        self._close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir",
                    help="directory for doubleanode_2d_df_partNNN.parquet")
    ap.add_argument("inputs", nargs="*",
                    default=["filelist_drift_full.txt"],
                    help="ROOT files, globs, or .txt/.list of paths. "
                         "Defaults to filelist_drift_full.txt (225k union).")
    ap.add_argument("--nproc", default="1")
    ap.add_argument("--maxfiles", type=int, default=None)
    ap.add_argument("--no-xroot", action="store_true",
                    help="read /pnfs files directly via NFS instead of xrootd")
    ap.add_argument("--cut-config",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "config", "cut_tuning",
                                         "cut_config_per_tpc.json"))
    ap.add_argument("--drift-frac", type=float, default=DRIFT_FRAC_DEFAULT)
    ap.add_argument("--rows-per-shard", type=int, default=100_000_000)
    args = ap.parse_args()
    set_use_xroot(not args.no_xroot)

    cuts = load_cut_config(args.cut_config)
    drift_thresh = args.drift_frac * FULL_DRIFT_TICKS
    os.makedirs(args.output_dir, exist_ok=True)

    paths = collect_filelist(args.inputs, args.maxfiles)
    if args.nproc == "auto":
        nproc = min(cpu_count(), len(paths))
    else:
        nproc = int(args.nproc)
    print(f"Processing {len(paths)} files with nproc={nproc}", flush=True)
    print(f"Output dir   : {args.output_dir}", flush=True)
    print(f"Cut window   : t_min in [{cuts['tmin_lo']:.1f}, "
          f"{cuts['tmin_hi']:.1f}] ticks per TPC", flush=True)
    print(f"Drift thresh : (t_max - t_min) >= {drift_thresh:.1f} ticks "
          f"per TPC ({args.drift_frac} * {FULL_DRIFT_TICKS})", flush=True)

    pool_args = [(p, i) for i, p in enumerate(paths)]
    t0 = time.time()
    writer = ShardWriter(args.output_dir, args.rows_per_shard)
    n_hits = 0
    n_tracks = 0

    def consume(r):
        nonlocal n_hits, n_tracks
        if r is None or r.empty:
            return
        n_hits += len(r)
        n_tracks += int(r.groupby(
            ["run", "subrun", "event", "trk_id"]).ngroups)
        writer.write(r)

    if nproc <= 1:
        _worker_init(cuts, drift_thresh)
        for a in tqdm(pool_args, unit="file"):
            consume(load_one_file(a))
    else:
        with Pool(processes=nproc, initializer=_worker_init,
                  initargs=(cuts, drift_thresh)) as pool:
            for r in tqdm(pool.imap_unordered(load_one_file, pool_args),
                          total=len(pool_args), unit="file", smoothing=0.6):
                consume(r)
    writer.close()

    elapsed = time.time() - t0
    print(f"\nRead+reduce wall time: {elapsed:.1f}s "
          f"({elapsed/max(1,len(paths)):.2f}s/file)", flush=True)
    print(f"Total hits     : {n_hits:,}", flush=True)
    print(f"Total tracks   : {n_tracks:,} (per-shard groupby)", flush=True)
    print(f"Shards written : {writer.idx}", flush=True)


if __name__ == "__main__":
    main()
