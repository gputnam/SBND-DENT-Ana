"""Build a per-hit DataFrame of TPC space-point vs anode-anode-line residuals.

CRT-INDEPENDENT analog of `make_crtdisp_df_sbnd.py`. For every TPC track
that pierces BOTH anodes (x = +/- 200 cm) and crosses the cathode in both
TPCs, take each collection-plane hit's space-point (x, y, z), project it
onto the line through the two anode-piercing points, and emit one row
with the hit position, the projected (foot) position, the (dx, dy, dz)
residual, the anode endpoints, and a few metadata fields.

Selection (per track, no CRT requirement):
  1. whicht0 != WHICHT0_NONE -- some t0 was applied so tick numbers are
     meaningful in the 2 MHz drift clock.
  2. There are >= 1 oncalo collection-plane hit in BOTH TPCs.
  3. Per TPC i in {0, 1}: t_min in [tmin_lo, tmin_hi] AND
     (t_max - t_min) >= DRIFT_FRAC * FULL_DRIFT_TICKS.

Endpoint per TPC: the (Y, Z) of the earliest-tick oncalo plane-2 hit in
that TPC (anode_y_i, anode_z_i). The anode-anode line is

    P0 = (-200, anode0_y, anode0_z)
    P1 = (+200, anode1_y, anode1_z)

For each oncalo plane-2 hit (in either TPC), foot = P0 + (((hit-P0).u))*u
with u = (P1 - P0)/|P1 - P0|, and (dx, dy, dz) = hit_xyz - foot.

Inputs:  SBND TrackCaloSkim ntuples (caloskim/TrackCaloSkim only -- no
         crtana branches needed).
Outputs: A directory of parquet shards, one row per oncalo plane-2 hit
         on a selected double-AC track. Sharded so no shard exceeds
         ~4 GB. With the expected ~7.6M total rows, one shard suffices.

Usage:
    python make_doubleanode_df_sbnd.py OUTDIR (filelist.txt | *.root ...)
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

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import uproot
from tqdm.auto import tqdm

from crtlib.constants import ANODE_X_NOMINAL, FULL_DRIFT_TICKS, WHICHT0_NONE
from crtlib.io import (collect_filelist, load_cut_config, pnfs_to_xroot,
                       set_use_xroot)
from crtlib.ntuple import TREE_TPC

DRIFT_FRAC_DEFAULT = 0.95

TCS_TRK_BRANCHES = ["meta.run", "meta.subrun", "meta.evt", "id", "whicht0"]
TCS_HIT_BRANCHES = ["hits2.h.time", "hits2.h.sp.x", "hits2.h.sp.y",
                    "hits2.h.sp.z", "hits2.h.tpc", "hits2.oncalo"]


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
            tcs_arr = f[TREE_TPC].arrays(TCS_TRK_BRANCHES + TCS_HIT_BRANCHES,
                                          library="ak")
    except Exception as e:
        print(f"[skip] {path}: {type(e).__name__}: {e}", flush=True)
        return None

    if len(tcs_arr) == 0:
        return None

    try:
        run = ak.to_numpy(tcs_arr["meta.run"]).astype(np.int64)
        sub = ak.to_numpy(tcs_arr["meta.subrun"]).astype(np.int64)
        evt = ak.to_numpy(tcs_arr["meta.evt"]).astype(np.int64)
        tid = ak.to_numpy(tcs_arr["id"]).astype(np.int64)
        wt0 = ak.to_numpy(tcs_arr["whicht0"]).astype(np.int32)

        rows = []
        for i in range(len(tcs_arr)):
            if int(wt0[i]) == WHICHT0_NONE:
                continue

            ht_raw = np.asarray(tcs_arr["hits2.h.time"][i], dtype=np.float64)
            hx_raw = np.asarray(tcs_arr["hits2.h.sp.x"][i], dtype=np.float64)
            hy_raw = np.asarray(tcs_arr["hits2.h.sp.y"][i], dtype=np.float64)
            hz_raw = np.asarray(tcs_arr["hits2.h.sp.z"][i], dtype=np.float64)
            htpc_raw = np.asarray(tcs_arr["hits2.h.tpc"][i], dtype=np.int32)
            hon = np.asarray(tcs_arr["hits2.oncalo"][i]).astype(bool)

            # Double-AC selection uses only the time mask (matches the
            # count_double_anode.py probe); requiring finite spacepoints here
            # would drop hits whose 2D->3D match failed and could lower t_max
            # below the drift threshold for otherwise-good tracks.
            ok_t = hon & np.isfinite(ht_raw)
            if not ok_t.any():
                continue
            ht_t = ht_raw[ok_t]
            htpc_t = htpc_raw[ok_t]
            mask0_t = (htpc_t == 0)
            mask1_t = (htpc_t == 1)
            if not (mask0_t.any() and mask1_t.any()):
                continue

            t0_arr = ht_t[mask0_t]
            t1_arr = ht_t[mask1_t]
            t0_min = float(t0_arr.min()); t0_max = float(t0_arr.max())
            t1_min = float(t1_arr.min()); t1_max = float(t1_arr.max())

            ac0 = (cuts["tmin_lo"] <= t0_min <= cuts["tmin_hi"]
                   and (t0_max - t0_min) >= drift_thresh)
            ac1 = (cuts["tmin_lo"] <= t1_min <= cuts["tmin_hi"]
                   and (t1_max - t1_min) >= drift_thresh)
            if not (ac0 and ac1):
                continue

            # Endpoint extraction and hit emission both require finite
            # spacepoints. Apply the stricter mask now.
            ok = ok_t & np.isfinite(hx_raw) & np.isfinite(hy_raw) & np.isfinite(hz_raw)
            if not ok.any():
                continue
            ht = ht_raw[ok]; hx = hx_raw[ok]; hy = hy_raw[ok]; hz = hz_raw[ok]
            htpc = htpc_raw[ok]
            mask0 = (htpc == 0)
            mask1 = (htpc == 1)
            if not (mask0.any() and mask1.any()):
                continue

            # Anode-piercing endpoint = earliest-tick spacepoint-valid hit per TPC.
            # In rare cases this is at a slightly later tick than the t_min
            # used for selection (if the t_min hit had a NaN spacepoint), but
            # it stays inside the anode peak window.
            j0 = int(np.argmin(ht[mask0]))
            j1 = int(np.argmin(ht[mask1]))
            ay0 = float(hy[mask0][j0]); az0 = float(hz[mask0][j0])
            ay1 = float(hy[mask1][j1]); az1 = float(hz[mask1][j1])

            P0 = np.array([ANODE_X_NOMINAL[0], ay0, az0], dtype=np.float64)
            P1 = np.array([ANODE_X_NOMINAL[1], ay1, az1], dtype=np.float64)
            line = P1 - P0
            line_norm = float(np.linalg.norm(line))
            if line_norm <= 0.0:
                continue
            u = line / line_norm

            hit_xyz = np.stack([hx, hy, hz], axis=1)            # (N, 3)
            delta = hit_xyz - P0[None, :]
            t_proj = delta @ u                                   # (N,)
            foot = P0[None, :] + t_proj[:, None] * u[None, :]    # (N, 3)
            d_vec = hit_xyz - foot                               # (N, 3)

            n = hx.size
            rows.append(pd.DataFrame({
                "run":      np.int32(run[i]),
                "subrun":   np.int32(sub[i]),
                "event":    np.int32(evt[i]),
                "trk_id":   np.int32(tid[i]),
                "hit_idx":  np.arange(n, dtype=np.int32),
                "hit_x":    hx.astype(np.float32),
                "hit_y":    hy.astype(np.float32),
                "hit_z":    hz.astype(np.float32),
                "hit_tpc":  htpc.astype(np.int8),
                "hit_time": ht.astype(np.float32),
                "anode0_y": np.full(n, ay0, dtype=np.float32),
                "anode0_z": np.full(n, az0, dtype=np.float32),
                "anode1_y": np.full(n, ay1, dtype=np.float32),
                "anode1_z": np.full(n, az1, dtype=np.float32),
                "foot_x":   foot[:, 0].astype(np.float32),
                "foot_y":   foot[:, 1].astype(np.float32),
                "foot_z":   foot[:, 2].astype(np.float32),
                "dx":       d_vec[:, 0].astype(np.float32),
                "dy":       d_vec[:, 1].astype(np.float32),
                "dz":       d_vec[:, 2].astype(np.float32),
                "whicht0":  np.uint8(int(wt0[i])),
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
    """Stream-write pandas frames into ParquetWriter shards capped by row count."""

    def __init__(self, out_dir, rows_per_shard, prefix="doubleanode_df_part",
                 compression="zstd"):
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
        else:
            table = table.cast(self.writer.schema_arrow,
                               safe=False) if hasattr(self.writer, "schema_arrow") \
                   else table
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
                    help="directory to write doubleanode_df_partNNN.parquet shards")
    ap.add_argument("inputs", nargs="*",
                    default=["filelist_nonempty_combined.txt"],
                    help="ROOT files, glob patterns, or .txt/.list of paths. "
                         "Defaults to filelist_nonempty_combined.txt.")
    ap.add_argument("--nproc", default="1")
    ap.add_argument("--maxfiles", type=int, default=None)
    ap.add_argument("--no-xroot", action="store_true",
                    help="read /pnfs files directly via NFS instead of xrootd")
    ap.add_argument("--cut-config",
                    default="cut_tuning/cut_config_per_tpc.json",
                    help="per-TPC anode-piercing cut config (default %(default)s)")
    ap.add_argument("--drift-frac", type=float, default=DRIFT_FRAC_DEFAULT,
                    help="(t_max - t_min) >= drift-frac * FULL_DRIFT_TICKS "
                         "required per TPC (default %(default)s)")
    ap.add_argument("--rows-per-shard", type=int, default=100_000_000,
                    help="rollover threshold per parquet shard "
                         "(default 100 M rows)")
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
    print(f"Rows / shard : {args.rows_per_shard:,}", flush=True)

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
        n_tracks += int(r.groupby(["run", "subrun", "event", "trk_id"]).ngroups)
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
    print(f"Total tracks   : {n_tracks:,} (per-shard groupby; "
          f"may slightly under-count if a track straddles shards)", flush=True)
    print(f"Shards written : {writer.idx}", flush=True)


if __name__ == "__main__":
    main()
