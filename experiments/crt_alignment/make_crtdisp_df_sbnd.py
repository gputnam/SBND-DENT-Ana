"""Build a per-hit DataFrame of TPC space-point vs CRT-track residuals.

For every TPC track that's matched to a CRT track in `crtana/tree`, take
each collection-plane hit's space-point (x, y, z), project it onto the
CRT line (closest-point foot, infinite-line extrapolation), and emit one
row with the hit position, the CRT-projected position, the (dx, dy, dz)
residual, and the matched CRT track's metadata (endpoints, score,
taggers, t0).

Input files:  SBND TrackCaloSkim ntuples (also contain `crtana/tree`).
Output:       A directory of parquet shards, one row per CRT-matched,
              oncalo, plane-2 hit. Sharded so no shard exceeds ~4.5 GB.

Join:
    Each `caloskim/TrackCaloSkim` entry is one TPC track and carries
    `meta.{run,subrun,evt}` + `id`. Each `crtana/tree` entry is one event
    and carries jagged `tpc_tr_*` arrays of length = # reco TPC tracks in
    that event. The TCS `id` indexes into those arrays (crtana stores
    *every* reco TPC track; TCS keeps a subset, so positional join is
    wrong — only `(run,subrun,event, id)` works).

Memory:
    Workers each hold one file's DataFrame (~few MB). The parent
    streams every returned DataFrame straight into a pyarrow
    ParquetWriter and rolls over to a new shard at ROWS_PER_SHARD rows.
    Peak resident memory is single-digit GB even with nproc=64.

Usage:
    python make_crtdisp_df_sbnd.py OUTDIR (filelist.txt | *.root ...)
        [--nproc N | auto] [--maxfiles N] [--no-xroot]
        [--rows-per-shard N]
"""

import argparse
import os
import sys
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

from crtlib.constants import SENTINEL
from crtlib.io import pnfs_to_xroot, set_use_xroot
from crtlib.ntuple import TREE_TPC, TREE_CRT

TCS_TRK_BRANCHES = ["meta.run", "meta.subrun", "meta.evt", "id", "t0PFP",
                    "xShiftCRT", "whicht0"]

# whicht0 values seen in SBND TrackCaloSkim:
#   0 = no t0 source (hits keep default trigger t0; uncorrected)
#   1 = CRT-Track-derived t0 (uses trk/t0CRTTrack)
#   3 = CRT-SpacePoint-derived t0 (uses trk/t0CRTSpacePoint)
# We require whicht0 == WHICHT0_CRTTRACK so the per-hit space points share
# the same CRT timing reference as the projection target (the matched CRT
# track endpoints from crtana). Mixing whicht0=3 hits with the CRT-track
# projection adds ~60 cm of residual scatter empirically.
WHICHT0_CRTTRACK = 1
TCS_HIT_BRANCHES = ["hits2.h.sp.x", "hits2.h.sp.y", "hits2.h.sp.z",
                    "hits2.oncalo"]

CRT_BRANCHES = [
    "run", "subrun", "event",
    "tpc_tr_matched", "tpc_tr_score",
    "tpc_tr_start_x", "tpc_tr_start_y", "tpc_tr_start_z",
    "tpc_tr_end_x", "tpc_tr_end_y", "tpc_tr_end_z",
    "tpc_tr_taggers", "tpc_tr_ts0", "tpc_tr_ts1",
]

TOP_TAGS = (5, 6)
SIDE_TAGS = (1, 2, 3, 4)
# SBND CRT wall enum (per CRTGeoService):
#   0 = Bottom, 1 = South, 2 = North, 3 = West, 4 = East,
#   5 = TopLow, 6 = TopHigh, 7 = (other / front)
TAG_S = 1
TAG_N = 2
TAG_W = 3
TAG_E = 4


def _classify_taggers(tags_list):
    """Classify a track by its CRT tagger list.

    Returns dict with:
      n_taggers (int), has_top, has_S, has_N, has_W, has_E, top_only, side_only.
    """
    n = len(tags_list)
    has_top = any(t in TOP_TAGS for t in tags_list)
    has_S = TAG_S in tags_list
    has_N = TAG_N in tags_list
    has_W = TAG_W in tags_list
    has_E = TAG_E in tags_list
    top_only = (n > 0) and all(t in TOP_TAGS for t in tags_list)
    side_only = (n > 0) and all(t in SIDE_TAGS for t in tags_list)
    return {
        "n_taggers": n,
        "has_top": has_top, "has_S": has_S, "has_N": has_N,
        "has_W": has_W, "has_E": has_E,
        "top_only": top_only, "side_only": side_only,
    }


def load_one_file(args):
    path, idx = args
    url = pnfs_to_xroot(path)
    try:
        with uproot.open(url, timeout=120) as f:
            tcs = f[TREE_TPC]
            crt = f[TREE_CRT]
            tcs_arr = tcs.arrays(TCS_TRK_BRANCHES + TCS_HIT_BRANCHES,
                                 library="ak")
            crt_arr = crt.arrays(CRT_BRANCHES, library="ak")
    except Exception as e:
        print(f"[skip] {path}: {type(e).__name__}: {e}", flush=True)
        return None

    if len(tcs_arr) == 0 or len(crt_arr) == 0:
        return None

    try:
        crt_run = ak.to_numpy(crt_arr["run"]).astype(np.int64)
        crt_sub = ak.to_numpy(crt_arr["subrun"]).astype(np.int64)
        crt_evt = ak.to_numpy(crt_arr["event"]).astype(np.int64)
        # Map (run, subrun, event) -> position in crtana arrays.
        ev_index = {(int(r), int(s), int(e)): i for i, (r, s, e)
                    in enumerate(zip(crt_run, crt_sub, crt_evt))}

        # Pull crtana fields once (still ak arrays of variable shape).
        cmatched = crt_arr["tpc_tr_matched"]
        cscore   = crt_arr["tpc_tr_score"]
        csx, csy, csz = (crt_arr["tpc_tr_start_x"],
                         crt_arr["tpc_tr_start_y"],
                         crt_arr["tpc_tr_start_z"])
        cex, cey, cez = (crt_arr["tpc_tr_end_x"],
                         crt_arr["tpc_tr_end_y"],
                         crt_arr["tpc_tr_end_z"])
        ctags = crt_arr["tpc_tr_taggers"]
        cts0  = crt_arr["tpc_tr_ts0"]
        cts1  = crt_arr["tpc_tr_ts1"]

        # TCS scalar columns as numpy.
        run = ak.to_numpy(tcs_arr["meta.run"]).astype(np.int64)
        sub = ak.to_numpy(tcs_arr["meta.subrun"]).astype(np.int64)
        evt = ak.to_numpy(tcs_arr["meta.evt"]).astype(np.int64)
        tid = ak.to_numpy(tcs_arr["id"]).astype(np.int64)
        t0pfp = ak.to_numpy(tcs_arr["t0PFP"]).astype(np.float32)
        xshift = ak.to_numpy(tcs_arr["xShiftCRT"]).astype(np.float32)
        wt0 = ak.to_numpy(tcs_arr["whicht0"]).astype(np.int32)

        rows = []
        for i in range(len(tcs_arr)):
            ev_key = (int(run[i]), int(sub[i]), int(evt[i]))
            ic = ev_index.get(ev_key)
            if ic is None:
                continue
            k = int(tid[i])
            if k < 0 or k >= len(cmatched[ic]):
                continue
            if not bool(cmatched[ic][k]):
                continue
            # Hits must have been reconstructed using the matched CRT TRACK's
            # t0 (whicht0 == 1), so the hit-position frame agrees with the
            # CRT-track endpoint frame we project to. whicht0 == 3 means
            # hits used a CRT space-point t0 from a different CRT object,
            # which puts them in an inconsistent frame.
            if int(wt0[i]) != WHICHT0_CRTTRACK:
                continue

            sx = float(csx[ic][k]); sy = float(csy[ic][k]); sz = float(csz[ic][k])
            ex = float(cex[ic][k]); ey = float(cey[ic][k]); ez = float(cez[ic][k])
            if (sx < SENTINEL or sy < SENTINEL or sz < SENTINEL
                    or ex < SENTINEL or ey < SENTINEL or ez < SENTINEL):
                continue
            dx0, dy0, dz0 = ex - sx, ey - sy, ez - sz
            dnorm2 = dx0 * dx0 + dy0 * dy0 + dz0 * dz0
            if dnorm2 <= 0:
                continue

            hx = np.asarray(tcs_arr["hits2.h.sp.x"][i], dtype=np.float64)
            hy = np.asarray(tcs_arr["hits2.h.sp.y"][i], dtype=np.float64)
            hz = np.asarray(tcs_arr["hits2.h.sp.z"][i], dtype=np.float64)
            oncalo = np.asarray(tcs_arr["hits2.oncalo"][i]).astype(bool)
            ok = (oncalo & np.isfinite(hx) & np.isfinite(hy) & np.isfinite(hz))
            if not ok.any():
                continue
            hx = hx[ok]; hy = hy[ok]; hz = hz[ok]
            # NOTE: hits2.h.sp.x already incorporates the CRT-derived t0
            # correction in TrackCaloSkim (verified empirically: applying
            # xShiftCRT in either sign INCREASES residuals). xShiftCRT is
            # stored as a column for audit but is not added to hit_x.
            xs_i = float(xshift[i])

            # Foot of perpendicular on the *infinite* line through start..end.
            # The CRT segment is short and lives well outside the TPC volume
            # (e.g. between two top-CRT walls at y > 600 cm); we use the
            # endpoints only to define the muon's trajectory direction and
            # extrapolate that line through the TPC.
            t = ((hx - sx) * dx0 + (hy - sy) * dy0 + (hz - sz) * dz0) / dnorm2
            px = sx + t * dx0
            py = sy + t * dy0
            pz = sz + t * dz0

            taggers = list(ak.to_list(ctags[ic][k]))
            tcl = _classify_taggers(taggers)

            n = hx.size
            rows.append(pd.DataFrame({
                "run":     np.int32(run[i]),
                "subrun":  np.int32(sub[i]),
                "event":   np.int32(evt[i]),
                "trk_id":  np.int32(k),
                "hit_idx": np.arange(n, dtype=np.int32),
                "hit_x":   hx.astype(np.float32),
                "hit_y":   hy.astype(np.float32),
                "hit_z":   hz.astype(np.float32),
                "crt_x":   px.astype(np.float32),
                "crt_y":   py.astype(np.float32),
                "crt_z":   pz.astype(np.float32),
                "dx":      (hx - px).astype(np.float32),
                "dy":      (hy - py).astype(np.float32),
                "dz":      (hz - pz).astype(np.float32),
                "crt_start_x": np.float32(sx),
                "crt_start_y": np.float32(sy),
                "crt_start_z": np.float32(sz),
                "crt_end_x":   np.float32(ex),
                "crt_end_y":   np.float32(ey),
                "crt_end_z":   np.float32(ez),
                "crt_score":   np.float32(cscore[ic][k]),
                "crt_ts0":     np.float32(cts0[ic][k]),
                "crt_ts1":     np.float32(cts1[ic][k]),
                "t0PFP":       np.float32(t0pfp[i]),
                "xshift_crt":  np.float32(xs_i),
                "whicht0":     np.uint8(int(wt0[i])),
                "crt_top_only":  np.bool_(tcl["top_only"]),
                "crt_side_only": np.bool_(tcl["side_only"]),
                "crt_has_top":   np.bool_(tcl["has_top"]),
                "crt_has_S":     np.bool_(tcl["has_S"]),
                "crt_has_N":     np.bool_(tcl["has_N"]),
                "crt_has_W":     np.bool_(tcl["has_W"]),
                "crt_has_E":     np.bool_(tcl["has_E"]),
                "crt_n_taggers": np.uint8(min(tcl["n_taggers"], 255)),
            }))
    except Exception as e:
        print(f"[reduce-skip] {path}: {type(e).__name__}: {e}", flush=True)
        return None

    if not rows:
        return None
    out = pd.concat(rows, axis=0, ignore_index=True)
    out["__file"] = np.int32(idx)
    return out


from crtlib.io import collect_filelist  # re-export for backwards compat


class ShardWriter:
    """Stream-write pandas frames into ParquetWriter shards capped by row count.

    Each call to ``write(df)`` converts to an Arrow Table and appends to
    the current shard. When the shard's row count reaches
    ``rows_per_shard``, the writer is closed and the next call starts a
    new shard.
    """

    def __init__(self, out_dir, rows_per_shard, prefix="crtdisp_df_part",
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
            # Cast new chunk to the writer's schema in case some int/bool
            # columns were inferred slightly differently between workers.
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
                    help="directory to write crtdisp_df_partNNN.parquet shards")
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
    ap.add_argument("--rows-per-shard", type=int, default=100_000_000,
                    help="rollover threshold per parquet shard "
                         "(default 100 M rows ≈ 4 GB at 39.5 B/row)")
    args = ap.parse_args()
    set_use_xroot(not args.no_xroot)

    os.makedirs(args.output_dir, exist_ok=True)

    paths = collect_filelist(args.inputs, args.maxfiles)
    if args.nproc == "auto":
        nproc = min(cpu_count(), len(paths))
    else:
        nproc = int(args.nproc)
    print(f"Processing {len(paths)} files with nproc={nproc}", flush=True)
    print(f"Output dir   : {args.output_dir}", flush=True)
    print(f"Rows / shard : {args.rows_per_shard:,}", flush=True)

    pool_args = [(p, i) for i, p in enumerate(paths)]
    t0 = time.time()

    writer = ShardWriter(args.output_dir, args.rows_per_shard)
    n_top = n_side = n_hits = 0

    def consume(r):
        nonlocal n_top, n_side, n_hits
        if r is None or r.empty:
            return
        n_hits += len(r)
        n_top  += int(r["crt_top_only"].sum())
        n_side += int(r["crt_side_only"].sum())
        writer.write(r)

    if nproc <= 1:
        for a in tqdm(pool_args, unit="file"):
            consume(load_one_file(a))
    else:
        with Pool(processes=nproc) as pool:
            for r in tqdm(pool.imap_unordered(load_one_file, pool_args),
                          total=len(pool_args), unit="file", smoothing=0.6):
                consume(r)
    writer.close()

    elapsed = time.time() - t0
    print(f"\nRead+reduce wall time: {elapsed:.1f}s "
          f"({elapsed/max(1,len(paths)):.2f}s/file)", flush=True)
    print(f"Total hits     : {n_hits:,}", flush=True)
    print(f"Top-only hits  : {n_top:,} ({n_top/max(1,n_hits):.4f})", flush=True)
    print(f"Side-only hits : {n_side:,} ({n_side/max(1,n_hits):.4f})", flush=True)
    print(f"Shards written : {writer.idx}", flush=True)


if __name__ == "__main__":
    main()
