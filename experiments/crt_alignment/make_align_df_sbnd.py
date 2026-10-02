"""Build the per-track DataFrame used to fit per-module CRT alignment.

For every TPC track that
  - is matched to a CRT track (`tpc_tr_matched == True`)
  - was reconstructed using the matched CRT track's t0 (`whicht0 == 1`)
  - is anode-piercing (anode-cathode crosser OR in-time anode crosser,
    by the cuts in cut_config.json)

we emit one row containing:
  - run, subrun, event, trk_id (and meta)
  - the two CRT space-points (start/end) and their tagger + module ID
  - the collection-plane anode-piercing point (Y, Z) in TPC0 and/or TPC1
    (whichever side(s) the track has hits at the anode tick)
  - selection flags (anode-cathode / in-time / both)

Module ID is taken from the FEB MAC5 of the constituent strip-hits in
the matched cluster. SBND has ~134 unique FEB MAC5 values that map 1:1
to physical modules — see make_align_df_sbnd notes / CRTAlignment
README. We pick the dominant FEB MAC per cluster (most frequent;
ties broken by smallest value) as that cluster's module ID.

Inputs:
  cut_tuning/cut_config.json  — produced by plot_track_cuts.py
  filelist_*.txt              — per CLAUDE.md
Outputs:
  align_df_partNNN.parquet    — sharded so no shard exceeds ~4 GB

Usage:
  python make_align_df_sbnd.py OUTDIR (filelist.txt | *.root ...)
      [--nproc N | auto] [--maxfiles N] [--no-xroot]
      [--cut-config cut_tuning/cut_config.json]
      [--rows-per-shard N]
"""

import argparse
import json
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

from crtlib.constants import (ANODE_X_NOMINAL, SENTINEL, WHICHT0_NONE,
                               SP_MATCH_TOL_CM)
from crtlib.io import pnfs_to_xroot, set_use_xroot, collect_filelist
from crtlib.ntuple import (
    TREE_TPC, TREE_CRT,
    TCS_TRK_BRANCHES_FULL as TCS_TRK_BRANCHES,
    TCS_HIT_BRANCHES_FULL as TCS_HIT_BRANCHES,
    CRT_BRANCHES_FULL as CRT_BRANCHES,
    dominant_feb_mac as _dominant_feb_mac,
    match_cluster_index as _match_cluster_index,
)

# whicht0 enum (from sbncode TrackCaloSkimmer_module.cc):
#   -1 : no T0 source (hits at default trigger T0)
#    0 : PFP T0 (Pandora flash match)
#    1 : CRT-Track T0
#    2 : CRT-Hit T0 (legacy)
#    3 : CRT-SpacePoint T0
# We accept any whicht0 != -1 (some t0 was applied). The per-TPC
# tpc_t_min ≈ anode_tick test then selects anode-piercing tracks
# regardless of muon arrival time. Empirically the anode peak in
# `tpc_t_min` sits at the same tick (~418-419) for whicht0 ∈ {0,1,3},
# so the cut is unbiased across these cases.


from crtlib.io import load_cut_config  # re-exported for backwards compat


# Shared cut config (set in main, copied to workers via initializer).
_CUTS = None


def _worker_init(cuts):
    global _CUTS
    _CUTS = cuts


def load_one_file(args):
    """Process one ROOT file, returning a per-track DataFrame.

    Returns an empty DataFrame (or None on error) if the file has no
    qualifying tracks. Columns are listed at the bottom of this module.
    """
    path, idx = args
    cuts = _CUTS
    url = pnfs_to_xroot(path)
    try:
        with uproot.open(url, timeout=120) as f:
            tcs = f[TREE_TPC]
            crt = f[TREE_CRT]
            tcs_arr = tcs.arrays(TCS_TRK_BRANCHES + TCS_HIT_BRANCHES, library="ak")
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
        ev_index = {(int(r), int(s), int(e)): i for i, (r, s, e)
                    in enumerate(zip(crt_run, crt_sub, crt_evt))}

        cmatched = crt_arr["tpc_tr_matched"]
        cscore   = crt_arr["tpc_tr_score"]
        csx, csy, csz = (crt_arr["tpc_tr_start_x"], crt_arr["tpc_tr_start_y"], crt_arr["tpc_tr_start_z"])
        cex, cey, cez = (crt_arr["tpc_tr_end_x"], crt_arr["tpc_tr_end_y"], crt_arr["tpc_tr_end_z"])
        ctags = crt_arr["tpc_tr_taggers"]
        cts0  = crt_arr["tpc_tr_ts0"]
        cts1  = crt_arr["tpc_tr_ts1"]
        cl_has_sp = crt_arr["cl_has_sp"]
        cl_sp_x   = crt_arr["cl_sp_x"]
        cl_sp_y   = crt_arr["cl_sp_y"]
        cl_sp_z   = crt_arr["cl_sp_z"]
        cl_tagger = crt_arr["cl_tagger"]
        cl_macs   = crt_arr["cl_sh_feb_mac5_set"]

        run = ak.to_numpy(tcs_arr["meta.run"]).astype(np.int64)
        sub = ak.to_numpy(tcs_arr["meta.subrun"]).astype(np.int64)
        evt = ak.to_numpy(tcs_arr["meta.evt"]).astype(np.int64)
        tid = ak.to_numpy(tcs_arr["id"]).astype(np.int64)
        t0pfp = ak.to_numpy(tcs_arr["t0PFP"]).astype(np.float32)
        xshift = ak.to_numpy(tcs_arr["xShiftCRT"]).astype(np.float32)
        wt0 = ak.to_numpy(tcs_arr["whicht0"]).astype(np.int32)
        sel = ak.to_numpy(tcs_arr["selected"]).astype(np.int32)
        ccm = ak.to_numpy(tcs_arr["clear_cosmic_muon"]).astype(bool)
        cryo = ak.to_numpy(tcs_arr["cryostat"]).astype(np.int32)
        sx_t = ak.to_numpy(tcs_arr["start.x"]).astype(np.float32)
        sy_t = ak.to_numpy(tcs_arr["start.y"]).astype(np.float32)
        sz_t = ak.to_numpy(tcs_arr["start.z"]).astype(np.float32)
        ex_t = ak.to_numpy(tcs_arr["end.x"]).astype(np.float32)
        ey_t = ak.to_numpy(tcs_arr["end.y"]).astype(np.float32)
        ez_t = ak.to_numpy(tcs_arr["end.z"]).astype(np.float32)
        dx_t = ak.to_numpy(tcs_arr["dir.x"]).astype(np.float32)
        dy_t = ak.to_numpy(tcs_arr["dir.y"]).astype(np.float32)
        dz_t = ak.to_numpy(tcs_arr["dir.z"]).astype(np.float32)
        ln_t = ak.to_numpy(tcs_arr["length"]).astype(np.float32)

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
            if int(wt0[i]) == WHICHT0_NONE:
                continue  # need SOME t0 applied; otherwise hits are uncorrected

            sx = float(csx[ic][k]); sy = float(csy[ic][k]); sz = float(csz[ic][k])
            ex = float(cex[ic][k]); ey = float(cey[ic][k]); ez = float(cez[ic][k])
            if (sx < SENTINEL or sy < SENTINEL or sz < SENTINEL
                    or ex < SENTINEL or ey < SENTINEL or ez < SENTINEL):
                continue

            # Per-TPC collection-plane min/max time and the (y, z) at
            # min-time per TPC. We do NOT use the t_span/t_min already
            # in drift_df_full because we want them recomputed per-file
            # without an extra join.
            ht = np.asarray(tcs_arr["hits2.h.time"][i], dtype=np.float64)
            hy = np.asarray(tcs_arr["hits2.h.sp.y"][i], dtype=np.float64)
            hz = np.asarray(tcs_arr["hits2.h.sp.z"][i], dtype=np.float64)
            htpc = np.asarray(tcs_arr["hits2.h.tpc"][i], dtype=np.int32)
            hon = np.asarray(tcs_arr["hits2.oncalo"][i]).astype(bool)
            ok = hon & np.isfinite(ht) & np.isfinite(hy) & np.isfinite(hz)
            if not ok.any():
                continue
            ht = ht[ok]; hy = hy[ok]; hz = hz[ok]; htpc = htpc[ok]

            mask0 = (htpc == 0)
            mask1 = (htpc == 1)
            n0, n1 = int(mask0.sum()), int(mask1.sum())

            # Per-TPC anode-piercing point = (y, z) at the earliest hit time.
            t_min_tpc0 = t_max_tpc0 = np.nan
            ay0 = az0 = np.nan
            if n0 > 0:
                j0 = int(np.argmin(ht[mask0]))
                t_min_tpc0 = float(ht[mask0][j0])
                t_max_tpc0 = float(ht[mask0].max())
                ay0 = float(hy[mask0][j0])
                az0 = float(hz[mask0][j0])
            t_min_tpc1 = t_max_tpc1 = np.nan
            ay1 = az1 = np.nan
            if n1 > 0:
                j1 = int(np.argmin(ht[mask1]))
                t_min_tpc1 = float(ht[mask1][j1])
                t_max_tpc1 = float(ht[mask1].max())
                ay1 = float(hy[mask1][j1])
                az1 = float(hz[mask1][j1])

            # Per-TPC anode-piercing test on the t0-corrected hit times:
            # the track is kept iff at least one TPC has its earliest
            # collection-plane hit inside the anode peak window.
            tpc0_anode_pierces = bool(
                np.isfinite(t_min_tpc0)
                and cuts["tmin_lo"] <= t_min_tpc0 <= cuts["tmin_hi"]
            )
            tpc1_anode_pierces = bool(
                np.isfinite(t_min_tpc1)
                and cuts["tmin_lo"] <= t_min_tpc1 <= cuts["tmin_hi"]
            )
            if not (tpc0_anode_pierces or tpc1_anode_pierces):
                continue
            # For backwards-compatibility we also classify the track as
            # "anode-cathode crosser" (track has anode-piercing AND
            # cathode-side hits in the SAME TPC, i.e. spans full drift)
            # vs "in-time anode crosser" (hits only on the anode side).
            FULL_DRIFT_TICKS = 2575.0
            DRIFT_FRAC = 0.95  # how close to a full drift counts
            is_ac = (
                (tpc0_anode_pierces and np.isfinite(t_max_tpc0)
                 and (t_max_tpc0 - t_min_tpc0) >= DRIFT_FRAC * FULL_DRIFT_TICKS)
                or
                (tpc1_anode_pierces and np.isfinite(t_max_tpc1)
                 and (t_max_tpc1 - t_min_tpc1) >= DRIFT_FRAC * FULL_DRIFT_TICKS)
            )
            is_it = (tpc0_anode_pierces or tpc1_anode_pierces) and not is_ac

            # Match CRT track endpoints to clusters in the same event so
            # we can read FEB MAC5 (= module ID) for each space-point.
            has_sp = ak.to_numpy(cl_has_sp[ic]).astype(bool)
            if not has_sp.any():
                continue
            cx = ak.to_numpy(cl_sp_x[ic])[has_sp]
            cy = ak.to_numpy(cl_sp_y[ic])[has_sp]
            cz = ak.to_numpy(cl_sp_z[ic])[has_sp]
            ctag = ak.to_numpy(cl_tagger[ic])[has_sp]
            cl_xyz = np.stack([cx, cy, cz], axis=1)
            # cl_macs[ic] is jagged (clusters x strip-hits); index by the
            # cumulative position within has_sp-True clusters.
            cluster_macs_all = list(cl_macs[ic])  # list of variable-len
            cluster_macs_sp = [cluster_macs_all[ii]
                               for ii, ok in enumerate(has_sp) if ok]

            sp1_idx = _match_cluster_index(
                np.array([sx, sy, sz]), cl_xyz, SP_MATCH_TOL_CM)
            sp2_idx = _match_cluster_index(
                np.array([ex, ey, ez]), cl_xyz, SP_MATCH_TOL_CM)
            if sp1_idx < 0 or sp2_idx < 0:
                # Endpoint isn't a cluster in this event — skip
                # (defensive: empirically this almost always matches).
                continue

            mac1 = _dominant_feb_mac(np.asarray(cluster_macs_sp[sp1_idx]))
            mac2 = _dominant_feb_mac(np.asarray(cluster_macs_sp[sp2_idx]))
            tag1 = int(ctag[sp1_idx])
            tag2 = int(ctag[sp2_idx])

            taggers = list(ak.to_list(ctags[ic][k]))

            rows.append({
                "run":     np.int32(run[i]),
                "subrun":  np.int32(sub[i]),
                "event":   np.int32(evt[i]),
                "trk_id":  np.int32(k),
                "cryostat": np.int8(cryo[i]),
                "selected": np.int32(sel[i]),
                "clear_cosmic_muon": np.bool_(ccm[i]),
                "length":  np.float32(ln_t[i]),
                "t0PFP":   np.float32(t0pfp[i]),
                "xshift_crt": np.float32(xshift[i]),
                "whicht0": np.uint8(int(wt0[i])),
                # Reconstructed track endpoints (lab frame, after CRT t0)
                "trk_start_x": np.float32(sx_t[i]),
                "trk_start_y": np.float32(sy_t[i]),
                "trk_start_z": np.float32(sz_t[i]),
                "trk_end_x":   np.float32(ex_t[i]),
                "trk_end_y":   np.float32(ey_t[i]),
                "trk_end_z":   np.float32(ez_t[i]),
                "trk_dir_x":   np.float32(dx_t[i]),
                "trk_dir_y":   np.float32(dy_t[i]),
                "trk_dir_z":   np.float32(dz_t[i]),
                # Per-TPC anode-piercing point (NaN if the track has no
                # plane-2 hits in that TPC).
                "tpc0_n_p2_hits": np.int32(n0),
                "tpc1_n_p2_hits": np.int32(n1),
                "tpc0_t_min": np.float32(t_min_tpc0),
                "tpc0_t_max": np.float32(t_max_tpc0),
                "tpc0_anode_y": np.float32(ay0),
                "tpc0_anode_z": np.float32(az0),
                "tpc1_t_min": np.float32(t_min_tpc1),
                "tpc1_t_max": np.float32(t_max_tpc1),
                "tpc1_anode_y": np.float32(ay1),
                "tpc1_anode_z": np.float32(az1),
                "is_anode_cathode_crosser": np.bool_(is_ac),
                "is_intime_anode_crosser":  np.bool_(is_it),
                # CRT space-points (start/end of the matched CRT track)
                "crt_sp1_x": np.float32(sx),
                "crt_sp1_y": np.float32(sy),
                "crt_sp1_z": np.float32(sz),
                "crt_sp1_tagger":    np.int8(tag1),
                "crt_sp1_module_id": np.int32(mac1),
                "crt_sp2_x": np.float32(ex),
                "crt_sp2_y": np.float32(ey),
                "crt_sp2_z": np.float32(ez),
                "crt_sp2_tagger":    np.int8(tag2),
                "crt_sp2_module_id": np.int32(mac2),
                "crt_score": np.float32(cscore[ic][k]),
                "crt_ts0":   np.float32(cts0[ic][k]),
                "crt_ts1":   np.float32(cts1[ic][k]),
                "crt_n_taggers": np.uint8(min(len(taggers), 255)),
                "crt_taggers_csv": ",".join(str(int(t)) for t in taggers),
            })
    except Exception as e:
        print(f"[reduce-skip] {path}: {type(e).__name__}: {e}", flush=True)
        return None

    if not rows:
        return None
    out = pd.DataFrame(rows)
    out["__file"] = np.int32(idx)
    return out


class ShardWriter:
    """Stream-write pandas frames into ParquetWriter shards capped by row count."""

    def __init__(self, out_dir, rows_per_shard, prefix="align_df_part",
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
              f"{sz/1e6:.2f} MB)", flush=True)
        self.idx += 1
        self.writer = None
        self.path = None
        self.rows_in_shard = 0

    def write(self, df):
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self._open(table.schema)
        else:
            try:
                table = table.cast(self.writer.schema_arrow, safe=False)
            except Exception:
                pass
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
                    help="directory to write align_df_partNNN.parquet shards")
    ap.add_argument("inputs", nargs="*",
                    default=["filelist_nonempty_combined.txt"],
                    help="ROOT files, glob patterns, or .txt/.list of paths. "
                         "Defaults to filelist_nonempty_combined.txt — the "
                         "FULL set of available calibration ntuples "
                         "(offbeamlight + bnblight, ~137k files).")
    ap.add_argument("--nproc", default="1")
    ap.add_argument("--maxfiles", type=int, default=None)
    ap.add_argument("--no-xroot", action="store_true")
    ap.add_argument("--cut-config", default="cut_tuning/cut_config_per_tpc.json",
                    help="JSON with the per-TPC tpc_t_min cut window. "
                         "Hits are CRT-t0-corrected for whicht0=1, so this "
                         "directly tests anode-piercing per TPC.")
    ap.add_argument("--rows-per-shard", type=int, default=10_000_000,
                    help="rollover threshold per parquet shard "
                         "(default 10M rows; align_df rows are ~120 B so "
                         "shards are roughly 1 GB)")
    args = ap.parse_args()
    set_use_xroot(not args.no_xroot)

    cuts = load_cut_config(args.cut_config)
    print(f"Per-TPC anode-piercing cut: t_min in "
          f"[{cuts['tmin_lo']}, {cuts['tmin_hi']}] ticks", flush=True)

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
    n_ac = n_it = n_both = 0

    def consume(r):
        nonlocal n_ac, n_it, n_both
        if r is None or r.empty:
            return
        n_ac   += int(r["is_anode_cathode_crosser"].sum())
        n_it   += int(r["is_intime_anode_crosser"].sum())
        n_both += int((r["is_anode_cathode_crosser"] & r["is_intime_anode_crosser"]).sum())
        writer.write(r)

    if nproc <= 1:
        _worker_init(cuts)
        for a in tqdm(pool_args, unit="file"):
            consume(load_one_file(a))
    else:
        with Pool(processes=nproc, initializer=_worker_init, initargs=(cuts,)) as pool:
            for r in tqdm(pool.imap_unordered(load_one_file, pool_args),
                          total=len(pool_args), unit="file", smoothing=0.6):
                consume(r)
    writer.close()

    elapsed = time.time() - t0
    print(f"\nRead+reduce wall time: {elapsed:.1f}s "
          f"({elapsed/max(1,len(paths)):.2f}s/file)", flush=True)
    print(f"Total rows         : {writer.total_rows:,}", flush=True)
    print(f"  anode-cathode    : {n_ac:,}", flush=True)
    print(f"  in-time anode    : {n_it:,}", flush=True)
    print(f"  both             : {n_both:,}", flush=True)
    print(f"Shards written     : {writer.idx}", flush=True)


if __name__ == "__main__":
    main()
