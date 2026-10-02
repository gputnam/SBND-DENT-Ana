#!/usr/bin/env python3
"""A/B/C parquets from the MC truth-matched hits, for pipeline forensics.

Builds three doubleanode_2d_df-format parquet sets from the SAME tracks and
wires, differing ONLY in the hit time fed to the 2D pipeline:

  pq_true_time/   time = 418.5 + 2 * tdrift            (TRUE arrival time;
                  contains the full simulated drift-time distortion but NO
                  hit-reconstruction effects)
  pq_reco_time/   time = matched reco hit PeakTime      (the production
                  content, restricted to truth-matched wires)
  pq_nosce_time/  time = 418.5 + 2 * |x_true - anode| / v_nom
                  (synthetic null: undistorted drift from the TRUE position;
                  the pipeline must measure dx ~ 0 ... up to the spatial
                  dy/dz displacement which is still in the wires? NO -- the
                  wire/s content is the true displaced charge either way, so
                  dy/dz remain measured; only the x information is nulled)

Running plot_doubleanode_2d_voxels.py on each and comparing the dx maps
splits the measured dx dome into: geometry/chord artifact (present in C),
+ true drift-time distortion (B-A ... present in A), + hit-reconstruction
time bias (present only in B).

Usage:
    python build_truthhit_parquets_mc.py --nfiles 400 --nproc 16 \
        --out /exp/sbnd/app/users/gputnam/cathode-study/truthhit_pq
"""

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

T0 = 418.5
V_NOM = 0.15627
TICK_US = 0.5
# production double-anode track selection (cut_tuning/cut_config_per_tpc.json
# + make_doubleanode_2d_df_sbnd defaults)
TMIN_LO, TMIN_HI = 414.5, 427.5
DRIFT_THRESH = 0.95 * 2575.0
WHICHT0_NONE = -1

_OUT = None


def _init(out):
    global _OUT
    _OUT = out


def process_file(args_):
    ifile, url = args_
    import uproot
    import awkward as ak
    rows = {"true": [], "reco": [], "nosce": []}
    try:
        with uproot.open(url) as f:
            t = f["caloskim/TrackCaloSkim"]
            br = ["whicht0"]
            for p in (0, 1, 2):
                br += [f"hits{p}.h.wire", f"hits{p}.h.tpc",
                       f"hits{p}.h.time", f"hits{p}.oncalo",
                       f"truth.p.truehits{p}.wire",
                       f"truth.p.truehits{p}.tpc",
                       f"truth.p.truehits{p}.time",
                       f"truth.p.truehits{p}.tdrift",
                       f"truth.p.truehits{p}.p.x"]
            arr = t.arrays(br, library="ak")
            wt0 = ak.to_numpy(arr["whicht0"]).astype(np.int64)
            ntrk = len(wt0)

            # per-TPC anode transit lag (constant): tdrift includes a ~12 us
            # offset; anchor the 'true' time variant so its anode tick is
            # 418.5 like the production reco hits.
            lag = np.zeros(2)
            lag_n = np.zeros(2)
            for p in (0, 1, 2):
                ttd = ak.to_numpy(ak.flatten(
                    arr[f"truth.p.truehits{p}.tdrift"], axis=None))
                ttp = ak.to_numpy(ak.flatten(
                    arr[f"truth.p.truehits{p}.tpc"], axis=None))
                txx = ak.to_numpy(ak.flatten(
                    arr[f"truth.p.truehits{p}.p.x"], axis=None))
                anx = np.where(ttp == 0, -200.0, 200.0)
                dd = np.abs(txx - anx)
                near = dd < 30
                for tt_ in (0, 1):
                    m = near & (ttp == tt_)
                    if m.sum() > 50:
                        lag[tt_] += np.median(ttd[m] - dd[m] / V_NOM)
                        lag_n[tt_] += 1
            lag = np.where(lag_n > 0, lag / np.maximum(lag_n, 1), 0.0)

            for i in range(ntrk):
                if wt0[i] == WHICHT0_NONE:
                    continue
                # production double-anode selection on the RECO collection
                # hits (same track sample for all three variants)
                ht2 = np.asarray(arr["hits2.h.time"][i], np.float64)
                tp2 = np.asarray(arr["hits2.h.tpc"][i], np.int64)
                on2 = np.asarray(arr["hits2.oncalo"][i]).astype(bool)
                ok2 = on2 & np.isfinite(ht2)
                if not ok2.any():
                    continue
                t2, tt2 = ht2[ok2], tp2[ok2]
                passed = True
                for tt_ in (0, 1):
                    m = tt2 == tt_
                    if not m.any():
                        passed = False
                        break
                    tmin, tmax = t2[m].min(), t2[m].max()
                    if not (TMIN_LO <= tmin <= TMIN_HI
                            and (tmax - tmin) >= DRIFT_THRESH):
                        passed = False
                        break
                if not passed:
                    continue
                for p in (0, 1, 2):
                    tw = np.asarray(arr[f"truth.p.truehits{p}.wire"][i],
                                    np.int64)
                    if not len(tw):
                        continue
                    tt = np.asarray(arr[f"truth.p.truehits{p}.tpc"][i],
                                    np.int64)
                    ttd = np.asarray(arr[f"truth.p.truehits{p}.tdrift"][i],
                                     np.float64)
                    tx = np.asarray(arr[f"truth.p.truehits{p}.p.x"][i],
                                    np.float64)
                    rw = np.asarray(arr[f"hits{p}.h.wire"][i], np.int64)
                    rt = np.asarray(arr[f"hits{p}.h.tpc"][i], np.int64)
                    rtime = np.asarray(arr[f"hits{p}.h.time"][i], np.float64)
                    onc = np.asarray(arr[f"hits{p}.oncalo"][i]).astype(bool)

                    # match reco hits to truehits by (tpc, wire)
                    tkey = tt * 10000 + tw
                    rkey = rt * 10000 + rw
                    order = np.argsort(tkey)
                    pos = np.clip(np.searchsorted(tkey[order], rkey), 0,
                                  len(order) - 1)
                    hit = order[pos]
                    ok = (tkey[hit] == rkey) & onc
                    if not ok.any():
                        continue
                    h = hit[ok]
                    base = dict(run=1, subrun=ifile, event=i, trk_id=0,
                                plane=np.int32(p))
                    anodex = np.where(tt[h] == 0, -200.0, 200.0)
                    t_true = T0 + 2.0 * (ttd[h] - lag[tt[h]])
                    t_nosce = T0 + np.abs(tx[h] - anodex) / (TICK_US * V_NOM)
                    for tag, tval in (("true", t_true),
                                      ("reco", rtime[ok]),
                                      ("nosce", t_nosce)):
                        rows[tag].append(pd.DataFrame({
                            **base,
                            "tpc": tt[h].astype(np.int32),
                            "wire": tw[h].astype(np.int32),
                            "time": tval.astype(np.float32)}))
    except Exception as e:
        print(f"skip {url.rsplit('/', 1)[-1][:40]}: {type(e).__name__}",
              file=sys.stderr)
        return 0
    n = 0
    for tag in rows:
        if not rows[tag]:
            continue
        df = pd.concat(rows[tag], ignore_index=True)
        n = len(df)
        d = os.path.join(_OUT, f"pq_{tag}_time")
        os.makedirs(d, exist_ok=True)
        df.to_parquet(os.path.join(d, f"part{ifile:05d}.parquet"),
                      index=False)
    return n


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filelist", default="filelist_mc_doubleanode_xroot.txt")
    ap.add_argument("--nfiles", type=int, default=400)
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--out", default="truthhit_pq")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    urls = [l.strip() for l in open(args.filelist) if l.strip()][:args.nfiles]
    print(f"building A/B/C parquets from {len(urls)} ntuples -> {args.out}/",
          flush=True)
    tot = 0
    with Pool(args.nproc, initializer=_init, initargs=(args.out,)) as pool:
        for n in tqdm(pool.imap_unordered(
                process_file, list(enumerate(urls))), total=len(urls),
                unit="file", file=sys.stderr):
            tot += n
    print(f"~{tot:,} hits per variant written", flush=True)


if __name__ == "__main__":
    main()
