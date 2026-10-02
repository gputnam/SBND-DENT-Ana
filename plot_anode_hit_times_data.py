#!/usr/bin/env python3
"""Per-track anode-candidate hit-time distributions, per (TPC, plane), and
the per-plane anode-peak offset determination.

For each (track, TPC, plane):
  * FIRST-HIT time:  min(time)
  * CLOSEST-TO-PLANE-PEAK time: the hit minimizing |time - (T_c + off_p)|,
    where T_c is the collection-plane anode tick (the track's earliest
    collection hit) and off_p is the per-(tpc, plane) anode-peak offset.

The offset off_p = median over tracks of (first_hit_p - T_c) is the fixed
per-plane anode-timing shift (the induction responses peak a few ticks off
collection). It is measured here per shard, averaged, and written to
<out>/anode_peak_offset.json for plot_doubleanode_2d_voxels.py
--anode-peak-offset-json, so each plane selects its anode hit at its OWN
peak instead of the collection peak.

Usage:
    python plot_anode_hit_times_data.py \
        --input <doubleanode_2d_df dir> --out anode_hit_times_data
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crtlib.constants import DATA_DIR

INPUT = os.path.join(DATA_DIR, "doubleanode_2d_df_v9")
WIN = (330.0, 600.0)
BIN = 1.0
ANODE_DT_WINDOW = 40.0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=INPUT)
    ap.add_argument("--out", default="anode_hit_times_data")
    ap.add_argument("--label", default="DATA double-anode (v9)")
    ap.add_argument("--max-shards", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    if args.max_shards:
        shards = shards[:args.max_shards]
    nb = int(round((WIN[1] - WIN[0]) / BIN))
    edges = np.linspace(*WIN, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    h_first = np.zeros((2, 3, nb), dtype=np.int64)
    h_peak = np.zeros((2, 3, nb), dtype=np.int64)
    off_sum = np.zeros((2, 3))
    off_n = np.zeros((2, 3))

    for path in shards:
        df = pd.read_parquet(path, columns=["run", "subrun", "event",
                                            "trk_id", "plane", "tpc", "time"])
        df["tkey"] = df.groupby(["run", "subrun", "event",
                                 "trk_id"]).ngroup()
        ntrk = int(df.tkey.max()) + 1
        # T_c = earliest collection (plane 2) hit per (tkey, tpc)
        col = df[df.plane == 2]
        tc = col.groupby(["tkey", "tpc"])["time"].min().reset_index()
        Tc = np.full((ntrk, 2), np.nan)
        Tc[tc.tkey.to_numpy(), tc.tpc.to_numpy()] = tc.time.to_numpy()
        tkey = df.tkey.to_numpy(); tp = df.tpc.to_numpy()
        pl = df.plane.to_numpy(); tm = df.time.to_numpy()
        tc_row = Tc[tkey, tp]

        # first hit per (tkey, tpc, plane)
        fi = df.groupby(["tkey", "tpc", "plane"])["time"].idxmin().to_numpy()
        first = df.loc[fi]
        # per-plane offset = median(first - T_c), accumulate (sum of medians-
        # by-shard is fine; offsets are stable). Use this shard's median for
        # selection within the shard.
        foff = (first.time.to_numpy() - Tc[first.tkey.to_numpy(),
                                           first.tpc.to_numpy()])
        off_shard = np.zeros((2, 3))
        for t in (0, 1):
            for p in (0, 1, 2):
                m = (first.tpc.to_numpy() == t) & (first.plane.to_numpy() == p)
                if m.any():
                    off_shard[t, p] = np.nanmedian(foff[m])
                    off_sum[t, p] += off_shard[t, p] * m.sum()
                    off_n[t, p] += m.sum()
        # closest to T_c + off_shard per (tkey, tpc, plane)
        ref = tc_row + off_shard[tp, pl]
        df["adt"] = np.abs(tm - ref)
        sel = df[np.isfinite(df.adt.to_numpy())]
        ni = sel.groupby(["tkey", "tpc", "plane"])["adt"].idxmin().to_numpy()
        near = df.loc[ni]
        near = near[np.abs(near.time.to_numpy()
                           - ref[ni]) < ANODE_DT_WINDOW]

        for seldf, acc in ((first, h_first), (near, h_peak)):
            t_ = seldf.tpc.to_numpy().astype(int)
            p_ = seldf.plane.to_numpy().astype(int)
            ib = np.floor((seldf.time.to_numpy() - WIN[0]) / BIN).astype(int)
            ok = (ib >= 0) & (ib < nb)
            np.add.at(acc, (t_[ok], p_[ok], ib[ok]), 1)
        print(f"{os.path.basename(path)}: ~{ntrk:,} tracks", flush=True)
        del df

    offsets = np.where(off_n > 0, off_sum / np.maximum(off_n, 1), 0.0)
    with open(os.path.join(args.out, "anode_peak_offset.json"), "w") as jf:
        json.dump({"offsets": offsets.tolist(),
                   "comment": "per-(tpc,plane) median(first_hit - T_c) "
                              "[ticks]; reference for the closest-to-peak "
                              "anode selection"}, jf, indent=1)
    print("\nper-plane anode-peak offsets [ticks] (median first - T_c):")
    for t in (0, 1):
        print(f"  TPC{t}: " + "  ".join(f"plane{p} {offsets[t,p]:+.2f}"
                                        for p in (0, 1, 2)), flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    for t in (0, 1):
        for p in (0, 1, 2):
            ax = axes[t, p]
            ax.step(centers, h_first[t, p], where="mid", color="0.5", lw=1.0,
                    label=f"first hit (N={h_first[t,p].sum():,})")
            ax.step(centers, h_peak[t, p], where="mid", color="navy", lw=1.2,
                    label=f"closest to plane peak "
                          f"(N={h_peak[t,p].sum():,})")
            pk = centers[np.argmax(h_first[t, p])]
            ax.axvline(pk, color="red", ls="--", lw=0.9,
                       label=f"first-hit peak {pk:.1f}")
            ax.set_yscale("log"); ax.set_ylim(0.5, None)
            ax.set_title(f"TPC{t} plane{p}"
                         + ("  (coll.)" if p == 2 else "  (ind.)")
                         + f": off {offsets[t,p]:+.1f} tk", fontsize=9)
            ax.grid(alpha=0.3)
            if t == 1:
                ax.set_xlabel("hit time [ticks]")
            if p == 0:
                ax.set_ylabel("tracks")
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{args.label}: anode-candidate hit times "
                 "(first hit vs closest to per-plane peak)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "anode_hit_times.png"), dpi=120)
    fig.savefig(os.path.join(args.out, "anode_hit_times.pdf"))

    # linear, t0 +/- 25
    T0 = 418.5
    m = (centers >= T0 - 25) & (centers <= T0 + 25)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    for t in (0, 1):
        for p in (0, 1, 2):
            ax = axes[t, p]
            ax.step(centers[m], h_first[t, p][m], where="mid", color="0.5",
                    lw=1.2, label="first hit")
            ax.step(centers[m], h_peak[t, p][m], where="mid", color="navy",
                    lw=1.4, label="closest to plane peak")
            pk = centers[np.argmax(h_first[t, p])]
            ax.axvline(pk, color="red", ls="--", lw=1.0,
                       label=f"peak {pk:.1f}")
            ax.set_ylim(0, None); ax.grid(alpha=0.3)
            ax.set_title(f"TPC{t} plane{p}"
                         + ("  (coll.)" if p == 2 else "  (ind.)"), fontsize=10)
            if t == 1:
                ax.set_xlabel("hit time [ticks]")
            if p == 0:
                ax.set_ylabel("tracks / tick")
            ax.legend(fontsize=8)
    ax.set_xlim(T0 - 25, T0 + 25)
    fig.suptitle(f"{args.label}: anode-candidate hit times, linear, "
                 "t0 +/- 25 ticks")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "anode_hit_times_linear.png"), dpi=120)
    np.savez_compressed(os.path.join(args.out, "anode_hit_times.npz"),
                        edges=edges, h_first=h_first, h_peak=h_peak,
                        offsets=offsets)
    print(f"\nwrote {args.out}/ (+ anode_peak_offset.json)", flush=True)


if __name__ == "__main__":
    main()
