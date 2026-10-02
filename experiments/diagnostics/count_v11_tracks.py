#!/usr/bin/env python3
"""Quote track/hit statistics for the v11 augmentation.

Reports, for the NEW-only augmentation sample and (optionally) the augmented
union:
  - double-anode-crossing tracks (unique run,subrun,event,trk_id) + hits + per-event
  - cathode-crossing tracks (rows in drift_df) + per-event
Baselines: v9 = 547,024,448 hits / 249,733 double-anode tracks; drift v3 = 52.6 M.
"""
import glob
import os
import sys
import pandas as pd

DATA = "/exp/sbnd/data/users/gputnam/efield-study"


def da_stats(parquet_dir, label):
    shards = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    n_hits = 0
    keys = set()
    ev = set()
    for s in shards:
        df = pd.read_parquet(s, columns=["run", "subrun", "event", "trk_id"])
        n_hits += len(df)
        keys.update(map(tuple, df[["run", "subrun", "event", "trk_id"]]
                        .drop_duplicates().to_numpy()))
        ev.update(map(tuple, df[["run", "subrun", "event"]]
                      .drop_duplicates().to_numpy()))
    n_trk = len(keys)
    n_ev = len(ev)
    print(f"\n=== DOUBLE-ANODE  [{label}]  ({len(shards)} shards) ===")
    print(f"  hits                 : {n_hits:,}")
    print(f"  double-anode tracks  : {n_trk:,}")
    print(f"  events (with tracks) : {n_ev:,}")
    if n_trk:
        print(f"  hits / track         : {n_hits/n_trk:,.1f}")
    if n_ev:
        print(f"  tracks / event       : {n_trk/n_ev:.3f}")
        print(f"  hits / event         : {n_hits/n_ev:,.1f}")
    return n_hits, n_trk, n_ev


def cathode_stats(parquet_path, label):
    if not os.path.exists(parquet_path):
        print(f"\n[cathode] {parquet_path} missing, skip")
        return
    cols = ["meta.run", "meta.subrun", "meta.evt"]
    df = pd.read_parquet(parquet_path, columns=cols)
    n_trk = len(df)
    n_ev = df.drop_duplicates().shape[0]
    print(f"\n=== DRIFT-DF (AC-candidate tracks)  [{label}] ===")
    print(f"  drift-df track rows : {n_trk:,}")
    print(f"  events              : {n_ev:,}")
    if n_ev:
        print(f"  tracks / event      : {n_trk/n_ev:.3f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "new"
    if which in ("new", "all"):
        da_stats(f"{DATA}/doubleanode_2d_df_v11_new", "NEW augmentation")
        cathode_stats(f"{DATA}/drift_df/drift_df_v11_new.parquet",
                      "NEW augmentation")
    if which in ("aug", "all"):
        da_stats(f"{DATA}/doubleanode_2d_df_v11", "AUGMENTED union (v9+new)")
        cathode_stats(f"{DATA}/drift_df/drift_df_full_v4.parquet",
                      "AUGMENTED union (v3+new)")
