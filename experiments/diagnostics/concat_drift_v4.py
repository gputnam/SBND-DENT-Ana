#!/usr/bin/env python3
"""Concatenate drift_df_full_v3 (existing union) + drift_df_v11_new
(2026Summer DENT-Study augmentation) -> drift_df_full_v4.parquet."""
import os
import pandas as pd

D = "/exp/sbnd/data/users/gputnam/efield-study/drift_df"
V3 = os.path.join(D, "drift_df_full_v3.parquet")
NEW = os.path.join(D, "drift_df_v11_new.parquet")
OUT = os.path.join(D, "drift_df_full_v4.parquet")

parts = [V3, NEW]
for p in parts:
    assert os.path.exists(p), f"missing {p}"
print("reading:", parts, flush=True)
dfs = [pd.read_parquet(p) for p in parts]
for p, d in zip(parts, dfs):
    print(f"  {os.path.basename(p)}: {len(d):,} tracks", flush=True)
df = pd.concat(dfs, ignore_index=True)
df.to_parquet(OUT, index=False)
print(f"wrote {OUT}: {len(df):,} tracks", flush=True)
