"""Cutflow report for the CRT-alignment input sample.

Walks the same filters as align_crt.py / bootstrap_wall_deformation.py
applied in sequence, reporting how many tracks (and residual rows)
survive each step.

  align_df.parquet
    -> per-track: both crt_sp module IDs resolved
    -> per-track: crt_len >= min_crt_len_cm (default 400)
    -> explode per-TPC: n_p2_hits>0, anode_y/z finite, t_min in [tmin_lo, tmin_hi]
    -> per-row: pre-fit residual <= max_pre_residual_cm (default 50)
    -> per-row: both modules above min_counts (default 30 in bootstrap)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/exp/sbnd/data/users/gputnam/efield-study/align_df")
    ap.add_argument("--cut-config", default="cut_tuning/cut_config_per_tpc.json")
    ap.add_argument("--min-crt-len-cm", type=float, default=400.0)
    ap.add_argument("--max-pre-residual-cm", type=float, default=50.0)
    ap.add_argument("--min-counts", type=int, default=30)
    args = ap.parse_args()

    if os.path.isdir(args.parquet):
        import glob
        shards = sorted(glob.glob(os.path.join(args.parquet, "*.parquet")))
        df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    else:
        df = pd.read_parquet(args.parquet)

    with open(args.cut_config) as f:
        cuts = json.load(f)
    tmin_lo = float(cuts["tpc_t_min_lo"])
    tmin_hi = float(cuts["tpc_t_min_hi"])

    rows = []
    n_tracks_initial = len(df)
    rows.append(("0. align_df.parquet (initial sample)",
                 n_tracks_initial, "tracks", None))

    # --- track-level cuts (must happen before explode) ---
    mask_modids = (df["crt_sp1_module_id"] >= 0) & (df["crt_sp2_module_id"] >= 0)
    n_modids = int(mask_modids.sum())
    rows.append(("1. CRT space-points have resolved module IDs",
                 n_modids, "tracks", n_tracks_initial))

    sp1 = df[["crt_sp1_x", "crt_sp1_y", "crt_sp1_z"]].to_numpy()
    sp2 = df[["crt_sp2_x", "crt_sp2_y", "crt_sp2_z"]].to_numpy()
    crt_len = np.linalg.norm(sp2 - sp1, axis=1)

    mask_len = mask_modids & (crt_len >= args.min_crt_len_cm)
    n_len = int(mask_len.sum())
    rows.append((f"2. CRT lever arm >= {args.min_crt_len_cm:.0f} cm",
                 n_len, "tracks", n_modids))

    # --- per-TPC explode ---
    df_keep = df.loc[mask_len].copy()
    df_keep["crt_len"] = crt_len[mask_len]
    expanded = []
    for tpc in (0, 1):
        sub = df_keep[[
            f"tpc{tpc}_n_p2_hits", f"tpc{tpc}_t_min",
            f"tpc{tpc}_anode_y", f"tpc{tpc}_anode_z",
            "crt_sp1_x", "crt_sp1_y", "crt_sp1_z",
            "crt_sp2_x", "crt_sp2_y", "crt_sp2_z",
            "crt_sp1_module_id", "crt_sp2_module_id",
        ]].copy()
        sub.columns = ["n_hits", "t_min", "anode_y", "anode_z",
                       "sp1x", "sp1y", "sp1z", "sp2x", "sp2y", "sp2z",
                       "mod1", "mod2"]
        sub["tpc"] = tpc
        sub["anode_x_nom"] = -200.0 if tpc == 0 else 200.0
        expanded.append(sub)
    exp = pd.concat(expanded, ignore_index=True)

    n_explode_total = len(exp)  # 2 × n_len
    rows.append(("3. Exploded per-TPC (each track → up to 2 rows)",
                 n_explode_total, "rows", n_len))

    # 3a. n_p2_hits > 0 & finite anode coords
    m_hits = (exp["n_hits"] > 0) & np.isfinite(exp["anode_y"]) & np.isfinite(exp["anode_z"])
    n_hits = int(m_hits.sum())
    rows.append(("4. TPC has collection-plane hits (n_p2_hits > 0, anode pos finite)",
                 n_hits, "rows", n_explode_total))

    # 3b. t_min in [tmin_lo, tmin_hi]
    m_t = m_hits & np.isfinite(exp["t_min"]) & (exp["t_min"] >= tmin_lo) & (exp["t_min"] <= tmin_hi)
    n_t = int(m_t.sum())
    rows.append((f"5. t_min in anode window [{tmin_lo:.1f}, {tmin_hi:.1f}] ticks",
                 n_t, "rows", n_hits))

    # --- pre-fit residual cut ---
    keep = exp.loc[m_t].copy()
    sp1k = keep[["sp1x", "sp1y", "sp1z"]].to_numpy()
    sp2k = keep[["sp2x", "sp2y", "sp2z"]].to_numpy()
    pa = np.stack([keep["anode_x_nom"].to_numpy(),
                   keep["anode_y"].to_numpy(),
                   keep["anode_z"].to_numpy()], axis=1)
    line = sp2k - sp1k
    L = np.linalg.norm(line, axis=1, keepdims=True)
    u = line / np.maximum(L, 1e-9)
    delta = pa - sp1k
    proj = np.sum(delta * u, axis=1, keepdims=True) * u
    pre_resid = np.linalg.norm(delta - proj, axis=1)
    keep["pre_resid"] = pre_resid

    m_resid = pre_resid <= args.max_pre_residual_cm
    n_resid = int(m_resid.sum())
    rows.append((f"6. pre-fit residual <= {args.max_pre_residual_cm:.0f} cm",
                 n_resid, "rows", n_t))

    keep_r = keep.loc[m_resid].copy()

    # --- per-module count cut (modules with < min_counts dropped, drop their rows) ---
    macs = pd.concat([keep_r["mod1"], keep_r["mod2"]]).astype(int)
    counts = macs.value_counts()
    keep_mods = set(counts[counts >= args.min_counts].index)
    m_minc = (keep_r["mod1"].astype(int).isin(keep_mods) &
              keep_r["mod2"].astype(int).isin(keep_mods))
    n_minc = int(m_minc.sum())
    rows.append((f"7. both modules above min_counts={args.min_counts} (after refresh)",
                 n_minc, "rows", n_resid))

    # ---- print ----
    print(f"\nCRT-alignment cutflow")
    print(f"  parquet           : {args.parquet}")
    print(f"  min_crt_len_cm    : {args.min_crt_len_cm:.0f}")
    print(f"  tmin window       : [{tmin_lo:.1f}, {tmin_hi:.1f}]")
    print(f"  max_pre_resid_cm  : {args.max_pre_residual_cm:.0f}")
    print(f"  min_counts/module : {args.min_counts}")
    print()

    head = f"{'step':<70s} {'survivors':>12s}  {'unit':<6s}  {'pass%':>6s}  {'cum%':>6s}"
    print(head)
    print("-" * len(head))
    n_init = rows[0][1]
    for label, n, unit, prev in rows:
        if prev is None:
            pct_step = ""
            pct_cum  = "100.0%"
        else:
            pct_step = f"{100.0 * n / prev:5.1f}%"
            pct_cum  = f"{100.0 * n / n_init:5.1f}%"
        print(f"{label:<70s} {n:>12,d}  {unit:<6s}  {pct_step:>6s}  {pct_cum:>6s}")


if __name__ == "__main__":
    main()
