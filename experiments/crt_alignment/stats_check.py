"""Pre-fit statistics check on align_df parquet.

Reports:
  - total residual-row count (after exploding to one row per anode-piercing
    on each track)
  - count per CRT module FEB MAC (a.k.a. module ID in this analysis)
  - count per (CRT module, anode TPC) pair
  - distribution of |CRT_sp2 - CRT_sp1| (track length in CRT space)
  - the median pre-alignment perpendicular residual

The goal is to surface any module that is sparsely populated (< 50 tracks)
before launching the full alignment fit. Per the plan we will refine
selection / increase data / merge sparse modules if so.

Usage:
  python stats_check.py /exp/sbnd/data/users/gputnam/efield-study/align_df
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from crtlib.constants import ANODE_X_NOMINAL
from crtlib.io import load_align_df


def explode_residual_rows(df, tmin_lo, tmin_hi):
    """Produce one row per anode-piercing point — only emit a TPC's anode
    point when that TPC's earliest collection-plane hit time is inside
    the in-time-anode window [tmin_lo, tmin_hi]. (Same logic as align_crt.py.)
    """
    rows = []
    for _, r in df.iterrows():
        sp1 = np.array([r["crt_sp1_x"], r["crt_sp1_y"], r["crt_sp1_z"]])
        sp2 = np.array([r["crt_sp2_x"], r["crt_sp2_y"], r["crt_sp2_z"]])
        m1, m2 = int(r["crt_sp1_module_id"]), int(r["crt_sp2_module_id"])
        if m1 < 0 or m2 < 0:
            continue
        for tpc in (0, 1):
            n_hits = r[f"tpc{tpc}_n_p2_hits"]
            ay, az = r[f"tpc{tpc}_anode_y"], r[f"tpc{tpc}_anode_z"]
            tmin = r[f"tpc{tpc}_t_min"]
            if n_hits <= 0 or not np.isfinite(ay) or not np.isfinite(az):
                continue
            if not (np.isfinite(tmin) and tmin_lo <= tmin <= tmin_hi):
                continue
            rows.append({
                "sp1_x": sp1[0], "sp1_y": sp1[1], "sp1_z": sp1[2],
                "sp2_x": sp2[0], "sp2_y": sp2[1], "sp2_z": sp2[2],
                "mod1": m1, "mod2": m2,
                "anode_x": ANODE_X_NOMINAL[tpc],
                "anode_y": ay, "anode_z": az, "anode_tpc": tpc,
            })
    return pd.DataFrame(rows)


def perp_residual(rows):
    """Pre-alignment perpendicular distance from (anode_xyz) to the line
    through (sp1_xyz, sp2_xyz)."""
    sp1 = rows[["sp1_x", "sp1_y", "sp1_z"]].to_numpy()
    sp2 = rows[["sp2_x", "sp2_y", "sp2_z"]].to_numpy()
    pa  = rows[["anode_x", "anode_y", "anode_z"]].to_numpy()
    line = sp2 - sp1
    n = np.linalg.norm(line, axis=1, keepdims=True)
    u = line / np.maximum(n, 1e-9)
    delta = pa - sp1
    proj = (delta * u).sum(axis=1, keepdims=True) * u
    return np.linalg.norm(delta - proj, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--outdir", default="stats_check")
    ap.add_argument("--cut-config", default="cut_tuning/cut_config_per_tpc.json")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.cut_config) as f:
        cuts = json.load(f)
    if "tpc_t_min_lo" in cuts:
        tmin_lo = float(cuts["tpc_t_min_lo"])
        tmin_hi = float(cuts["tpc_t_min_hi"])
    else:
        tmin_lo = float(cuts["intime_anode_crosser"]["t_min_p2_lo"])
        tmin_hi = float(cuts["intime_anode_crosser"]["t_min_p2_hi"])

    df = load_align_df(args.input)
    print(f"\nalign_df rows: {len(df):,}")
    print(f"  anode-cathode      : {int(df['is_anode_cathode_crosser'].sum()):,}")
    print(f"  in-time anode      : {int(df['is_intime_anode_crosser'].sum()):,}")
    print(f"  both               : {int((df['is_anode_cathode_crosser'] & df['is_intime_anode_crosser']).sum()):,}")
    print(f"  per-TPC anode gate : t_min in [{tmin_lo}, {tmin_hi}] ticks")

    rows = explode_residual_rows(df, tmin_lo, tmin_hi)
    print(f"\nresidual rows (one per anode-piercing): {len(rows):,}")
    print(f"  TPC0 anode points  : {int((rows['anode_tpc']==0).sum()):,}")
    print(f"  TPC1 anode points  : {int((rows['anode_tpc']==1).sum()):,}")

    macs = pd.concat([rows["mod1"], rows["mod2"]])
    counts = macs.value_counts().sort_index()
    print(f"\nunique CRT modules (FEB MAC values): {len(counts)}")
    print(f"  per-module track count: min={counts.min()}, "
          f"median={int(counts.median())}, max={counts.max()}, "
          f"mean={counts.mean():.0f}")

    sparse = counts[counts < 50]
    print(f"  modules with < 50 tracks: {len(sparse)}")
    if 0 < len(sparse) <= 20:
        for mac, n in sparse.items():
            print(f"     mac {int(mac):4d}: {int(n):4d}")

    # Pre-alignment residuals
    resid = perp_residual(rows)
    print(f"\npre-alignment perpendicular residual (cm):")
    qs = np.percentile(resid, [10, 50, 90])
    print(f"  10/50/90 pct: {qs[0]:.2f} / {qs[1]:.2f} / {qs[2]:.2f}")
    print(f"  mean: {resid.mean():.2f}, max: {resid.max():.2f}")

    # CRT line length
    crt_len = np.linalg.norm(
        rows[["sp1_x", "sp1_y", "sp1_z"]].to_numpy()
        - rows[["sp2_x", "sp2_y", "sp2_z"]].to_numpy(), axis=1)
    print(f"\nCRT-track length (cm): min={crt_len.min():.0f}, "
          f"median={np.median(crt_len):.0f}, max={crt_len.max():.0f}")

    # Plots
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(counts.values, bins=50, histtype="step", linewidth=1.5)
    ax.set_xlabel("residuals attributing to a module")
    ax.set_ylabel("modules")
    ax.set_title(f"Per-module residual count "
                 f"(median {int(counts.median())}, min {counts.min()}, "
                 f"{len(sparse)} modules below 50)")
    ax.axvline(50, color="r", linestyle="--", label="min-counts = 50")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.outdir, f"per_module_counts.{ext}"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, np.percentile(resid, 99.9), 100)
    ax.hist(resid, bins=bins, histtype="step", linewidth=1.5)
    ax.axvline(np.median(resid), color="r", linestyle="--",
               label=f"median {np.median(resid):.2f} cm")
    ax.set_xlabel("perpendicular residual (cm)")
    ax.set_ylabel("residual rows")
    ax.set_yscale("log")
    ax.set_title("Pre-alignment anode-to-CRT-line residual")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.outdir, f"pre_alignment_residual.{ext}"), dpi=150)
    plt.close(fig)

    # Save the per-module count CSV.
    pd.DataFrame({"module_id": counts.index.astype(int),
                  "n_residuals": counts.values}).to_csv(
        os.path.join(args.outdir, "per_module_counts.csv"), index=False)

    print(f"\n  plots/CSV written to {args.outdir}/")


if __name__ == "__main__":
    main()
