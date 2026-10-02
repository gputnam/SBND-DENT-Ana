"""For each top intra-wall outlier in the GDML-pivot module fit, compare
its pull to:
  - the pulls of its 3 nearest physical neighbors in the same wall, and
  - the per-wall pull statistics (mean / std / count of modules)

Reads the existing GDML-pivot module bootstrap outputs.

Usage:
    python intra_wall_neighbors.py [--top 12]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align_crt import (
    load_align_df, explode_to_residual_rows, build_module_index,
    pack_inputs_for_jax, load_gdml_centroids,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-dir", default="bootstrap_module_1000_gdml")
    ap.add_argument("--top", type=int, default=12,
                    help="how many top |pull| outliers to detail")
    ap.add_argument("--n-neighbors", type=int, default=3,
                    help="number of nearest in-wall neighbors to show")
    ap.add_argument("--parquet",
                    default="/exp/sbnd/data/users/gputnam/efield-study/align_df")
    ap.add_argument("--cut-config", default="cut_tuning/cut_config_per_tpc.json")
    ap.add_argument("--max-pre-residual-cm", type=float, default=50.0)
    ap.add_argument("--min-counts", type=int, default=30)
    args = ap.parse_args()

    # 1. Load intra-wall pulls and bootstrap thetas.
    summ = pd.read_csv(os.path.join(args.module_dir, "intra_wall_summary.csv"))
    print(f"intra_wall_summary rows: {len(summ):,}", flush=True)

    # 2. Re-derive each module's GDML position (the fit's rotation pivot).
    print("Reproducing module GDML positions ...", flush=True)
    df = load_align_df(args.parquet)
    with open(args.cut_config) as f:
        cuts = json.load(f)
    rows = explode_to_residual_rows(df, cuts["tpc_t_min_lo"],
                                    cuts["tpc_t_min_hi"], 0.0)
    s1 = rows[["crt_sp1_x", "crt_sp1_y", "crt_sp1_z"]].to_numpy()
    s2 = rows[["crt_sp2_x", "crt_sp2_y", "crt_sp2_z"]].to_numpy()
    pa = np.stack([rows["anode_x_nom"], rows["anode_y"], rows["anode_z"]],
                  axis=1)
    line = s2 - s1
    u = line / np.maximum(np.linalg.norm(line, axis=1, keepdims=True), 1e-9)
    delta = pa - s1
    proj = np.sum(delta * u, axis=1, keepdims=True) * u
    pre = np.linalg.norm(delta - proj, axis=1)
    rows = rows.loc[pre <= args.max_pre_residual_cm].reset_index(drop=True)

    mod_ids, mod_to_idx, emp_centroids, counts, taggers = build_module_index(rows, df)
    keep = mod_ids[counts >= args.min_counts]
    keep_mask = (rows["mod1_id"].isin(keep) & rows["mod2_id"].isin(keep))
    rows = rows[keep_mask].reset_index(drop=True)
    mod_ids, mod_to_idx, emp_centroids, counts, taggers = build_module_index(rows, df)
    gdml_centroids = load_gdml_centroids(mod_ids, emp_centroids)

    pos_df = pd.DataFrame({
        "module_id": mod_ids.astype(int),
        "tagger":    taggers.astype(int),
        "n_residuals": counts.astype(int),
        "x": gdml_centroids[:, 0],
        "y": gdml_centroids[:, 1],
        "z": gdml_centroids[:, 2],
    })

    # 3. Pick top outliers (by |pull|).
    sub = summ.dropna(subset=["pull"]).copy()
    sub["abs_pull"] = sub["pull"].abs()
    sub = sub.sort_values("abs_pull", ascending=False).reset_index(drop=True)
    top = sub.head(args.top)

    # 4. For each outlier, find neighbors in the same wall by Euclidean
    #    distance (within the wall — i.e. between modules in that tagger).
    print(f"\nTop {args.top} intra-wall outliers (GDML pivot, B=1000):\n",
          flush=True)
    for _, r in top.iterrows():
        mac = int(r["module_id"])
        comp = r["component"]
        wall = r["wall"]
        pos = pos_df[pos_df.module_id == mac].iloc[0]
        wall_modules = pos_df[pos_df.tagger == int(r["tagger"])]
        wall_modules = wall_modules[wall_modules.module_id != mac].copy()
        # Distance to outlier in 3D.
        wall_modules["dist"] = np.sqrt(
            (wall_modules.x - pos.x) ** 2
            + (wall_modules.y - pos.y) ** 2
            + (wall_modules.z - pos.z) ** 2
        )
        wall_modules = wall_modules.sort_values("dist")
        neighbors = wall_modules.head(args.n_neighbors)

        # Pulls of neighbors AND the same wall for the same component.
        same_comp = summ[(summ.component == comp) & (summ.tagger == int(r["tagger"]))]
        wall_pulls = same_comp["pull"].dropna().to_numpy()

        print(f"=== {wall} mac{mac:3d}  {comp}  "
              f"@ ({pos.x:+7.1f}, {pos.y:+7.1f}, {pos.z:+7.1f}) cm  "
              f"(n_residuals={int(pos.n_residuals)})", flush=True)
        print(f"   outlier:  Δ = {r['delta_central']:+8.3f}  "
              f"σ = {r['delta_boot_std']:.3f}  pull = {r['pull']:+6.2f}",
              flush=True)
        # Neighbors' pulls for the same component.
        neighbor_macs = neighbors["module_id"].astype(int).tolist()
        for nmac in neighbor_macs:
            row = same_comp[same_comp.module_id == nmac]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            d = float(neighbors[neighbors.module_id == nmac].iloc[0]["dist"])
            print(f"   neighbor mac{nmac:3d}  dist {d:6.1f} cm  "
                  f"Δ = {row['delta_central']:+8.3f}  "
                  f"σ = {row['delta_boot_std']:.3f}  pull = {row['pull']:+6.2f}",
                  flush=True)
        # Wall-wide stats for this DOF.
        print(f"   wall {wall} on {comp}: mean(pull)={wall_pulls.mean():+5.2f}  "
              f"std={wall_pulls.std(ddof=1):5.2f}  "
              f"|pull|>2: {int((np.abs(wall_pulls)>2).sum())} of "
              f"{len(wall_pulls)}", flush=True)
        print()


if __name__ == "__main__":
    main()
