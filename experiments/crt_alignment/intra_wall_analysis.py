"""Identify statistically significant intra-wall module relative shifts.

Reads the per-module bootstrap output and computes, for each module m:
    Δ_m = θ_m − ⟨θ_modules in same wall⟩
i.e. the module's correction *relative to its wall's mean correction*.
The bootstrap distribution of Δ_m gives a per-module "intra-wall pull"
whose σ accounts for the correlated motion of all modules within a
wall (those should bootstrap-shift together; only deviations from the
wall mean count).

A module is flagged "intra-wall significant" if for any of its 6 DOFs
    |Δ_m| / σ(Δ_m) > --threshold (default 3)

Outputs:
  intra_wall_summary.csv      one row per (module, dof): central, mean,
                              std, pull = mean/std, flag
  intra_wall_outliers.csv     subset with |pull| > threshold
  intra_wall_pulls.{png,pdf}  per-DOF histograms of pulls, with
                              outliers labeled

Usage:
  python intra_wall_analysis.py [--module-dir bootstrap_module_1000] \\
                                [--threshold 3]
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COMP = ["dx", "dy", "dz", "dthx", "dthy", "dthz"]
TAG_NAME = {0: "Bot", 1: "S", 2: "N", 3: "W", 4: "E", 5: "TopLow", 6: "TopHigh"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-dir", default="bootstrap_module_1000")
    ap.add_argument("--threshold", type=float, default=3.0,
                    help="|pull| threshold for flagging intra-wall outliers")
    ap.add_argument("--out", default=None,
                    help="output dir (default: alongside --module-dir)")
    args = ap.parse_args()
    out = args.out or args.module_dir

    print(f"Loading {args.module_dir} ...", flush=True)
    thetas = np.load(os.path.join(args.module_dir, "bootstrap_thetas.npy"))
    central = np.load(os.path.join(args.module_dir, "central_theta.npy"))
    idx = pd.read_csv(os.path.join(args.module_dir, "bootstrap_param_index.csv"))
    B, P = thetas.shape
    print(f"  B = {B}, n_params = {P}", flush=True)

    mod_idx = idx[idx["kind"] == "module"].copy()
    print(f"  modules: {mod_idx['unit_id'].nunique()}", flush=True)
    walls_present = sorted(mod_idx["tagger"].unique())
    print(f"  walls present: {walls_present} "
          f"({[TAG_NAME.get(int(w), '?') for w in walls_present]})",
          flush=True)

    # For each (wall, dof), build the wall-mean theta(b) across modules in
    # that wall, then for each module compute Δ_m(b) = θ_m(b) − wall_mean(b).
    rows = []
    for w in walls_present:
        wall_modules = mod_idx[mod_idx["tagger"] == w]
        wall_name = TAG_NAME.get(int(w), f"tag{int(w)}")
        for c in COMP:
            sub = wall_modules[wall_modules["component"] == c]
            if len(sub) == 0:
                continue
            param_indices = sub["param_idx"].to_numpy()
            wall_thetas_b = thetas[:, param_indices]   # (B, n_in_wall)
            wall_mean_b = wall_thetas_b.mean(axis=1, keepdims=True)  # (B, 1)
            delta_b = wall_thetas_b - wall_mean_b      # (B, n_in_wall)

            # Per-module Δ summary.
            delta_mean = delta_b.mean(axis=0)
            delta_std  = delta_b.std(axis=0, ddof=1)
            # central Δ uses the central full-sample fit.
            wall_mean_central = central[param_indices].mean()
            delta_central = central[param_indices] - wall_mean_central

            for k, (_, r) in enumerate(sub.iterrows()):
                pull = float(delta_central[k] / delta_std[k]) if delta_std[k] > 0 else np.nan
                rows.append({
                    "tagger": int(w), "wall": wall_name,
                    "module_id": int(r["unit_id"]),
                    "param": r["param"], "component": c,
                    "wall_mean_central": float(wall_mean_central),
                    "delta_central": float(delta_central[k]),
                    "delta_boot_mean": float(delta_mean[k]),
                    "delta_boot_std":  float(delta_std[k]),
                    "pull": pull,
                    "abs_pull": float(abs(pull)) if np.isfinite(pull) else np.nan,
                    "outlier": (np.isfinite(pull) and abs(pull) > args.threshold),
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "intra_wall_summary.csv"), index=False)

    out_df = df[df["outlier"]].sort_values("abs_pull", ascending=False)
    out_df.to_csv(os.path.join(out, "intra_wall_outliers.csv"), index=False)
    print(f"\nflagged outliers: {len(out_df)} of {len(df)} (module, DOF) rows",
          flush=True)
    print(f"  threshold |pull| > {args.threshold}", flush=True)
    print()
    if len(out_df):
        print("top 30 by |pull|:")
        print(out_df.head(30)[
            ["wall", "module_id", "component", "delta_central",
             "delta_boot_std", "pull"]
        ].to_string(index=False, float_format="%+.3f"))

    # Per-component pull histogram.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, c in zip(axes.flat, COMP):
        sub = df[df["component"] == c]
        if len(sub) == 0:
            continue
        v = sub["pull"].dropna().to_numpy()
        ax.hist(v, bins=40, range=(-8, 8), histtype="step", linewidth=1.5)
        ax.axvline(args.threshold, color="r", linestyle="--", linewidth=0.8)
        ax.axvline(-args.threshold, color="r", linestyle="--", linewidth=0.8)
        ax.set_xlabel(f"intra-wall pull on {c}")
        ax.set_ylabel("modules")
        ax.set_title(f"{c}  ({len(sub)} modules)")
        # Annotate outliers.
        outs = sub[sub["outlier"]].sort_values("abs_pull", ascending=False)
        for _, r in outs.head(8).iterrows():
            ax.text(r["pull"], 0.5, f"{r['wall']}/{int(r['module_id'])}",
                    fontsize=6, rotation=90, ha="center", va="bottom",
                    color="C3", clip_on=True)
    fig.suptitle(f"Per-DOF intra-wall pulls (B = {B})", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"intra_wall_pulls.{ext}"), dpi=140)
    plt.close(fig)

    print(f"\n  wrote intra_wall_summary.csv, intra_wall_outliers.csv, "
          f"intra_wall_pulls.{{png,pdf}}", flush=True)
    print(f"  in {out}/", flush=True)


if __name__ == "__main__":
    main()
