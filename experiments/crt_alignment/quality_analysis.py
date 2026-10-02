"""Per-module quality analysis of pre-alignment residuals.

For every selected residual row we have a 3D residual VECTOR — the
component of (P_anode − P_crt1) perpendicular to the line through
(P_crt1, P_crt2). That vector points from the foot of perpendicular
(on the line) to the anode point, so it captures the directional
disagreement between the matched CRT track and the TPC-anode point.

For each CRT module M, we collect all residual vectors from tracks
where M contributed at least one space-point, and compute:

  - n_residuals:  count
  - perp_med:     median |perp residual| (cm)
  - mean_dx, dy, dz:  mean of each component of the perp residual
                  vector (signed)
  - bias_axis:    the axis (x|y|z) with the largest |mean_d*| component
  - mean_dot_axis: the projection of mean residual along the dominant
                  CRT-line axis (helps isolate "drift-direction"
                  biases from "transverse" biases)

Outputs (in --outdir, default ./quality/):
  per_module_residuals.csv        one row per module
  per_module_perp_med.{png,pdf}   sorted bar of per-module |perp|
                                  median, color-coded by tagger
  per_module_bias_xyz.{png,pdf}   per-axis bias scatter, flagged
                                  outliers labeled
  per_tagger_summary.{png,pdf}    stacked per-tagger bias histograms
  per_module_residuals_summary.txt

Usage:
  python quality_analysis.py [--parquet ...] [--outdir quality]
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from crtlib.constants import ANODE_X_NOMINAL
from crtlib.io import load_align_df


def build_residual_rows(df, tmin_lo, tmin_hi):
    """One row per (track, anode-piercing TPC), with per-axis residual
    components."""
    out = []
    for _, r in df.iterrows():
        sp1 = np.array([r["crt_sp1_x"], r["crt_sp1_y"], r["crt_sp1_z"]])
        sp2 = np.array([r["crt_sp2_x"], r["crt_sp2_y"], r["crt_sp2_z"]])
        mod1 = int(r["crt_sp1_module_id"])
        mod2 = int(r["crt_sp2_module_id"])
        t1   = int(r["crt_sp1_tagger"])
        t2   = int(r["crt_sp2_tagger"])
        if mod1 < 0 or mod2 < 0:
            continue
        line = sp2 - sp1
        L = np.linalg.norm(line)
        if L < 1e-3:
            continue
        u = line / L
        for tpc in (0, 1):
            n_hits = r[f"tpc{tpc}_n_p2_hits"]
            ay = r[f"tpc{tpc}_anode_y"]; az = r[f"tpc{tpc}_anode_z"]
            tmin = r[f"tpc{tpc}_t_min"]
            if n_hits <= 0 or not np.isfinite(ay) or not np.isfinite(az):
                continue
            if not (np.isfinite(tmin) and tmin_lo <= tmin <= tmin_hi):
                continue
            pa = np.array([ANODE_X_NOMINAL[tpc], ay, az])
            delta = pa - sp1
            perp = delta - (delta @ u) * u
            out.append({
                "mod1": mod1, "mod2": mod2,
                "tag1": t1, "tag2": t2,
                "anode_tpc": tpc,
                "crt_len": float(L),
                "perp_x": float(perp[0]),
                "perp_y": float(perp[1]),
                "perp_z": float(perp[2]),
                "perp": float(np.linalg.norm(perp)),
            })
    return pd.DataFrame(out)


def per_module_summary(rows, max_perp_cm=None):
    """Group residual rows by module (each row contributes to BOTH
    mod1 and mod2). Optionally cap |perp| ≤ max_perp_cm to focus on
    geometrically constrained tracks (and avoid the long-tail
    contamination from short-lever-arm matches)."""
    if max_perp_cm is not None:
        rows = rows[rows["perp"] <= max_perp_cm].reset_index(drop=True)
    # Long form: one entry per (residual row, module-end).
    a = rows[["mod1", "tag1", "perp_x", "perp_y", "perp_z", "perp"]] \
        .rename(columns={"mod1": "module_id", "tag1": "tagger"})
    b = rows[["mod2", "tag2", "perp_x", "perp_y", "perp_z", "perp"]] \
        .rename(columns={"mod2": "module_id", "tag2": "tagger"})
    long = pd.concat([a, b], ignore_index=True)

    g = long.groupby("module_id")
    summary = pd.DataFrame({
        "tagger":    g["tagger"].agg(lambda s: int(s.mode().iloc[0])),
        "n":         g.size(),
        "perp_med":  g["perp"].median(),
        "perp_mean": g["perp"].mean(),
        "perp_std":  g["perp"].std(),
        "mean_dx":   g["perp_x"].mean(),
        "mean_dy":   g["perp_y"].mean(),
        "mean_dz":   g["perp_z"].mean(),
        "std_dx":    g["perp_x"].std(),
        "std_dy":    g["perp_y"].std(),
        "std_dz":    g["perp_z"].std(),
    }).reset_index()

    # Bias axis: which of (dx, dy, dz) has the largest mean-magnitude.
    abs_means = summary[["mean_dx", "mean_dy", "mean_dz"]].abs().to_numpy()
    axis_names = np.array(["x", "y", "z"])
    summary["bias_axis"] = axis_names[abs_means.argmax(axis=1)]
    summary["bias_mag"] = abs_means.max(axis=1)
    summary["mean_3d_mag"] = np.linalg.norm(
        summary[["mean_dx", "mean_dy", "mean_dz"]].to_numpy(), axis=1)
    return summary


def flag_outliers(summary, key="mean_3d_mag", k_mad=4.0):
    """Modules where summary[key] > median + k_mad * MAD."""
    v = summary[key].to_numpy()
    med = np.median(v); mad = np.median(np.abs(v - med))
    threshold = med + k_mad * mad
    summary["is_outlier"] = v > threshold
    return threshold


# Tagger ID → name (from earlier exploration: 0=Bot, 1=South, 2=North,
# 3=West, 4=East, 5=TopLow, 6=TopHigh).
TAGGER_NAMES = {0: "Bot", 1: "S", 2: "N", 3: "W", 4: "E",
                5: "TopLow", 6: "TopHigh"}
TAGGER_COLORS = {0: "tab:brown", 1: "tab:purple", 2: "tab:orange",
                 3: "tab:green", 4: "tab:red", 5: "tab:cyan",
                 6: "tab:blue"}


def plot_per_module_perp_med(summary, threshold, outdir):
    s = summary.sort_values("perp_med").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [TAGGER_COLORS.get(t, "gray") for t in s["tagger"]]
    ax.bar(np.arange(len(s)), s["perp_med"], color=colors, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("modules sorted by median |perp residual|")
    ax.set_ylabel("median |perp residual| (cm)")
    ax.set_title("Per-module median pre-alignment perpendicular residual")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=TAGGER_NAMES.get(t, "?"))
               for t, c in TAGGER_COLORS.items() if t in s["tagger"].unique()]
    ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=9)
    ax.set_yscale("log")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"per_module_perp_med.{ext}"), dpi=140)
    plt.close(fig)


def plot_per_module_bias_xyz(summary, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, key, name in zip(axes, ("mean_dx", "mean_dy", "mean_dz"),
                             ("mean dx (cm)", "mean dy (cm)", "mean dz (cm)")):
        for tag in sorted(summary["tagger"].unique()):
            sub = summary[summary["tagger"] == tag]
            ax.scatter(sub[key], sub["perp_med"],
                       color=TAGGER_COLORS.get(int(tag), "gray"),
                       label=TAGGER_NAMES.get(int(tag), "?"),
                       s=24, edgecolor="k", linewidth=0.3)
        # outliers labeled
        for _, r in summary[summary["is_outlier"]].iterrows():
            ax.annotate(f'mac{int(r["module_id"])}',
                        (r[key], r["perp_med"]),
                        fontsize=7, alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel(name)
        ax.set_ylabel("median |perp residual| (cm)")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle("Per-module residual bias vs median magnitude (pre-alignment)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"per_module_bias_xyz.{ext}"), dpi=140)
    plt.close(fig)


def plot_per_tagger_summary(summary, outdir):
    """Per-tagger histograms of (mean_dx, mean_dy, mean_dz) and perp_med."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, key, label in zip(
        axes.flat,
        ("mean_dx", "mean_dy", "mean_dz", "perp_med"),
        ("per-module mean dx (cm)", "mean dy (cm)", "mean dz (cm)",
         "per-module median |perp| (cm)"),
    ):
        for tag in sorted(summary["tagger"].unique()):
            sub = summary[summary["tagger"] == tag]
            ax.hist(sub[key], bins=20, histtype="step", linewidth=1.3,
                    color=TAGGER_COLORS.get(int(tag), "gray"),
                    label=TAGGER_NAMES.get(int(tag), "?"))
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel("modules")
    axes[0, 0].legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"per_tagger_summary.{ext}"), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet",
                    default="/exp/sbnd/data/users/gputnam/efield-study/align_df")
    ap.add_argument("--outdir", default="quality")
    ap.add_argument("--cut-config", default="cut_tuning/cut_config_per_tpc.json")
    ap.add_argument("--max-perp-cm", type=float, default=50.0,
                    help="cap on |perp residual| for the per-module summary "
                         "(matches the pre-residual cut used in align_crt.py)")
    ap.add_argument("--mad-k", type=float, default=4.0,
                    help="outlier threshold = median + k × MAD (in mean_3d_mag)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.cut_config) as f:
        cuts = json.load(f)
    tmin_lo = float(cuts["tpc_t_min_lo"])
    tmin_hi = float(cuts["tpc_t_min_hi"])

    print(f"Loading {args.parquet} ...", flush=True)
    df = load_align_df(args.parquet)
    print(f"  {len(df):,} matched tracks", flush=True)

    print("Building residual rows ...", flush=True)
    rows = build_residual_rows(df, tmin_lo, tmin_hi)
    print(f"  {len(rows):,} residual rows", flush=True)
    print(f"  before |perp| cut: P10/P50/P90 = "
          f"{np.percentile(rows['perp'], [10,50,90])}", flush=True)

    summary = per_module_summary(rows, max_perp_cm=args.max_perp_cm)
    print(f"\nper-module summary (after |perp|<={args.max_perp_cm} cm):", flush=True)
    print(f"  {len(summary)} unique modules contributing", flush=True)
    print(f"  median(perp_med) = {summary['perp_med'].median():.2f} cm", flush=True)
    print(f"  median(mean_3d_mag) = {summary['mean_3d_mag'].median():.2f} cm",
          flush=True)

    threshold = flag_outliers(summary, key="mean_3d_mag", k_mad=args.mad_k)
    n_outliers = int(summary["is_outlier"].sum())
    print(f"\noutlier flag: mean_3d_mag > median + {args.mad_k}×MAD "
          f"(threshold {threshold:.2f} cm) → {n_outliers} flagged",
          flush=True)

    summary = summary.sort_values("mean_3d_mag", ascending=False).reset_index(drop=True)
    out_csv = os.path.join(args.outdir, "per_module_residuals.csv")
    summary.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}", flush=True)

    # Show top outliers and bias-axis breakdown.
    print(f"\nTop 20 modules by mean_3d_mag (outliers in particular):", flush=True)
    top20 = summary.head(20)
    print(top20[["module_id", "tagger", "n", "perp_med",
                 "mean_dx", "mean_dy", "mean_dz", "bias_axis",
                 "is_outlier"]].to_string(index=False, float_format="%.2f"))

    print(f"\nbias-axis distribution (over all modules):", flush=True)
    print(summary["bias_axis"].value_counts().to_string())

    print(f"\nbias-axis distribution per tagger:", flush=True)
    for tag in sorted(summary["tagger"].unique()):
        sub = summary[summary["tagger"] == tag]
        print(f"  {TAGGER_NAMES.get(int(tag), '?'):8s}: "
              f"{dict(sub['bias_axis'].value_counts())}", flush=True)

    plot_per_module_perp_med(summary, threshold, args.outdir)
    plot_per_module_bias_xyz(summary, args.outdir)
    plot_per_tagger_summary(summary, args.outdir)
    print(f"\n  plots written to {args.outdir}/", flush=True)


if __name__ == "__main__":
    main()
