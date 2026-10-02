"""Compare empirical CRT-module centroids (per FEB MAC, from align_df)
to nominal centroids (per volCRTModuleN, from the SBND GDML).

For each FEB MAC observed in the data we compute:
  empirical_centroid = mean of all CRT space-points attributed to that MAC

For each volCRTModuleN we have its nominal world-frame centroid from
extract_gdml_crt_centroids.py. We match each FEB MAC to the closest
GDML module by Euclidean distance — this is robust to alignment shifts
of order ~few cm because module-to-module spacing in SBND is ~tens of
cm. A mismatch fraction near zero means the assignment is unambiguous.

Outputs (in --outdir, default ./centroid_compare/):
  centroid_deltas.csv             — one row per FEB MAC: empirical (x,y,z),
                                    matched module name, nominal (x,y,z),
                                    delta (dx,dy,dz, |Δ|)
  centroid_deltas.{png,pdf}       — histograms of dx, dy, dz, |Δ|
  centroid_2d_per_tagger.{png,pdf} — empirical vs nominal positions, per tagger

Usage:
  python compare_centroids.py \
      /exp/sbnd/data/users/gputnam/efield-study/align_df \
      crt_module_gdml_centroids.csv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from crtlib.io import load_align_df


def empirical_centroids(df):
    """Return DataFrame indexed by FEB MAC with both mean and median
    (x, y, z) and count.

    Median is more robust to angular-distribution bias than the mean
    (cosmic muons are not uniformly distributed across a CRT module —
    the mean drifts toward the high-flux side of the module).
    """
    sp1 = df[["crt_sp1_module_id", "crt_sp1_x", "crt_sp1_y", "crt_sp1_z"]] \
        .rename(columns=lambda c: c.replace("crt_sp1_", ""))
    sp2 = df[["crt_sp2_module_id", "crt_sp2_x", "crt_sp2_y", "crt_sp2_z"]] \
        .rename(columns=lambda c: c.replace("crt_sp2_", ""))
    sp = pd.concat([sp1, sp2], ignore_index=True)
    sp = sp[(sp["module_id"] >= 0)
            & np.isfinite(sp["x"]) & np.isfinite(sp["y"]) & np.isfinite(sp["z"])]
    g = sp.groupby("module_id").agg(
        emp_x_mean=("x", "mean"), emp_y_mean=("y", "mean"), emp_z_mean=("z", "mean"),
        emp_x_med=("x", "median"), emp_y_med=("y", "median"), emp_z_med=("z", "median"),
        emp_n=("x", "size"))
    g = g.reset_index().rename(columns={"module_id": "feb_mac"})
    # Use median as the primary centroid (more robust); keep mean for diagnostics.
    g["emp_x"] = g["emp_x_med"]
    g["emp_y"] = g["emp_y_med"]
    g["emp_z"] = g["emp_z_med"]
    return g


def match_to_nominal(emp_df, nom_df):
    """Optimal one-to-one assignment via the Hungarian algorithm.

    Minimizes the total sum of distances. With per-axis alignment shifts
    of a few cm and per-module spacing of ~tens of cm, this resolves
    boundary cases where greedy matching would mis-assign neighbors.
    """
    from scipy.optimize import linear_sum_assignment
    emp_xyz = emp_df[["emp_x", "emp_y", "emp_z"]].to_numpy()
    nom_xyz = nom_df[["world_x", "world_y", "world_z"]].to_numpy()

    # Cost matrix (m_emp × n_nom) of Euclidean distances.
    diff = emp_xyz[:, None, :] - nom_xyz[None, :, :]
    cost = np.linalg.norm(diff, axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)

    out = []
    for i, j in zip(row_ind, col_ind):
        nom_row = nom_df.iloc[int(j)]
        out.append({
            "feb_mac":     int(emp_df.iloc[i]["feb_mac"]),
            "emp_n":       int(emp_df.iloc[i]["emp_n"]),
            "emp_x":       float(emp_df.iloc[i]["emp_x"]),
            "emp_y":       float(emp_df.iloc[i]["emp_y"]),
            "emp_z":       float(emp_df.iloc[i]["emp_z"]),
            "emp_x_mean":  float(emp_df.iloc[i]["emp_x_mean"]),
            "emp_y_mean":  float(emp_df.iloc[i]["emp_y_mean"]),
            "emp_z_mean":  float(emp_df.iloc[i]["emp_z_mean"]),
            "module_idx":  int(nom_row["module_idx"]),
            "module_name": str(nom_row["module_name"]),
            "tagger":      str(nom_row["tagger"]),
            "nom_x":       float(nom_row["world_x"]),
            "nom_y":       float(nom_row["world_y"]),
            "nom_z":       float(nom_row["world_z"]),
            "match_dist":  float(cost[i, j]),
        })
    out_df = pd.DataFrame(out)
    out_df["dx"] = out_df["emp_x"] - out_df["nom_x"]
    out_df["dy"] = out_df["emp_y"] - out_df["nom_y"]
    out_df["dz"] = out_df["emp_z"] - out_df["nom_z"]
    out_df["dmag"] = np.sqrt(out_df["dx"]**2 + out_df["dy"]**2 + out_df["dz"]**2)
    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("align_df")
    ap.add_argument("gdml_centroids")
    ap.add_argument("--outdir", default="centroid_compare")
    ap.add_argument("--ambiguous-frac", type=float, default=0.5,
                    help="if match_dist > this × distance to next-closest, "
                         "flag as ambiguous match")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading align_df from {args.align_df} ...", flush=True)
    df = load_align_df(args.align_df)
    print(f"  {len(df):,} matched anode-piercing rows", flush=True)

    print("Computing empirical centroids ...", flush=True)
    emp = empirical_centroids(df)
    print(f"  {len(emp)} unique FEB MACs", flush=True)

    print(f"Loading GDML centroids from {args.gdml_centroids} ...", flush=True)
    nom = pd.read_csv(args.gdml_centroids)
    print(f"  {len(nom)} GDML modules", flush=True)

    print("Matching empirical to nominal ...", flush=True)
    res = match_to_nominal(emp, nom)
    print(f"\nmatch distance distribution (cm): "
          f"min={res['match_dist'].min():.2f}, "
          f"median={res['match_dist'].median():.2f}, "
          f"max={res['match_dist'].max():.2f}", flush=True)
    print(f"per-axis Δ (empirical − nominal), median±MAD (cm):")
    for ax in ("dx", "dy", "dz", "dmag"):
        med = res[ax].median()
        mad = (res[ax] - med).abs().median()
        print(f"  {ax}: median {med:+.3f}, MAD {mad:.3f}, "
              f"min {res[ax].min():+.3f}, max {res[ax].max():+.3f}")

    res = res.sort_values("feb_mac").reset_index(drop=True)
    out_csv = os.path.join(args.outdir, "centroid_deltas.csv")
    res.to_csv(out_csv, index=False)
    print(f"\n  wrote {out_csv}", flush=True)

    # Plots: 1D histograms of dx, dy, dz, |Δ|
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, key, label in zip(axes.flat,
                              ("dx", "dy", "dz", "dmag"),
                              ("Δx (cm)", "Δy (cm)", "Δz (cm)", "|Δ| (cm)")):
        v = res[key]
        lim = max(abs(v.min()), abs(v.max())) * 1.05 if key != "dmag" \
            else v.max() * 1.05
        if key == "dmag":
            ax.hist(v, bins=50, range=(0, lim), histtype="step", linewidth=1.5)
        else:
            ax.hist(v, bins=50, range=(-lim, lim), histtype="step", linewidth=1.5)
            ax.axvline(0, color="grey", lw=0.5)
        ax.axvline(v.median(), color="r", ls="--", lw=0.8,
                   label=f"median {v.median():+.2f} cm")
        ax.set_xlabel(label); ax.set_ylabel("modules"); ax.legend()
    fig.suptitle("Empirical (data) − Nominal (GDML) centroid deltas, per FEB MAC")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.outdir, f"centroid_deltas.{ext}"), dpi=150)
    plt.close(fig)

    # Per-tagger 2D scatter (empirical vs nominal in two relevant axes).
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    taggers = sorted(res["tagger"].unique())
    for ax, tag in zip(axes.flat, taggers):
        sub = res[res["tagger"] == tag]
        # Choose the two most-spread axes for this tagger.
        spreads = [(sub["nom_x"].max() - sub["nom_x"].min(), 0, "x"),
                   (sub["nom_y"].max() - sub["nom_y"].min(), 1, "y"),
                   (sub["nom_z"].max() - sub["nom_z"].min(), 2, "z")]
        spreads.sort(reverse=True)
        ix1, ix2 = spreads[0][1], spreads[1][1]
        labels = ["x", "y", "z"]
        nom_axes = [sub["nom_x"], sub["nom_y"], sub["nom_z"]]
        emp_axes = [sub["emp_x"], sub["emp_y"], sub["emp_z"]]
        for nx, ny, ex, ey in zip([nom_axes[ix1]], [nom_axes[ix2]],
                                  [emp_axes[ix1]], [emp_axes[ix2]]):
            ax.scatter(nx, ny, s=20, c="grey", marker="o",
                       label="nominal", edgecolor="none")
            ax.scatter(ex, ey, s=20, c="C3", marker="x", label="empirical")
            for a, b, c, d in zip(nx, ny, ex, ey):
                ax.plot([a, c], [b, d], "k-", lw=0.4, alpha=0.5)
        ax.set_title(f"{tag} (n={len(sub)})")
        ax.set_xlabel(f"{labels[ix1]} (cm)")
        ax.set_ylabel(f"{labels[ix2]} (cm)")
        ax.set_aspect("equal", adjustable="datalim")
    # Hide unused axes if fewer than 8 taggers.
    for ax in axes.flat[len(taggers):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8, loc="best")
    fig.suptitle("Per-tagger empirical vs nominal CRT-module centroids")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.outdir, f"centroid_2d_per_tagger.{ext}"), dpi=150)
    plt.close(fig)

    print(f"  plots written to {args.outdir}/")


if __name__ == "__main__":
    main()
