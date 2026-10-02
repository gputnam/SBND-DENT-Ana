"""Comprehensive comparison across all alignment-fit variants run in this
study. Produces a single overview plot of residual distributions, a
per-wall results table, and an intra-wall outlier comparison.

Inputs (created by earlier runs):
  crt_alignment_137k                    module fit, empirical pivot, +rot
  crt_alignment_137k_wall               wall fit,   empirical pivot, +rot
  crt_alignment_137k_wall_norot         wall fit,   empirical pivot, translation-only
  crt_alignment_137k_gdml               module fit, gdml pivot,      +rot
  crt_alignment_137k_wall_gdml          wall fit,   gdml pivot,      +rot
  bootstrap_wall_1000                   wall bootstrap, empirical
  bootstrap_module_1000                 module bootstrap, empirical
  bootstrap_wall_1000_gdml              wall bootstrap, gdml
  bootstrap_module_1000_gdml            module bootstrap, gdml

Outputs:
  final_residual_overlay.{png,pdf}              all six residual histograms
  final_residual_overlay_cdf.{png,pdf}          same as CDF
  final_wall_table.csv                          per-wall central + boot σ
                                                across the 4 bootstrapped variants
  final_intra_wall_outliers_compare.csv         intra-wall outliers under
                                                empirical vs GDML pivot
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS = [
    # (label, alignment_dir, color)
    ("module fit, empirical pivot (636 params)",
     "crt_alignment_137k",            "tab:blue"),
    ("wall fit, empirical pivot (48 params)",
     "crt_alignment_137k_wall",       "tab:orange"),
    ("wall, empirical, translation-only (24 params)",
     "crt_alignment_137k_wall_norot", "tab:green"),
    ("module fit, GDML pivot (636 params)",
     "crt_alignment_137k_gdml",       "tab:red"),
    ("wall fit, GDML pivot (48 params)",
     "crt_alignment_137k_wall_gdml",  "tab:purple"),
]

BOOT_VARIANTS = [
    # (label, bootstrap_dir, kind: 'wall'|'module')
    ("wall, empirical",  "bootstrap_wall_1000",        "wall"),
    ("module, empirical","bootstrap_module_1000",      "module"),
    ("wall, GDML",       "bootstrap_wall_1000_gdml",   "wall"),
    ("module, GDML",     "bootstrap_module_1000_gdml", "module"),
]

TAG = {0: "Bot", 1: "S", 2: "N", 3: "W", 4: "E", 5: "TopLow", 6: "TopHigh"}


# ----- 1) Residual-distribution overlay -----
def plot_residuals(out_basename="final_residual_overlay"):
    pre = np.load(os.path.join(VARIANTS[0][1], "residuals_pre.npy"))
    print(f"Loaded pre residuals: {len(pre):,}")
    posts = []
    for label, d, color in VARIANTS:
        p = np.load(os.path.join(d, "residuals_post.npy"))
        posts.append((label, p, color))
        print(f"  {label:55s} -> median {np.median(p):.3f} cm")

    # Histogram
    edges = np.linspace(0, 50, 101)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(10, 6))
    h_pre, _ = np.histogram(pre, bins=edges)
    ax.step(centers, h_pre, where="mid", lw=2.0, color="0.35",
            label=f"pre-alignment (median {np.median(pre):.2f} cm)")
    ax.axvline(np.median(pre), color="0.35", ls=":", lw=0.8)
    for label, p, color in posts:
        h, _ = np.histogram(p, bins=edges)
        ax.step(centers, h, where="mid", lw=1.4, color=color,
                label=f"{label} (med {np.median(p):.2f} cm)")
        ax.axvline(np.median(p), color=color, ls=":", lw=0.7)
    ax.set_xlim(0, 50)
    ax.set_yscale("log")
    ax.set_xlabel("perpendicular residual |Pa − line(P1, P2)| (cm)")
    ax.set_ylabel("residual rows / bin")
    ax.set_title(f"Anode-to-CRT-line straightness — all alignment variants "
                 f"(N = {len(pre):,})")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_basename}.{ext}", dpi=140)
    plt.close(fig)

    # CDF
    fig, ax = plt.subplots(figsize=(10, 6))
    F = np.linspace(0, 1, len(pre), endpoint=True)
    ax.plot(np.sort(pre), F, color="0.35", lw=2.0, label="pre-alignment")
    for label, p, color in posts:
        ax.plot(np.sort(p), F, color=color, lw=1.4, label=label)
    ax.set_xlim(0, 50)
    ax.set_xlabel("perpendicular residual (cm)")
    ax.set_ylabel("cumulative fraction")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_basename}_cdf.{ext}", dpi=140)
    plt.close(fig)
    print(f"\nWrote {out_basename}.png/.pdf, {out_basename}_cdf.png/.pdf")


# ----- 2) Per-wall summary table across the 4 bootstrapped variants -----
def wall_summary(out_csv="final_wall_table.csv"):
    """For each wall and each DOF, list the central value and boot_std
    under each of the 4 bootstrap variants. For module-mode bootstraps
    we aggregate by averaging across modules in the wall (per replicate)
    so the wall-level number is comparable to wall-mode fits."""
    rows = []

    for label, d, kind in BOOT_VARIANTS:
        thetas = np.load(os.path.join(d, "bootstrap_thetas.npy"))
        central = np.load(os.path.join(d, "central_theta.npy"))
        idx = pd.read_csv(os.path.join(d, "bootstrap_param_index.csv"))

        for tag in sorted(TAG.keys()):
            sub = idx[(idx["kind"] == kind) & (idx["tagger"] == tag)]
            if len(sub) == 0:
                continue
            for comp in ["dx", "dy", "dz", "dthx", "dthy", "dthz"]:
                ssub = sub[sub["component"] == comp]
                if len(ssub) == 0:
                    continue
                pidx = ssub["param_idx"].to_numpy()
                # Per-replicate wall-mean across modules in this wall.
                vals_b = thetas[:, pidx].mean(axis=1)
                cen = float(central[pidx].mean())
                rows.append({
                    "variant": label, "wall_tag": tag, "wall": TAG[tag],
                    "component": comp,
                    "n_units_in_wall": int(len(ssub)),
                    "central":  cen,
                    "boot_mean": float(np.mean(vals_b)),
                    "boot_std":  float(np.std(vals_b, ddof=1)),
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows)")

    # Print pivot table: central ± std, organized by (wall, component)
    print("\nPer-wall translation summary (central ± boot_std, cm):")
    for comp in ("dx", "dy", "dz"):
        print(f"\n  Component {comp}:")
        sub = df[df["component"] == comp]
        head = "    " + f"{'wall':<8s}" + "".join(
            f"{l:>30s}" for l, _, _ in BOOT_VARIANTS)
        print(head)
        for tag in sorted(sub["wall_tag"].unique()):
            line = f"    {TAG[int(tag)]:<8s}"
            for l, _, _ in BOOT_VARIANTS:
                rec = sub[(sub["wall_tag"] == tag) & (sub["variant"] == l)]
                if len(rec) == 0:
                    line += f"{'—':>30s}"
                else:
                    r = rec.iloc[0]
                    line += f"{r['central']:+12.3f} ± {r['boot_std']:>5.2f}        "
            print(line)

    print("\nPer-wall rotation summary (central ± boot_std, mrad):")
    for comp in ("dthx", "dthy", "dthz"):
        print(f"\n  Component {comp}:")
        sub = df[df["component"] == comp]
        head = "    " + f"{'wall':<8s}" + "".join(
            f"{l:>30s}" for l, _, _ in BOOT_VARIANTS)
        print(head)
        for tag in sorted(sub["wall_tag"].unique()):
            line = f"    {TAG[int(tag)]:<8s}"
            for l, _, _ in BOOT_VARIANTS:
                rec = sub[(sub["wall_tag"] == tag) & (sub["variant"] == l)]
                if len(rec) == 0:
                    line += f"{'—':>30s}"
                else:
                    r = rec.iloc[0]
                    line += f"{r['central']*1e3:+11.2f} ± {r['boot_std']*1e3:>5.2f}        "
            print(line)


# ----- 3) Intra-wall outlier consistency across pivots -----
def outlier_compare(out_csv="final_intra_wall_outliers_compare.csv"):
    emp = pd.read_csv("bootstrap_module_1000/intra_wall_outliers.csv")
    gdml = pd.read_csv("bootstrap_module_1000_gdml/intra_wall_outliers.csv")
    print(f"\nIntra-wall outliers (|pull|>3):")
    print(f"  empirical pivot: {len(emp)}")
    print(f"  GDML pivot     : {len(gdml)}")

    emp["key"] = emp["module_id"].astype(str) + "_" + emp["component"]
    gdml["key"] = gdml["module_id"].astype(str) + "_" + gdml["component"]
    common = sorted(set(emp["key"]) & set(gdml["key"]))
    only_emp = sorted(set(emp["key"]) - set(gdml["key"]))
    only_gdml = sorted(set(gdml["key"]) - set(emp["key"]))
    print(f"  in both : {len(common)}")
    print(f"  only emp: {len(only_emp)}")
    print(f"  only gdml: {len(only_gdml)}")

    # Build a side-by-side comparison.
    rows = []
    keys = sorted(set(emp["key"]) | set(gdml["key"]))
    for k in keys:
        e = emp[emp["key"] == k].iloc[0] if k in emp["key"].values else None
        g = gdml[gdml["key"] == k].iloc[0] if k in gdml["key"].values else None
        ref = e if e is not None else g
        rows.append({
            "module_id": int(ref["module_id"]),
            "wall":      ref["wall"],
            "component": ref["component"],
            "emp_delta_central": float(e["delta_central"]) if e is not None else np.nan,
            "emp_pull":         float(e["pull"]) if e is not None else np.nan,
            "gdml_delta_central": float(g["delta_central"]) if g is not None else np.nan,
            "gdml_pull":          float(g["pull"]) if g is not None else np.nan,
            "in_both": (e is not None) and (g is not None),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}")

    print("\nTop 15 outliers in BOTH pivot variants (by max(|emp_pull|,|gdml_pull|)):")
    both = df[df["in_both"]].copy()
    both["max_pull"] = np.fmax(both["emp_pull"].abs(), both["gdml_pull"].abs())
    both = both.sort_values("max_pull", ascending=False).head(15)
    for _, r in both.iterrows():
        print(f"   {r['wall']:<8s} mac{int(r['module_id']):>3d}  {r['component']:<5s}  "
              f"emp Δ {r['emp_delta_central']:+6.2f} pull {r['emp_pull']:+5.2f}  |  "
              f"gdml Δ {r['gdml_delta_central']:+6.2f} pull {r['gdml_pull']:+5.2f}")


def main():
    plot_residuals()
    wall_summary()
    outlier_compare()


if __name__ == "__main__":
    main()
