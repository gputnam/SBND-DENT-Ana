"""Build a table of median voxel-fit mu per X-slice, per (subset, component,
alignment configuration).

Reads voxel_data.npz from the three crtdisp_maps_20cm* directories and
emits:
  - a long-form CSV with one row per (slice, subset, component) and one
    median-mu column per alignment configuration
  - a stdout dump of NS / EW per slice for quick inspection
"""

import argparse
import os

import numpy as np
import pandas as pd


VARIANTS = [
    ("no_alignment", "crtdisp_maps_20cm"),
    ("wall_gdml",    "crtdisp_maps_20cm_wall_gdml"),
    ("module_gdml",  "crtdisp_maps_20cm_module_gdml"),
]

SUBSETS = ("side", "ns", "ew", "topside", "topn", "tops", "tope", "topw")
COMPONENTS = ("dx", "dy", "dz")

MIN_COUNTS = 30


def median_for_slice(d, sub, comp, ix):
    grid = d[f"mu_{sub}_{comp}"][ix]
    cs = d[f"count_{sub}"][ix]
    valid = np.isfinite(grid) & (cs >= MIN_COUNTS)
    if not valid.any():
        return float("nan"), 0
    return float(np.median(grid[valid])), int(valid.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="compare_distortion/median_per_slice.csv")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    datasets = {}
    for label, d in VARIANTS:
        path = os.path.join(d, "voxel_data.npz")
        if not os.path.exists(path):
            print(f"  WARNING: {path} missing")
            continue
        datasets[label] = np.load(path)
    if len(datasets) < 2:
        raise SystemExit("Need at least 2 variants present.")

    any_d = next(iter(datasets.values()))
    x_edges = any_d["x_edges"]
    nx = len(x_edges) - 1

    rows = []
    for ix in range(nx):
        for sub in SUBSETS:
            for comp in COMPONENTS:
                row = {
                    "x_slice_idx": ix,
                    "x_lo_cm": float(x_edges[ix]),
                    "x_hi_cm": float(x_edges[ix + 1]),
                    "subset": sub,
                    "component": comp,
                }
                for label, d in datasets.items():
                    med, nv = median_for_slice(d, sub, comp, ix)
                    row[f"median_mu_{label}_cm"] = med
                    row[f"n_vox_{label}"] = nv
                rows.append(row)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False, float_format="%.4f")
    print(f"  wrote {args.out}  ({len(out_df)} rows, "
          f"{nx} slices x {len(SUBSETS)} subsets x {len(COMPONENTS)} comps)")

    # Compact stdout dump for NS / EW (two-by-two-look most relevant after
    # the 6-pane request).
    cols_med = [c for c in out_df.columns if c.startswith("median_mu_")]
    label_short = {
        "median_mu_no_alignment_cm": "noalign",
        "median_mu_wall_gdml_cm":    "wall",
        "median_mu_module_gdml_cm":  "module",
    }
    for sub in ("ns", "ew"):
        print(f"\n{'='*78}\n{sub.upper()}-only median mu (cm) per X slice\n{'='*78}")
        sub_df = out_df[out_df.subset == sub].copy()
        for comp in COMPONENTS:
            print(f"\n  component = {comp}")
            print(f"   {'ix':>3s}  {'x[cm]':>14s}  "
                  + "  ".join(f"{label_short[c]:>9s}" for c in cols_med))
            cd = sub_df[sub_df.component == comp].sort_values("x_slice_idx")
            for _, r in cd.iterrows():
                xrange = f"[{r.x_lo_cm:+4.0f},{r.x_hi_cm:+4.0f}]"
                vals = "  ".join(
                    f"{r[c]:>+9.2f}" if np.isfinite(r[c]) else f"{'nan':>9s}"
                    for c in cols_med)
                print(f"   {int(r.x_slice_idx):>3d}  {xrange:>14s}  {vals}")


if __name__ == "__main__":
    main()
