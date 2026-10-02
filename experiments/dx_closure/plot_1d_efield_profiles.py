"""1D profiles of the inferred E-field distortion, vs the simulation truth.

For each E-field-distortion component (dEx, dEy, dEz) plotted against each
detector axis (x, y, z) -- 9 panels total -- the two non-plotted detector
axes are averaged over (nanmean +/- sem). Each panel carries solid measured
curves per TPC (TPC0 = x<0, TPC1 = x>0) and, by default, dashed "truth"
(space-charge-only) curves per TPC sampled from the nominal SCE simulation
ROOT (`True_ElecField_*`). Drawn in the raw detector (lab) frame.

Reads the E-field voxel npz written by build_efield_2d_voxels.py
(`dE_all` V/cm lab frame, `dE_frac_all` = dE/E0, `keep_mask`, edges).

The truth E-field is stored TPC-local (drift-axis component symmetric across
the cathode); it is converted back to the lab frame (local = sign*lab, with
sign +1 for x<0/TPC0, -1 for x>0/TPC1) to match the measured map, the inverse
of the convention written by fit_efield_2d_spline.py. The dual map's per-TPC
`*_E` (x<0) / `*_W` (x>0) halves are selected by the sign of x.

Usage:
    python plot_1d_efield_profiles.py \
        --input efield_2d_maps_20cm_v10c/voxel_data.npz \
        --out efield_profiles_1d_v10c --label "data v10c"
    # fractional dE/E0 instead of V/cm:
    python plot_1d_efield_profiles.py --input ... --frac
    # disable the truth overlay:
    python plot_1d_efield_profiles.py --input ... --no-truth
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import build_sce_input as B
import compare_2d_vs_truth as C

COMPONENTS = ("dEx", "dEy", "dEz")
AXES = ("x", "y", "z")
C_TPC0 = "#1f77b4"
C_TPC1 = "#d62728"


def sample_truth_efield(vx, vy, vz, truth_root, e0_vcm, want_frac):
    """Sample the nominal-SCE truth E-field at voxel centers -> (nx,ny,nz,3),
    converted to the lab frame and the requested unit (dE/E0 or V/cm).

    True_ElecField_{X,Y,Z} are stored as fractional dE/E0 in the TPC-local
    convention (local = sign*lab, sign +1 for x<0, -1 for x>0); we invert that
    to the lab frame so it overlays the measured `dE_*_all`. The dual map
    carries per-TPC `*_E` (x<0/TPC0) / `*_W` (x>0/TPC1) halves."""
    sx, sy, sz = B.sce_bin_centers()
    X, Y, Z = np.meshgrid(vx, vy, vz, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    shape = (len(vx), len(vy), len(vz))
    east = (np.asarray(vx) < 0)[:, None, None]      # TPC0
    sign = np.where(np.asarray(vx) < 0.0, 1.0, -1.0)[:, None, None]
    with uproot.open(truth_root) as f:
        keys = set(k.split(";")[0] for k in f.keys())
        dual = "True_ElecField_X_E" in keys
        truth = np.full((*shape, 3), np.nan)
        for c, comp in enumerate("XYZ"):
            if dual:
                gE = f[f"True_ElecField_{comp}_E"].values().astype(np.float64)
                gW = f[f"True_ElecField_{comp}_W"].values().astype(np.float64)
                local = np.where(east,
                                 C._interp(gE, sx, sy, sz, pts, shape),
                                 C._interp(gW, sx, sy, sz, pts, shape))
            else:
                g = f[f"True_ElecField_{comp}"].values().astype(np.float64)
                local = C._interp(g, sx, sy, sz, pts, shape)
            lab = sign * local                       # TPC-local -> lab frame
            truth[..., c] = lab
    if not want_frac:
        truth = truth * e0_vcm                       # dE/E0 -> V/cm
    return truth


def profile(arr3d, axis, region):
    """nanmean +/- sem of arr3d along `axis`, averaging the other two axes,
    restricted to a boolean `region` (broadcast to arr3d shape)."""
    a = np.where(region, arr3d, np.nan)
    other = tuple(ax for ax in (0, 1, 2) if ax != axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(a, axis=other)
        std = np.nanstd(a, axis=other)
    n = np.sum(np.isfinite(a), axis=other)
    sem = std / np.sqrt(np.maximum(n, 1))
    sem[n == 0] = np.nan
    return mean, sem


def panel(ax, comp_i, axis_i, dE, vx, vy, vz, unit, truth=None):
    centers = (vx, vy, vz)[axis_i]
    tpc0 = np.broadcast_to((vx < 0)[:, None, None], dE.shape[:3])
    tpc1 = np.broadcast_to((vx > 0)[:, None, None], dE.shape[:3])
    # truth (space-charge-only) expectation, dashed, per TPC
    if truth is not None:
        for region, color in ((tpc0, C_TPC0), (tpc1, C_TPC1)):
            tt, _ = profile(truth[..., comp_i], axis_i, region)
            ax.plot(centers, tt, color=color, lw=2.2, ls="--", zorder=3)
        ax.plot([], [], color=C_TPC0, ls="--", lw=2.2, label="truth TPC0")
        ax.plot([], [], color=C_TPC1, ls="--", lw=2.2, label="truth TPC1")
    for region, color, lab in ((tpc0, C_TPC0, "measured TPC0"),
                               (tpc1, C_TPC1, "measured TPC1")):
        mm, sem = profile(dE[..., comp_i], axis_i, region)
        ax.plot(centers, mm, color=color, lw=1.6, marker="o", ms=3,
                label=lab)
        ax.fill_between(centers, mm - sem, mm + sem, color=color, alpha=0.20,
                        lw=0)
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xlabel(f"{AXES[axis_i]} [cm]")
    ax.set_ylabel(f"$\\Delta E_{{{COMPONENTS[comp_i][2]}}}$ {unit}")
    ax.set_title(f"$\\Delta E_{{{COMPONENTS[comp_i][2]}}}$ vs {AXES[axis_i]}")
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",
                    default="efield_2d_maps_20cm_v10c/voxel_data.npz")
    ap.add_argument("--out", default="efield_profiles_1d_v10c")
    ap.add_argument("--frac", action="store_true",
                    help="plot fractional dE/E0 instead of lab-frame V/cm")
    ap.add_argument("--truth-root",
                    default="SCEoffsets_SBND_E500_dualmap_voxelTH3.root",
                    help="nominal SCE ROOT carrying True_ElecField_* (truth)")
    ap.add_argument("--no-truth", action="store_true",
                    help="omit the space-charge-only truth overlay")
    ap.add_argument("--label", default="",
                    help="extra text appended to the figure title")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.input)
    key = "dE_frac_all" if args.frac else "dE_all"
    unit = "$/E_0$" if args.frac else "[V/cm]"
    dE = np.array(d[key], dtype=float)            # (nx,ny,nz,3)
    keep = np.array(d["keep_mask"], dtype=bool)
    dE[~keep] = np.nan                            # drop un-inverted voxels
    vx = B.voxel_centers(d["x_edges"])
    vy = B.voxel_centers(d["y_edges"])
    vz = B.voxel_centers(d["z_edges"])
    e0 = float(d["E0_Vcm"])
    print(f"[efield] {args.input}: {int(keep.sum())} voxels, "
          f"plotting {'dE/E0' if args.frac else 'dE [V/cm]'}", flush=True)

    truth = None
    if not args.no_truth:
        truth = sample_truth_efield(vx, vy, vz, args.truth_root, e0, args.frac)
        print(f"  truth overlay from {args.truth_root}", flush=True)

    # ---- combined 3x3 grid (rows=component, cols=axis) ----
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for ci in range(3):
        for ai in range(3):
            panel(axes[ci][ai], ci, ai, dE, vx, vy, vz, unit, truth)
    axes[0][0].legend(loc="best", fontsize=9)
    title = ("double-anode inferred E-field distortion 1D profiles "
             "(solid=measured, dashed=truth; per TPC; non-plotted axes averaged)")
    if args.label:
        title += f"\n{args.label}"
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(args.out, "efield_profiles_grid.png"), dpi=120)
    fig.savefig(os.path.join(args.out, "efield_profiles_grid.pdf"))
    plt.close(fig)

    # ---- individual panels ----
    for ci in range(3):
        for ai in range(3):
            f, a = plt.subplots(figsize=(6.5, 5))
            panel(a, ci, ai, dE, vx, vy, vz, unit, truth)
            a.legend(loc="best", fontsize=9)
            f.tight_layout()
            tag = f"{COMPONENTS[ci]}_vs_{AXES[ai]}"
            f.savefig(os.path.join(args.out, f"prof_{tag}.png"), dpi=120)
            plt.close(f)

    print(f"wrote {args.out}/  (efield_profiles_grid.png + 9 prof_*.png)",
          flush=True)


if __name__ == "__main__":
    main()
