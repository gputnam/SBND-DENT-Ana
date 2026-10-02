"""1D profiles of the measured distortion vs the true (simulation) distortion.

For each distortion component (dx, dy, dz) plotted against each detector axis
(x, y, z) -- 9 panels total -- the two non-plotted detector axes are averaged
over. Every panel carries FOUR curves (solid = measured, dashed = expected;
colour = TPC): measured+truth for TPC0 (x<0) and measured+truth for TPC1
(x>0). The expectation is drawn PER TPC because the per-TPC (dual) map
differs between the two drift volumes; for a single symmetric map the two
truth curves coincide.

Drift-aligned frame
-------------------
The SCE field obeys Delta_x(-x,y,z) = -Delta_x(x,y,z) (anti-symmetric) and
Delta_{y,z}(-x,y,z) = +Delta_{y,z}(x,y,z) (symmetric). To make the truth one
shared curve, each component is brought into a TPC-local "drift-aligned"
frame: for components whose truth is anti-symmetric across the cathode the
TPC1 (x>0) values are sign-flipped -- applied IDENTICALLY to measurement and
truth so their agreement is preserved. The per-component flip is auto-detected
from the truth's reflection symmetry and printed. `--frame detector` disables
the flip (raw detector coordinates; the truth dx then cancels when averaged
over x).

Measurement and truth use the same convention as compare_2d_vs_truth.py
(measured `delta_mu_all` matches the truth's RAW stored TrueFwd_X; no un-flip).

Usage:
    python plot_1d_profiles_vs_truth.py --input doubleanode_2d_maps_20cm_mc/voxel_data.npz
    python plot_1d_profiles_vs_truth.py --input .../voxel_data.npz \
        --use-spline doubleanode_2d_spline_20cm_mc/spline_fit.npz --out prof_mc_spline
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sce_input as B
import compare_2d_vs_truth as C

COMPONENTS = ("dx", "dy", "dz")
AXES = ("x", "y", "z")
C_MEAS0 = "#1f77b4"   # measured TPC0
C_MEAS1 = "#d62728"   # measured TPC1
C_TRUTH = "black"     # expectation


def detect_flip(truth, vx):
    """Per-component sign that maps TPC1 onto TPC0 (so truth is equivalent).

    Returns array sign[c] in {+1,-1}; -1 means the component is anti-symmetric
    across the cathode and TPC1 must be flipped to align with TPC0."""
    tpc1 = vx > 0
    signs = np.ones(3)
    for c in range(3):
        t = truth[..., c]
        mirror = t[::-1, :, :]          # x -> -x
        a, b = t[tpc1].ravel(), mirror[tpc1].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() and np.nansum(a[ok] * b[ok]) < 0:
            signs[c] = -1.0
    return signs


def apply_alignment(arr, vx, signs):
    """Flip the TPC1 (x>0) values of anti-symmetric components in place-safe."""
    out = arr.copy()
    tpc1 = vx > 0
    for c in range(3):
        if signs[c] < 0:
            out[tpc1, :, :, c] *= -1.0
    return out


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


def panel(ax, comp_i, axis_i, meas, truth, vx, vy, vz):
    centers = (vx, vy, vz)[axis_i]
    tpc0 = np.broadcast_to((vx < 0)[:, None, None], meas.shape[:3])
    tpc1 = np.broadcast_to((vx > 0)[:, None, None], meas.shape[:3])

    # Per-TPC expectation: the dual map differs between TPCs, so draw the
    # truth restricted to each TPC (dashed). For a symmetric single map the
    # two truth curves coincide.
    for region, color in ((tpc0, C_MEAS0), (tpc1, C_MEAS1)):
        tt, _ = profile(truth[..., comp_i], axis_i, region)
        ax.plot(centers, tt, color=color, lw=2.2, ls="--", zorder=3)
    # measured per TPC (solid + markers)
    for region, color, lab in ((tpc0, C_MEAS0, "TPC0"),
                               (tpc1, C_MEAS1, "TPC1")):
        mm, sem = profile(meas[..., comp_i], axis_i, region)
        ax.plot(centers, mm, color=color, lw=1.6, marker="o", ms=3,
                label=f"measured {lab}")
        ax.fill_between(centers, mm - sem, mm + sem, color=color, alpha=0.20,
                        lw=0)
    # legend proxies for the dashed truth curves
    ax.plot([], [], color=C_MEAS0, ls="--", lw=2.2, label="truth TPC0")
    ax.plot([], [], color=C_MEAS1, ls="--", lw=2.2, label="truth TPC1")

    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xlabel(f"{AXES[axis_i]} [cm]")
    ax.set_ylabel(f"$\\Delta_{{{COMPONENTS[comp_i][1]}}}$ [cm]")
    ax.set_title(f"$\\Delta_{{{COMPONENTS[comp_i][1]}}}$ vs {AXES[axis_i]}")
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="doubleanode_2d_maps_20cm_mc/voxel_data.npz")
    ap.add_argument("--truth-root", default=B.REFERENCE_SCE_ROOT_DEFAULT)
    ap.add_argument("--use-spline", default=None,
                    help="compare a fitted spline_fit.npz instead of raw voxels")
    ap.add_argument("--out", default="profiles_1d_mc")
    ap.add_argument("--cond-max", type=float, default=100.0)
    ap.add_argument("--max-resolution-cm", type=float, default=2.0)
    ap.add_argument("--min-counts", type=int, default=30)
    ap.add_argument("--frame", choices=("drift", "detector"), default="drift")
    ap.add_argument("--label", default="",
                    help="extra text appended to the figure title")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.use_spline:
        meas, vx, keep, st = C.load_measured_spline(args.use_spline)
        d = np.load(args.input)
        vy = B.voxel_centers(d["y_edges"]); vz = B.voxel_centers(d["z_edges"])
        print(f"[spline] {args.use_spline}: {st['n_keep']} voxels kept")
    else:
        meas, vx, vy, vz, st = C.load_measured_voxels(
            args.input, args.cond_max, args.max_resolution_cm, args.min_counts)
        print(f"[voxel] {args.input}: {st['n_keep']}/{st['n_inverted']} kept")
    truth = C.sample_truth_at(vx, vy, vz, args.truth_root, unflip_x=False)

    if args.frame == "drift":
        signs = detect_flip(truth, vx)
        print("  drift-aligned flips (TPC1):",
              {COMPONENTS[c]: ("flip" if signs[c] < 0 else "keep")
               for c in range(3)})
        meas = apply_alignment(meas, vx, signs)
        truth = apply_alignment(truth, vx, signs)
    else:
        print("  detector frame: no per-TPC alignment")

    # ---- combined 3x3 grid (rows=component, cols=axis) ----
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for ci in range(3):
        for ai in range(3):
            panel(axes[ci][ai], ci, ai, meas, truth, vx, vy, vz)
    axes[0][0].legend(loc="best", fontsize=9)
    title = ("double-anode measured vs truth 1D profiles "
             f"({args.frame}-aligned; non-plotted axes averaged)")
    if args.label:
        title += f"\n{args.label}"
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(args.out, "profiles_grid.png"), dpi=120)
    fig.savefig(os.path.join(args.out, "profiles_grid.pdf"))
    plt.close(fig)

    # ---- individual panels ----
    for ci in range(3):
        for ai in range(3):
            f, a = plt.subplots(figsize=(6.5, 5))
            panel(a, ci, ai, meas, truth, vx, vy, vz)
            a.legend(loc="best", fontsize=9)
            f.tight_layout()
            tag = f"{COMPONENTS[ci]}_vs_{AXES[ai]}"
            f.savefig(os.path.join(args.out, f"prof_{tag}.png"), dpi=120)
            plt.close(f)

    print(f"wrote {args.out}/  (profiles_grid.png + 9 prof_*.png)", flush=True)


if __name__ == "__main__":
    main()
