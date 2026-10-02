#!/usr/bin/env python3
"""1D profile comparison (measurement vs expectation) for EVERY dx-campaign
variant, one figure per recipe.

For each campaign run (and each offline post-processing variant -- bare /
calibrated x-ridge / calibrated scale, reconstructed exactly as in
rank_dx_interventions.py: tuned on the odd-z voxel half, validated on
even-z), draws the y/z-averaged dx, dy, dz profiles vs X of the measured
map against the as-applied truth (`delta_applied` of
build_apparent_truth_map.py). Output: one PNG per recipe in --out, named
<run>__<variant>.png, annotated with the validated dx slope/RMS.

Usage:
    python make_campaign_profiles.py --out campaign_profiles
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import build_sce_input as B
import rank_dx_interventions as R

COMPONENTS = ("dx", "dy", "dz")

RUNS = [
    ("R0", "no intervention", "campaign/R0"),
    ("R1_cut08", "X-only cut tan#theta_xw>=0.8", "campaign/R1_cut08"),
    ("R1b_cut12", "X-only cut tan#theta_xw>=1.2", "campaign/R1b_cut12"),
    ("R2_lut1", "1-angle hit-time LUT", "campaign/R2_lut1"),
    ("R3_lut2", "2-angle hit-time LUT", "campaign/R3_lut2"),
    ("R4_detmean", "global per-track detrend (mean)", "campaign/R4_detmean"),
    ("R5_detlin", "global per-track detrend (linear)", "campaign/R5_detlin"),
    ("R6_det_cut", "detrend mean + cut 0.8 (X-only)", "campaign/R6_det_cut"),
    ("R6b_detl_cut", "detrend linear + cut 0.8 (X-only)",
     "campaign/R6b_detl_cut"),
    ("R7_lut2_cut", "2-angle LUT + cut 0.8", "campaign/R7_lut2_cut"),
    ("R8_det_lut2", "detrend mean + 2-angle LUT", "campaign/R8_det_lut2"),
    ("R9_ramp_hits", "in-situ ramp correction, hits mode (4 iter)",
     "campaign/R9_ramp_hits"),
    ("R0sub", "no intervention (production subset)", "campaign/R0sub"),
    ("R1sub", "cut 0.8 (production subset)", "campaign/R1sub"),
    ("FULLcut08", "cut 0.8 (FULL statistics)",
     "doubleanode_2d_maps_20cm_mc_full_thxw08"),
    ("FULLnocut", "no cut (FULL statistics, pre-chord-fix)",
     "doubleanode_2d_maps_20cm_mc_full"),
]


def profile(arr, keep):
    return np.nanmean(np.where(keep, arr, np.nan), axis=(1, 2))


def make_fig(vx, mu, truth, keep4, title, metrics_txt, path):
    keep = keep4[..., 0]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for c, (ax, comp) in enumerate(zip(axes, COMPONENTS)):
        ax.plot(vx, profile(mu[..., c], keep), "o-", ms=3.5, lw=1.2,
                color="navy", label="measured")
        ax.plot(vx, profile(truth[..., c], keep), "r-", lw=2.0,
                label="expectation (as-applied field)")
        ax.axhline(0, color="0.6", ls=":", lw=0.7)
        ax.axvline(0, color="0.85", lw=0.6)
        ax.set_xlabel("X [cm]")
        ax.set_ylabel(f"$\\Delta_{{{comp[1]}}}$ [cm]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{title}\n{metrics_txt}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_diff_fig(vx, mu, truth, keep4, title, path):
    """(measurement - expectation) vs X per component; the dx panel gets
    per-TPC linear fits to quantify the (linear-in-drift) excess."""
    keep = keep4[..., 0]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for c, (ax, comp) in enumerate(zip(axes, COMPONENTS)):
        diff = profile(mu[..., c], keep) - profile(truth[..., c], keep)
        ax.plot(vx, diff, "o-", ms=3.5, lw=1.2, color="navy",
                label="measured - expectation")
        if c == 0:
            for tsel, color, lab in ((vx < 0, "tab:orange", "TPC0"),
                                     (vx > 0, "tab:green", "TPC1")):
                m = tsel & np.isfinite(diff)
                if m.sum() > 3:
                    b, a = np.polyfit(vx[m], diff[m], 1)
                    ax.plot(vx[m], a + b * vx[m], "--", lw=1.6, color=color,
                            label=f"{lab}: {b*100:+.2f} cm/m, "
                                  f"{a:+.2f} cm @ x=0")
        ax.axhline(0, color="0.6", ls=":", lw=0.7)
        ax.axvline(0, color="0.85", lw=0.6)
        ax.set_xlabel("X [cm]")
        ax.set_ylabel(f"$\\Delta_{{{comp[1]}}}$ (meas - exp) [cm]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", default="apparent_truth_map_mc_v2.npz")
    ap.add_argument("--min-counts", type=int, default=30)
    ap.add_argument("--out", default="campaign_profiles")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t = np.load(args.truth)
    truth = t["delta_applied"].astype(np.float64)
    sh = truth.shape[:3]
    odd, even = R.halves(sh)
    vx = B.voxel_centers(t["x_edges"])

    nfig = 0
    for label, desc, rdir in RUNS:
        npz = os.path.join(rdir, "voxel_data.npz")
        if not os.path.exists(npz):
            print(f"skip {label}: no {npz}")
            continue
        d = np.load(npz)
        mu = d["delta_mu_all"].astype(np.float64)
        cnt = d["count_all"]
        keep = (np.isfinite(mu).all(-1) & np.isfinite(truth).all(-1)
                & (cnt >= args.min_counts))
        keep4 = keep[..., None]
        dump_path = os.path.join(rdir, "accum.npz")
        dump = R.load_dump(dump_path) if os.path.exists(dump_path) else None

        variants = {}
        variants["bare"] = (mu, "")
        # in-situ ramp correction (data-applicable)
        try:
            import fit_dx_ramp as FR
            rrf = FR.fit_ramp_npz(npz, min_counts=args.min_counts)
            eps_f = np.nan_to_num(np.array(
                [rrf["fits"][0].get("eps", np.nan),
                 rrf["fits"][1].get("eps", np.nan)]))
            vx_l = B.voxel_centers(d["x_edges"])
            mu_p = mu.copy()
            mu_p[..., 0] = mu[..., 0] - FR.ramp_grid(vx_l, eps_f)[:, None,
                                                                  None]
            variants[f"ramp_e{eps_f[0]*100:+.2f}_{eps_f[1]*100:+.2f}"] = \
                (mu_p, "")
        except Exception:
            pass
        # calibrated scale
        s_odd, _, _ = R.slope_rms(mu[..., 0], truth[..., 0], keep & odd, 0.15)
        if np.isfinite(s_odd) and abs(s_odd) > 0.05:
            mu_s = mu.copy()
            mu_s[..., 0] = mu[..., 0] / s_odd
            variants[f"scale_k{1.0/s_odd:.3f}"] = (mu_s, "")
        # calibrated x-ridge
        if dump is not None:
            sl = []
            for lam in R.RIDGE_GRID:
                dxl = R.solve_dx_ridge(dump, mu[..., 1], mu[..., 2], lam,
                                       args.min_counts)
                s, _, _ = R.slope_rms(dxl, truth[..., 0], keep & odd, 0.15)
                sl.append(s)
            sl = np.array(sl)
            okl = np.isfinite(sl)
            if okl.sum() > 2 and sl[okl].min() < 1.0 < sl[okl].max():
                lam_star = float(np.interp(1.0, sl[okl][::-1],
                                           R.RIDGE_GRID[okl][::-1]))
                mu_r = mu.copy()
                mu_r[..., 0] = R.solve_dx_ridge(dump, mu[..., 1], mu[..., 2],
                                                lam_star, args.min_counts)
                variants[f"ridge_lamx{lam_star:.3f}"] = (mu_r, "")

        for vname, (mv, _) in variants.items():
            s_v, r_v, _ = R.slope_rms(mv[..., 0], truth[..., 0],
                                      keep & even, 0.15)
            sy, ry, _ = R.slope_rms(mv[..., 1], truth[..., 1],
                                    keep & even, 0.3)
            sz, rz, _ = R.slope_rms(mv[..., 2], truth[..., 2],
                                    keep & even, 0.3)
            mtxt = (f"validated: dx slope {s_v:+.3f}, rms {r_v:.3f} cm | "
                    f"dy {sy:+.3f}/{ry:.3f} | dz {sz:+.3f}/{rz:.3f} | "
                    f"{int(keep.sum())} voxels")
            path = os.path.join(args.out, f"{label}__{vname}.png")
            make_fig(vx, mv, truth, keep4, f"{label}: {desc} [{vname}]",
                     mtxt, path)
            make_diff_fig(vx, mv, truth, keep4,
                          f"{label}: {desc} [{vname}] -- "
                          f"measurement - expectation\n{mtxt}",
                          os.path.join(args.out,
                                       f"{label}__{vname}__diff.png"))
            nfig += 2
    print(f"wrote {nfig} figures to {args.out}/", flush=True)


if __name__ == "__main__":
    main()
