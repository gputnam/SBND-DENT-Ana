"""Knot-count scan for the 3-D doubleanode SCE forward-displacement
spline fit (build_sce_input.py).

Replicates the in-fit pipeline of build_sce_input.main() per-half but
without writing the ROOT file or the diagnostic plots — just runs the
ridge LSQ at a list of (nx, ny, nz) knot triples and reports:

    chi2 / dof per component on the fitted voxels
    component-wise RMS of (fit - measured) at the voxel centres
    SCE-grid eval range and the number of "out of physical scale" SCE
        pixels (|fit| > --sanity-clip cm; default 10 cm)
    elapsed seconds per fit

Writes a summary CSV and a small RMS vs knots plot.
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import build_sce_input as B

OUT_DIR = "/exp/sbnd/data/users/gputnam/efield-study/sce_input_doubleanode/knot_scan"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=B.NPZ_DEFAULT)
    ap.add_argument("--knots", type=str,
                    default="3,5,7;4,6,8;6,8,12;8,10,14;10,12,18;12,14,22",
                    help="semicolon-separated 'nx,ny,nz' triples")
    ap.add_argument("--ridge-lambda", type=float, default=1.0)
    ap.add_argument("--sigma-floor", type=float, default=0.5)
    ap.add_argument("--min-count", type=int, default=200)
    ap.add_argument("--cond-max", type=float, default=300.0)
    ap.add_argument("--k-mad", type=float, default=5.0)
    ap.add_argument("--abs-floor", type=float, default=2.0)
    ap.add_argument("--sanity-clip", type=float, default=10.0,
                    help="SCE-grid pixels with |fit| > this many cm flagged "
                         "as overshoot (default %(default)s)")
    args = ap.parse_args()

    pairs = []
    for tok in args.knots.split(";"):
        nx, ny, nz = tok.split(",")
        pairs.append((int(nx), int(ny), int(nz)))

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- load (mirror build_sce_input.main)
    d = np.load(args.npz)
    mu     = d["delta_mu_all"].astype(np.float64)
    cov    = d["delta_cov_all"].astype(np.float64)
    count  = d["count_all"]
    cond   = d["cond_all"]
    x_edges = d["x_edges"]; y_edges = d["y_edges"]; z_edges = d["z_edges"]
    vx_c = B.voxel_centers(x_edges); vy_c = B.voxel_centers(y_edges); vz_c = B.voxel_centers(z_edges)
    sigma_v = np.sqrt(np.maximum(cov[..., :3], 0.0))

    keep_mask, _ = B.x_neighbor_outlier_mask(
        mu, count, cond,
        min_count=args.min_count, cond_max=args.cond_max,
        k_mad=args.k_mad, abs_floor=args.abs_floor)
    mu_clean    = np.where(keep_mask[..., None], mu,      np.nan)
    sigma_clean = np.where(keep_mask[..., None], sigma_v, np.nan)

    tpc0_mask = vx_c < 0
    tpc1_mask = vx_c > 0

    def collect(half_mask):
        idx_x = np.where(half_mask)[0]
        x_v = np.repeat(vx_c[idx_x], len(vy_c) * len(vz_c))
        y_v = np.tile(np.repeat(vy_c, len(vz_c)), len(idx_x))
        z_v = np.tile(vz_c, len(idx_x) * len(vy_c))
        sub_mu = mu_clean   [idx_x].reshape(-1, 3)
        sub_sig = sigma_clean[idx_x].reshape(-1, 3)
        return x_v, y_v, z_v, sub_mu, sub_sig

    sx, sy, sz = B.sce_bin_centers()

    rows = []
    for tag, half_mask, xa, xb, anode_side, x_eval_mask in [
        ("TPC0", tpc0_mask, B.ANODE_X_TPC0, B.CATHODE_X, "low",  sx < 0),
        ("TPC1", tpc1_mask, B.CATHODE_X,    B.ANODE_X_TPC1, "high", sx > 0),
    ]:
        x_v, y_v, z_v, vals, sigs = collect(half_mask)
        n_voxels = int(np.isfinite(vals[:, 0]).sum())
        x_eval = sx[x_eval_mask]
        for (nx, ny, nz) in pairs:
            t0 = time.time()
            kx, ky, kz, coeffs, chi2 = B.fit_spline_half(
                x_v, y_v, z_v, vals, sigs,
                xa=xa, xb=xb, anode_side=anode_side,
                nx_int=nx, ny_int=ny, nz_int=nz,
                sigma_floor=args.sigma_floor,
                ridge_lambda=args.ridge_lambda)
            # eval the fit at voxel-center coords for residual RMS
            for c_idx, comp_name in enumerate(("dx", "dy", "dz")):
                vox_grid = np.full((len(vx_c), len(vy_c), len(vz_c)), np.nan)
                # only the half's x indices
                ix_half = np.where(half_mask)[0]
                grid = B.eval_spline_grid(vx_c[ix_half], vy_c, vz_c,
                                           kx, ky, kz, coeffs[c_idx])
                vox_grid[ix_half] = grid
                meas = mu_clean[..., c_idx]
                meas_half = np.where(half_mask[:, None, None], meas, np.nan)
                resid = vox_grid - meas_half
                rms = float(np.sqrt(np.nanmean(resid ** 2)))
                # SCE-grid eval
                sce_grid = B.eval_spline_grid(x_eval, sy, sz, kx, ky, kz, coeffs[c_idx])
                sce_min, sce_max = float(np.nanmin(sce_grid)), float(np.nanmax(sce_grid))
                n_overshoot = int(np.sum(np.abs(sce_grid) > args.sanity_clip))
                chi2_v, dof = chi2[c_idx]
                rows.append({
                    "half": tag, "nx": nx, "ny": ny, "nz": nz,
                    "component": comp_name,
                    "chi2": chi2_v, "dof": dof,
                    "rms_resid_cm": rms,
                    "sce_min": sce_min, "sce_max": sce_max,
                    "n_overshoot": n_overshoot,
                })
                print(f"  {tag} {nx}x{ny}x{nz} {comp_name}: "
                      f"chi2/dof = {chi2_v:.0f}/{dof}  RMS = {rms:.3f} cm  "
                      f"SCE [{sce_min:+.2f}, {sce_max:+.2f}]  "
                      f"|fit|>{args.sanity_clip:g}cm: {n_overshoot}",
                      flush=True)
            print(f"  {tag} {nx}x{ny}x{nz}: dt = {time.time()-t0:.1f}s",
                  flush=True)

    # save summary csv + plot
    import pandas as pd
    df = pd.DataFrame(rows)
    csv = os.path.join(OUT_DIR, "knot_scan.csv")
    df.to_csv(csv, index=False)
    print(f"\nwrote {csv}", flush=True)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, tag in zip(axs, ("TPC0", "TPC1")):
        sub = df[df.half == tag]
        for comp, marker in (("dx", "o"), ("dy", "s"), ("dz", "^")):
            ssub = sub[sub.component == comp]
            xs = [f"{r.nx}x{r.ny}x{r.nz}" for _, r in ssub.iterrows()]
            ax.plot(range(len(xs)), ssub.rms_resid_cm.values,
                    marker=marker, label=comp)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=20)
        ax.set_title(tag); ax.set_xlabel("nx x ny x nz")
        ax.set_ylabel("voxel RMS (fit - measured) [cm]")
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "knot_scan_rms.png"), dpi=130)
    plt.close(fig)
    print(f"wrote {OUT_DIR}/knot_scan_rms.png", flush=True)


if __name__ == "__main__":
    main()
