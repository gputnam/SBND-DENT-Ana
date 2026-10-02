#!/usr/bin/env python3
"""Globally-constrained spline solve of the double-anode displacement field.

The production pipeline inverts each voxel independently (3x3 LSQ on
M = Sigma n n^T, v = Sigma n m) and THEN fits a spline to the noisy
per-voxel map with diagonal weights. The drift component dx is weakly
constrained per voxel (small n_x leverage), so the per-voxel step amplifies
a coherent residual systematic ~2x and the previous mitigation (an
anisotropic x-ridge, lambda_x ~ 0.1) is an empirically-tuned damping.

This script replaces both steps with ONE constrained linear solve, directly
for the tensor-product cubic-B-spline coefficients c of the displacement
field Delta(r):

    chi^2(c) = Sigma_vox [ Delta(r_v)^T M_v Delta(r_v) - 2 Delta(r_v)^T v_v ]

(the exact per-hit chi^2 up to the voxel-center approximation already made
by the accumulator). The per-voxel M_v propagates the full y/z-x covariance
into the fit, voxels share leverage through the smooth basis, and -- the
physics -- hard boundary conditions pin the dx dome:

  * Delta = 0 (all components) at the anode face   [zero drift time;
    MicroBooNE laser paper arXiv:1910.01430 sec 5.2 uses the same condition]
  * Delta_x = 0 at the cathode face                 [self-enforced by the
    cathode-timing v_drift calibration: v = 4 m / <dt> puts full-drift
    charge exactly at the cathode; verified on MC in CATHODE_DRIFT_MC.md]

With clamped knots the boundary value is carried by exactly one x-basis
column, so each condition is exact column elimination (same mechanism as
build_sce_input.fit_spline_half's anode clamp).

Input is an accumulator dump (plot_doubleanode_2d_voxels.py
--dump-accumulator) or any npz carrying nnT/sum_nm/n + grid edges, e.g.
diagnose_dx_offsets.py's diagnose_dx.npz. Output spline_fit.npz is
compatible with compare_2d_vs_truth.py --use-spline.

Usage:
    python fit_global_spline_from_accum.py --accum accumulator_mc_subset.npz \
        --truth-root SCEoffsets_SBND_E500_dualmap_voxelTH3.root \
        --out global_spline_mc
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
import compare_2d_vs_truth as C
import explore_inversion as E

COMPONENTS = ("dx", "dy", "dz")
CATHODE_X = 0.0


def half_basis(vx, vy, vz, mask3, xa, xb, nx_int, ny_int, nz_int):
    """Dense tensor-product basis rows for the voxel centers selected by
    mask3 (3-D bool). Returns (B, kx, ky, kz, flat_idx) with B (Nv, K)."""
    kx = B.clamped_knots(xa, xb, nx_int)
    ky = B.clamped_knots(-200.0, 200.0, ny_int)
    kz = B.clamped_knots(0.0, 500.0, nz_int)
    ii, jj, kk = np.nonzero(mask3)
    Bx = B.univariate_basis(vx[ii], kx)
    By = B.univariate_basis(vy[jj], ky)
    Bz = B.univariate_basis(vz[kk], kz)
    Bv = np.einsum("va,vb,vc->vabc", Bx, By, Bz)
    Bv = Bv.reshape(len(ii), -1)
    return Bv, kx, ky, kz, (ii, jj, kk)


def solve_half(Mv, vv, Bv, nb, anode_idx, cathode_idx, cathode_bc, ridge):
    """Constrained normal-equation solve for one TPC half.

    Mv (Nv,3,3), vv (Nv,3), Bv (Nv,K); nb = (nx_b, ny_b, nz_b).
    Returns coeffs (3, nx_b, ny_b, nz_b)."""
    nx_b, ny_b, nz_b = nb
    K = nx_b * ny_b * nz_b
    A = np.zeros((3 * K, 3 * K))
    rhs = np.zeros(3 * K)
    for i in range(3):
        rhs[i * K:(i + 1) * K] = Bv.T @ vv[:, i]
        for j in range(i, 3):
            blk = Bv.T @ (Mv[:, i, j][:, None] * Bv)
            A[i * K:(i + 1) * K, j * K:(j + 1) * K] = blk
            if j > i:
                A[j * K:(j + 1) * K, i * K:(i + 1) * K] = blk.T

    # column elimination: x-basis index -> flat columns
    xidx = np.arange(K) // (ny_b * nz_b)
    keep = np.ones(3 * K, dtype=bool)
    for c in range(3):
        drop = (xidx == anode_idx)
        if cathode_bc and c == 0:
            drop |= (xidx == cathode_idx)
        keep[c * K:(c + 1) * K] = ~drop

    A_red = A[np.ix_(keep, keep)]
    rhs_red = rhs[keep]
    # relative ridge for numerical conditioning only
    lam = ridge * np.trace(A_red) / max(A_red.shape[0], 1)
    c_red = np.linalg.solve(A_red + lam * np.eye(A_red.shape[0]), rhs_red)
    c_full = np.zeros(3 * K)
    c_full[keep] = c_red
    return c_full.reshape(3, nx_b, ny_b, nz_b)


def fit_global(accum_npz, nx_int, ny_int, nz_int, min_counts, cathode_bc,
               ridge):
    """Fit both halves; returns dict with knots, coeffs, eval map, etc."""
    d = np.load(accum_npz)
    T = d["nnT"].astype(np.float64)
    v4 = d["sum_nm"].astype(np.float64)
    n = d["n"].astype(np.float64)
    vx = B.voxel_centers(d["x_edges"])
    vy = B.voxel_centers(d["y_edges"])
    vz = B.voxel_centers(d["z_edges"])
    nx, ny, nz = n.shape

    M = np.zeros((nx, ny, nz, 3, 3))
    M[..., 0, 0] = T[..., 0]; M[..., 1, 1] = T[..., 1]; M[..., 2, 2] = T[..., 2]
    M[..., 0, 1] = M[..., 1, 0] = T[..., 3]
    M[..., 0, 2] = M[..., 2, 0] = T[..., 4]
    M[..., 1, 2] = M[..., 2, 1] = T[..., 5]

    halves = {}
    spline_mu = np.full((nx, ny, nz, 3), np.nan)
    for t, (xa, xb) in enumerate(((-200.0, CATHODE_X), (CATHODE_X, 200.0))):
        inhalf = ((vx[:, None, None] >= xa) & (vx[:, None, None] < xb)
                  if t == 0 else
                  (vx[:, None, None] >= xa) & (vx[:, None, None] <= xb))
        mask3 = np.broadcast_to(inhalf, n.shape) & (n >= min_counts)
        Bv, kx, ky, kz, _ = half_basis(vx, vy, vz, mask3, xa, xb,
                                       nx_int, ny_int, nz_int)
        Mv = M[mask3]
        vv = v4[mask3]
        nb = (len(kx) - 4, len(ky) - 4, len(kz) - 4)
        # anode at low end of TPC0 half, high end of TPC1 half
        anode_idx = 0 if t == 0 else nb[0] - 1
        cathode_idx = nb[0] - 1 if t == 0 else 0
        coeffs = solve_half(Mv, vv, Bv, nb, anode_idx, cathode_idx,
                            cathode_bc, ridge)
        halves[t] = dict(kx=kx, ky=ky, kz=kz, coeffs=coeffs,
                         n_vox=int(mask3.sum()))
        # evaluate on the half's voxel centers
        hx = np.where((vx >= xa) & ((vx < xb) if t == 0 else (vx <= xb)))[0]
        for c in range(3):
            g = B.eval_spline_grid(vx[hx], vy, vz, kx, ky, kz, coeffs[c])
            spline_mu[hx, :, :, c] = g
    return dict(halves=halves, spline_mu=spline_mu, vx=vx, vy=vy, vz=vz,
                x_edges=d["x_edges"], y_edges=d["y_edges"],
                z_edges=d["z_edges"], n=n)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accum", default="accumulator_mc_subset.npz")
    ap.add_argument("--truth-root",
                    default="SCEoffsets_SBND_E500_dualmap_voxelTH3.root")
    ap.add_argument("--out", default="global_spline_mc")
    ap.add_argument("--nx-knots", type=int, default=5,
                    help="interior X knots per half (default %(default)s, "
                         "matching the production spatial spline)")
    ap.add_argument("--ny-knots", type=int, default=6)
    ap.add_argument("--nz-knots", type=int, default=9)
    ap.add_argument("--min-counts", type=int, default=30)
    ap.add_argument("--ridge", type=float, default=1e-8,
                    help="relative numerical ridge (x trace(A)/dim)")
    ap.add_argument("--no-cathode-bc", action="store_true",
                    help="drop the Delta_x=0 cathode boundary condition")
    ap.add_argument("--ridge-scan", action="store_true",
                    help="scan the numerical ridge to demonstrate "
                         "insensitivity")
    ap.add_argument("--knot-scan", action="store_true",
                    help="scan x-knot counts (dome resolution)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cathode_bc = not args.no_cathode_bc

    # ---- fit -----------------------------------------------------------------
    fit = fit_global(args.accum, args.nx_knots, args.ny_knots, args.nz_knots,
                     args.min_counts, cathode_bc, args.ridge)
    vx, vy, vz = fit["vx"], fit["vy"], fit["vz"]
    print(f"global constrained spline: knots ({args.nx_knots},{args.ny_knots},"
          f"{args.nz_knots}) interior/half, anode BC all comps, cathode BC "
          f"{'dx' if cathode_bc else 'none'}; voxels used "
          f"{fit['halves'][0]['n_vox']}+{fit['halves'][1]['n_vox']}",
          flush=True)

    # ---- score vs truth on the explore_inversion fixed voxel set -------------
    M, v2, n2, sh, _, _, _ = E.build_MV(args.accum)
    base = E.solve(M, v2, R=1e-6 * n2[:, None, None] * np.eye(3))
    cond = np.full(M.shape[0], np.inf)
    valid = n2 >= args.min_counts
    cond[valid] = np.linalg.cond(M[valid])
    truth = C.sample_truth_at(vx, vy, vz, args.truth_root,
                              unflip_x=False).reshape(-1, 3)
    keep = (valid & (cond <= 100.0) & np.isfinite(base).all(1)
            & np.isfinite(truth).all(1))
    print(f"voxels scored: {keep.sum()}", flush=True)

    delta = fit["spline_mu"].reshape(-1, 3)
    rows = {}
    for name, dd in (("baseline_pervoxel", base),
                     ("global_spline", delta)):
        mr = E.metrics(dd, truth, keep, vx, sh)
        rows[name] = mr
        print(f"\n{name}:")
        for cmp_ in COMPONENTS:
            s, r, _ = mr[(cmp_, "all")]
            s0, r0, _ = mr[(cmp_, "T0")]
            s1, r1, _ = mr[(cmp_, "T1")]
            print(f"  {cmp_}: slope {s:+.3f} rms {r:.3f} "
                  f"(T0 {s0:+.3f}/{r0:.3f}, T1 {s1:+.3f}/{r1:.3f})",
                  flush=True)

    # ---- ridge / knot scans ---------------------------------------------------
    if args.ridge_scan:
        print("\nridge scan (dx slope / rms):")
        for rg in (1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
            f2 = fit_global(args.accum, args.nx_knots, args.ny_knots,
                            args.nz_knots, args.min_counts, cathode_bc, rg)
            mr = E.metrics(f2["spline_mu"].reshape(-1, 3), truth, keep, vx, sh)
            s, r, _ = mr[("dx", "all")]
            print(f"  ridge {rg:.0e}: dx slope {s:+.3f} rms {r:.3f}",
                  flush=True)
    if args.knot_scan:
        print("\nx-knot scan (dx & dy slope / rms):")
        for nxk in (2, 3, 4, 5, 7, 9):
            f2 = fit_global(args.accum, nxk, args.ny_knots, args.nz_knots,
                            args.min_counts, cathode_bc, args.ridge)
            mr = E.metrics(f2["spline_mu"].reshape(-1, 3), truth, keep, vx, sh)
            sx, rx, _ = mr[("dx", "all")]
            sy, ry, _ = mr[("dy", "all")]
            print(f"  nx_int {nxk}: dx {sx:+.3f}/{rx:.3f}  "
                  f"dy {sy:+.3f}/{ry:.3f}", flush=True)

    # ---- npz (compare_2d_vs_truth --use-spline compatible) --------------------
    keep3 = keep.reshape(sh)
    np.savez_compressed(
        os.path.join(args.out, "spline_fit.npz"),
        spline_mu=fit["spline_mu"].astype(np.float32),
        keep_mask=keep3,
        x_edges=fit["x_edges"], y_edges=fit["y_edges"],
        z_edges=fit["z_edges"],
        kx_t0=fit["halves"][0]["kx"], ky_t0=fit["halves"][0]["ky"],
        kz_t0=fit["halves"][0]["kz"],
        coeffs_t0=fit["halves"][0]["coeffs"],
        kx_t1=fit["halves"][1]["kx"], ky_t1=fit["halves"][1]["ky"],
        kz_t1=fit["halves"][1]["kz"],
        coeffs_t1=fit["halves"][1]["coeffs"],
        cathode_bc=np.bool_(cathode_bc),
        nx_int=np.int32(args.nx_knots), ny_int=np.int32(args.ny_knots),
        nz_int=np.int32(args.nz_knots))
    print(f"\nsaved {args.out}/spline_fit.npz", flush=True)

    # ---- 1D dx(x) profile plot -------------------------------------------------
    t3 = truth.reshape(*sh, 3)
    d3 = fit["spline_mu"]
    b3 = base.reshape(*sh, 3)
    k3 = keep3
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    for ci, (ax, comp) in enumerate(zip(axes, COMPONENTS)):
        for arr, label, sty in ((b3, "per-voxel LSQ (baseline)", "s--"),
                                (d3, "global constrained spline", "o-"),
                                (t3, "truth (dual map)", "r-")):
            prof = np.nanmean(np.where(k3[..., None], arr, np.nan)[..., ci],
                              axis=(1, 2))
            if label.startswith("truth"):
                ax.plot(vx, prof, sty, lw=2.2, label=label)
            else:
                ax.plot(vx, prof, sty, ms=4, lw=1.3, label=label)
        ax.axhline(0, color="0.6", ls=":", lw=0.7)
        ax.set_xlabel("X [cm]")
        ax.set_ylabel(f"$\\Delta_{{{comp[1]}}}$ [cm]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("global constrained spline vs per-voxel LSQ vs truth "
                 "(y/z-averaged profiles)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "profiles_vs_x.png"), dpi=120)
    fig.savefig(os.path.join(args.out, "profiles_vs_x.pdf"))
    plt.close(fig)
    print(f"wrote {args.out}/", flush=True)


if __name__ == "__main__":
    main()
