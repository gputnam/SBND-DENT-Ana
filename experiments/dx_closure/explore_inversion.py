"""Explore inversion / regularization strategies to reduce the dx measurement
bias of the 2D double-anode method, given its weak x-direction handle.

The double-anode tracks are near-parallel to drift, so the per-voxel normal
matrix M = Sigma n_hat n_hat^T is weakly constrained in x; the straight
anode-to-anode reference biases dx, and the weak x-leverage amplifies it
(measured dx ~ 2-3x truth). M and v = Sigma n_hat m are the sufficient
statistics, dumped by `plot_doubleanode_2d_voxels.py --dump-accumulator`, so
any LINEAR regularization can be tested offline without re-reading hits.

Strategies (each a per-voxel solve delta = (M + R)^-1 v, or an SVD variant):
  baseline      R = 1e-6 n I             (production: essentially none)
  iso_ridge     R = lam n I              (isotropic Tikhonov, scan lam)
  x_ridge       R = diag(lamx n, 0, 0)   (penalize ONLY dx, scan lamx)
  svd_floor     floor eigenvalues to f*emax (scan f)
  svd_trunc     drop the smallest eigenvalue direction (rank-2 pinv)
  dx_zero       solve the 2x2 (y,z) block, dx := 0   (reference bound)

Metric vs the dual-map truth (the as-applied per-TPC field): per component the
robust slope k = <m.t>/<t.t> (ideal 1.0) and RMS(meas-truth), over the SAME
voxel set (baseline cond<=100, n>=min). Goal: dx slope -> 1 and dx RMS down,
WITHOUT harming dy/dz.

Usage:
    python explore_inversion.py --accum accumulator_mc_subset.npz \
        --truth-root SCEoffsets_SBND_E500_dualmap_voxelTH3.root --out explore_inversion_mc
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import build_sce_input as B
import compare_2d_vs_truth as C

COMPONENTS = ("dx", "dy", "dz")


def build_MV(accum):
    d = np.load(accum)
    T = d["nnT"].astype(np.float64)           # (nx,ny,nz,6) xx,yy,zz,xy,xz,yz
    v = d["sum_nm"].astype(np.float64)        # (nx,ny,nz,3)
    n = d["n"].astype(np.float64)             # (nx,ny,nz)
    sh = n.shape
    nvox = n.size
    M = np.zeros((nvox, 3, 3))
    Tf = T.reshape(nvox, 6)
    M[:, 0, 0] = Tf[:, 0]; M[:, 1, 1] = Tf[:, 1]; M[:, 2, 2] = Tf[:, 2]
    M[:, 0, 1] = M[:, 1, 0] = Tf[:, 3]
    M[:, 0, 2] = M[:, 2, 0] = Tf[:, 4]
    M[:, 1, 2] = M[:, 2, 1] = Tf[:, 5]
    return (M, v.reshape(nvox, 3), n.reshape(nvox), sh,
            B.voxel_centers(d["x_edges"]), B.voxel_centers(d["y_edges"]),
            B.voxel_centers(d["z_edges"]))


def solve(M, v, R=None):
    """Per-voxel solve delta=(M+R)^-1 v; NaN where singular."""
    A = M if R is None else M + R
    out = np.full((M.shape[0], 3), np.nan)
    try:
        out = np.linalg.solve(A, v[..., None])[..., 0]
    except np.linalg.LinAlgError:
        for i in range(M.shape[0]):
            try:
                out[i] = np.linalg.solve(A[i], v[i])
            except np.linalg.LinAlgError:
                pass
    return out


def solve_svd(M, v, floor_frac=None, truncate=False):
    """SVD/eigen solve: floor eigenvalues to floor_frac*emax, or drop the
    smallest eigenvalue (truncate)."""
    w, Q = np.linalg.eigh(M)                   # ascending eigenvalues
    wmax = w[:, -1:]
    winv = np.zeros_like(w)
    if truncate:
        keep = np.zeros_like(w, dtype=bool)
        keep[:, 1:] = True                     # drop smallest
        winv[keep] = 1.0 / w[keep]
    else:
        wf = np.maximum(w, floor_frac * wmax)
        winv = np.where(wf > 0, 1.0 / wf, 0.0)
    # delta = Q diag(winv) Q^T v
    vt = np.einsum("vij,vi->vj", Q, v)         # Q^T v
    xt = winv * vt
    return np.einsum("vij,vj->vi", Q, xt)


def solve_yz(M, v):
    """dx := 0; solve the 2x2 (y,z) block."""
    out = np.full((M.shape[0], 3), np.nan)
    A = M[:, 1:, 1:]; b = v[:, 1:]
    try:
        sol = np.linalg.solve(A, b[..., None])[..., 0]
        out[:, 1:] = sol; out[:, 0] = 0.0
    except np.linalg.LinAlgError:
        pass
    return out


def metrics(delta, truth, keep, vx, sh):
    """Per-component slope and RMS(meas-truth) over kept voxels, all + per TPC."""
    nx = sh[0]
    tpc = np.broadcast_to((np.repeat(vx, sh[1] * sh[2]) < 0), (delta.shape[0],))
    rows = {}
    for ci, c in enumerate(COMPONENTS):
        m = delta[:, ci]; t = truth[:, ci]
        for reg, msk in (("all", keep), ("T0", keep & tpc), ("T1", keep & ~tpc)):
            mm, tt = m[msk], t[msk]
            ok = np.isfinite(mm) & np.isfinite(tt)
            mm, tt = mm[ok], tt[ok]
            sig = np.abs(tt) > 0.3
            slope = (np.sum(mm[sig] * tt[sig]) / np.sum(tt[sig] ** 2)
                     if sig.sum() > 5 else np.nan)
            rms = np.sqrt(np.mean((mm - tt) ** 2)) if mm.size else np.nan
            rows[(c, reg)] = (slope, rms, mm.size)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accum", default="accumulator_mc_subset.npz")
    ap.add_argument("--truth-root",
                    default="SCEoffsets_SBND_E500_dualmap_voxelTH3.root")
    ap.add_argument("--out", default="explore_inversion_mc")
    ap.add_argument("--min-counts", type=int, default=30)
    ap.add_argument("--cond-max", type=float, default=100.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    M, v, n, sh, vx, vy, vz = build_MV(args.accum)
    truth4 = C.sample_truth_at(vx, vy, vz, args.truth_root, unflip_x=False)
    truth = truth4.reshape(-1, 3)

    # fixed voxel set: baseline cond<=cond_max & n>=min (so all strategies are
    # scored on the same voxels)
    base = solve(M, v, R=1e-6 * n[:, None, None] * np.eye(3))
    cond = np.full(M.shape[0], np.inf)
    valid = n >= args.min_counts
    cond[valid] = np.linalg.cond(M[valid])
    keep = valid & (cond <= args.cond_max) & np.isfinite(base).all(1) \
        & np.isfinite(truth).all(1)
    print(f"voxels scored: {keep.sum()}", flush=True)

    nI = n[:, None, None] * np.eye(3)
    ex = np.zeros((M.shape[0], 3, 3)); ex[:, 0, 0] = n   # x-only penalty
    strategies = [("baseline", lambda: base)]
    for lam in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        strategies.append((f"iso_ridge_{lam}", (lambda l: (lambda: solve(M, v, l * nI)))(lam)))
    for lx in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        strategies.append((f"x_ridge_{lx}", (lambda l: (lambda: solve(M, v, l * ex)))(lx)))
    for f in (0.05, 0.1, 0.2, 0.35, 0.5):
        strategies.append((f"svd_floor_{f}", (lambda ff: (lambda: solve_svd(M, v, floor_frac=ff)))(f)))
    strategies.append(("svd_trunc", lambda: solve_svd(M, v, truncate=True)))
    strategies.append(("dx_zero", lambda: solve_yz(M, v)))

    records = []
    for name, fn in strategies:
        delta = fn()
        mr = metrics(delta, truth, keep, vx, sh)
        rec = {"strategy": name}
        for c in COMPONENTS:
            rec[f"{c}_slope"] = mr[(c, "all")][0]
            rec[f"{c}_rms"] = mr[(c, "all")][1]
            rec[f"{c}_rms_T0"] = mr[(c, "T0")][1]
            rec[f"{c}_rms_T1"] = mr[(c, "T1")][1]
            rec[f"{c}_slope_T0"] = mr[(c, "T0")][0]
            rec[f"{c}_slope_T1"] = mr[(c, "T1")][0]
        records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(args.out, "strategy_metrics.csv"), index=False)
    pd.set_option("display.width", 200)
    show = df[["strategy", "dx_slope", "dx_rms", "dy_slope", "dy_rms",
               "dz_slope", "dz_rms"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"),
          flush=True)

    # plot: dx slope & dx RMS vs strength for the ridge families
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for fam, marker in (("iso_ridge", "o"), ("x_ridge", "s"),
                        ("svd_floor", "^")):
        sub = df[df.strategy.str.startswith(fam)]
        strengths = [float(s.rsplit("_", 1)[1]) for s in sub.strategy]
        axes[0].plot(strengths, sub.dx_slope, marker=marker, label=fam)
        axes[1].plot(strengths, sub.dx_rms, marker=marker, label=fam)
    for ax, ttl, yl in ((axes[0], "dx slope (ideal 1)", "dx slope"),
                        (axes[1], "dx RMS(meas-truth)", "dx RMS [cm]")):
        ax.set_xscale("log"); ax.set_xlabel("regularization strength")
        ax.set_ylabel(yl); ax.set_title(ttl); ax.grid(alpha=0.3); ax.legend()
    axes[0].axhline(1.0, color="k", ls=":")
    bl = df[df.strategy == "baseline"].iloc[0]
    for ax, key in ((axes[0], "dx_slope"), (axes[1], "dx_rms")):
        ax.axhline(bl[key], color="gray", ls="--", alpha=0.6, label="baseline")
    fig.suptitle("Reducing dx bias via regularization (dy/dz preserved — see CSV)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "dx_bias_vs_regularization.png"), dpi=120)
    plt.close(fig)
    print(f"\nwrote {args.out}/", flush=True)


if __name__ == "__main__":
    main()
