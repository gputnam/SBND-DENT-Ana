#!/usr/bin/env python3
"""Per-hit diagnostics of the double-anode dx (drift-direction) measurement.

The 2D double-anode method over-measures dx ~2x on MC at the correct drift
velocity, while dy/dz close against the dual truth map. Each hit measures
m = n_hat . Delta with a small drift component n_x = -slope/denom, so the
LSQ dx is a leverage-weighted average that amplifies any coherent residual
systematic. This script opens up that middle step:

Pass 1 re-reads the parquet shards (bit-identical geometry via
plot_doubleanode_2d_voxels.shard_hits, including the production iteration-2
outlier rejection) and accumulates the per-voxel normal equations, from
which the robust (y,z)-block solution (Delta_y, Delta_z with dx ignored)
is taken.

Pass 2 re-reads the shards and, with (Delta_y, Delta_z) fixed, forms per hit
the yz-subtracted residual and the per-hit x-offset estimate

    resid_yz = m - n_y Delta_y - n_z Delta_z      (= n_x * dx + noise)
    x_est    = resid_yz / n_x                     (|n_x| > --nx-min only)

and accumulates:
  * per-voxel histograms of x_est        -> the "distribution of X-offsets
    in each voxel" (gallery + median/weighted-mean maps vs LSQ dx vs truth)
  * profile sums of resid_yz vs n_x split by (tpc, plane, x-column)
    -> per-plane dx(x) profiles: the slope of resid_yz vs n_x in an
    x-column IS the dx measured by that plane there. Plane disagreement
    localizes a plane-level systematic; agreement means all planes see the
    same (over-)measurement.
  * per-(tpc, plane) mean final-fit offset r = m - n_hat.Delta_final vs x
    -> a nonzero plane-dependent mean is a fake-Delta source.
  * 2D (n_x, resid_yz) histograms per (tpc, plane).

The per-voxel median of x_est plus the yz-block (Delta_y, Delta_z) is also
saved as `delta_median` -- the MicroBooNE-cosmics-style robust estimator
(arXiv:2008.09765 uses per-voxel medians) -- for scoring against truth.

Usage (MC subset):
    python diagnose_dx_offsets.py \
        --input /exp/sbnd/data/users/gputnam/efield-study/doubleanode_2d_df_mc \
        --maps doubleanode_2d_maps_20cm_mc/voxel_data.npz \
        --truth-root SCEoffsets_SBND_E500_dualmap_voxelTH3.root \
        --out diagnose_dx_mc
"""

import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import plot_doubleanode_2d_voxels as P2
from crtlib import voxel_invert

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import compare_2d_vs_truth as C

# n_x and resid_yz binning for the 2D coherence histograms
NX_RANGE = 0.8
NX_BIN = 0.02
RES_RANGE = 8.0
RES_BIN = 0.2

# worker context (set by _init)
_CTX = None


def _init(geom, params, ctx):
    P2._worker_init(geom, params)
    global _CTX
    _CTX = ctx


def _kept_hits(path):
    """shard_hits + the production iteration-2 outlier rejection (if the map
    npz carried an iteration-1 map), so the hit set matches the final map."""
    ix, iy, iz, mk, nk, _atick, stats, ext = P2.shard_hits(path, extra=True)
    delta_prev = _CTX.get("delta_prev")
    if delta_prev is not None and stats["n_hits"]:
        keep = P2._iter_keep_mask(ix, iy, iz, mk, nk, delta_prev,
                                  _CTX["iter_resid_cm"])
        ix, iy, iz = ix[keep], iy[keep], iz[keep]
        mk, nk = mk[keep], nk[keep]
        ext = {k: v[keep] for k, v in ext.items()}
        stats["n_hits"] = int(keep.sum())
    return ix, iy, iz, mk, nk, ext, stats


# ----------------------------------------------------------------------------
# pass 1: accumulate the per-voxel normal equations
# ----------------------------------------------------------------------------
def pass1_shard(path):
    nx, ny, nz = _CTX["shape"]
    acc = voxel_invert.make_scalar_accumulator(nx, ny, nz)
    ix, iy, iz, mk, nk, _ext, stats = _kept_hits(path)
    if stats["n_hits"]:
        voxel_invert.add_scalar_hits(acc, ix, iy, iz, mk, nk)
    return acc, stats


def solve_yz_block(acc, min_n):
    """Per-voxel (Delta_y, Delta_z) ignoring dx: solve the 2x2 (y,z) block of
    M = Sigma n n^T against v = Sigma n m. NaN below min_n / singular."""
    T = acc["nnT"]                                  # xx,yy,zz,xy,xz,yz
    v = acc["sum_nm"]
    n = acc["n"]
    A00 = T[..., 1]; A11 = T[..., 2]; A01 = T[..., 5]
    b0 = v[..., 1]; b1 = v[..., 2]
    det = A00 * A11 - A01 ** 2
    ok = (n >= min_n) & (np.abs(det) > 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        dy = np.where(ok, (A11 * b0 - A01 * b1) / det, np.nan)
        dz = np.where(ok, (A00 * b1 - A01 * b0) / det, np.nan)
    return dy, dz


# ----------------------------------------------------------------------------
# pass 2: x_est histograms + coherence profiles
# ----------------------------------------------------------------------------
def pass2_shard(path):
    nx, ny, nz = _CTX["shape"]
    nvox = nx * ny * nz
    nb_xe = _CTX["n_xe"]
    lo_xe = -_CTX["xest_range"]
    w_xe = _CTX["xest_bin"]
    n_nx = int(round(2 * NX_RANGE / NX_BIN))
    n_re = int(round(2 * RES_RANGE / RES_BIN))

    out = dict(
        xest_hist=np.zeros((nvox, nb_xe), dtype=np.int64),
        s_w=np.zeros(nvox), s_wx=np.zeros(nvox), s_r2yz=np.zeros(nvox),
        n_xest=np.zeros(nvox, dtype=np.int64),
        # (tpc, plane, x-column) profile sums of resid_yz vs n_x
        col_nx2=np.zeros((2, 3, nx)), col_nxr=np.zeros((2, 3, nx)),
        col_n=np.zeros((2, 3, nx), dtype=np.int64),
        # (tpc, plane, x-column) final-fit offset r sums
        col_r=np.zeros((2, 3, nx)), col_r2=np.zeros((2, 3, nx)),
        col_rn=np.zeros((2, 3, nx), dtype=np.int64),
        # (tpc, plane, n_x bin, resid bin) 2D histogram
        h2=np.zeros((2, 3, n_nx, n_re), dtype=np.int64),
    )
    ix, iy, iz, mk, nk, ext, stats = _kept_hits(path)
    if not stats["n_hits"]:
        return out, stats

    dy = _CTX["dyz"][0][ix, iy, iz]
    dz = _CTX["dyz"][1][ix, iy, iz]
    resid = mk - nk[:, 1] * dy - nk[:, 2] * dz
    nxh = nk[:, 0]
    tpc = ext["tpc"].astype(np.int64)
    pln = ext["plane"].astype(np.int64)
    okr = np.isfinite(resid)

    # --- per-voxel x_est ----------------------------------------------------
    base = ix * (ny * nz) + iy * nz + iz
    sel = okr & (np.abs(nxh) > _CTX["nx_min"])
    xest = resid[sel] / nxh[sel]
    bsel = base[sel]
    ib = np.floor((xest - lo_xe) / w_xe).astype(np.int64)
    inr = (ib >= 0) & (ib < nb_xe)
    out["xest_hist"] += np.bincount(
        bsel[inr] * nb_xe + ib[inr],
        minlength=nvox * nb_xe).reshape(nvox, nb_xe)
    out["n_xest"] += np.bincount(bsel, minlength=nvox)
    # weighted (w = n_x^2) sums over the |n_x| > nx_min selection:
    # w*x_est = n_x*resid, so s_wx/s_w is the conditional LSQ dx from the
    # selected (high-leverage) hits with (dy,dz) fixed from ALL hits --
    # with nx_min ~ 0.7 this is the "dx from high-|slope| tracks only"
    # estimator (the reco peak-time bias is flat there).
    out["s_w"] += np.bincount(bsel, weights=nxh[sel] ** 2,
                              minlength=nvox)
    out["s_wx"] += np.bincount(bsel, weights=nxh[sel] * resid[sel],
                               minlength=nvox)
    out["s_r2yz"] += np.bincount(base[okr], weights=resid[okr] ** 2,
                                 minlength=nvox)

    # --- (tpc, plane, x-column) profiles ------------------------------------
    colflat = (tpc * 3 + pln) * nx + ix
    ncol = 2 * 3 * nx
    out["col_nx2"] += np.bincount(colflat[okr], weights=nxh[okr] ** 2,
                                  minlength=ncol).reshape(2, 3, nx)
    out["col_nxr"] += np.bincount(colflat[okr],
                                  weights=nxh[okr] * resid[okr],
                                  minlength=ncol).reshape(2, 3, nx)
    out["col_n"] += np.bincount(colflat[okr],
                                minlength=ncol).reshape(2, 3, nx)

    # --- final-fit offset r per (tpc, plane, x-column) ----------------------
    proj = np.einsum("hc,hc->h", nk, _CTX["delta_final"][ix, iy, iz])
    r = mk - proj
    okf = np.isfinite(r)
    out["col_r"] += np.bincount(colflat[okf], weights=r[okf],
                                minlength=ncol).reshape(2, 3, nx)
    out["col_r2"] += np.bincount(colflat[okf], weights=r[okf] ** 2,
                                 minlength=ncol).reshape(2, 3, nx)
    out["col_rn"] += np.bincount(colflat[okf],
                                 minlength=ncol).reshape(2, 3, nx)

    # --- 2D (n_x, resid_yz) histogram per (tpc, plane) ----------------------
    inx = np.floor((nxh + NX_RANGE) / NX_BIN).astype(np.int64)
    ire = np.floor((resid + RES_RANGE) / RES_BIN).astype(np.int64)
    ok2 = okr & (inx >= 0) & (inx < n_nx) & (ire >= 0) & (ire < n_re)
    flat2 = ((tpc[ok2] * 3 + pln[ok2]) * n_nx + inx[ok2]) * n_re + ire[ok2]
    out["h2"] += np.bincount(flat2, minlength=2 * 3 * n_nx * n_re
                             ).reshape(2, 3, n_nx, n_re)
    return out, stats


def hist_median(hist, edges):
    """Median per row of `hist` (N, nbins), linear interpolation within the
    median bin. NaN for empty rows."""
    csum = np.cumsum(hist, axis=1).astype(float)
    tot = csum[:, -1]
    out = np.full(hist.shape[0], np.nan)
    ok = tot > 0
    half = 0.5 * tot[ok]
    idx = (csum[ok] >= half[:, None]).argmax(axis=1)
    below = np.where(idx > 0, csum[ok, np.maximum(idx - 1, 0)], 0.0)
    cnt = hist[ok, idx].astype(float)
    frac = np.where(cnt > 0, (half - below) / cnt, 0.5)
    w = edges[1] - edges[0]
    out[ok] = edges[idx] + frac * w
    return out


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------
def _gallery_fig(task):
    (xc, yc, zc, hist, centers, width, ann, png) = task
    h = hist.astype(float)
    tot = h.sum()
    if tot <= 0:
        return None
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.bar(centers, h, width=width, color="steelblue",
           edgecolor="navy", linewidth=0.3)
    ax.axvline(0.0, color="0.4", ls=":", lw=0.8)
    for val, color, label in ann:
        if np.isfinite(val):
            ax.axvline(val, color=color, ls="--", lw=1.0, label=label)
    ax.set_xlabel(r"per-hit $x_{\rm est}=({m-n_y\Delta_y-n_z\Delta_z})/n_x$"
                  " [cm]")
    ax.set_ylabel(f"hits / {width:g} cm")
    ax.set_ylim(bottom=0.0)
    ax.set_title(f"voxel  X={xc:+.0f}  Y={yc:+.0f}  Z={zc:+.0f} cm   "
                 f"N={int(tot):,}", fontsize=9)
    ax.legend(fontsize=6, loc="upper right")
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(png, dpi=90)
    plt.close(fig)
    return png


def plot_dx_profiles(vxc, col_nx2, col_nxr, dx_lsq_prof, dx_truth_prof, out):
    """Per-plane dx(x) = slope of resid_yz vs n_x per x-column, vs the truth
    dome and the production LSQ dx profile."""
    with np.errstate(divide="ignore", invalid="ignore"):
        dx_col = np.where(col_nx2 > 0, col_nxr / col_nx2, np.nan)
        comb = np.where(col_nx2.sum(1) > 0,
                        col_nxr.sum(1) / col_nx2.sum(1), np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    colors = ("tab:blue", "tab:orange", "tab:green")
    for t, ax in enumerate(axes):
        for p in range(3):
            ax.plot(vxc, dx_col[t, p], "o-", ms=3, lw=1, color=colors[p],
                    alpha=0.7, label=f"plane {p}"
                    + (" (coll.)" if p == 2 else " (ind.)"))
        ax.plot(vxc, comb[t], "k.-", ms=5, lw=1.8, label="3-plane combined")
        ax.plot(vxc, dx_lsq_prof[t], "s--", ms=4, lw=1.2, color="purple",
                alpha=0.8, label="production LSQ dx")
        ax.plot(vxc, dx_truth_prof[t], "r-", lw=2.2, label="truth (dual map)")
        ax.axhline(0, color="0.5", ls=":", lw=0.7)
        ax.set_xlabel("X [cm]")
        ax.set_title(f"TPC{t}  ({'x<0' if t == 0 else 'x>0'})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$\Delta_x$ [cm]")
    fig.suptitle("double-anode dx vs X: per-plane column slope of "
                 r"$(m-n_y\Delta_y-n_z\Delta_z)$ vs $n_x$, vs truth")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "dx_vs_x_per_plane.png"), dpi=120)
    fig.savefig(os.path.join(out, "dx_vs_x_per_plane.pdf"))
    plt.close(fig)
    return dx_col, comb


def plot_resid_vs_nx(h2, out):
    """2D (n_x, resid_yz) histogram + profile per (tpc, plane)."""
    n_nx = h2.shape[2]
    nx_edges = np.linspace(-NX_RANGE, NX_RANGE, n_nx + 1)
    nx_c = 0.5 * (nx_edges[:-1] + nx_edges[1:])
    re_edges = np.linspace(-RES_RANGE, RES_RANGE,  h2.shape[3] + 1)
    re_c = 0.5 * (re_edges[:-1] + re_edges[1:])
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True, sharey=True)
    for t in (0, 1):
        for p in (0, 1, 2):
            ax = axes[t, p]
            H = h2[t, p].astype(float)
            ax.imshow(H.T, origin="lower", aspect="auto",
                      extent=(nx_edges[0], nx_edges[-1],
                              re_edges[0], re_edges[-1]),
                      cmap="viridis",
                      norm=matplotlib.colors.LogNorm(vmin=1,
                                                     vmax=max(1, H.max())))
            tot = H.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                prof = (H * re_c[None, :]).sum(axis=1) / tot
            okp = tot > 200
            ax.plot(nx_c[okp], prof[okp], "w.-", ms=3, lw=1.2)
            s_nx2 = (tot * nx_c ** 2).sum()
            slope = ((tot * nx_c * prof)[okp].sum()
                     / (tot * nx_c ** 2)[okp].sum()) if s_nx2 > 0 else np.nan
            ax.plot(nx_c, slope * nx_c, "r--", lw=1.2)
            ax.axhline(0, color="0.7", ls=":", lw=0.6)
            ax.set_title(f"TPC{t} plane{p}: slope (=dx) {slope:+.2f} cm",
                         fontsize=10)
            if t == 1:
                ax.set_xlabel(r"$n_x$")
            if p == 0:
                ax.set_ylabel(r"$m-n_y\Delta_y-n_z\Delta_z$ [cm]")
    fig.suptitle("yz-subtracted residual vs drift-direction leverage "
                 r"$n_x$ (white: profile mean; red: fitted slope)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "resid_vs_nx.png"), dpi=120)
    fig.savefig(os.path.join(out, "resid_vs_nx.pdf"))
    plt.close(fig)


def plot_plane_offset_vs_x(vxc, col_r, col_r2, col_rn, out):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    colors = ("tab:blue", "tab:orange", "tab:green")
    for t, ax in enumerate(axes):
        for p in range(3):
            n = col_rn[t, p].astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(n > 0, col_r[t, p] / n, np.nan)
                rms = np.sqrt(np.maximum(
                    col_r2[t, p] / np.maximum(n, 1) - mean ** 2, 0))
                err = rms / np.sqrt(np.maximum(n, 1))
            ax.errorbar(vxc, mean, yerr=err, fmt="o-", ms=3, lw=1,
                        color=colors[p], alpha=0.8, label=f"plane {p}")
        ax.axhline(0, color="0.5", ls=":", lw=0.7)
        ax.set_xlabel("X [cm]")
        ax.set_title(f"TPC{t}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"mean $r = m-\hat n\cdot\hat\Delta_{\rm final}$ [cm]")
    fig.suptitle("per-plane mean final-fit offset vs X "
                 "(nonzero plane splitting = fake-$\\Delta$ source)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "plane_offset_vs_x.png"), dpi=120)
    fig.savefig(os.path.join(out, "plane_offset_vs_x.pdf"))
    plt.close(fig)


def plot_xest_maps(vxc, vyc, vzc, dx_med, dx_lsq, dx_truth, out, vrange):
    """Per-X-slice 4-pane [median x_est | LSQ dx | truth dx | median-truth]."""
    mdir = os.path.join(out, "xest_maps")
    os.makedirs(mdir, exist_ok=True)
    extent = (vzc[0] - 10, vzc[-1] + 10, vyc[0] - 10, vyc[-1] + 10)
    cmap = matplotlib.colormaps["RdYlBu_r"].copy()
    cmap.set_bad("white")
    for ix in range(len(vxc)):
        panes = ((r"median $x_{\rm est}$", dx_med[ix]),
                 ("LSQ dx (production)", dx_lsq[ix]),
                 ("truth dx (dual map)", dx_truth[ix]),
                 (r"median $x_{\rm est}$ - truth", dx_med[ix] - dx_truth[ix]))
        if not any(np.isfinite(g).any() for _, g in panes):
            continue
        fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
        for ax, (label, grid) in zip(axes, panes):
            im = ax.imshow(grid, origin="lower", aspect="equal",
                           extent=extent, cmap=cmap, vmin=-vrange,
                           vmax=vrange, interpolation="nearest")
            fig.colorbar(im, ax=ax, extend="both", label="[cm]")
            ax.set_title(label)
            ax.set_xlabel("Z [cm]"); ax.set_ylabel("Y [cm]")
        fig.suptitle(f"dx estimators, X in [{vxc[ix]-10:+.0f}, "
                     f"{vxc[ix]+10:+.0f}] cm")
        fig.tight_layout()
        fig.savefig(os.path.join(mdir, f"x{int(round(vxc[ix])):+04d}.png"),
                    dpi=110)
        plt.close(fig)


def plot_leverage(acc, vxc, out):
    """Leverage diagnostics from the accumulated normal equations."""
    n = acc["n"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        nx2 = np.where(n > 0, acc["nnT"][..., 0] / n, np.nan)   # <n_x^2>
        ny2 = np.where(n > 0, acc["nnT"][..., 1] / n, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ok = np.isfinite(nx2) & (n >= 30)
    axes[0].hist(nx2[ok].ravel(), bins=60, color="steelblue")
    axes[0].set_xlabel(r"per-voxel $\langle n_x^2\rangle$")
    axes[0].set_title(f"median {np.nanmedian(nx2[ok]):.3f}")
    axes[1].hist((nx2[ok] / ny2[ok]).ravel(), bins=60, range=(0, 2),
                 color="steelblue")
    axes[1].set_xlabel(r"$\langle n_x^2\rangle/\langle n_y^2\rangle$")
    axes[1].set_title(f"median {np.nanmedian(nx2[ok]/ny2[ok]):.3f}")
    prof = np.nanmean(np.where(ok, nx2, np.nan), axis=(1, 2))
    axes[2].plot(vxc, prof, "o-")
    axes[2].set_xlabel("X [cm]")
    axes[2].set_ylabel(r"$\langle n_x^2\rangle$ (column mean)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("drift-direction leverage of the per-voxel LSQ")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "leverage.png"), dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    import multiprocessing as _mp
    try:
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",
                    default="/exp/sbnd/data/users/gputnam/efield-study/"
                            "doubleanode_2d_df_mc")
    ap.add_argument("--maps", default="doubleanode_2d_maps_20cm_mc/voxel_data.npz")
    ap.add_argument("--truth-root",
                    default="SCEoffsets_SBND_E500_dualmap_voxelTH3.root")
    ap.add_argument("--out", default="diagnose_dx_mc")
    ap.add_argument("--wire-geom", default=P2.WIRE_GEOM)
    ap.add_argument("--anode-cut-config", default=P2.ANODE_CUT_CONFIG)
    ap.add_argument("--max-r-cm", type=float, default=25.0)
    ap.add_argument("--anode-dt-window", type=float, default=40.0)
    ap.add_argument("--max-anode-fom-cm", type=float, default=None)
    ap.add_argument("--nx-min", type=float, default=0.05,
                    help="min |n_x| for a hit to enter the x_est histogram")
    ap.add_argument("--xest-range", type=float, default=15.0)
    ap.add_argument("--xest-bin", type=float, default=0.25)
    ap.add_argument("--min-counts", type=int, default=30)
    ap.add_argument("--min-xest", type=int, default=20,
                    help="min x_est entries for a per-voxel median")
    ap.add_argument("--vrange", type=float, default=2.0,
                    help="color range of the dx estimator maps [cm]")
    ap.add_argument("--accum-nproc", type=int, default=4)
    ap.add_argument("--nproc", default="auto")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--no-gallery", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- map npz: grid, build params, final + iter1 maps --------------------
    d = np.load(args.maps)
    delta_final = d["delta_mu_all"].astype(np.float64)
    nx, ny, nz, _ = delta_final.shape
    x_edges, y_edges, z_edges = d["x_edges"], d["y_edges"], d["z_edges"]
    vxc = 0.5 * (x_edges[:-1] + x_edges[1:])
    vyc = 0.5 * (y_edges[:-1] + y_edges[1:])
    vzc = 0.5 * (z_edges[:-1] + z_edges[1:])
    size = float(d["voxel_size"])
    vdrift = float(d["vdrift"])
    t0 = float(d["anode_t0"])
    delta_prev = (d["delta_mu_all_iter1"].astype(np.float64)
                  if "delta_mu_all_iter1" in d else None)
    iter_resid = float(d["iter_resid_cm"]) if "iter_resid_cm" in d else 4.0

    max_fom = args.max_anode_fom_cm
    if max_fom is None:
        try:
            max_fom = float(json.load(open(args.anode_cut_config))["cut_cm"])
        except Exception:
            max_fom = 5.0
    geom = P2.load_wire_geom(args.wire_geom)
    params = dict(size=size, t0=t0, vdrift=vdrift, max_r_cm=args.max_r_cm,
                  max_anode_fom=max_fom, anode_dt_window=args.anode_dt_window)

    if os.path.isdir(args.input):
        shards = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    else:
        shards = sorted(glob.glob(args.input))
    if args.max_shards:
        shards = shards[:args.max_shards]
    if not shards:
        print(f"no parquet shards at {args.input}", file=sys.stderr)
        sys.exit(1)

    nb_xe = int(round(2 * args.xest_range / args.xest_bin))
    xe_edges = np.linspace(-args.xest_range, args.xest_range, nb_xe + 1)
    xe_centers = 0.5 * (xe_edges[:-1] + xe_edges[1:])

    print(f"dx diagnostics: {len(shards)} shards from {args.input}")
    print(f"  build: vdrift={vdrift} t0={t0} max_r={args.max_r_cm} "
          f"fom={max_fom:.3f} dt={args.anode_dt_window} "
          f"iter2-reject={'on (' + str(iter_resid) + ' cm)' if delta_prev is not None else 'off'}",
          flush=True)

    ctx = dict(shape=(nx, ny, nz), delta_prev=delta_prev,
               iter_resid_cm=iter_resid, nx_min=args.nx_min,
               xest_range=args.xest_range, xest_bin=args.xest_bin,
               n_xe=nb_xe)

    # ---- pass 1: normal equations -> yz-block solution ----------------------
    acc = voxel_invert.make_scalar_accumulator(nx, ny, nz)
    n_hits = 0
    print("\n[pass 1/2] accumulating per-voxel normal equations", flush=True)
    if args.accum_nproc > 1:
        with Pool(args.accum_nproc, initializer=_init,
                  initargs=(geom, params, ctx)) as pool:
            for a, st in tqdm(pool.imap_unordered(pass1_shard, shards),
                              total=len(shards), unit="shard",
                              file=sys.stderr):
                voxel_invert.combine_scalar_accumulators(acc, a)
                n_hits += st["n_hits"]
    else:
        _init(geom, params, ctx)
        for path in tqdm(shards, unit="shard", file=sys.stderr):
            a, st = pass1_shard(path)
            voxel_invert.combine_scalar_accumulators(acc, a)
            n_hits += st["n_hits"]
    print(f"  hits: {n_hits:,}", flush=True)

    dyz = solve_yz_block(acc, args.min_counts)
    n_yz = int(np.isfinite(dyz[0]).sum())
    print(f"  yz-block solved voxels: {n_yz:,}", flush=True)

    # cross-check the re-derived accumulator against the production map: the
    # full 3x3 solve here should reproduce delta_mu_all
    res_chk = voxel_invert.invert_scalar_accumulator(
        acc, min_n=args.min_counts, cond_max=100.0)
    both = (np.isfinite(res_chk["delta_mu"][..., 0])
            & np.isfinite(delta_final[..., 0]))
    if both.sum():
        dd = res_chk["delta_mu"][both] - delta_final[both]
        print(f"  re-derived LSQ vs production map: max|diff| = "
              f"{np.abs(dd).max():.4f} cm over {both.sum():,} voxels "
              f"(should be ~0 if build params match)", flush=True)

    # ---- pass 2: x_est + profiles -------------------------------------------
    ctx2 = dict(ctx)
    ctx2["dyz"] = dyz
    ctx2["delta_final"] = delta_final
    print("\n[pass 2/2] x_est histograms + coherence profiles", flush=True)
    tot = None
    if args.accum_nproc > 1:
        with Pool(args.accum_nproc, initializer=_init,
                  initargs=(geom, params, ctx2)) as pool:
            for out_s, _st in tqdm(pool.imap_unordered(pass2_shard, shards),
                                   total=len(shards), unit="shard",
                                   file=sys.stderr):
                if tot is None:
                    tot = out_s
                else:
                    for k in tot:
                        tot[k] += out_s[k]
    else:
        _init(geom, params, ctx2)
        for path in tqdm(shards, unit="shard", file=sys.stderr):
            out_s, _st = pass2_shard(path)
            if tot is None:
                tot = out_s
            else:
                for k in tot:
                    tot[k] += out_s[k]

    nvox = nx * ny * nz
    # ---- estimators ----------------------------------------------------------
    dx_med_f = hist_median(tot["xest_hist"], xe_edges)
    dx_med_f[tot["n_xest"] < args.min_xest] = np.nan
    dx_med = dx_med_f.reshape(nx, ny, nz)
    with np.errstate(invalid="ignore", divide="ignore"):
        dx_wmean = np.where(tot["s_w"] > 0, tot["s_wx"] / tot["s_w"],
                            np.nan).reshape(nx, ny, nz)
    # mask to inverted voxels
    inverted = np.isfinite(delta_final[..., 0])
    dx_med[~inverted] = np.nan
    dx_wmean[~inverted] = np.nan
    dx_lsq = delta_final[..., 0].copy()

    truth = C.sample_truth_at(vxc, vyc, vzc, args.truth_root, unflip_x=False)
    dx_truth = truth[..., 0].copy()
    dx_truth[~inverted] = np.nan

    # delta_median: robust estimator map for scoring (S3)
    delta_median = np.stack([dx_med, dyz[0], dyz[1]], axis=-1)
    delta_median[~inverted] = np.nan

    # ---- per-x-column LSQ/truth profiles for the dx_vs_x plot ---------------
    tmask = inverted
    dx_lsq_prof = np.full((2, nx), np.nan)
    dx_truth_prof = np.full((2, nx), np.nan)
    for ix in range(nx):
        t = 0 if vxc[ix] < 0 else 1
        sl = np.where(tmask[ix], dx_lsq[ix], np.nan)
        st_ = np.where(tmask[ix], dx_truth[ix], np.nan)
        dx_lsq_prof[t, ix] = np.nanmean(sl)
        dx_truth_prof[t, ix] = np.nanmean(st_)

    # ---- summary table -------------------------------------------------------
    print("\nglobal dx slope estimates (cm) per TPC / plane "
          "[slope of resid_yz vs n_x]:", flush=True)
    print(f"  {'':6s} {'plane0':>8s} {'plane1':>8s} {'plane2':>8s} "
          f"{'combined':>9s} {'truth-rms-x':>12s}")
    for t in (0, 1):
        vals = [tot["col_nxr"][t, p].sum() / tot["col_nx2"][t, p].sum()
                for p in range(3)]
        comb = tot["col_nxr"][t].sum() / tot["col_nx2"][t].sum()
        tr = np.sqrt(np.nanmean(dx_truth_prof[t] ** 2))
        print(f"  TPC{t:1d}  {vals[0]:>8.3f} {vals[1]:>8.3f} {vals[2]:>8.3f} "
              f"{comb:>9.3f} {tr:>12.3f}", flush=True)

    # ---- npz -----------------------------------------------------------------
    np.savez_compressed(
        os.path.join(args.out, "diagnose_dx.npz"),
        x_edges=x_edges, y_edges=y_edges, z_edges=z_edges,
        xest_edges=xe_edges, nx_min=np.float32(args.nx_min),
        xest_hist=tot["xest_hist"].reshape(nx, ny, nz, nb_xe).astype(np.int32),
        n_xest=tot["n_xest"].reshape(nx, ny, nz),
        dx_median=dx_med, dx_wmean=dx_wmean, dx_lsq=dx_lsq,
        dx_truth=dx_truth, dy_yzblock=dyz[0], dz_yzblock=dyz[1],
        delta_median=delta_median.astype(np.float32),
        s_w=tot["s_w"].reshape(nx, ny, nz),
        s_wx=tot["s_wx"].reshape(nx, ny, nz),
        s_r2yz=tot["s_r2yz"].reshape(nx, ny, nz),
        col_nx2=tot["col_nx2"], col_nxr=tot["col_nxr"],
        col_n=tot["col_n"], col_r=tot["col_r"], col_r2=tot["col_r2"],
        col_rn=tot["col_rn"], h2=tot["h2"],
        nnT=acc["nnT"], sum_nm=acc["sum_nm"], sum_m2=acc["sum_m2"],
        n=acc["n"],
        vdrift=np.float32(vdrift), anode_t0=np.float32(t0))
    print(f"\nsaved {args.out}/diagnose_dx.npz", flush=True)

    # ---- plots ---------------------------------------------------------------
    nproc = cpu_count() if args.nproc == "auto" else int(args.nproc)
    plot_dx_profiles(vxc, tot["col_nx2"], tot["col_nxr"],
                     dx_lsq_prof, dx_truth_prof, args.out)
    plot_resid_vs_nx(tot["h2"], args.out)
    plot_plane_offset_vs_x(vxc, tot["col_r"], tot["col_r2"], tot["col_rn"],
                           args.out)
    plot_xest_maps(vxc, vyc, vzc, dx_med, dx_lsq, dx_truth, args.out,
                   args.vrange)
    plot_leverage(acc, vxc, args.out)

    # ---- per-voxel gallery ----------------------------------------------------
    if not args.no_gallery:
        gdir = os.path.join(args.out, "xest_gallery")
        hist4 = tot["xest_hist"].reshape(nx, ny, nz, nb_xe)
        tasks = []
        for ix in range(nx):
            sub = os.path.join(gdir, f"x{int(round(vxc[ix])):+04d}")
            made = False
            for iy in range(ny):
                for iz in range(nz):
                    if not inverted[ix, iy, iz]:
                        continue
                    if hist4[ix, iy, iz].sum() < args.min_xest:
                        continue
                    if not made:
                        os.makedirs(sub, exist_ok=True)
                        made = True
                    ann = ((dx_lsq[ix, iy, iz], "purple", "LSQ dx"),
                           (dx_med[ix, iy, iz], "darkorange", "median"),
                           (dx_truth[ix, iy, iz], "red", "truth"))
                    tasks.append((vxc[ix], vyc[iy], vzc[iz],
                                  hist4[ix, iy, iz], xe_centers,
                                  args.xest_bin, ann,
                                  os.path.join(
                                      sub, f"y{int(round(vyc[iy])):+04d}"
                                           f"_z{int(round(vzc[iz])):+04d}.png")))
        print(f"\nrendering {len(tasks):,} per-voxel x_est figures "
              f"on {nproc} workers", flush=True)
        if nproc > 1:
            with Pool(nproc) as pool:
                for _ in tqdm(pool.imap_unordered(_gallery_fig, tasks,
                                                  chunksize=16),
                              total=len(tasks), unit="fig", file=sys.stderr):
                    pass
        else:
            for t_ in tasks:
                _gallery_fig(t_)

    print(f"\nwrote {args.out}/", flush=True)


if __name__ == "__main__":
    main()
