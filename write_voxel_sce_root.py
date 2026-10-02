#!/usr/bin/env python3
"""Write an SBND SpaceCharge TH3F ROOT from the DIRECT voxel measurement
(`delta_mu_all`), i.e. the raw per-voxel rank-1 inversion -- no spline.

The companion to `fit_doubleanode_2d_spline.py --write-sce-root` (which
ships the smooth fitted map). This ships the measured voxel map sampled
onto the 49x49x61 SCE grid by trilinear interpolation of the voxel
centers, per TPC half (no cross-cathode blending). NaN voxels (un-inverted
LSQ cells) are first filled by iterative 26-neighbour averaging within the
half. Forward = the measured map; backward + E-field are copied from the
reference SCE ROOT (same as the forward-only spline ROOT). The per-TPC
X sign flip (x<0 -> store -Delta_x) matches the SpaceChargeServiceSBND
loader convention, identical to the spline writer.

Usage:
    python write_voxel_sce_root.py --input <maps>/voxel_data.npz \
        --sce-root-name sbnd_sce_fwd_voxel_doubleanode_2d_v10.root
"""

import argparse
import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import build_sce_input as B

COMPONENTS = ("dx", "dy", "dz")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="voxel_data.npz with delta_mu_all (nx,ny,nz,3)")
    ap.add_argument("--sce-outdir", default="sce_input_doubleanode_2d")
    ap.add_argument("--sce-root-name", required=True)
    ap.add_argument("--reference-root", default=B.REFERENCE_SCE_ROOT_DEFAULT)
    ap.add_argument("--fill-iters", type=int, default=30,
                    help="26-neighbour NaN-fill passes per TPC half")
    ap.add_argument("--cond-max", type=float, default=100.0,
                    help="drop voxels with LSQ cond above this (matches the "
                         "spline writer); set 0 to disable")
    ap.add_argument("--max-resolution-cm", type=float, default=2.0,
                    help="drop voxels with worst-axis 1-sigma above this; "
                         "set 0 to disable")
    args = ap.parse_args()
    os.makedirs(args.sce_outdir, exist_ok=True)

    d = np.load(args.input)
    mu = d["delta_mu_all"].astype(np.float64)          # (nx,ny,nz,3)
    sig = np.zeros_like(mu)
    vx = B.voxel_centers(d["x_edges"])
    vy = B.voxel_centers(d["y_edges"])
    vz = B.voxel_centers(d["z_edges"])
    n_fin = int(np.isfinite(mu[..., 0]).sum())
    print(f"loaded {args.input}: {mu.shape} voxels, {n_fin} finite", flush=True)

    # Same quality gate as the spline writer: reject ill-conditioned / poorly
    # resolved voxels (their raw inversion is garbage); they become holes to
    # be neighbour-filled below. This keeps the DIRECT measurement but drops
    # the LSQ-degenerate cells the spline fit also discards.
    if args.cond_max > 0 or args.max_resolution_cm > 0:
        cond = d["cond_all"]
        cov = d["delta_cov_all"]                       # xx,yy,zz,...
        res = np.sqrt(np.maximum(cov[..., :3], 0.0)).max(axis=-1)
        bad = np.zeros(mu.shape[:3], dtype=bool)
        if args.cond_max > 0:
            bad |= ~(np.isfinite(cond) & (cond <= args.cond_max))
        if args.max_resolution_cm > 0:
            bad |= ~(np.isfinite(res) & (res <= args.max_resolution_cm))
        n_drop = int((bad & np.isfinite(mu[..., 0])).sum())
        mu[bad] = np.nan
        print(f"  quality gate (cond<={args.cond_max:g}, "
              f"res<={args.max_resolution_cm:g} cm): dropped {n_drop} "
              f"voxels -> {int(np.isfinite(mu[...,0]).sum())} kept", flush=True)

    # Fill NaN holes separately within each TPC half so the fill never
    # averages across the cathode (x=0).
    filled = np.full_like(mu, np.nan)
    for tag, hmask in (("TPC0", vx < 0), ("TPC1", vx > 0)):
        sub = mu[hmask].copy()
        fsub, _, fmask, _ = B.iterative_neighbor_fill(
            sub, sig[hmask], n_iter=args.fill_iters)
        rem = int((~np.isfinite(fsub[..., 0])).sum())
        print(f"  {tag}: filled {int(fmask.sum())} holes, "
              f"{rem} still empty after {args.fill_iters} passes", flush=True)
        if rem:  # last resort: global-mean fill so the grid has no NaN
            for c in range(3):
                m = ~np.isfinite(fsub[..., c])
                fsub[..., c][m] = np.nanmean(fsub[..., c])
        filled[hmask] = fsub

    # Sample onto the SCE grid, per TPC half (trilinear, edge-clamped).
    sx, sy, sz = B.sce_bin_centers()
    fwd = np.zeros((B.SCE_NX, B.SCE_NY, B.SCE_NZ, 3), dtype=np.float64)
    for tag, vmask, gmask in (("TPC0", vx < 0, sx < 0),
                              ("TPC1", vx > 0, sx > 0)):
        vxx = vx[vmask]
        gx = sx[gmask]
        gxc = np.clip(gx, vxx.min(), vxx.max())        # clamp to voxel span
        gyc = np.clip(sy, vy.min(), vy.max())
        gzc = np.clip(sz, vz.min(), vz.max())
        pts = np.stack(np.meshgrid(gxc, gyc, gzc, indexing="ij"), -1)
        for c in range(3):
            interp = RegularGridInterpolator(
                (vxx, vy, vz), filled[vmask, :, :, c],
                method="linear", bounds_error=False, fill_value=None)
            fwd[np.where(gmask)[0], :, :, c] = interp(pts)

    # Same per-TPC X flip as the spline writer (loader un-flips for x<0).
    fwd[sx < 0, :, :, 0] *= -1.0
    for ci, comp in enumerate(COMPONENTS):
        g = fwd[..., ci]
        print(f"  SCE grid {comp}: range [{g.min():+.2f}, {g.max():+.2f}] cm",
              flush=True)

    xe = np.linspace(*B.SCE_X_EDGE, B.SCE_NX + 1)
    ye = np.linspace(*B.SCE_Y_EDGE, B.SCE_NY + 1)
    ze = np.linspace(*B.SCE_Z_EDGE, B.SCE_NZ + 1)
    out = os.path.join(args.sce_outdir, args.sce_root_name)
    B.write_th3f_root(out, fwd[..., 0], fwd[..., 1], fwd[..., 2],
                      xe, ye, ze, reference_root=args.reference_root)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
