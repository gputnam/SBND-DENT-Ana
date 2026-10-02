#!/usr/bin/env python3
"""Build the SBND electric-field distortion voxel map from the v5 2D
double-anode *spatial* distortion map, using the MicroBooNE technique
(arXiv:2008.09765, section 4.5).

Method
------
The space-charge effect distorts both the reconstructed positions of drift
charge and the drift field itself. The spatial map `delta_mu_all` (cm) is the
forward displacement Delta = x_reco - x_true. Reconstruction places charge in
x from its drift time at the uniform nominal velocity v0; tracing the distorted
drift trajectory shows the local drift-velocity vector is

    v = +/- v0 * ( xhat - dDelta/dx )          (+ TPC1, - TPC0)

with the lab-x derivative g = dDelta/dx. Per voxel:

    norm  = sqrt( (1-gx)^2 + gy^2 + gz^2 )
    speed = v0 * norm
    |E|   = E(speed)        via the SBND-adapted MicroBooNE v(E)  [drift_model]

The field is (anti)parallel to the drift velocity. In the lab frame the
nominal field is E0 = (+500,0,0) V/cm in TPC0 and (-500,0,0) in TPC1, and the
distortion DeltaE = E - E0 works out to

    DeltaE = sign * ( |E|(1-gx)/norm - E0,  -|E|gy/norm,  -|E|gz/norm )

with sign = +1 for TPC0 (x<0), -1 for TPC1 (x>0). This is stored lab-frame,
the same convention `delta_mu_all` uses, so the divergence (check_gauss_2d.py)
and the SCE ROOT writer get a proper lab-coordinate field.

dDelta/dx is taken analytically from the fitted *spatial* B-spline (smooth;
the raw-voxel finite difference amplifies the noisy double-anode Delta_x) --
`--dx-method finitediff` is a noisy cross-check fallback.

Usage
-----
    python build_efield_2d_voxels.py
    python build_efield_2d_voxels.py --dx-method finitediff --no-gif
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sce_input as B
import drift_model as DM
import fit_doubleanode_2d_spline as F
from stitch_gifs import stitch_one

NPZ_DEFAULT = "doubleanode_2d_maps_20cm/voxel_data.npz"
KNOTS_DEFAULT = "doubleanode_2d_spline_20cm/spline_knots.json"
OUTDIR_DEFAULT = "efield_2d_maps_20cm"

Y_RANGE = (-200.0, 200.0)
Z_RANGE = (0.0, 500.0)
E0_VCM = DM.SBND_E0_VCM        # 500 V/cm nominal drift field
COMPONENTS = ("dEx", "dEy", "dEz")
GRADS = ("gx", "gy", "gz")


# --------------------------------------------------------------------------
def gradient_from_spline(mu_clean, sigma_clean, vx_c, vy_c, vz_c,
                         nx_k, ny_k, nz_k, ridge_lambda, sigma_floor):
    """g = dDelta/dx at every voxel center, from the analytic x-derivative
    of the spatial cubic B-spline (re-fit in-memory). Shape (nx,ny,nz,3)."""
    halves = F.fit_both_halves(vx_c, vy_c, vz_c, mu_clean, sigma_clean,
                               nx_k, ny_k, nz_k, ridge_lambda, sigma_floor)
    g = np.full(mu_clean.shape, np.nan)
    for tag, (kx, ky, kz, coeffs, _) in halves.items():
        ix = np.where(vx_c < 0)[0] if tag == "TPC0" else np.where(vx_c > 0)[0]
        for c in range(3):
            g[ix, :, :, c] = B.eval_spline_deriv(
                vx_c[ix], vy_c, vz_c, kx, ky, kz, coeffs[c], dx=1)
    return g


def gradient_from_finitediff(mu_clean, vx_c):
    """g = dDelta/dx by central finite difference of the raw voxel map,
    computed per TPC half so it never straddles the cathode. Noisy."""
    g = np.full(mu_clean.shape, np.nan)
    dx = float(vx_c[1] - vx_c[0])
    for ix in (np.where(vx_c < 0)[0], np.where(vx_c > 0)[0]):
        for c in range(3):
            g[ix, :, :, c] = np.gradient(mu_clean[ix, :, :, c], dx, axis=0)
    return g


def invert_to_efield(g, vx_c, v0):
    """MicroBooNE inversion: g -> (dE_all, Emag, speed) in the lab frame.

    Returns dE_all (nx,ny,nz,3) V/cm, Emag (nx,ny,nz) V/cm, speed cm/us."""
    gx, gy, gz = g[..., 0], g[..., 1], g[..., 2]
    norm = np.sqrt((1.0 - gx) ** 2 + gy ** 2 + gz ** 2)
    speed = v0 * norm                                   # cm/us
    Emag = 1000.0 * DM.E_of_v_array(speed)              # V/cm (NaN if extreme)

    dE_drift = Emag * (1.0 - gx) / norm - E0_VCM        # field-strength change
    dE_perp_y = Emag * (-gy) / norm
    dE_perp_z = Emag * (-gz) / norm
    sign = np.where(vx_c < 0.0, 1.0, -1.0)[:, None, None]   # +TPC0, -TPC1
    dE_all = np.stack([sign * dE_drift,
                       sign * dE_perp_y,
                       sign * dE_perp_z], axis=-1)
    return dE_all.astype(np.float64), Emag, speed


# --------------------------------------------------------------------------
def _heat_worker(task):
    """Render one per-X-slice heatmap (png+pdf). `grid` indexed [iy, iz]."""
    grid, vmin, vmax, cmap_name, extend, clabel, title, png, pdf = task
    cmap = matplotlib.colormaps[cmap_name].copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid, origin="lower", aspect="equal",
                   extent=(Z_RANGE[0], Z_RANGE[1], Y_RANGE[0], Y_RANGE[1]),
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.colorbar(im, ax=ax, extend=extend, label=clabel)
    ax.set_xlabel("Z [cm]")
    ax.set_ylabel("Y [cm]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    fig.savefig(pdf)
    plt.close(fig)
    return png


def make_map_set(arr, vx_c, outdir, sub, comp_names, *, vrange, center,
                 cmap, label_tmpl, title_tmpl, symmetric):
    """Queue per-X-slice heatmaps for every component of `arr` (nx,ny,nz,nc).
    Returns {comp: [png paths in X order]} and the task list."""
    by_comp = {c: [] for c in comp_names}
    tasks = []
    vmin = center - vrange
    vmax = center + vrange
    extend = "both"
    for ci, comp in enumerate(comp_names):
        pdir = os.path.join(outdir, sub, comp, "png")
        qdir = os.path.join(outdir, sub, comp, "pdf")
        os.makedirs(pdir, exist_ok=True)
        os.makedirs(qdir, exist_ok=True)
        for ix, xc in enumerate(vx_c):
            tpc = "TPC0" if xc < 0 else "TPC1"
            tag = f"x{int(round(xc)):+04d}"
            png = os.path.join(pdir, f"{tag}.png")
            pdf = os.path.join(qdir, f"{tag}.pdf")
            grid = arr[ix, :, :, ci] if arr.ndim == 4 else arr[ix]
            sc = comp[-1]
            title = (f"{title_tmpl.format(comp=comp, sub=sc)}\n"
                     f"X = {xc:+.0f} cm ({tpc})")
            tasks.append((grid.astype(float), vmin, vmax, cmap, extend,
                          label_tmpl.format(comp=comp, sub=sc),
                          title, png, pdf))
            by_comp[comp].append(png)
    return by_comp, tasks


def build_gifs(by_comp, outdir, sub, stack_name, label):
    """Per-component GIFs + a vertically-stacked dEx/dEy/dEz GIF."""
    gif_paths = {}
    for comp, files in by_comp.items():
        gif = os.path.join(outdir, sub, comp, f"{comp}.gif")
        B.make_gif(files, gif, duration_ms=200)
        gif_paths[comp] = gif
    keys = list(by_comp.keys())
    if len(keys) == 3:
        combined = os.path.join(outdir, stack_name)
        stitch_one(gif_paths[keys[0]], gif_paths[keys[1]], gif_paths[keys[2]],
                   combined, label)
        print(f"  GIF: {combined}", flush=True)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=NPZ_DEFAULT,
                    help="spatial 2D double-anode voxel_data.npz")
    ap.add_argument("--spline-knots-json", default=KNOTS_DEFAULT,
                    help="spatial spline_knots.json (knot counts + cuts)")
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    ap.add_argument("--cond-max", type=float, default=None,
                    help="voxel LSQ cond cut (default: from knots JSON)")
    ap.add_argument("--max-resolution-cm", type=float, default=None,
                    help="voxel worst-axis sigma cut (default: from knots JSON)")
    ap.add_argument("--dx-method", choices=("spline", "finitediff"),
                    default="spline")
    ap.add_argument("--nx-knots", type=int, default=None)
    ap.add_argument("--ny-knots", type=int, default=None)
    ap.add_argument("--nz-knots", type=int, default=None)
    ap.add_argument("--ridge-lambda", type=float, default=1.0)
    ap.add_argument("--sigma-floor", type=float, default=0.5)
    ap.add_argument("--vrange-abs", type=float, default=50.0,
                    help="color half-range for absolute dE maps [V/cm]")
    ap.add_argument("--vrange-frac", type=float, default=0.10,
                    help="color half-range for fractional dE/E0 maps")
    ap.add_argument("--emag-vrange", type=float, default=50.0,
                    help="color half-range for |E| maps about 500 V/cm")
    ap.add_argument("--grad-vrange", type=float, default=0.15,
                    help="color half-range for g = dDelta/dx maps")
    ap.add_argument("--nproc", default="auto")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    knots = json.load(open(args.spline_knots_json))
    ik = knots.get("interior_knots", {})
    cuts = knots.get("cuts", {})
    nx_k = args.nx_knots if args.nx_knots is not None else ik.get("nx", 5)
    ny_k = args.ny_knots if args.ny_knots is not None else ik.get("ny", 6)
    nz_k = args.nz_knots if args.nz_knots is not None else ik.get("nz", 9)
    cond_max = (args.cond_max if args.cond_max is not None
                else cuts.get("cond_max", 100.0))
    max_res = (args.max_resolution_cm if args.max_resolution_cm is not None
               else cuts.get("max_resolution_cm", 2.0))
    nproc = cpu_count() if args.nproc == "auto" else int(args.nproc)

    d = np.load(args.input)
    v0 = float(d["vdrift"])
    voxel_size = float(d["voxel_size"])
    print(f"E-field distortion build  (MicroBooNE arXiv:2008.09765 method)",
          flush=True)
    print(f"  spatial map : {args.input}", flush=True)
    print(f"  v0          : {v0:.5f} cm/us   E0 = {E0_VCM:.0f} V/cm", flush=True)
    print(f"  drift FUDGE : {DM.FUDGE:.6f}   ve_log_slope = "
          f"{DM.ve_log_slope():.5f}", flush=True)
    print(f"  dx-method   : {args.dx_method}", flush=True)

    mu_clean, sigma_clean, vx_c, vy_c, vz_c, keep, stats = F.load_and_cut(
        args.input, cond_max, max_res)
    print(f"  voxel cuts  : cond<={cond_max:g}, res<={max_res:g} cm  -> "
          f"{stats['n_keep']} / {stats['n_inverted']} kept", flush=True)

    # ---- g = dDelta/dx ----
    if args.dx_method == "spline":
        print(f"  fitting spatial spline nx={nx_k} ny={ny_k} nz={nz_k} "
              f"ridge={args.ridge_lambda} for the analytic dDelta/dx ...",
              flush=True)
        g = gradient_from_spline(mu_clean, sigma_clean, vx_c, vy_c, vz_c,
                                 nx_k, ny_k, nz_k, args.ridge_lambda,
                                 args.sigma_floor)
    else:
        g = gradient_from_finitediff(mu_clean, vx_c)

    # ---- MicroBooNE inversion ----
    dE_all, Emag, speed = invert_to_efield(g, vx_c, v0)
    dE_frac = dE_all / E0_VCM
    alpha = DM.ve_log_slope()
    sign = np.where(vx_c < 0.0, 1.0, -1.0)[:, None, None]
    dE_lin_frac = np.stack([sign * (-g[..., 0] / alpha),
                            sign * (-g[..., 1]),
                            sign * (-g[..., 2])], axis=-1)

    finite = np.isfinite(dE_all).all(axis=-1)
    n_nan = int((~finite).sum())
    em = Emag[np.isfinite(Emag)]
    lin_rms = float(np.sqrt(np.nanmean((dE_frac - dE_lin_frac) ** 2)))
    print(f"  voxels with finite dE       : {int(finite.sum())} / "
          f"{finite.size}  ({n_nan} NaN)", flush=True)
    print(f"  |E| range  : [{em.min():.1f}, {em.max():.1f}] V/cm  "
          f"mean {em.mean():.1f}", flush=True)
    print(f"  dE/E0 range: dEx [{np.nanmin(dE_frac[...,0]):+.3f}, "
          f"{np.nanmax(dE_frac[...,0]):+.3f}]  "
          f"dEy [{np.nanmin(dE_frac[...,1]):+.3f}, "
          f"{np.nanmax(dE_frac[...,1]):+.3f}]  "
          f"dEz [{np.nanmin(dE_frac[...,2]):+.3f}, "
          f"{np.nanmax(dE_frac[...,2]):+.3f}]", flush=True)
    print(f"  nonlinear vs linearized dE/E0 RMS difference = {lin_rms:.4f}",
          flush=True)

    # ---- save npz ----
    os.makedirs(args.outdir, exist_ok=True)
    npz = os.path.join(args.outdir, "voxel_data.npz")
    np.savez_compressed(
        npz,
        dE_all=dE_all.astype(np.float32),
        dE_frac_all=dE_frac.astype(np.float32),
        Emag_all=Emag.astype(np.float32),
        speed_all=speed.astype(np.float32),
        grad_all=g.astype(np.float32),
        dE_lin_frac_all=dE_lin_frac.astype(np.float32),
        keep_mask=keep,
        x_edges=d["x_edges"], y_edges=d["y_edges"], z_edges=d["z_edges"],
        voxel_size=np.float32(voxel_size),
        E0_Vcm=np.float32(E0_VCM),
        v0_cmus=np.float32(v0),
        drift_fudge=np.float32(DM.FUDGE),
        ve_log_slope=np.float32(alpha),
        dx_method=np.array(args.dx_method),
    )
    print(f"  wrote {npz}  ({os.path.getsize(npz)/1e3:.0f} kB)", flush=True)

    # ---- plots ----
    print("plotting per-X-slice maps ...", flush=True)
    tasks = []
    by_abs, t = make_map_set(
        dE_all, vx_c, args.outdir, "dE", COMPONENTS,
        vrange=args.vrange_abs, center=0.0, cmap="RdYlBu_r",
        label_tmpl="$\\Delta E_{sub}$ [V/cm]",
        title_tmpl="SBND 2D double-anode  $\\Delta E_{sub}$ (lab frame)",
        symmetric=True)
    tasks += t
    by_frac, t = make_map_set(
        dE_frac, vx_c, args.outdir, "dE_frac", COMPONENTS,
        vrange=args.vrange_frac, center=0.0, cmap="RdYlBu_r",
        label_tmpl="$\\Delta E_{sub}/E_0$",
        title_tmpl="SBND 2D double-anode  $\\Delta E_{sub}/E_0$ (lab frame)",
        symmetric=True)
    tasks += t
    by_emag, t = make_map_set(
        Emag[..., None], vx_c, args.outdir, "emag", ("emag",),
        vrange=args.emag_vrange, center=E0_VCM, cmap="RdYlBu_r",
        label_tmpl="$|E|$ [V/cm]",
        title_tmpl="SBND 2D double-anode  field magnitude $|E|$",
        symmetric=False)
    tasks += t
    by_grad, t = make_map_set(
        g, vx_c, args.outdir, "grad", GRADS,
        vrange=args.grad_vrange, center=0.0, cmap="RdYlBu_r",
        label_tmpl="$\\partial\\Delta_{sub}/\\partial x$",
        title_tmpl="SBND 2D double-anode  $g_{sub}="
                   "\\partial\\Delta_{sub}/\\partial x$",
        symmetric=True)
    tasks += t

    if nproc > 1 and len(tasks) > 1:
        with Pool(nproc) as pool:
            pool.map(_heat_worker, tasks, chunksize=4)
    else:
        for t in tasks:
            _heat_worker(t)
    print(f"  wrote {len(tasks)} heatmaps (png+pdf)", flush=True)

    if not args.no_gif:
        print("building GIFs ...", flush=True)
        build_gifs(by_abs, args.outdir, "dE", "dExyz_stack.gif",
                   "E-field distortion dE [V/cm]")
        build_gifs(by_frac, args.outdir, "dE_frac", "dExyz_frac_stack.gif",
                   "E-field distortion dE/E0")
        build_gifs(by_grad, args.outdir, "grad", "grad_xyz_stack.gif",
                   "spatial-distortion gradient dDelta/dx")
        for comp, files in by_emag.items():
            B.make_gif(files, os.path.join(args.outdir, "emag",
                                           f"{comp}.gif"), duration_ms=200)

    print(f"\nwrote {args.outdir}/", flush=True)


if __name__ == "__main__":
    main()
