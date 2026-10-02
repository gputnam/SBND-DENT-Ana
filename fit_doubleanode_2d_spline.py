"""Cubic B-spline fit of the 2D double-anode field-distortion voxel map.

Loads the `voxel_data.npz` written by `plot_doubleanode_2d_voxels.py`,
drops voxels failing two quality cuts, and fits a per-TPC-half,
anode-pinned, ridge-regularised cubic tensor-product B-spline to each of
Delta_x / Delta_y / Delta_z.

Quality cuts (drop the voxel from the fit if EITHER fails):
  * LSQ condition number  > --cond-max          (default 100)
  * position resolution   > --max-resolution-cm (default 2 cm)
    where the position resolution of a voxel is the largest of its three
    LSQ 1-sigma uncertainties sqrt(diag(delta_cov)).

The spline machinery is reused from `build_sce_input.py`
(`fit_spline_half` — two splines, x in [-200,0] and [0,+200], each
clamped to zero at its anode plane; `eval_spline_grid`; `make_gif`).

Modes:
  --knot-scan   scan (nx,ny,nz) interior-knot triples; write an
                RMS-vs-knots plot + CSV; pick the knot counts from it.
  (default)     fit at --nx/ny/nz-knots; write per-X-slice 3-pane plots
                (measured voxel / spline / residual), one GIF per
                component, and the combined dx/dy/dz stacked GIF.

Usage:
    python fit_doubleanode_2d_spline.py --knot-scan
    python fit_doubleanode_2d_spline.py --nx-knots 6 --ny-knots 8 --nz-knots 12
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make sibling top-level modules importable regardless of the CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sce_input as B
from stitch_gifs import stitch_one

NPZ_DEFAULT = "doubleanode_2d_maps_20cm/voxel_data.npz"
OUTDIR_DEFAULT = "doubleanode_2d_spline_20cm"
COMPONENTS = ("dx", "dy", "dz")
KNOT_SCAN_DEFAULT = "3,4,5;4,5,7;5,6,9;6,8,12;8,10,16;9,12,20"

HEATMAP = matplotlib.colormaps["RdYlBu_r"].copy()
HEATMAP.set_bad("white")


def load_and_cut(npz_path, cond_max, max_res):
    """Load the voxel map, return (mu_clean, sigma_clean, vx_c, vy_c, vz_c,
    stats). Voxels failing the cuts are set to NaN in mu_clean."""
    d = np.load(npz_path)
    mu = d["delta_mu_all"].astype(np.float64)            # (nx,ny,nz,3)
    cov = d["delta_cov_all"].astype(np.float64)          # (nx,ny,nz,6)
    cond = d["cond_all"]                                 # (nx,ny,nz)
    vx_c = B.voxel_centers(d["x_edges"])
    vy_c = B.voxel_centers(d["y_edges"])
    vz_c = B.voxel_centers(d["z_edges"])

    sigma = np.sqrt(np.maximum(cov[..., :3], 0.0))       # (nx,ny,nz,3)
    resolution = sigma.max(axis=-1)                      # worst-axis sigma

    finite = np.isfinite(mu).all(axis=-1)
    pass_cond = np.isfinite(cond) & (cond <= cond_max)
    pass_res = np.isfinite(resolution) & (resolution <= max_res)
    keep = finite & pass_cond & pass_res

    stats = dict(
        n_inverted=int(finite.sum()),
        n_fail_cond=int((finite & ~pass_cond).sum()),
        n_fail_res=int((finite & pass_cond & ~pass_res).sum()),
        n_keep=int(keep.sum()),
    )
    mu_clean = np.where(keep[..., None], mu, np.nan)
    sigma_clean = np.where(keep[..., None], sigma, np.nan)
    return mu_clean, sigma_clean, vx_c, vy_c, vz_c, keep, stats


def collect_half(half_mask, vx_c, vy_c, vz_c, mu_clean, sigma_clean):
    """Flat per-voxel (x,y,z,vals,sigmas) arrays for one TPC half."""
    idx_x = np.where(half_mask)[0]
    x_v = np.repeat(vx_c[idx_x], len(vy_c) * len(vz_c))
    y_v = np.tile(np.repeat(vy_c, len(vz_c)), len(idx_x))
    z_v = np.tile(vz_c, len(idx_x) * len(vy_c))
    vals = mu_clean[idx_x].reshape(-1, 3)
    sigs = sigma_clean[idx_x].reshape(-1, 3)
    return x_v, y_v, z_v, vals, sigs


def fit_both_halves(vx_c, vy_c, vz_c, mu_clean, sigma_clean,
                    nx_k, ny_k, nz_k, ridge_lambda, sigma_floor):
    """Fit the TPC0 and TPC1 splines. Returns {tag: (kx,ky,kz,coeffs,chi2)}."""
    halves = {}
    for tag, half_mask, xa, xb, side in [
        ("TPC0", vx_c < 0, B.ANODE_X_TPC0, B.CATHODE_X, "low"),
        ("TPC1", vx_c > 0, B.CATHODE_X, B.ANODE_X_TPC1, "high"),
    ]:
        x_v, y_v, z_v, vals, sigs = collect_half(
            half_mask, vx_c, vy_c, vz_c, mu_clean, sigma_clean)
        kx, ky, kz, coeffs, chi2 = B.fit_spline_half(
            x_v, y_v, z_v, vals, sigs,
            xa=xa, xb=xb, anode_side=side,
            nx_int=nx_k, ny_int=ny_k, nz_int=nz_k,
            sigma_floor=sigma_floor, ridge_lambda=ridge_lambda)
        halves[tag] = (kx, ky, kz, coeffs, chi2)
    return halves


def eval_on_voxel_grid(halves, vx_c, vy_c, vz_c):
    """Evaluate the fitted spline at every voxel center. (nx,ny,nz,3)."""
    out = np.full((len(vx_c), len(vy_c), len(vz_c), 3), np.nan)
    for tag, (kx, ky, kz, coeffs, _) in halves.items():
        mask = (vx_c < 0) if tag == "TPC0" else (vx_c > 0)
        ix = np.where(mask)[0]
        for c in range(3):
            out[ix, :, :, c] = B.eval_spline_grid(
                vx_c[ix], vy_c, vz_c, kx, ky, kz, coeffs[c])
    return out


# --------------------------------------------------------------------------
def run_knot_scan(args, mu_clean, sigma_clean, vx_c, vy_c, vz_c):
    triples = []
    for tok in args.knots.split(";"):
        a, b, c = tok.split(",")
        triples.append((int(a), int(b), int(c)))
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for (nx_k, ny_k, nz_k) in triples:
        t0 = time.time()
        halves = fit_both_halves(vx_c, vy_c, vz_c, mu_clean, sigma_clean,
                                 nx_k, ny_k, nz_k, args.ridge_lambda,
                                 args.sigma_floor)
        fit_grid = eval_on_voxel_grid(halves, vx_c, vy_c, vz_c)
        dt = time.time() - t0
        for tag in ("TPC0", "TPC1"):
            hmask = (vx_c < 0) if tag == "TPC0" else (vx_c > 0)
            kx, ky, kz, coeffs, chi2 = halves[tag]
            # spline coefficient count (dof denominator)
            n_coeff = sum(int(np.size(c)) for c in coeffs) // 3
            for ci, comp in enumerate(COMPONENTS):
                meas = np.where(hmask[:, None, None], mu_clean[..., ci],
                                np.nan)
                resid = fit_grid[..., ci] - meas
                rms = float(np.sqrt(np.nanmean(resid ** 2)))
                n_vox = int(np.isfinite(meas).sum())
                chi2_v, dof = chi2[ci]
                rows.append(dict(half=tag, nx=nx_k, ny=ny_k, nz=nz_k,
                                 component=comp, rms_resid_cm=rms,
                                 chi2=chi2_v, dof=dof, n_voxels=n_vox,
                                 n_coeff=n_coeff))
        print(f"  {nx_k}x{ny_k}x{nz_k}: "
              + "  ".join(f"{r['half']}/{r['component']}={r['rms_resid_cm']:.3f}"
                          for r in rows[-6:])
              + f"   ({dt:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    csv = os.path.join(args.outdir, "knot_scan.csv")
    df.to_csv(csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, tag in zip(axes, ("TPC0", "TPC1")):
        sub = df[df.half == tag]
        labels = [f"{r.nx}x{r.ny}x{r.nz}"
                  for _, r in sub[sub.component == "dx"].iterrows()]
        for comp, mk in (("dx", "o"), ("dy", "s"), ("dz", "^")):
            ss = sub[sub.component == comp]
            ax.plot(range(len(labels)), ss.rms_resid_cm.values,
                    marker=mk, label=comp)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(tag)
        ax.set_xlabel("interior knots  nx x ny x nz")
        ax.set_ylabel("voxel RMS(spline - measured) [cm]")
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Double-anode 2D spline: knot-count scan")
    fig.tight_layout()
    png = os.path.join(args.outdir, "knot_scan_rms.png")
    fig.savefig(png, dpi=130)
    fig.savefig(png.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"\nwrote {csv}\nwrote {png}", flush=True)


# --------------------------------------------------------------------------
def make_slice_plots(mu_clean, halves, vx_c, vy_c, vz_c, outdir, vrange,
                     fine_cm):
    """Per-X-slice 3-pane (measured / spline / residual). Returns
    {component: [png paths in X order]}."""
    fy = np.arange(vy_c[0], vy_c[-1] + 1e-6, fine_cm)
    fz = np.arange(vz_c[0], vz_c[-1] + 1e-6, fine_cm)
    fit_vox = eval_on_voxel_grid(halves, vx_c, vy_c, vz_c)
    by_comp = {c: [] for c in COMPONENTS}
    ext_vox = (vz_c[0] - 10, vz_c[-1] + 10, vy_c[0] - 10, vy_c[-1] + 10)
    ext_fine = (fz[0], fz[-1], fy[0], fy[-1])

    for ci, comp in enumerate(COMPONENTS):
        pdir = os.path.join(outdir, comp, "png")
        qdir = os.path.join(outdir, comp, "pdf")
        os.makedirs(pdir, exist_ok=True)
        os.makedirs(qdir, exist_ok=True)
        for ix, xc in enumerate(vx_c):
            tag = "TPC0" if xc < 0 else "TPC1"
            kx, ky, kz, coeffs, _ = halves[tag]
            meas = mu_clean[ix, :, :, ci]
            fine = B.eval_spline_grid(np.array([xc]), fy, fz,
                                      kx, ky, kz, coeffs[ci])[0]
            resid = fit_vox[ix, :, :, ci] - meas
            fig, axs = plt.subplots(1, 3, figsize=(19, 5.4))
            for ax, grid, ext, title in (
                (axs[0], meas, ext_vox, f"measured voxel {comp}"),
                (axs[1], fine, ext_fine, f"cubic-spline fit {comp}"),
                (axs[2], resid, ext_vox, f"spline - measured {comp}")):
                # grids are indexed [Y, Z]; imshow with extent=(Z..,Y..)
                # and origin="lower" plots them directly (no transpose).
                im = ax.imshow(grid, origin="lower", aspect="equal",
                               extent=ext, cmap=HEATMAP,
                               vmin=-vrange, vmax=vrange,
                               interpolation="nearest")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             label=f"$\\Delta_{{{comp[1]}}}$ [cm]")
                ax.set_xlabel("Z [cm]"); ax.set_ylabel("Y [cm]")
                ax.set_title(title)
            fig.suptitle(f"SBND 2D double-anode spline  —  "
                         f"X = {xc:+.0f} cm ({tag})")
            fig.tight_layout()
            tagn = f"x{int(round(xc)):+04d}"
            p = os.path.join(pdir, f"{tagn}.png")
            fig.savefig(p, dpi=110)
            fig.savefig(os.path.join(qdir, f"{tagn}.pdf"))
            plt.close(fig)
            by_comp[comp].append(p)
    return by_comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=NPZ_DEFAULT)
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    ap.add_argument("--cond-max", type=float, default=100.0,
                    help="drop voxels with LSQ condition number > this")
    ap.add_argument("--max-resolution-cm", type=float, default=2.0,
                    help="drop voxels whose worst-axis LSQ sigma exceeds this")
    ap.add_argument("--nx-knots", type=int, default=6)
    ap.add_argument("--ny-knots", type=int, default=8)
    ap.add_argument("--nz-knots", type=int, default=12)
    ap.add_argument("--ridge-lambda", type=float, default=1.0)
    ap.add_argument("--sigma-floor", type=float, default=0.5)
    ap.add_argument("--vrange", type=float, default=4.0)
    ap.add_argument("--fine-cm", type=float, default=5.0,
                    help="Y-Z grid step for the smooth spline display pane")
    ap.add_argument("--knot-scan", action="store_true")
    ap.add_argument("--knots", default=KNOT_SCAN_DEFAULT,
                    help="semicolon-separated nx,ny,nz triples for --knot-scan")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--write-sce-root", action="store_true",
                    help="also write the SBND SpaceChargeServiceSBND TH3F "
                         "input: forward displacement = this spline fit; "
                         "backward displacement and E-field copied as "
                         "defaults from --reference-root")
    ap.add_argument("--sce-outdir", default="sce_input_doubleanode_2d")
    ap.add_argument("--sce-root-name",
                    default="sbnd_sce_fwd_doubleanode_2d.root")
    ap.add_argument("--reference-root", default=B.REFERENCE_SCE_ROOT_DEFAULT,
                    help="reference SCE ROOT supplying the default E-field "
                         "and backward-displacement maps")
    args = ap.parse_args()

    mu_clean, sigma_clean, vx_c, vy_c, vz_c, keep, stats = load_and_cut(
        args.input, args.cond_max, args.max_resolution_cm)
    print(f"voxel quality cuts (cond <= {args.cond_max:g}, "
          f"resolution <= {args.max_resolution_cm:g} cm):", flush=True)
    print(f"  inverted voxels       : {stats['n_inverted']}", flush=True)
    print(f"  dropped by cond cut   : {stats['n_fail_cond']}", flush=True)
    print(f"  dropped by resolution : {stats['n_fail_res']}", flush=True)
    print(f"  kept for the fit      : {stats['n_keep']}", flush=True)

    if args.knot_scan:
        print(f"\nknot scan: {args.knots}", flush=True)
        run_knot_scan(args, mu_clean, sigma_clean, vx_c, vy_c, vz_c)
        return

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\nfitting splines: interior knots "
          f"nx={args.nx_knots} ny={args.ny_knots} nz={args.nz_knots}, "
          f"ridge={args.ridge_lambda}", flush=True)
    halves = fit_both_halves(vx_c, vy_c, vz_c, mu_clean, sigma_clean,
                             args.nx_knots, args.ny_knots, args.nz_knots,
                             args.ridge_lambda, args.sigma_floor)

    # fit summary + residual RMS
    fit_vox = eval_on_voxel_grid(halves, vx_c, vy_c, vz_c)
    summary = []
    for tag in ("TPC0", "TPC1"):
        hmask = (vx_c < 0) if tag == "TPC0" else (vx_c > 0)
        _, _, _, _, chi2 = halves[tag]
        for ci, comp in enumerate(COMPONENTS):
            meas = np.where(hmask[:, None, None], mu_clean[..., ci], np.nan)
            resid = fit_vox[..., ci] - meas
            rms = float(np.sqrt(np.nanmean(resid ** 2)))
            chi2_v, dof = chi2[ci]
            summary.append(dict(half=tag, component=comp, chi2=chi2_v,
                                dof=dof, rms_resid_cm=rms,
                                n_voxels=int(np.isfinite(meas).sum())))
            print(f"  {tag} {comp}: chi2/dof = {chi2_v:.0f}/{dof}  "
                  f"RMS(spline-measured) = {rms:.3f} cm", flush=True)
    pd.DataFrame(summary).to_csv(
        os.path.join(args.outdir, "fit_summary.csv"), index=False)

    # save knots + fitted voxel grid
    knots_json = {tag: {"x": list(map(float, h[0])),
                        "y": list(map(float, h[1])),
                        "z": list(map(float, h[2]))}
                  for tag, h in halves.items()}
    knots_json["interior_knots"] = dict(nx=args.nx_knots, ny=args.ny_knots,
                                        nz=args.nz_knots)
    knots_json["cuts"] = dict(cond_max=args.cond_max,
                              max_resolution_cm=args.max_resolution_cm,
                              n_voxels_kept=stats["n_keep"])
    with open(os.path.join(args.outdir, "spline_knots.json"), "w") as fh:
        json.dump(knots_json, fh, indent=2)
    np.savez_compressed(os.path.join(args.outdir, "spline_fit.npz"),
                        spline_mu=fit_vox.astype(np.float32),
                        x_edges=np.append(vx_c - 10, vx_c[-1] + 10),
                        keep_mask=keep)

    # ---- SBND SpaceCharge TH3F input ----
    if args.write_sce_root:
        os.makedirs(args.sce_outdir, exist_ok=True)
        sx, sy, sz = B.sce_bin_centers()
        fwd = np.zeros((B.SCE_NX, B.SCE_NY, B.SCE_NZ, 3), dtype=np.float64)
        for tag, (kx, ky, kz, coeffs, _) in halves.items():
            x_idx = (np.where(sx < 0)[0] if tag == "TPC0"
                     else np.where(sx > 0)[0])
            for c in range(3):
                fwd[x_idx, :, :, c] = B.eval_spline_grid(
                    sx[x_idx], sy, sz, kx, ky, kz, coeffs[c])
        # SpaceChargeServiceSBND flips TrueFwd_Displacement_X for x<0;
        # store -Delta_x there so the loader reproduces our measured map.
        fwd[sx < 0, :, :, 0] *= -1.0
        for ci, comp in enumerate(COMPONENTS):
            g = fwd[..., ci]
            print(f"  SCE grid {comp}: range [{g.min():+.2f}, "
                  f"{g.max():+.2f}] cm", flush=True)
        xe = np.linspace(*B.SCE_X_EDGE, B.SCE_NX + 1)
        ye = np.linspace(*B.SCE_Y_EDGE, B.SCE_NY + 1)
        ze = np.linspace(*B.SCE_Z_EDGE, B.SCE_NZ + 1)
        root_path = os.path.join(args.sce_outdir, args.sce_root_name)
        B.write_th3f_root(root_path, fwd[..., 0], fwd[..., 1], fwd[..., 2],
                          xe, ye, ze, reference_root=args.reference_root)

    # per-X-slice plots
    print("\nplotting per-X-slice 3-pane maps ...", flush=True)
    by_comp = make_slice_plots(mu_clean, halves, vx_c, vy_c, vz_c,
                               args.outdir, args.vrange, args.fine_cm)

    if not args.no_gif:
        gif_paths = {}
        for comp, files in by_comp.items():
            gif = os.path.join(args.outdir, comp, f"{comp}.gif")
            B.make_gif(files, gif, duration_ms=200)
            gif_paths[comp] = gif
            print(f"  GIF: {gif}", flush=True)
        combined = os.path.join(args.outdir, "dxyz_stack.gif")
        stitch_one(gif_paths["dx"], gif_paths["dy"], gif_paths["dz"],
                   combined, "double-anode 2D spline fit (dx/dy/dz)")

    print(f"\nwrote {args.outdir}/", flush=True)


if __name__ == "__main__":
    main()
