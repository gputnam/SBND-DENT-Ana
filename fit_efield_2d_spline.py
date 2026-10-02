#!/usr/bin/env python3
"""Cubic B-spline fit of the 2D double-anode *electric-field* distortion
voxel map produced by build_efield_2d_voxels.py.

This mirrors fit_doubleanode_2d_spline.py (the *spatial* spline) with two
physics differences:

  * The spline is NOT anode-clamped (anode_side="none"). Only the spatial
    displacement vanishes at the anode; the E-field distortion is generally
    nonzero everywhere, including at the anode plane.
  * The fitted quantity is dE (V/cm, lab frame) or dE/E0 (--quantity).

Per-TPC-half, ridge-regularised cubic tensor-product B-splines (x in
[-200,0] and [0,+200]) are fit to dE_x / dE_y / dE_z. Outputs: per-X-slice
3-pane plots (measured / spline / residual), per-component + stacked GIFs,
fit_summary.csv, spline_knots.json, and spline_fit.npz (which also stores the
spline coefficients so check_gauss_2d.py can differentiate analytically).

--write-sce-root additionally evaluates the spline on the SBND SCE service's
49x49x61 TH3F grid and writes True_ElecField_{X,Y,Z,Mag} as the *fractional*
offset dE/E0 in the TPC-local convention the SpaceChargeServiceSBND loader
expects (verified against the reference SCE ROOT: True_ElecField_X is the
fractional drift-field change, stored symmetric across the cathode).

Usage:
    python fit_efield_2d_spline.py --knot-scan
    python fit_efield_2d_spline.py --nx-knots 5 --ny-knots 6 --nz-knots 9
    python fit_efield_2d_spline.py --write-sce-root
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sce_input as B
import fit_doubleanode_2d_spline as F
from stitch_gifs import stitch_one

NPZ_DEFAULT = "efield_2d_maps_20cm/voxel_data.npz"
OUTDIR_DEFAULT = "efield_2d_spline_20cm"
COMPONENTS = ("dEx", "dEy", "dEz")
KNOT_SCAN_DEFAULT = "3,4,5;4,5,7;5,6,9;6,8,12;8,10,16"

HEATMAP = matplotlib.colormaps["RdYlBu_r"].copy()
HEATMAP.set_bad("white")


def load_efield(npz_path, quantity):
    """Load the E-field voxel map. Returns (mu_clean, sigma_clean, vx_c,
    vy_c, vz_c, keep, stats, meta). Voxels with any non-finite component
    are NaN in mu_clean; sigma is uniform (unweighted ridge fit)."""
    d = np.load(npz_path)
    key = "dE_all" if quantity == "abs" else "dE_frac_all"
    mu = d[key].astype(np.float64)                       # (nx,ny,nz,3)
    vx_c = B.voxel_centers(d["x_edges"])
    vy_c = B.voxel_centers(d["y_edges"])
    vz_c = B.voxel_centers(d["z_edges"])
    keep = np.isfinite(mu).all(axis=-1)
    mu_clean = np.where(keep[..., None], mu, np.nan)
    sigma_clean = np.where(keep[..., None], np.ones_like(mu), np.nan)
    stats = dict(n_finite=int(keep.sum()), n_total=int(keep.size))
    meta = dict(E0_Vcm=float(d["E0_Vcm"]), v0_cmus=float(d["v0_cmus"]),
                dx_method=str(d["dx_method"]))
    return mu_clean, sigma_clean, vx_c, vy_c, vz_c, keep, stats, meta


def fit_both_halves(vx_c, vy_c, vz_c, mu_clean, sigma_clean,
                    nx_k, ny_k, nz_k, ridge_lambda, sigma_floor):
    """Fit TPC0 and TPC1 splines with NO anode clamp (anode_side='none').
    Returns {tag: (kx,ky,kz,coeffs,chi2)}."""
    halves = {}
    for tag, half_mask, xa, xb in [
        ("TPC0", vx_c < 0, B.ANODE_X_TPC0, B.CATHODE_X),
        ("TPC1", vx_c > 0, B.CATHODE_X, B.ANODE_X_TPC1),
    ]:
        x_v, y_v, z_v, vals, sigs = F.collect_half(
            half_mask, vx_c, vy_c, vz_c, mu_clean, sigma_clean)
        kx, ky, kz, coeffs, chi2 = B.fit_spline_half(
            x_v, y_v, z_v, vals, sigs,
            xa=xa, xb=xb, anode_side="none",
            nx_int=nx_k, ny_int=ny_k, nz_int=nz_k,
            sigma_floor=sigma_floor, ridge_lambda=ridge_lambda)
        halves[tag] = (kx, ky, kz, coeffs, chi2)
    return halves


# --------------------------------------------------------------------------
def run_knot_scan(args, mu_clean, sigma_clean, vx_c, vy_c, vz_c):
    triples = [tuple(int(v) for v in tok.split(","))
               for tok in args.knots.split(";")]
    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for (nx_k, ny_k, nz_k) in triples:
        t0 = time.time()
        halves = fit_both_halves(vx_c, vy_c, vz_c, mu_clean, sigma_clean,
                                 nx_k, ny_k, nz_k, args.ridge_lambda,
                                 args.sigma_floor)
        fit_grid = F.eval_on_voxel_grid(halves, vx_c, vy_c, vz_c)
        dt = time.time() - t0
        for tag in ("TPC0", "TPC1"):
            hmask = (vx_c < 0) if tag == "TPC0" else (vx_c > 0)
            for ci, comp in enumerate(COMPONENTS):
                meas = np.where(hmask[:, None, None], mu_clean[..., ci],
                                np.nan)
                resid = fit_grid[..., ci] - meas
                rms = float(np.sqrt(np.nanmean(resid ** 2)))
                rows.append(dict(half=tag, nx=nx_k, ny=ny_k, nz=nz_k,
                                 component=comp, rms_resid=rms,
                                 n_voxels=int(np.isfinite(meas).sum())))
        print(f"  {nx_k}x{ny_k}x{nz_k}: "
              + "  ".join(f"{r['half']}/{r['component']}={r['rms_resid']:.3g}"
                          for r in rows[-6:])
              + f"   ({dt:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    csv = os.path.join(args.outdir, "knot_scan.csv")
    df.to_csv(csv, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, tag in zip(axes, ("TPC0", "TPC1")):
        sub = df[df.half == tag]
        labels = [f"{r.nx}x{r.ny}x{r.nz}"
                  for _, r in sub[sub.component == "dEx"].iterrows()]
        for comp, mk in (("dEx", "o"), ("dEy", "s"), ("dEz", "^")):
            ss = sub[sub.component == comp]
            ax.plot(range(len(labels)), ss.rms_resid.values,
                    marker=mk, label=comp)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(tag)
        ax.set_xlabel("interior knots  nx x ny x nz")
        ax.set_ylabel("voxel RMS(spline - measured)")
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("E-field 2D spline: knot-count scan")
    fig.tight_layout()
    png = os.path.join(args.outdir, "knot_scan_rms.png")
    fig.savefig(png, dpi=130)
    fig.savefig(png.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"\nwrote {csv}\nwrote {png}", flush=True)


# --------------------------------------------------------------------------
def make_slice_plots(mu_clean, halves, vx_c, vy_c, vz_c, outdir, vrange,
                     fine_cm, unit_label):
    """Per-X-slice 3-pane (measured / spline / residual). Returns
    {component: [png paths in X order]}."""
    fy = np.arange(vy_c[0], vy_c[-1] + 1e-6, fine_cm)
    fz = np.arange(vz_c[0], vz_c[-1] + 1e-6, fine_cm)
    fit_vox = F.eval_on_voxel_grid(halves, vx_c, vy_c, vz_c)
    by_comp = {c: [] for c in COMPONENTS}
    ext_vox = (vz_c[0] - 10, vz_c[-1] + 10, vy_c[0] - 10, vy_c[-1] + 10)
    ext_fine = (fz[0], fz[-1], fy[0], fy[-1])

    for ci, comp in enumerate(COMPONENTS):
        pdir = os.path.join(outdir, comp, "png")
        qdir = os.path.join(outdir, comp, "pdf")
        os.makedirs(pdir, exist_ok=True)
        os.makedirs(qdir, exist_ok=True)
        sc = comp[-1]
        clabel = f"$\\Delta E_{{{sc}}}${unit_label}"
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
                im = ax.imshow(grid, origin="lower", aspect="equal",
                               extent=ext, cmap=HEATMAP,
                               vmin=-vrange, vmax=vrange,
                               interpolation="nearest")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             label=clabel)
                ax.set_xlabel("Z [cm]"); ax.set_ylabel("Y [cm]")
                ax.set_title(title)
            fig.suptitle(f"SBND 2D double-anode E-field spline  —  "
                         f"X = {xc:+.0f} cm ({tag})")
            fig.tight_layout()
            tagn = f"x{int(round(xc)):+04d}"
            p = os.path.join(pdir, f"{tagn}.png")
            fig.savefig(p, dpi=110)
            fig.savefig(os.path.join(qdir, f"{tagn}.pdf"))
            plt.close(fig)
            by_comp[comp].append(p)
    return by_comp


# --------------------------------------------------------------------------
def write_sce_root(args, halves, meta):
    """Evaluate the E-field spline on the SCE 49x49x61 grid and write the
    SBND SpaceChargeServiceSBND TH3F input.

    True_ElecField_{X,Y,Z} are the *fractional* offset dE/E0 in the TPC-local
    convention (drift-axis component symmetric across the cathode) -- the
    representation verified in the reference SCE ROOT. Forward displacement
    is copied from --displacement-root (the spatial SCE ROOT, else the
    reference); backward displacement from the reference."""
    os.makedirs(args.sce_outdir, exist_ok=True)
    E0 = meta["E0_Vcm"]
    sx, sy, sz = B.sce_bin_centers()

    # E-field spline on the SCE grid, lab frame (NaN at x==0 -> 0).
    dE_lab = np.nan_to_num(F.eval_on_voxel_grid(halves, sx, sy, sz), nan=0.0)
    if args.quantity == "abs":          # spline is V/cm -> fractional
        dE_lab = dE_lab / E0
    # lab -> TPC-local (drift-axis component symmetric across the cathode)
    sign = np.where(sx < 0.0, 1.0, -1.0)[:, None, None]
    efx = sign * dE_lab[..., 0]
    efy = sign * dE_lab[..., 1]
    efz = sign * dE_lab[..., 2]
    # fractional magnitude offset (|E_total| - E0)/E0, TPC-local components
    et = np.sqrt((E0 + efx * E0) ** 2 + (efy * E0) ** 2 + (efz * E0) ** 2)
    efmag = et / E0 - 1.0
    for nm, g in (("X", efx), ("Y", efy), ("Z", efz), ("Mag", efmag)):
        print(f"  SCE True_ElecField_{nm}: range "
              f"[{g.min():+.4f}, {g.max():+.4f}]", flush=True)

    # forward displacement: measured spatial SCE ROOT if present, else ref.
    import uproot
    disp_src = (args.displacement_root
                if os.path.exists(args.displacement_root)
                else args.reference_root)
    with uproot.open(disp_src) as df:
        fwd_x = df["TrueFwd_Displacement_X"].values().astype(np.float32)
        fwd_y = df["TrueFwd_Displacement_Y"].values().astype(np.float32)
        fwd_z = df["TrueFwd_Displacement_Z"].values().astype(np.float32)
    print(f"  forward displacement copied from {disp_src}", flush=True)

    xe = np.linspace(*B.SCE_X_EDGE, B.SCE_NX + 1)
    ye = np.linspace(*B.SCE_Y_EDGE, B.SCE_NY + 1)
    ze = np.linspace(*B.SCE_Z_EDGE, B.SCE_NZ + 1)
    root_path = os.path.join(args.sce_outdir, args.sce_root_name)
    B.write_th3f_root(root_path, fwd_x, fwd_y, fwd_z, xe, ye, ze,
                      reference_root=args.reference_root,
                      efield=(efx, efy, efz), efield_mag=efmag)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=NPZ_DEFAULT)
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    ap.add_argument("--quantity", choices=("abs", "frac"), default="abs",
                    help="fit dE [V/cm] (abs) or dE/E0 (frac)")
    ap.add_argument("--nx-knots", type=int, default=8)
    ap.add_argument("--ny-knots", type=int, default=10)
    ap.add_argument("--nz-knots", type=int, default=16)
    ap.add_argument("--ridge-lambda", type=float, default=0.1,
                    help="ridge scaled for the V/cm E-field (the spatial "
                         "spline's lambda=1 is calibrated for cm-scale data)")
    ap.add_argument("--sigma-floor", type=float, default=0.5)
    ap.add_argument("--vrange", type=float, default=None,
                    help="3-pane color half-range (default 50 abs / 0.10 frac)")
    ap.add_argument("--fine-cm", type=float, default=5.0)
    ap.add_argument("--knot-scan", action="store_true")
    ap.add_argument("--knots", default=KNOT_SCAN_DEFAULT)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--write-sce-root", action="store_true")
    ap.add_argument("--sce-outdir", default="sce_input_doubleanode_2d")
    ap.add_argument("--sce-root-name",
                    default="sbnd_sce_efield_doubleanode_2d.root")
    ap.add_argument("--displacement-root",
                    default="sce_input_doubleanode_2d/"
                            "sbnd_sce_fwd_doubleanode_2d.root",
                    help="SCE ROOT supplying TrueFwd_Displacement_* "
                         "(falls back to --reference-root if absent)")
    ap.add_argument("--reference-root", default=B.REFERENCE_SCE_ROOT_DEFAULT)
    args = ap.parse_args()

    vrange = args.vrange
    if vrange is None:
        vrange = 50.0 if args.quantity == "abs" else 0.10
    unit_label = " [V/cm]" if args.quantity == "abs" else "$/E_0$"

    mu_clean, sigma_clean, vx_c, vy_c, vz_c, keep, stats, meta = load_efield(
        args.input, args.quantity)
    print(f"E-field spline fit  (quantity={args.quantity}, "
          f"dx-method={meta['dx_method']})", flush=True)
    print(f"  finite voxels : {stats['n_finite']} / {stats['n_total']}",
          flush=True)

    if args.knot_scan:
        print(f"\nknot scan: {args.knots}", flush=True)
        run_knot_scan(args, mu_clean, sigma_clean, vx_c, vy_c, vz_c)
        return

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\nfitting splines (anode_side=none): nx={args.nx_knots} "
          f"ny={args.ny_knots} nz={args.nz_knots} ridge={args.ridge_lambda}",
          flush=True)
    halves = fit_both_halves(vx_c, vy_c, vz_c, mu_clean, sigma_clean,
                             args.nx_knots, args.ny_knots, args.nz_knots,
                             args.ridge_lambda, args.sigma_floor)

    fit_vox = F.eval_on_voxel_grid(halves, vx_c, vy_c, vz_c)
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
                                dof=dof, rms_resid=rms,
                                n_voxels=int(np.isfinite(meas).sum())))
            print(f"  {tag} {comp}: chi2/dof = {chi2_v:.0f}/{dof}  "
                  f"RMS(spline-measured) = {rms:.4g}", flush=True)
    pd.DataFrame(summary).to_csv(
        os.path.join(args.outdir, "fit_summary.csv"), index=False)

    # knots + coefficients + fitted grid
    knots_json = {tag: {"x": list(map(float, h[0])),
                        "y": list(map(float, h[1])),
                        "z": list(map(float, h[2]))}
                  for tag, h in halves.items()}
    knots_json["interior_knots"] = dict(nx=args.nx_knots, ny=args.ny_knots,
                                        nz=args.nz_knots)
    knots_json["quantity"] = args.quantity
    with open(os.path.join(args.outdir, "spline_knots.json"), "w") as fh:
        json.dump(knots_json, fh, indent=2)

    save = dict(spline_mu=fit_vox.astype(np.float32), keep_mask=keep,
                quantity=np.array(args.quantity),
                E0_Vcm=np.float32(meta["E0_Vcm"]),
                v0_cmus=np.float32(meta["v0_cmus"]),
                x_edges=np.append(vx_c - 10, vx_c[-1] + 10))
    for tag, (kx, ky, kz, coeffs, _) in halves.items():
        save[f"kx_{tag}"] = kx
        save[f"ky_{tag}"] = ky
        save[f"kz_{tag}"] = kz
        for c in range(3):
            save[f"coeffs_{tag}_{c}"] = coeffs[c].astype(np.float64)
    np.savez_compressed(os.path.join(args.outdir, "spline_fit.npz"), **save)

    if args.write_sce_root:
        print("\nwriting SBND SCE E-field ROOT ...", flush=True)
        write_sce_root(args, halves, meta)

    print("\nplotting per-X-slice 3-pane maps ...", flush=True)
    by_comp = make_slice_plots(mu_clean, halves, vx_c, vy_c, vz_c,
                               args.outdir, vrange, args.fine_cm, unit_label)

    if not args.no_gif:
        gif_paths = {}
        for comp, files in by_comp.items():
            gif = os.path.join(args.outdir, comp, f"{comp}.gif")
            B.make_gif(files, gif, duration_ms=200)
            gif_paths[comp] = gif
            print(f"  GIF: {gif}", flush=True)
        combined = os.path.join(args.outdir, "dExyz_stack.gif")
        stitch_one(gif_paths["dEx"], gif_paths["dEy"], gif_paths["dEz"],
                   combined, "double-anode 2D E-field spline (dEx/dEy/dEz)")

    print(f"\nwrote {args.outdir}/", flush=True)


if __name__ == "__main__":
    main()
