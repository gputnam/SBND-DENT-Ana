#!/usr/bin/env python3
"""Gauss's-law consistency check of the measured 2D double-anode E-field
distortion map.

A real electrostatic field obeys the differential Gauss's law
div(E) = rho / (eps0 eps_r). In the absence of free charge ("vacuum" Gauss's
law) div(E) = 0, so the per-voxel divergence of the *measured* field is a
direct discrepancy-from-zero diagnostic: a self-consistent measurement gives
a zero-centred div(E); its spread quantifies measurement noise, and any
coherent structure is the implied space-charge density.

The nominal uniform field E0 has zero divergence, so div(E) = div(Delta E):
the divergence is computed straight from dE_all (lab frame, V/cm). The
x-derivative is taken per TPC half so it never crosses the central cathode.

    div(E)   [V/cm^2]   = d(dEx)/dx + d(dEy)/dy + d(dEz)/dz
    rho      [C/cm^3]   = eps0 * eps_r * div(E)
    rho/e    [e-/cm^3]  = rho / 1.602e-19

Two estimators:
  * voxel  -- central finite differences of the 20-cm voxel map (the honest
              "what the data says" check; noisier).
  * spline -- analytic derivatives of the fitted E-field B-spline (smooth;
              needs efield_2d_spline_20cm/spline_fit.npz). Produced in
              addition whenever --spline-npz is available.

Usage:
    python check_gauss_2d.py
    python check_gauss_2d.py --source spline
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sce_input as B
from build_efield_2d_voxels import _heat_worker
from stitch_gifs import stitch_one

NPZ_DEFAULT = "efield_2d_maps_20cm/voxel_data.npz"
SPLINE_DEFAULT = "efield_2d_spline_20cm/spline_fit.npz"
OUTDIR_DEFAULT = "efield_2d_gauss_20cm"

EPS0 = 8.8541878128e-14        # F/cm  (vacuum permittivity)
ECHARGE = 1.602176634e-19      # C


def divergence_voxel(dE, vx_c, vy_c, vz_c):
    """Finite-difference div(E) of the lab-frame dE map on the 20-cm grid.
    The x-derivative is taken per TPC half. Returns div (nx,ny,nz) V/cm^2."""
    dx = float(vx_c[1] - vx_c[0])
    dy = float(vy_c[1] - vy_c[0])
    dz = float(vz_c[1] - vz_c[0])
    ddx = np.full(dE.shape[:3], np.nan)
    for ixh in (np.where(vx_c < 0)[0], np.where(vx_c > 0)[0]):
        ddx[ixh] = np.gradient(dE[ixh, :, :, 0], dx, axis=0)
    ddy = np.gradient(dE[:, :, :, 1], dy, axis=1)
    ddz = np.gradient(dE[:, :, :, 2], dz, axis=2)
    return ddx + ddy + ddz


def divergence_spline(spline_npz, vx_c, vy_c, vz_c):
    """Analytic div(E) from the fitted E-field B-spline. Returns V/cm^2."""
    d = np.load(spline_npz, allow_pickle=True)
    scale = float(d["E0_Vcm"]) if str(d["quantity"]) == "frac" else 1.0
    div = np.full((len(vx_c), len(vy_c), len(vz_c)), np.nan)
    for tag in ("TPC0", "TPC1"):
        kx, ky, kz = d[f"kx_{tag}"], d[f"ky_{tag}"], d[f"kz_{tag}"]
        ix = np.where(vx_c < 0)[0] if tag == "TPC0" else np.where(vx_c > 0)[0]
        ddx = B.eval_spline_deriv(vx_c[ix], vy_c, vz_c, kx, ky, kz,
                                  d[f"coeffs_{tag}_0"], dx=1)
        ddy = B.eval_spline_deriv(vx_c[ix], vy_c, vz_c, kx, ky, kz,
                                  d[f"coeffs_{tag}_1"], dy=1)
        ddz = B.eval_spline_deriv(vx_c[ix], vy_c, vz_c, kx, ky, kz,
                                  d[f"coeffs_{tag}_2"], dz=1)
        div[ix] = scale * (ddx + ddy + ddz)
    return div


def interior_mask(nx, ny, nz):
    """True for voxels away from every grid/cathode edge (one-sided finite
    differences bias the outer layer); used for the summary statistics."""
    m = np.zeros((nx, ny, nz), dtype=bool)
    half = nx // 2
    m[1:half - 1, 1:ny - 1, 1:nz - 1] = True          # TPC0 interior
    m[half + 1:nx - 1, 1:ny - 1, 1:nz - 1] = True      # TPC1 interior
    return m


def queue_maps(arr, vx_c, outdir, sub, vrange, clabel, title_base):
    """Per-X-slice heatmap tasks for a scalar (nx,ny,nz) field."""
    pdir = os.path.join(outdir, sub, "png")
    qdir = os.path.join(outdir, sub, "pdf")
    os.makedirs(pdir, exist_ok=True)
    os.makedirs(qdir, exist_ok=True)
    tasks, files = [], []
    for ix, xc in enumerate(vx_c):
        tpc = "TPC0" if xc < 0 else "TPC1"
        tag = f"x{int(round(xc)):+04d}"
        png = os.path.join(pdir, f"{tag}.png")
        pdf = os.path.join(qdir, f"{tag}.pdf")
        title = f"{title_base}\nX = {xc:+.0f} cm ({tpc})"
        tasks.append((arr[ix].astype(float), -vrange, vrange, "RdYlBu_r",
                      "both", clabel, title, png, pdf))
        files.append(png)
    return tasks, files


def summary_row(name, div, keep, interior):
    """Headline statistics for one div(E) estimator."""
    sel = keep & interior & np.isfinite(div)
    v = div[sel]
    return dict(estimator=name, n_voxels=int(v.size),
                mean=float(v.mean()), rms=float(v.std()),
                abs_median=float(np.median(np.abs(v))),
                p99_abs=float(np.percentile(np.abs(v), 99)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=NPZ_DEFAULT,
                    help="E-field voxel_data.npz from build_efield_2d_voxels")
    ap.add_argument("--spline-npz", default=SPLINE_DEFAULT,
                    help="E-field spline_fit.npz (analytic-divergence source)")
    ap.add_argument("--source", choices=("voxel", "spline"), default="voxel",
                    help="estimator used for the rho map / headline number")
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    ap.add_argument("--epsilon-r", type=float, default=1.504,
                    help="LAr static dielectric constant (rho = eps0 eps_r "
                         "div E); use 1.0 for the strict vacuum law")
    ap.add_argument("--vrange", type=float, default=None,
                    help="div(E) color half-range [V/cm^2] (default: auto)")
    ap.add_argument("--nproc", default="auto")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    from multiprocessing import Pool, cpu_count
    nproc = cpu_count() if args.nproc == "auto" else int(args.nproc)

    d = np.load(args.input)
    dE = d["dE_all"].astype(np.float64)              # (nx,ny,nz,3) V/cm
    keep = d["keep_mask"]
    vx_c = B.voxel_centers(d["x_edges"])
    vy_c = B.voxel_centers(d["y_edges"])
    vz_c = B.voxel_centers(d["z_edges"])
    nx, ny, nz = dE.shape[:3]
    interior = interior_mask(nx, ny, nz)

    print("Gauss's-law check of the measured E-field distortion", flush=True)
    print(f"  input      : {args.input}", flush=True)

    estimators = {"voxel": divergence_voxel(dE, vx_c, vy_c, vz_c)}
    have_spline = os.path.exists(args.spline_npz)
    if have_spline:
        estimators["spline"] = divergence_spline(args.spline_npz,
                                                 vx_c, vy_c, vz_c)
    else:
        print(f"  (no spline npz at {args.spline_npz}; voxel estimator only)",
              flush=True)
        if args.source == "spline":
            args.source = "voxel"

    rows = [summary_row(nm, dv, keep, interior)
            for nm, dv in estimators.items()]
    print("\n  div(E) over interior, well-measured voxels  [V/cm^2]:",
          flush=True)
    for r in rows:
        rho_rms = EPS0 * args.epsilon_r * r["rms"]
        print(f"    {r['estimator']:7s}: mean {r['mean']:+.4f}  "
              f"rms {r['rms']:.4f}  |median| {r['abs_median']:.4f}  "
              f"(rho rms {rho_rms:.2e} C/cm^3 = "
              f"{rho_rms/ECHARGE:.2e} e-/cm^3)", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(args.outdir, "gauss_summary.csv"), index=False)

    # ---- divergence histogram ----
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    lim = max(r["p99_abs"] for r in rows) * 1.3
    bins = np.linspace(-lim, lim, 81)
    for nm, dv in estimators.items():
        sel = keep & interior & np.isfinite(dv)
        ax.hist(dv[sel], bins=bins, histtype="step", linewidth=1.6,
                label=f"{nm}: mean={dv[sel].mean():+.3f}, "
                      f"rms={dv[sel].std():.3f}")
    ax.axvline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel(r"$\nabla\cdot E$  [V/cm$^2$]")
    ax.set_ylabel("interior voxels")
    ax.set_title("SBND 2D double-anode  —  vacuum Gauss's-law discrepancy\n"
                 r"($\nabla\cdot E$ should be 0; spread = discrepancy)")
    ax.legend()
    fig.tight_layout()
    hist_png = os.path.join(args.outdir, "divE_histogram.png")
    fig.savefig(hist_png, dpi=130)
    fig.savefig(hist_png.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"\n  wrote {hist_png}", flush=True)

    # ---- per-X-slice maps ----
    div_primary = estimators[args.source]
    if args.vrange is not None:
        vr = args.vrange
    else:
        vr = float(np.nanpercentile(np.abs(div_primary[keep & interior]), 99))
        vr = max(vr, 1e-6)
    rho = EPS0 * args.epsilon_r * div_primary
    rho_e = rho / ECHARGE

    tasks = []
    for nm, dv in estimators.items():
        sub = "divE" if nm == "voxel" else "divE_spline"
        t, f_div = queue_maps(dv, vx_c, args.outdir, sub, vr,
                              r"$\nabla\cdot E$ [V/cm$^2$]",
                              f"SBND 2D double-anode  $\\nabla\\cdot E$ ({nm})")
        tasks += t
        if nm == args.source:
            div_files = f_div
    t, rho_files = queue_maps(
        rho_e, vx_c, args.outdir, "rho",
        EPS0 * args.epsilon_r * vr / ECHARGE,
        r"$\rho/e$ [e$^-$/cm$^3$]",
        "SBND 2D double-anode  implied charge density "
        f"($\\epsilon_r$={args.epsilon_r})")
    tasks += t

    if nproc > 1:
        with Pool(nproc) as pool:
            pool.map(_heat_worker, tasks, chunksize=4)
    else:
        for t in tasks:
            _heat_worker(t)
    print(f"  wrote {len(tasks)} heatmaps (png+pdf)", flush=True)

    if not args.no_gif:
        B.make_gif(div_files, os.path.join(args.outdir, "divE.gif"),
                   duration_ms=200)
        B.make_gif(rho_files, os.path.join(args.outdir, "rho.gif"),
                   duration_ms=200)
        print(f"  wrote {args.outdir}/divE.gif, rho.gif", flush=True)
        if have_spline:
            spl_files = [os.path.join(args.outdir, "divE_spline", "png",
                                      f"x{int(round(xc)):+04d}.png")
                         for xc in vx_c]
            spl_gif = os.path.join(args.outdir, "divE_spline.gif")
            B.make_gif(spl_files, spl_gif, duration_ms=200)
            stitch_one(os.path.join(args.outdir, "divE.gif"), spl_gif,
                       os.path.join(args.outdir, "rho.gif"),
                       os.path.join(args.outdir, "gauss_stack.gif"),
                       "Gauss check: divE voxel / divE spline / rho")
            print(f"  wrote {args.outdir}/gauss_stack.gif", flush=True)

    print(f"\nwrote {args.outdir}/", flush=True)


if __name__ == "__main__":
    main()
