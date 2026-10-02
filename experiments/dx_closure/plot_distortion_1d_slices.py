"""1-D slice plots of the v7 double-anode field-distortion voxel map.

Reads `doubleanode_2d_maps_20cm/voxel_data.npz` and refits the per-TPC
cubic-B-spline halves on the fly (matching `fit_doubleanode_2d_spline.py`
defaults) so a smooth fine-grid spline overlay can be drawn on every
slice. Per-voxel 1-sigma error bars come from sqrt(diag) of
`delta_cov_all` (xx,yy,zz are the first three packed entries).

For each component (dx, dy, dz) we produce:
  * per-cell vs X       at fixed (Y,Z)  -> two segmented lines (TPC0 X<0,
                                          TPC1 X>0). 20x25 = 500 files.
  * per-cell vs Y       at fixed (X,Z)  -> one line (TPC determined by X).
                                          20x25 = 500 files.
  * per-cell vs Z       at fixed (X,Y)  -> one line.
                                          20x20 = 400 files.
  * averaged vs X / Y / Z (other two voxel dims averaged in)  -> 2 lines.

Two versions per category: `no_spline/` and `with_spline/`.

TPC bookkeeping: X-bins with x<0 = TPC0, x>0 = TPC1. Plots never average
across the cathode. The user explicitly asked for per-(row,column) cell
slicing (full grid), not marginals.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import build_sce_input as B

VOXEL_NPZ_DEFAULT = "doubleanode_2d_maps_20cm/voxel_data.npz"
SPLINE_DIR_DEFAULT = "doubleanode_2d_spline_20cm"
OUTDIR_DEFAULT = "distortion_1d_slices"
COMPONENTS = ("dx", "dy", "dz")
COMP_LABEL = {"dx": r"$\Delta_x$ [cm]",
              "dy": r"$\Delta_y$ [cm]",
              "dz": r"$\Delta_z$ [cm]"}

C_TPC0 = "tab:blue"
C_TPC1 = "tab:red"


def _load_voxels(npz_path, cond_max, max_resolution_cm):
    """Return mu_clean (nx,ny,nz,3), sigma (nx,ny,nz,3), keep (nx,ny,nz),
    counts (nx,ny,nz), and voxel-center axes."""
    d = np.load(npz_path)
    mu = d["delta_mu_all"].astype(np.float64)
    cov = d["delta_cov_all"].astype(np.float64)
    cond = d["cond_all"].astype(np.float64)
    counts = d["count_all"].astype(np.int64)
    vx_c = B.voxel_centers(d["x_edges"])
    vy_c = B.voxel_centers(d["y_edges"])
    vz_c = B.voxel_centers(d["z_edges"])

    # cov packing is xx,yy,zz,xy,xz,yz -> diagonals are [..., :3]
    sigma = np.sqrt(np.maximum(cov[..., :3], 0.0))
    resolution = sigma.max(axis=-1)
    finite = np.isfinite(mu).all(axis=-1)
    pass_cond = np.isfinite(cond) & (cond <= cond_max)
    pass_res = np.isfinite(resolution) & (resolution <= max_resolution_cm)
    keep = finite & pass_cond & pass_res

    mu_clean = np.where(keep[..., None], mu, np.nan)
    sigma_clean = np.where(keep[..., None], sigma, np.nan)
    return mu_clean, sigma_clean, keep, counts, vx_c, vy_c, vz_c


def _collect_half(half_mask, vx_c, vy_c, vz_c, mu_clean, sigma_clean):
    idx_x = np.where(half_mask)[0]
    x_v = np.repeat(vx_c[idx_x], len(vy_c) * len(vz_c))
    y_v = np.tile(np.repeat(vy_c, len(vz_c)), len(idx_x))
    z_v = np.tile(vz_c, len(idx_x) * len(vy_c))
    vals = mu_clean[idx_x].reshape(-1, 3)
    sigs = sigma_clean[idx_x].reshape(-1, 3)
    return x_v, y_v, z_v, vals, sigs


def _refit_halves(vx_c, vy_c, vz_c, mu_clean, sigma_clean,
                  nx_k, ny_k, nz_k, ridge_lambda, sigma_floor):
    halves = {}
    for tag, half_mask, xa, xb, side in [
        ("TPC0", vx_c < 0, B.ANODE_X_TPC0, B.CATHODE_X,    "low"),
        ("TPC1", vx_c > 0, B.CATHODE_X,    B.ANODE_X_TPC1, "high"),
    ]:
        x_v, y_v, z_v, vals, sigs = _collect_half(
            half_mask, vx_c, vy_c, vz_c, mu_clean, sigma_clean)
        kx, ky, kz, coeffs, _ = B.fit_spline_half(
            x_v, y_v, z_v, vals, sigs,
            xa=xa, xb=xb, anode_side=side,
            nx_int=nx_k, ny_int=ny_k, nz_int=nz_k,
            sigma_floor=sigma_floor, ridge_lambda=ridge_lambda)
        halves[tag] = dict(kx=kx, ky=ky, kz=kz, coeffs=coeffs)
    return halves


def _eval_spline_at_voxel_grid(halves, vx_c, vy_c, vz_c):
    out = np.full((len(vx_c), len(vy_c), len(vz_c), 3), np.nan)
    for tag, h in halves.items():
        mask = (vx_c < 0) if tag == "TPC0" else (vx_c > 0)
        ix = np.where(mask)[0]
        for c in range(3):
            out[ix, :, :, c] = B.eval_spline_grid(
                vx_c[ix], vy_c, vz_c, h["kx"], h["ky"], h["kz"], h["coeffs"][c])
    return out


# --------------------------------------------------------------------------
# multiprocessing setup
_W = {}  # populated by _init


def _init(state):
    _W.update(state)


def _spline_curve_vs_X(tag, iy, iz, ci, x_fine):
    h = _W["halves"][tag]
    vy = np.array([_W["vy_c"][iy]])
    vz = np.array([_W["vz_c"][iz]])
    return B.eval_spline_grid(x_fine, vy, vz,
                              h["kx"], h["ky"], h["kz"],
                              h["coeffs"][ci])[:, 0, 0]


def _spline_curve_vs_Y(tag, ix, iz, ci, y_fine):
    h = _W["halves"][tag]
    vx = np.array([_W["vx_c"][ix]])
    vz = np.array([_W["vz_c"][iz]])
    return B.eval_spline_grid(vx, y_fine, vz,
                              h["kx"], h["ky"], h["kz"],
                              h["coeffs"][ci])[0, :, 0]


def _spline_curve_vs_Z(tag, ix, iy, ci, z_fine):
    h = _W["halves"][tag]
    vx = np.array([_W["vx_c"][ix]])
    vy = np.array([_W["vy_c"][iy]])
    return B.eval_spline_grid(vx, vy, z_fine,
                              h["kx"], h["ky"], h["kz"],
                              h["coeffs"][ci])[0, 0, :]


def _spline_averaged_vs_X(tag, ci, x_fine):
    """Evaluate spline at fine X for every (Y,Z) voxel center and average."""
    h = _W["halves"][tag]
    vy_c = _W["vy_c"]; vz_c = _W["vz_c"]
    grid = B.eval_spline_grid(x_fine, vy_c, vz_c,
                              h["kx"], h["ky"], h["kz"], h["coeffs"][ci])
    return np.nanmean(grid, axis=(1, 2))


def _spline_averaged_vs_Y(tag, ci, y_fine):
    h = _W["halves"][tag]
    vx_c = _W["vx_c"]; vz_c = _W["vz_c"]
    half_mask = (vx_c < 0) if tag == "TPC0" else (vx_c > 0)
    vx_half = vx_c[half_mask]
    grid = B.eval_spline_grid(vx_half, y_fine, vz_c,
                              h["kx"], h["ky"], h["kz"], h["coeffs"][ci])
    return np.nanmean(grid, axis=(0, 2))


def _spline_averaged_vs_Z(tag, ci, z_fine):
    h = _W["halves"][tag]
    vx_c = _W["vx_c"]; vy_c = _W["vy_c"]
    half_mask = (vx_c < 0) if tag == "TPC0" else (vx_c > 0)
    vx_half = vx_c[half_mask]
    grid = B.eval_spline_grid(vx_half, vy_c, z_fine,
                              h["kx"], h["ky"], h["kz"], h["coeffs"][ci])
    return np.nanmean(grid, axis=(0, 1))


def _save_dual_versions(fig_no, fig_with, path_no, path_with,
                        want_spline):
    fig_no.tight_layout(); fig_no.savefig(path_no, dpi=110); plt.close(fig_no)
    if want_spline:
        fig_with.tight_layout(); fig_with.savefig(path_with, dpi=110)
        plt.close(fig_with)


def _new_fig():
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.axhline(0.0, color="gray", lw=0.6, ls="--", alpha=0.6)
    ax.grid(alpha=0.3)
    return fig, ax


def _format_axes(ax, primary, title, comp):
    ax.set_xlabel(f"{primary} [cm]")
    ax.set_ylabel(COMP_LABEL[comp])
    ax.set_title(title)


def _draw_errbar(ax, x, y, yerr, color, label):
    finite = np.isfinite(y) & np.isfinite(yerr)
    if not finite.any():
        return
    ax.errorbar(x[finite], y[finite], yerr=yerr[finite],
                fmt="o", ms=3.0, lw=0.7, color=color, label=label)


# --------------------------------------------------------------------------
def _plot_vs_X_cell(task):
    """Per-cell vs X at fixed (iy, iz). Two segments: TPC0 (x<0), TPC1 (x>0)."""
    ci, comp, iy, iz, want_spline, root_no, root_with = task
    vx_c = _W["vx_c"]; vy_c = _W["vy_c"]; vz_c = _W["vz_c"]
    mu = _W["mu"][:, iy, iz, ci]
    sig = _W["sigma"][:, iy, iz, ci]
    yv = float(vy_c[iy]); zv = float(vz_c[iz])

    figs = []
    for spline_on in (False, True) if want_spline else (False,):
        fig, ax = _new_fig()
        m_lo = vx_c < 0; m_hi = vx_c > 0
        _draw_errbar(ax, vx_c[m_lo], mu[m_lo], sig[m_lo], C_TPC0, "TPC0 (x<0)")
        _draw_errbar(ax, vx_c[m_hi], mu[m_hi], sig[m_hi], C_TPC1, "TPC1 (x>0)")
        if spline_on:
            xf0 = np.linspace(B.ANODE_X_TPC0, B.CATHODE_X,    121)
            xf1 = np.linspace(B.CATHODE_X,    B.ANODE_X_TPC1, 121)
            s0 = _spline_curve_vs_X("TPC0", iy, iz, ci, xf0)
            s1 = _spline_curve_vs_X("TPC1", iy, iz, ci, xf1)
            ax.plot(xf0, s0, "-", color=C_TPC0, lw=1.3, alpha=0.85,
                    label="TPC0 spline")
            ax.plot(xf1, s1, "-", color=C_TPC1, lw=1.3, alpha=0.85,
                    label="TPC1 spline")
        ax.set_xlim(B.ANODE_X_TPC0 - 5, B.ANODE_X_TPC1 + 5)
        ax.legend(loc="upper right", fontsize=9, ncol=2 if spline_on else 1)
        _format_axes(ax, "X", f"{comp}  Y = {yv:+.0f} cm,  Z = {zv:+.0f} cm",
                     comp)
        figs.append(fig)

    fname = f"Y_{int(round(yv)):+04d}_Z_{int(round(zv)):+04d}.png"
    if want_spline:
        figs[0].tight_layout()
        figs[0].savefig(os.path.join(root_no, fname), dpi=110); plt.close(figs[0])
        figs[1].tight_layout()
        figs[1].savefig(os.path.join(root_with, fname), dpi=110); plt.close(figs[1])
    else:
        figs[0].tight_layout()
        figs[0].savefig(os.path.join(root_no, fname), dpi=110); plt.close(figs[0])


def _plot_vs_Y_cell(task):
    """Per-cell vs Y at fixed (ix, iz). One TPC determined by ix."""
    ci, comp, ix, iz, want_spline, root_no, root_with = task
    vx_c = _W["vx_c"]; vy_c = _W["vy_c"]; vz_c = _W["vz_c"]
    mu = _W["mu"][ix, :, iz, ci]
    sig = _W["sigma"][ix, :, iz, ci]
    xv = float(vx_c[ix]); zv = float(vz_c[iz])
    tag = "TPC0" if xv < 0 else "TPC1"
    color = C_TPC0 if tag == "TPC0" else C_TPC1

    figs = []
    for spline_on in (False, True) if want_spline else (False,):
        fig, ax = _new_fig()
        _draw_errbar(ax, vy_c, mu, sig, color, f"{tag} (X={xv:+.0f} cm)")
        if spline_on:
            yf = np.linspace(-200.0, +200.0, 201)
            s = _spline_curve_vs_Y(tag, ix, iz, ci, yf)
            ax.plot(yf, s, "-", color=color, lw=1.3, alpha=0.85,
                    label=f"{tag} spline")
        ax.set_xlim(-205, 205)
        ax.legend(loc="upper right", fontsize=9)
        _format_axes(ax, "Y", f"{comp}  X = {xv:+.0f} cm,  Z = {zv:+.0f} cm",
                     comp)
        figs.append(fig)

    fname = f"X_{int(round(xv)):+04d}_Z_{int(round(zv)):+04d}.png"
    if want_spline:
        figs[0].tight_layout()
        figs[0].savefig(os.path.join(root_no, fname), dpi=110); plt.close(figs[0])
        figs[1].tight_layout()
        figs[1].savefig(os.path.join(root_with, fname), dpi=110); plt.close(figs[1])
    else:
        figs[0].tight_layout()
        figs[0].savefig(os.path.join(root_no, fname), dpi=110); plt.close(figs[0])


def _plot_vs_Z_cell(task):
    """Per-cell vs Z at fixed (ix, iy)."""
    ci, comp, ix, iy, want_spline, root_no, root_with = task
    vx_c = _W["vx_c"]; vy_c = _W["vy_c"]; vz_c = _W["vz_c"]
    mu = _W["mu"][ix, iy, :, ci]
    sig = _W["sigma"][ix, iy, :, ci]
    xv = float(vx_c[ix]); yv = float(vy_c[iy])
    tag = "TPC0" if xv < 0 else "TPC1"
    color = C_TPC0 if tag == "TPC0" else C_TPC1

    figs = []
    for spline_on in (False, True) if want_spline else (False,):
        fig, ax = _new_fig()
        _draw_errbar(ax, vz_c, mu, sig, color, f"{tag} (X={xv:+.0f} cm)")
        if spline_on:
            zf = np.linspace(0.0, 500.0, 251)
            s = _spline_curve_vs_Z(tag, ix, iy, ci, zf)
            ax.plot(zf, s, "-", color=color, lw=1.3, alpha=0.85,
                    label=f"{tag} spline")
        ax.set_xlim(-5, 505)
        ax.legend(loc="upper right", fontsize=9)
        _format_axes(ax, "Z", f"{comp}  X = {xv:+.0f} cm,  Y = {yv:+.0f} cm",
                     comp)
        figs.append(fig)

    fname = f"X_{int(round(xv)):+04d}_Y_{int(round(yv)):+04d}.png"
    if want_spline:
        figs[0].tight_layout()
        figs[0].savefig(os.path.join(root_no, fname), dpi=110); plt.close(figs[0])
        figs[1].tight_layout()
        figs[1].savefig(os.path.join(root_with, fname), dpi=110); plt.close(figs[1])
    else:
        figs[0].tight_layout()
        figs[0].savefig(os.path.join(root_no, fname), dpi=110); plt.close(figs[0])


# --------------------------------------------------------------------------
# averaged plots (run in main process)

def _averaged_mean_sigma_along(mu_clean, sigma_clean, primary_axis):
    """Average over the non-primary voxel dims (NaN-safe). Returns
    {tag: (mean, err)} where err = sqrt(mean(sigma**2) / N_valid)."""
    nx, ny, nz, nc = mu_clean.shape
    out = {}
    for tag, mask in (("TPC0", _W["vx_c"] < 0), ("TPC1", _W["vx_c"] > 0)):
        sub_mu = mu_clean[mask]
        sub_sig = sigma_clean[mask]
        if primary_axis == "X":
            axes_to_avg = (1, 2)
            primary = _W["vx_c"][mask]
        elif primary_axis == "Y":
            axes_to_avg = (0, 2)
            primary = _W["vy_c"]
        elif primary_axis == "Z":
            axes_to_avg = (0, 1)
            primary = _W["vz_c"]
        mean = np.nanmean(sub_mu, axis=axes_to_avg)              # (..., 3)
        # error of the mean: sqrt(sum(sigma^2) / N^2) = sqrt(mean(sigma^2)/N)
        n_valid = np.sum(np.isfinite(sub_mu), axis=axes_to_avg)
        msq = np.nanmean(sub_sig ** 2, axis=axes_to_avg)
        with np.errstate(invalid="ignore", divide="ignore"):
            err = np.sqrt(msq / np.maximum(n_valid, 1))
        err = np.where(n_valid > 0, err, np.nan)
        out[tag] = (primary, mean, err)
    return out


def _make_averaged_plots(mu_clean, sigma_clean, halves, outdir, want_spline):
    for primary_axis in ("X", "Y", "Z"):
        avg = _averaged_mean_sigma_along(mu_clean, sigma_clean, primary_axis)
        if primary_axis == "X":
            x_fine_tpc0 = np.linspace(B.ANODE_X_TPC0, B.CATHODE_X,    121)
            x_fine_tpc1 = np.linspace(B.CATHODE_X,    B.ANODE_X_TPC1, 121)
            xlim = (B.ANODE_X_TPC0 - 5, B.ANODE_X_TPC1 + 5)
        elif primary_axis == "Y":
            y_fine = np.linspace(-200.0, +200.0, 201)
            xlim = (-205, 205)
        else:  # Z
            z_fine = np.linspace(0.0, 500.0, 251)
            xlim = (-5, 505)

        for ci, comp in enumerate(COMPONENTS):
            for spline_on in (False, True) if want_spline else (False,):
                fig, ax = _new_fig()
                for tag, color in (("TPC0", C_TPC0), ("TPC1", C_TPC1)):
                    primary, mean, err = avg[tag]
                    _draw_errbar(ax, primary, mean[:, ci], err[:, ci],
                                 color, tag)
                    if spline_on:
                        if primary_axis == "X":
                            xf = x_fine_tpc0 if tag == "TPC0" else x_fine_tpc1
                            s = _spline_averaged_vs_X(tag, ci, xf)
                            ax.plot(xf, s, "-", color=color, lw=1.3,
                                    alpha=0.85, label=f"{tag} spline")
                        elif primary_axis == "Y":
                            s = _spline_averaged_vs_Y(tag, ci, y_fine)
                            ax.plot(y_fine, s, "-", color=color, lw=1.3,
                                    alpha=0.85, label=f"{tag} spline")
                        else:
                            s = _spline_averaged_vs_Z(tag, ci, z_fine)
                            ax.plot(z_fine, s, "-", color=color, lw=1.3,
                                    alpha=0.85, label=f"{tag} spline")
                ax.set_xlim(*xlim)
                ax.legend(loc="upper right", fontsize=9,
                          ncol=2 if spline_on else 1)
                _format_axes(ax, primary_axis,
                             f"{comp} averaged over other two voxel dims",
                             comp)
                tag_dir = "with_spline" if spline_on else "no_spline"
                outd = os.path.join(outdir, f"vs_{primary_axis}",
                                    "averaged", tag_dir)
                os.makedirs(outd, exist_ok=True)
                fig.tight_layout()
                fig.savefig(os.path.join(outd, f"{comp}.png"), dpi=110)
                plt.close(fig)


# --------------------------------------------------------------------------
def _make_cell_outdirs(outdir, want_spline):
    """Create per-axis/per-comp/[with_spline,no_spline] directories."""
    paths = {}
    for axis in ("vs_X", "vs_Y", "vs_Z"):
        for comp in COMPONENTS:
            for sub in (("no_spline", "with_spline") if want_spline
                        else ("no_spline",)):
                p = os.path.join(outdir, axis, comp, sub)
                os.makedirs(p, exist_ok=True)
                paths[(axis, comp, sub)] = p
    return paths


def _build_tasks(want_spline, outdir, nx, ny, nz):
    """Return list of (axis, comp, ci, slice indices..., want_spline,
    no_spline_dir, with_spline_dir) tuples."""
    tasks_vsX, tasks_vsY, tasks_vsZ = [], [], []
    for ci, comp in enumerate(COMPONENTS):
        no_X = os.path.join(outdir, "vs_X", comp, "no_spline")
        wi_X = os.path.join(outdir, "vs_X", comp, "with_spline")
        no_Y = os.path.join(outdir, "vs_Y", comp, "no_spline")
        wi_Y = os.path.join(outdir, "vs_Y", comp, "with_spline")
        no_Z = os.path.join(outdir, "vs_Z", comp, "no_spline")
        wi_Z = os.path.join(outdir, "vs_Z", comp, "with_spline")
        for iy in range(ny):
            for iz in range(nz):
                tasks_vsX.append((ci, comp, iy, iz, want_spline, no_X, wi_X))
        for ix in range(nx):
            for iz in range(nz):
                tasks_vsY.append((ci, comp, ix, iz, want_spline, no_Y, wi_Y))
        for ix in range(nx):
            for iy in range(ny):
                tasks_vsZ.append((ci, comp, ix, iy, want_spline, no_Z, wi_Z))
    return tasks_vsX, tasks_vsY, tasks_vsZ


def main():
    mp.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxel-npz", default=VOXEL_NPZ_DEFAULT)
    ap.add_argument("--spline-dir", default=SPLINE_DIR_DEFAULT)
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    ap.add_argument("--nproc", type=int, default=32)
    ap.add_argument("--cond-max", type=float, default=100.0)
    ap.add_argument("--max-resolution-cm", type=float, default=2.0)
    ap.add_argument("--nx-knots", type=int, default=5)
    ap.add_argument("--ny-knots", type=int, default=6)
    ap.add_argument("--nz-knots", type=int, default=9)
    ap.add_argument("--ridge-lambda", type=float, default=1.0)
    ap.add_argument("--sigma-floor", type=float, default=0.5)
    ap.add_argument("--no-cells", action="store_true",
                    help="skip per-cell plots, only do the averaged plots")
    ap.add_argument("--no-spline-overlay", action="store_true",
                    help="skip the with_spline duplicate set")
    args = ap.parse_args()

    want_spline = not args.no_spline_overlay
    mu_clean, sigma_clean, keep, counts, vx_c, vy_c, vz_c = _load_voxels(
        args.voxel_npz, args.cond_max, args.max_resolution_cm)
    nx, ny, nz, _ = mu_clean.shape
    print(f"loaded voxel map: nx={nx} ny={ny} nz={nz}, "
          f"n_keep={int(keep.sum())} of {nx*ny*nz}", flush=True)

    halves = _refit_halves(vx_c, vy_c, vz_c, mu_clean, sigma_clean,
                           args.nx_knots, args.ny_knots, args.nz_knots,
                           args.ridge_lambda, args.sigma_floor)

    # self-check: refit must match the saved spline at voxel centers
    saved_path = os.path.join(args.spline_dir, "spline_fit.npz")
    if os.path.exists(saved_path):
        saved = np.load(saved_path)["spline_mu"]
        ours = _eval_spline_at_voxel_grid(halves, vx_c, vy_c, vz_c)
        diff = np.nanmax(np.abs(saved.astype(np.float64) - ours))
        print(f"spline-refit self-check: max |refit - saved| = "
              f"{diff:.3e} cm", flush=True)

    # populate worker state
    state = dict(mu=mu_clean, sigma=sigma_clean,
                 vx_c=vx_c, vy_c=vy_c, vz_c=vz_c, halves=halves)
    _W.update(state)

    os.makedirs(args.outdir, exist_ok=True)
    _make_cell_outdirs(args.outdir, want_spline)

    # averaged plots (always single-process; cheap)
    print("plotting averaged-over-other-dims slices...", flush=True)
    _make_averaged_plots(mu_clean, sigma_clean, halves, args.outdir,
                         want_spline)

    if args.no_cells:
        print("(--no-cells; skipping per-cell plots)", flush=True)
        return

    tasks_vsX, tasks_vsY, tasks_vsZ = _build_tasks(
        want_spline, args.outdir, nx, ny, nz)
    print(f"plotting per-cell slices: "
          f"vs_X={len(tasks_vsX)}  vs_Y={len(tasks_vsY)}  "
          f"vs_Z={len(tasks_vsZ)}  (each yields {2 if want_spline else 1} "
          f"PNG)", flush=True)

    work = [(_plot_vs_X_cell, tasks_vsX, "vs_X"),
            (_plot_vs_Y_cell, tasks_vsY, "vs_Y"),
            (_plot_vs_Z_cell, tasks_vsZ, "vs_Z")]

    if args.nproc <= 1:
        for fn, tasks, label in work:
            print(f"  {label}: {len(tasks)} cells (serial)", flush=True)
            for t in tasks:
                fn(t)
    else:
        with mp.Pool(args.nproc, initializer=_init, initargs=(state,)) as pool:
            for fn, tasks, label in work:
                print(f"  {label}: {len(tasks)} cells (pool of "
                      f"{args.nproc})", flush=True)
                # imap_unordered for progress
                done = 0
                for _ in pool.imap_unordered(fn, tasks, chunksize=8):
                    done += 1
                    if done % 250 == 0:
                        print(f"    [{label}] {done}/{len(tasks)}",
                              flush=True)

    print(f"\nwrote outputs under {args.outdir}/", flush=True)


if __name__ == "__main__":
    main()
