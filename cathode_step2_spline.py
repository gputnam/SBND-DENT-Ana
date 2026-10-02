"""STEP 2 — fit each TPC's spatial cathode-displacement map with a 2-D
cubic LSQ spline, after dropping pixels that are far from ALL of their
8 neighbors.

Input: step 1 NPZs (`tpc{0,1}_displacement.npz`) carrying the per-pixel
x_cathode in cm on the native 2-cm grid.

Outlier rule: for each pixel, compute the minimum absolute distance to
any finite neighbor in the 8-pixel surrounding (Chebyshev radius 1). If
that minimum exceeds `--neighbor-thresh` cm, the pixel is "far from all
of its neighbors" and is dropped from the LSQ fit. Pixels with no finite
neighbors are kept (they may be on the edge of the support).

Spline: `scipy.interpolate.LSQBivariateSpline` with uniform interior
knots, cubic order in both y and z. Weights = 1 / max(mu_err_cm,
weight-floor) where mu_err_cm = mu_err * cm_per_tick (the propagated
LSQ-fit uncertainty from step 1's CB fit).

Outputs (under --outdir):
    tpc{0,1}_spline.npz
        keys: x_measured  – step-1 input, NaN where mu_err > err_drop
              x_clean     – step-1 input with neighbor-outliers also NaN'd
              outlier_mask – True where the neighbor filter dropped a pixel
              x_fit_native – spline evaluated on the native 2-cm grid
              residual_native = x_clean - x_fit_native
              x_fit_v6     – spline evaluated on the 160x200 v6 grid
              y_native, z_native, y_v6, z_v6
              ny_knots, nz_knots, fit_rms_cm
    png/tpc{0,1}_compare_3pane.png
        measured | spline fit | residual (fit - measured), same colour scale
    png/tpc{0,1}_outliers.png
        binary map of which pixels the neighbor filter dropped
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Shared tensor-product B-spline machinery (consolidated into crtlib),
# previously duplicated byte-for-byte here from build_sce_input.
from crtlib.spline import clamped_knots, univariate_basis, fit_tensor_spline


NATIVE_Y_LO, NATIVE_Y_HI = -200.0, +200.0
NATIVE_Z_LO, NATIVE_Z_HI =    0.0, +500.0
NATIVE_BIN = 2.0

V6_NY, V6_NZ = 160, 200
V6_BIN = 2.5


def neighbor_isolation_mask(grid, *, threshold):
    """Returns (keep_mask, closest_neighbor_dist).

    A pixel passes (True) unless it has at least one finite neighbor and
    the closest of those neighbors is more than `threshold` away.
    """
    ny, nz = grid.shape
    pad = np.pad(grid, 1, constant_values=np.nan)
    closest = np.full_like(grid, np.nan)
    for dy in (-1, 0, 1):
        for dz in (-1, 0, 1):
            if dy == 0 and dz == 0:
                continue
            neighbor = pad[1+dy:1+dy+ny, 1+dz:1+dz+nz]
            dist = np.abs(grid - neighbor)
            update = np.isfinite(dist) & (
                np.isnan(closest) | (dist < closest))
            closest[update] = dist[update]
    has_neighbor = np.isfinite(closest)
    keep = np.isfinite(grid).copy()
    # only drop pixels that have a neighbor AND that neighbor is far
    keep &= ~(has_neighbor & (closest > threshold))
    return keep, closest


# _clamped_knots / _univariate_basis / _khatri_rao_sparse were byte-identical
# to build_sce_input's; both now import them from crtlib.spline.


class TensorRidgeSpline:
    """Tensor-product cubic B-spline fitted with weighted ridge LSQ.

    Provides scipy-LSQBivariateSpline-compatible __call__(y, z) that
    evaluates on a (len(y), len(z)) tensor grid.
    """

    def __init__(self, ky, kz, coeffs):
        self.ky = ky        # y knot vector
        self.kz = kz        # z knot vector
        self.coeffs = coeffs  # (ny_b, nz_b)

    def __call__(self, y, z):
        Ay = univariate_basis(y, self.ky)  # (n_y_eval, ny_b)
        Az = univariate_basis(z, self.kz)  # (n_z_eval, nz_b)
        return Ay @ self.coeffs @ Az.T


def fit_spline(x, mu_err_cm, y_c, z_c, *, ny_knots, nz_knots,
               weight_floor, ridge_lambda=0.0):
    """Tensor-product cubic B-spline fit, weighted LSQ + Tikhonov ridge.

    With ridge_lambda=0 this is a plain LSQ in the explicit basis (close
    to but not identical to scipy.LSQBivariateSpline). With ridge_lambda
    > 0 we penalize the coefficient norm, which suppresses the wild
    boundary oscillations LSQBivariateSpline produces when the support
    has gaps or thin edge coverage.

    The 2-D tensor-product solve is delegated to
    ``crtlib.spline.fit_tensor_spline`` (identical assembly: row-scaled
    Khatri-Rao design + ridge + LSMR).
    """
    Y, Z = np.meshgrid(y_c, z_c, indexing="ij")
    finite = np.isfinite(x)
    y_p = Y[finite]; z_p = Z[finite]; v = x[finite]
    e = np.where(np.isfinite(mu_err_cm), mu_err_cm, np.nan)
    e = e[finite]; e = np.where(np.isfinite(e), e, weight_floor)
    e = np.clip(e, weight_floor, None)
    w = 1.0 / e

    ky = clamped_knots(NATIVE_Y_LO, NATIVE_Y_HI, ny_knots)
    kz = clamped_knots(NATIVE_Z_LO, NATIVE_Z_HI, nz_knots)
    coeffs = fit_tensor_spline([y_p, z_p], [ky, kz], v, w,
                               ridge_lambda=ridge_lambda)
    return TensorRidgeSpline(ky, kz, coeffs)


def plot_3pane(measured, fit, residual, *, title, out_png, vrange):
    cm = matplotlib.colormaps["RdYlBu_r"].copy()
    cm.set_bad("white")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.4))
    for ax, grid, sub, vr in zip(
            axs,
            [measured, fit, residual],
            ["measured (cm)", "spline fit (cm)", "fit - measured (cm)"],
            [vrange, vrange, max(vrange * 0.5, 0.5)]):
        im = ax.imshow(grid, origin="lower", aspect="equal",
                       extent=(NATIVE_Z_LO, NATIVE_Z_HI,
                               NATIVE_Y_LO, NATIVE_Y_HI),
                       cmap=cm, vmin=-vr, vmax=+vr, interpolation="nearest")
        fig.colorbar(im, ax=ax, extend="both", fraction=0.046, pad=0.04)
        ax.set_xlabel("Z [cm]"); ax.set_ylabel("Y [cm]")
        ax.set_title(sub)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130); plt.close(fig)


def plot_outliers(measured, outlier_mask, *, title, out_png):
    fig, ax = plt.subplots(figsize=(9, 6))
    bg = np.where(np.isfinite(measured), 0.5, np.nan)
    bg[outlier_mask] = 1.0
    cm = matplotlib.colormaps["gray_r"].copy(); cm.set_bad("white")
    ax.imshow(bg, origin="lower", aspect="equal",
              extent=(NATIVE_Z_LO, NATIVE_Z_HI, NATIVE_Y_LO, NATIVE_Y_HI),
              cmap=cm, vmin=0, vmax=1, interpolation="nearest")
    n_outlier = int(np.sum(outlier_mask))
    n_pixels  = int(np.isfinite(measured).sum())
    ax.set_xlabel("Z [cm]"); ax.set_ylabel("Y [cm]")
    ax.set_title(f"{title}\n"
                 f"black = outlier ({n_outlier:,d} / {n_pixels:,d} finite pixels)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1-dir",
                    default="/exp/sbnd/data/users/gputnam/efield-study/cathode_pipeline/step1_displacement")
    ap.add_argument("--outdir",
                    default="/exp/sbnd/data/users/gputnam/efield-study/cathode_pipeline/step2_spline")
    ap.add_argument("--neighbor-thresh", type=float, default=0.7,
                    help="drop a pixel if its closest finite neighbor is "
                         "more than this many cm away (default %(default)s)")
    ap.add_argument("--ny-knots", type=int, default=8)
    ap.add_argument("--nz-knots", type=int, default=12)
    ap.add_argument("--weight-floor-cm", type=float, default=0.1,
                    help="floor on per-pixel sigma_cm in LSQ weights "
                         "(default %(default)s cm)")
    ap.add_argument("--ridge-lambda", type=float, default=2.0,
                    help="Tikhonov ridge multiplier on spline coefficient "
                         "norm. Higher = smoother / better-conditioned "
                         "edges. (default %(default)s)")
    ap.add_argument("--vrange-cm", type=float, default=2.5,
                    help="comparison plot colour scale half-range in cm "
                         "(default %(default)s)")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.outdir, "png"), exist_ok=True)

    for tpc in (0, 1):
        npz_in = os.path.join(args.step1_dir, f"tpc{tpc}_displacement.npz")
        print(f"\n[TPC{tpc}] loading {npz_in}", flush=True)
        d = np.load(npz_in)
        x_measured = d["x"].astype(np.float64)
        mu_err     = d["mu_err"].astype(np.float64)
        cm_per_tick = float(d["cm_per_tick"])
        y_native   = d["y_centers"]
        z_native   = d["z_centers"]
        mu_err_cm  = mu_err * abs(cm_per_tick)

        # outlier filter on the spatial map
        keep, closest = neighbor_isolation_mask(
            x_measured, threshold=args.neighbor_thresh)
        outlier_mask = np.isfinite(x_measured) & (~keep)
        x_clean = np.where(keep, x_measured, np.nan)
        n_in = int(np.isfinite(x_measured).sum())
        n_drop = int(outlier_mask.sum())
        print(f"  neighbor-isolation filter: dropped {n_drop:,d} / "
              f"{n_in:,d} finite pixels (threshold {args.neighbor_thresh:g} cm)",
              flush=True)

        # spline fit (ridge-regularized tensor-product B-spline)
        spline = fit_spline(x_clean, mu_err_cm, y_native, z_native,
                            ny_knots=args.ny_knots, nz_knots=args.nz_knots,
                            weight_floor=args.weight_floor_cm,
                            ridge_lambda=args.ridge_lambda)

        # evaluate on the native and v6 grids
        x_fit_native = spline(y_native, z_native)
        residual_native = x_clean - x_fit_native  # NaN where x_clean is NaN

        rms_cm = float(np.sqrt(np.nanmean(residual_native ** 2)))
        print(f"  fit RMS over kept native pixels: {rms_cm:.3f} cm "
              f"(ny_knots={args.ny_knots}, nz_knots={args.nz_knots})",
              flush=True)

        y_v6 = -200.0 + (np.arange(V6_NY) + 0.5) * V6_BIN
        z_v6 =    0.0 + (np.arange(V6_NZ) + 0.5) * V6_BIN
        x_fit_v6 = spline(y_v6, z_v6)

        # save
        out_npz = os.path.join(args.outdir, f"tpc{tpc}_spline.npz")
        np.savez(out_npz,
                 x_measured=x_measured.astype(np.float32),
                 x_clean=x_clean.astype(np.float32),
                 outlier_mask=outlier_mask,
                 x_fit_native=x_fit_native.astype(np.float32),
                 residual_native=residual_native.astype(np.float32),
                 x_fit_v6=x_fit_v6.astype(np.float32),
                 y_native=y_native, z_native=z_native,
                 y_v6=y_v6, z_v6=z_v6,
                 ny_knots=np.int32(args.ny_knots),
                 nz_knots=np.int32(args.nz_knots),
                 neighbor_thresh_cm=np.float32(args.neighbor_thresh),
                 fit_rms_cm=np.float32(rms_cm),
                 tpc=np.int32(tpc))
        print(f"  wrote {out_npz}", flush=True)

        # plots
        plot_3pane(x_clean, x_fit_native, residual_native,
                   title=f"TPC{tpc}: native cathode displacement, "
                         f"spline fit, residual",
                   out_png=os.path.join(args.outdir, "png",
                                         f"tpc{tpc}_compare_3pane.png"),
                   vrange=args.vrange_cm)
        plot_outliers(x_measured, outlier_mask,
                      title=f"TPC{tpc}: neighbor-isolation outliers",
                      out_png=os.path.join(args.outdir, "png",
                                            f"tpc{tpc}_outliers.png"))

    print(f"\nDone. {args.outdir}/", flush=True)


if __name__ == "__main__":
    main()
