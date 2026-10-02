"""Validate the 2D wire-based double-anode method against the original
3D space-point double-anode method.

Both `plot_doubleanode_2d_voxels.py` (new, wire/tick only) and
`plot_doubleanode_voxels.py` (old, Pandora space-points) write a
`voxel_data.npz` with the SAME grid (20 cm, X/Y/Z) and the SAME keys
(`mu_all_{dx,dy,dz}`, `delta_mu_all`, `count_all`, `cond_all`). This
script overlays them voxel-for-voxel:

  * per X-slice, per component: a 3-pane figure
        [ 2D wire method | 3D space-point method | 2D - 3D difference ]
  * a correlation summary: 2D vs 3D scatter for each component, with the
    voxel-wise mean/RMS difference.

A close agreement is the validation that dropping the Pandora
space-points (and the CRT) costs nothing.

Usage:
    python compare_doubleanode_2d_vs_3d.py
        [--map2d doubleanode_2d_maps/voxel_data.npz]
        [--map3d doubleanode_maps_v4_20cm_singleg/voxel_data.npz]
        [--out compare_2d_vs_3d] [--vrange 5] [--min-counts 30]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COMPONENTS = ("dx", "dy", "dz")
HEATMAP = matplotlib.colormaps["RdYlBu_r"].copy()
HEATMAP.set_bad("white")


def load(path, min_counts):
    d = np.load(path)
    cnt = d["count_all"]
    mu = {c: np.where(cnt >= min_counts, d[f"mu_all_{c}"], np.nan)
          for c in COMPONENTS}
    # also drop voxels the LSQ rejected (mu already NaN there)
    edges = (d["x_edges"], d["y_edges"], d["z_edges"])
    return mu, cnt, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map2d", default="doubleanode_2d_maps/voxel_data.npz")
    ap.add_argument("--map3d",
                    default="doubleanode_maps_v4_20cm_singleg/voxel_data.npz")
    ap.add_argument("--out", default="compare_2d_vs_3d")
    ap.add_argument("--vrange", type=float, default=5.0)
    ap.add_argument("--min-counts", type=int, default=30)
    args = ap.parse_args()

    mu2, cnt2, edges = load(args.map2d, args.min_counts)
    mu3, cnt3, _ = load(args.map3d, args.min_counts)
    xe, ye, ze = edges
    nx = len(xe) - 1
    os.makedirs(args.out, exist_ok=True)

    extent = (ze[0], ze[-1], ye[0], ye[-1])
    panes = [("2D wire method", mu2), ("3D space-point method", mu3)]

    for comp in COMPONENTS:
        cdir = os.path.join(args.out, comp)
        os.makedirs(cdir, exist_ok=True)
        for ix in range(nx):
            g2 = mu2[comp][ix]
            g3 = mu3[comp][ix]
            diff = g2 - g3
            if not (np.isfinite(g2).any() or np.isfinite(g3).any()):
                continue
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            for ax, (label, grid) in zip(axes[:2], [(panes[0][0], g2),
                                                    (panes[1][0], g3)]):
                im = ax.imshow(grid, origin="lower", aspect="equal",
                               extent=extent, cmap=HEATMAP,
                               vmin=-args.vrange, vmax=args.vrange,
                               interpolation="nearest")
                fig.colorbar(im, ax=ax, extend="both",
                             label=f"$\\hat{{\\Delta}}_{{{comp[1]}}}$ [cm]")
                ax.set_title(label)
                ax.set_xlabel("Z [cm]"); ax.set_ylabel("Y [cm]")
            im = axes[2].imshow(diff, origin="lower", aspect="equal",
                                extent=extent, cmap=HEATMAP,
                                vmin=-args.vrange, vmax=args.vrange,
                                interpolation="nearest")
            fig.colorbar(im, ax=axes[2], extend="both",
                         label="2D - 3D [cm]")
            dv = diff[np.isfinite(diff)]
            axes[2].set_title(f"2D - 3D  (N={dv.size}, "
                              f"mean={dv.mean() if dv.size else 0:+.2f}, "
                              f"rms={dv.std() if dv.size else 0:.2f} cm)")
            axes[2].set_xlabel("Z [cm]"); axes[2].set_ylabel("Y [cm]")
            x_lo = xe[ix]
            fig.suptitle(f"double-anode $\\hat{{\\Delta}}_{{{comp[1]}}}$, "
                         f"X in [{x_lo:+.0f}, {xe[ix+1]:+.0f}] cm")
            fig.tight_layout()
            tag = f"x{int(round(x_lo)):+04d}"
            fig.savefig(os.path.join(cdir, f"{tag}.png"), dpi=110)
            fig.savefig(os.path.join(cdir, f"{tag}.pdf"))
            plt.close(fig)

    # ---- correlation summary ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, comp in zip(axes, COMPONENTS):
        a = mu2[comp].ravel()
        b = mu3[comp].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        ax.scatter(b, a, s=4, alpha=0.3, color="navy")
        lim = args.vrange * 1.6
        ax.plot([-lim, lim], [-lim, lim], "r-", lw=1)
        if a.size > 2:
            r = np.corrcoef(a, b)[0, 1]
            d = a - b
            ax.set_title(f"$\\Delta_{{{comp[1]}}}$: N={a.size}, r={r:.3f}, "
                         f"<2D-3D>={d.mean():+.2f}, rms={d.std():.2f} cm")
        ax.set_xlabel(f"3D space-point $\\Delta_{{{comp[1]}}}$ [cm]")
        ax.set_ylabel(f"2D wire $\\Delta_{{{comp[1]}}}$ [cm]")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.grid(alpha=0.3)
    fig.suptitle("double-anode 2D wire method vs 3D space-point method "
                 "(per-voxel)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "correlation_summary.png"), dpi=120)
    fig.savefig(os.path.join(args.out, "correlation_summary.pdf"))
    plt.close(fig)
    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
