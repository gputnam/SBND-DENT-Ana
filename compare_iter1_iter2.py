"""Before/after comparison of the two-iteration double-anode displacement map.

`plot_doubleanode_2d_voxels.py --iterations 2` writes a single
`voxel_data.npz` carrying BOTH the iteration-1 (pre-rejection) map
(`delta_mu_all_iter1`, `count_all_iter1`) and the final iteration-2
(post-rejection) map (`delta_mu_all`, `count_all`). This script overlays
the two voxel-for-voxel:

  * per X-slice, per component: a 3-pane figure
        [ iteration 1 | iteration 2 | iteration 2 - iteration 1 ]
  * a correlation summary: iter1 vs iter2 scatter for each component, with
    the voxel-wise mean/RMS shift.

The shift is the effect of dropping every hit whose residual against the
iteration-1 fit exceeded `--iter-resid-cm` (default 4 cm).

Usage:
    python compare_iter1_iter2.py
        [--input doubleanode_2d_maps_20cm_iter2/voxel_data.npz]
        [--out compare_iter1_iter2] [--vrange 5] [--min-counts 30]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COMPONENTS = ("dx", "dy", "dz")
HEATMAP = matplotlib.colormaps["RdYlBu_r"].copy()
HEATMAP.set_bad("white")


def comp_maps(delta, cnt, min_counts):
    """(nx,ny,nz,3) delta + (nx,ny,nz) count -> {comp: (nx,ny,nz) masked}."""
    return {c: np.where(cnt >= min_counts, delta[..., i], np.nan)
            for i, c in enumerate(COMPONENTS)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",
                    default="doubleanode_2d_maps_20cm_iter2/voxel_data.npz",
                    help="voxel_data.npz from a >= 2 iteration run")
    ap.add_argument("--out", default="compare_iter1_iter2")
    ap.add_argument("--vrange", type=float, default=5.0)
    ap.add_argument("--min-counts", type=int, default=30)
    args = ap.parse_args()

    d = np.load(args.input)
    if "delta_mu_all_iter1" not in d:
        print(f"{args.input} has no 'delta_mu_all_iter1' -- it was built with "
              f"a single iteration (n_iterations="
              f"{int(d['n_iterations']) if 'n_iterations' in d else 1}). "
              f"Nothing to compare.", file=sys.stderr)
        sys.exit(1)

    iter_cut = float(d["iter_resid_cm"]) if "iter_resid_cm" in d else None
    mu1 = comp_maps(d["delta_mu_all_iter1"].astype(np.float64),
                    d["count_all_iter1"], args.min_counts)
    mu2 = comp_maps(d["delta_mu_all"].astype(np.float64),
                    d["count_all"], args.min_counts)
    xe, ye, ze = d["x_edges"], d["y_edges"], d["z_edges"]
    nx = len(xe) - 1
    os.makedirs(args.out, exist_ok=True)
    extent = (ze[0], ze[-1], ye[0], ye[-1])

    # ---- per-X-slice 3-pane figures ----
    for comp in COMPONENTS:
        cdir = os.path.join(args.out, comp)
        os.makedirs(cdir, exist_ok=True)
        for ix in range(nx):
            g1 = mu1[comp][ix]
            g2 = mu2[comp][ix]
            diff = g2 - g1
            if not (np.isfinite(g1).any() or np.isfinite(g2).any()):
                continue
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            for ax, (label, grid) in zip(axes[:2],
                                         [("iteration 1", g1),
                                          ("iteration 2", g2)]):
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
                         label="iter2 - iter1 [cm]")
            dv = diff[np.isfinite(diff)]
            axes[2].set_title(f"iter2 - iter1  (N={dv.size}, "
                              f"mean={dv.mean() if dv.size else 0:+.2f}, "
                              f"rms={dv.std() if dv.size else 0:.2f} cm)")
            axes[2].set_xlabel("Z [cm]"); axes[2].set_ylabel("Y [cm]")
            x_lo = xe[ix]
            cut_txt = f", |r| cut {iter_cut:g} cm" if iter_cut else ""
            fig.suptitle(f"double-anode $\\hat{{\\Delta}}_{{{comp[1]}}}$, "
                         f"X in [{x_lo:+.0f}, {xe[ix+1]:+.0f}] cm{cut_txt}")
            fig.tight_layout()
            tag = f"x{int(round(x_lo)):+04d}"
            fig.savefig(os.path.join(cdir, f"{tag}.png"), dpi=110)
            fig.savefig(os.path.join(cdir, f"{tag}.pdf"))
            plt.close(fig)

    # ---- correlation summary ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, comp in zip(axes, COMPONENTS):
        a = mu2[comp].ravel()
        b = mu1[comp].ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        ax.scatter(b, a, s=4, alpha=0.3, color="navy")
        lim = args.vrange * 1.6
        ax.plot([-lim, lim], [-lim, lim], "r-", lw=1)
        if a.size > 2:
            r = np.corrcoef(a, b)[0, 1]
            shift = a - b
            ax.set_title(f"$\\Delta_{{{comp[1]}}}$: N={a.size}, r={r:.4f}, "
                         f"<iter2-iter1>={shift.mean():+.3f}, "
                         f"rms={shift.std():.3f} cm")
        ax.set_xlabel(f"iteration 1 $\\Delta_{{{comp[1]}}}$ [cm]")
        ax.set_ylabel(f"iteration 2 $\\Delta_{{{comp[1]}}}$ [cm]")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.grid(alpha=0.3)
    cut_txt = f" (|r| > {iter_cut:g} cm dropped)" if iter_cut else ""
    fig.suptitle("double-anode displacement: iteration 1 vs iteration 2 "
                 f"(per-voxel){cut_txt}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "correlation_summary.png"), dpi=120)
    fig.savefig(os.path.join(args.out, "correlation_summary.pdf"))
    plt.close(fig)
    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
