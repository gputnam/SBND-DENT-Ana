"""Linear-y residual histogram: wall-GDML vs module-GDML vs unoptimized.

Loads residuals_pre.npy / residuals_post.npy from the two GDML-pivot
fits and overlays them on one linear-y figure (and a CDF).

  pre-alignment            : θ = 0 (nominal GDML)
  wall fit, GDML pivot     : crt_alignment_137k_wall_gdml/post
  module fit, GDML pivot   : crt_alignment_137k_gdml/post
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-dir", default="crt_alignment_137k_gdml")
    ap.add_argument("--wall-dir",   default="crt_alignment_137k_wall_gdml")
    ap.add_argument("--out",        default="residual_compare_gdml")
    ap.add_argument("--xmax",       type=float, default=50.0)
    ap.add_argument("--bins",       type=int,   default=50)
    args = ap.parse_args()

    pre   = np.load(os.path.join(args.module_dir, "residuals_pre.npy"))
    pre_w = np.load(os.path.join(args.wall_dir,   "residuals_pre.npy"))
    if pre.shape == pre_w.shape:
        md = float(np.max(np.abs(pre - pre_w)))
        print(f"  pre array consistency: max diff {md:.2e} cm")
    else:
        print(f"  WARNING: pre shapes differ ({pre.shape} vs {pre_w.shape})")

    post_module = np.load(os.path.join(args.module_dir, "residuals_post.npy"))
    post_wall   = np.load(os.path.join(args.wall_dir,   "residuals_post.npy"))

    print(f"\nN residuals : {len(pre):,}")
    print(f"medians (cm):  pre={np.median(pre):.3f}  "
          f"wall-GDML={np.median(post_wall):.3f}  "
          f"module-GDML={np.median(post_module):.3f}")

    edges = np.linspace(0, args.xmax, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    pre_label  = f"pre-alignment (median {np.median(pre):.2f} cm)"
    wall_label = (f"wall fit, GDML pivot, 6 DOF×7 walls (48 params) "
                  f"(median {np.median(post_wall):.2f} cm)")
    mod_label  = (f"module fit, GDML pivot, 6 DOF×105 modules (636 params) "
                  f"(median {np.median(post_module):.2f} cm)")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    h_pre, _  = np.histogram(pre,         bins=edges)
    h_wall, _ = np.histogram(post_wall,   bins=edges)
    h_mod, _  = np.histogram(post_module, bins=edges)
    ax.step(centers, h_pre,  where="mid", lw=2.0, color="0.35",
            label=pre_label)
    ax.step(centers, h_wall, where="mid", lw=1.7, color="tab:orange",
            label=wall_label)
    ax.step(centers, h_mod,  where="mid", lw=1.7, color="tab:blue",
            label=mod_label)
    ax.set_xlim(0, args.xmax)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("perpendicular residual |Pa − line(P1, P2)| (cm)")
    ax.set_ylabel("residual rows / bin")
    ax.set_title(f"Anode-to-CRT-line residual: GDML-pivot fits vs unoptimized "
                 f"(N = {len(pre):,})")
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    F = np.linspace(0, 1, len(pre), endpoint=True)
    ax.plot(np.sort(pre),         F, color="0.35",       lw=2.0, label=pre_label)
    ax.plot(np.sort(post_wall),   F, color="tab:orange", lw=1.7, label=wall_label)
    ax.plot(np.sort(post_module), F, color="tab:blue",   lw=1.7, label=mod_label)
    ax.set_xlim(0, args.xmax)
    ax.set_xlabel("perpendicular residual (cm)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("CDF of straightness residual")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}_cdf.{ext}", dpi=140)
    plt.close(fig)

    print(f"\n  wrote {args.out}.png/.pdf and {args.out}_cdf.png/.pdf")


if __name__ == "__main__":
    main()
