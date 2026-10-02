"""Compare pre- and post-alignment residual distributions for the
per-module and wall-only fits.

Reads residuals_pre.npy and residuals_post.npy from each fit's output
directory and overlays them on a single figure-of-merit histogram.
The pre-fit residuals from the two fits are bitwise-identical (same
data, same θ=0 starting point), so we plot one "pre" curve.

Usage:
    python plot_residual_comparison.py \\
        --module-dir crt_alignment_137k \\
        --wall-dir   crt_alignment_137k_wall \\
        --out residual_comparison
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CURVES = [
    # (label_for_legend, dirname_default, color, linewidth)
    ("post: module fit, 6 DOF/module (636 params)",
     "crt_alignment_137k",       "tab:blue",   1.7),
    ("post: wall fit, 6 DOF/wall (48 params)",
     "crt_alignment_137k_wall",  "tab:orange", 1.7),
    ("post: wall fit, translation only (24 params)",
     "crt_alignment_137k_wall_norot", "tab:green", 1.7),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module-dir", default="crt_alignment_137k")
    ap.add_argument("--wall-dir", default="crt_alignment_137k_wall")
    ap.add_argument("--wall-norot-dir", default="crt_alignment_137k_wall_norot")
    ap.add_argument("--out", default="residual_comparison",
                    help="basename (no extension) for the output figure(s)")
    ap.add_argument("--xmax", type=float, default=50.0)
    ap.add_argument("--bins", type=int, default=100)
    args = ap.parse_args()

    dirs = {"module": args.module_dir, "wall": args.wall_dir,
            "wall_norot": args.wall_norot_dir}
    pre = np.load(os.path.join(dirs["module"], "residuals_pre.npy"))

    # Pre arrays should be identical across all directories.
    for name, d in dirs.items():
        a = np.load(os.path.join(d, "residuals_pre.npy"))
        if a.shape == pre.shape:
            md = float(np.max(np.abs(pre - a)))
            print(f"  pre array consistency vs {name}: max diff {md:.2e} cm")
        else:
            print(f"  WARNING: {name} pre shape {a.shape} != {pre.shape}")

    posts = {
        "module":     np.load(os.path.join(dirs["module"], "residuals_post.npy")),
        "wall":       np.load(os.path.join(dirs["wall"], "residuals_post.npy")),
        "wall_norot": np.load(os.path.join(dirs["wall_norot"], "residuals_post.npy")),
    }

    print(f"\nN entries: {len(pre):,}")
    print(f"Median residuals (cm):")
    print(f"  pre                                         : {np.median(pre):.3f}")
    for k in ("module", "wall", "wall_norot"):
        print(f"  post — {k:<10s} : {np.median(posts[k]):.3f}")

    # ---- Overlaid step-histogram, log-y ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    edges = np.linspace(0, args.xmax, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    h_pre, _ = np.histogram(pre, bins=edges)
    ax.step(centers, h_pre, where="mid", linewidth=2.0, color="0.35",
            label=f"pre-alignment (median {np.median(pre):.2f} cm)")
    ax.axvline(np.median(pre), color="0.35", linestyle=":", linewidth=0.9)

    palette = {"module": "tab:blue", "wall": "tab:orange",
               "wall_norot": "tab:green"}
    label = {
        "module":     f"module fit, 6 DOF×105 modules (636 params)",
        "wall":       f"wall fit, 6 DOF×7 walls (48 params)",
        "wall_norot": f"wall fit, translation only (24 params)",
    }
    for k in ("module", "wall", "wall_norot"):
        h, _ = np.histogram(posts[k], bins=edges)
        med = np.median(posts[k])
        ax.step(centers, h, where="mid", linewidth=1.6,
                color=palette[k],
                label=f"{label[k]} (median {med:.2f} cm)")
        ax.axvline(med, color=palette[k], linestyle=":", linewidth=0.9)

    ax.set_xlim(0, args.xmax)
    ax.set_yscale("log")
    ax.set_xlabel("perpendicular residual |Pa − line(P1, P2)| (cm)")
    ax.set_ylabel("residual rows / bin")
    ax.set_title(f"Anode-to-CRT-line straightness: alignment-mode comparison "
                 f"(N = {len(pre):,} residuals)")
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=140)
    plt.close(fig)

    # Cumulative on the same data.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    F = np.linspace(0, 1, len(pre), endpoint=True)
    ax.plot(np.sort(pre), F, color="0.35", lw=2.0, label="pre-alignment")
    for k in ("module", "wall", "wall_norot"):
        ax.plot(np.sort(posts[k]), F, color=palette[k], lw=1.6, label=label[k])
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
