#!/usr/bin/env python3
"""Compare the A/B/C truth-hit pipeline maps (build_truthhit_parquets_mc.py).

  A = maps_true   (true arrival times: geometry + drift-time distortion)
  B = maps_reco   (reco hit times:     geometry + drift-time + hit reco)
  C = maps_nosce  (synthetic nominal times from true x: geometry only)

Overlays the y/z-averaged dx/dy/dz profiles of the three maps with the
as-applied spatial field (dual map, physical convention) so the measured
dome decomposes visibly:  C = chord/geometry artifact;  A-C = drift-time
distortion;  B-A = hit-reconstruction time bias.

Usage:
    python compare_abc_maps.py --base truthhit_pq --out truthhit_pq/compare
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import build_sce_input as B
import compare_2d_vs_truth as C2

COMPONENTS = ("dx", "dy", "dz")


def load(npz):
    d = np.load(npz)
    mu = d["delta_mu_all"].astype(np.float64)
    cnt = d["count_all"]
    keep = np.isfinite(mu).all(-1) & (cnt >= 20)
    return np.where(keep[..., None], mu, np.nan), \
        B.voxel_centers(d["x_edges"]), keep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="truthhit_pq")
    ap.add_argument("--truth-root",
                    default="SCEoffsets_SBND_E500_dualmap_voxelTH3.root")
    ap.add_argument("--out", default="truthhit_pq/compare")
    ap.add_argument("--suffix", default="",
                    help="maps dir suffix, e.g. _extrap for maps_true_extrap")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    maps = {}
    keep_all = None
    for tag in ("true", "reco", "nosce"):
        npz = os.path.join(args.base, f"maps_{tag}{args.suffix}",
                           "voxel_data.npz")
        mu, vx, keep = load(npz)
        maps[tag] = mu
        keep_all = keep if keep_all is None else (keep_all & keep)
    d0 = np.load(os.path.join(args.base, f"maps_true{args.suffix}",
                              "voxel_data.npz"))
    vy = B.voxel_centers(d0["y_edges"])
    vz = B.voxel_centers(d0["z_edges"])

    # spatial truth (dual map raw); for dx also show the PHYSICAL sign
    truth = C2.sample_truth_at(vx, vy, vz, args.truth_root, unflip_x=False)

    labels = {"nosce": "C: nominal times (geometry only)",
              "true": "A: true arrival times (+drift-time dist.)",
              "reco": "B: reco hit times (+hit reco)"}
    colors = {"nosce": "tab:green", "true": "tab:blue", "reco": "tab:red"}

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    for ci, (ax, comp) in enumerate(zip(axes, COMPONENTS)):
        for tag in ("nosce", "true", "reco"):
            arr = np.where(keep_all[..., None], maps[tag], np.nan)[..., ci]
            prof = np.nanmean(arr, axis=(1, 2))
            ax.plot(vx, prof, "o-", ms=3.5, lw=1.3, color=colors[tag],
                    label=labels[tag])
        tprof = np.nanmean(np.where(keep_all, truth[..., ci], np.nan),
                           axis=(1, 2))
        ax.plot(vx, tprof, "k--", lw=1.8,
                label="dual map (raw stored)")
        ax.axhline(0, color="0.6", ls=":", lw=0.7)
        ax.axvline(0, color="0.85", lw=0.6)
        ax.set_xlabel("X [cm]")
        ax.set_ylabel(f"$\\Delta_{{{comp[1]}}}$ [cm]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("A/B/C truth-hit forensics: same tracks/wires, different "
                 "hit times through the full 2D pipeline")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "abc_profiles.png"), dpi=120)
    fig.savefig(os.path.join(args.out, "abc_profiles.pdf"))
    plt.close(fig)

    # summary numbers: dx amplitude per variant
    print(f"{'variant':28s} {'dx min':>8s} {'dx max':>8s} "
          f"{'dy slope vs dual':>17s}")
    tr = truth.reshape(-1, 3)
    for tag in ("nosce", "true", "reco"):
        m3 = np.where(keep_all[..., None], maps[tag], np.nan)
        prof = np.nanmean(m3[..., 0], axis=(1, 2))
        mm = m3.reshape(-1, 3)
        ok = np.isfinite(mm[:, 1]) & np.isfinite(tr[:, 1])
        sig = ok & (np.abs(tr[:, 1]) > 0.3)
        sl = (mm[sig, 1] * tr[sig, 1]).sum() / (tr[sig, 1] ** 2).sum()
        print(f"{labels[tag]:28s} {np.nanmin(prof):>8.3f} "
              f"{np.nanmax(prof):>8.3f} {sl:>17.3f}")
    print(f"wrote {args.out}/", flush=True)


if __name__ == "__main__":
    main()
