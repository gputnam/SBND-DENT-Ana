"""Knot-count scan for cathode_step2_spline.

Runs the same fit pipeline as cathode_step2_spline.py at several (ny, nz)
knot settings, prints RMS, and writes a single summary plot. No per-knot
NPZs/PNGs are written — that's still cathode_step2_spline.py's job.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cathode_step2_spline import (
    NATIVE_Y_LO, NATIVE_Y_HI, NATIVE_Z_LO, NATIVE_Z_HI, NATIVE_BIN,
    neighbor_isolation_mask, fit_spline,
)

STEP1_DIR = "/exp/sbnd/data/users/gputnam/efield-study/cathode_pipeline/step1_displacement"
OUT_PNG   = "/exp/sbnd/data/users/gputnam/efield-study/cathode_pipeline/step2_spline/png/knot_scan_rms.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neighbor-thresh", type=float, default=0.7)
    ap.add_argument("--weight-floor-cm", type=float, default=0.1)
    ap.add_argument("--knots", type=str,
                    default="8,12;12,18;16,24;20,30;24,36;28,42;32,48",
                    help="semicolon-separated 'ny_knots,nz_knots' pairs")
    args = ap.parse_args()

    pairs = []
    for tok in args.knots.split(";"):
        ny, nz = tok.split(",")
        pairs.append((int(ny.strip()), int(nz.strip())))

    results = {0: [], 1: []}
    for tpc in (0, 1):
        d = np.load(os.path.join(STEP1_DIR, f"tpc{tpc}_displacement.npz"))
        x_measured  = d["x"].astype(np.float64)
        mu_err      = d["mu_err"].astype(np.float64)
        cm_per_tick = float(d["cm_per_tick"])
        y_native    = d["y_centers"]; z_native = d["z_centers"]
        mu_err_cm   = mu_err * abs(cm_per_tick)
        keep, _ = neighbor_isolation_mask(x_measured,
                                          threshold=args.neighbor_thresh)
        x_clean = np.where(keep, x_measured, np.nan)
        for ny, nz in pairs:
            spline = fit_spline(x_clean, mu_err_cm, y_native, z_native,
                                ny_knots=ny, nz_knots=nz,
                                weight_floor=args.weight_floor_cm)
            x_fit = spline(y_native, z_native)
            res = x_clean - x_fit
            rms = float(np.sqrt(np.nanmean(res ** 2)))
            results[tpc].append((ny, nz, rms))
            print(f"  TPC{tpc} (ny={ny:2d}, nz={nz:2d}): RMS = {rms:.4f} cm",
                  flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for tpc, marker in ((0, "o"), (1, "s")):
        ny_arr = [r[0] for r in results[tpc]]
        nz_arr = [r[1] for r in results[tpc]]
        rms_arr = [r[2] for r in results[tpc]]
        xs = np.arange(len(rms_arr))
        ax.plot(xs, rms_arr, marker=marker, lw=1.2, label=f"TPC{tpc}")
        for x, r in zip(xs, rms_arr):
            ax.annotate(f"{r:.3f}", (x, r), xytext=(3, 3),
                        textcoords="offset points", fontsize=8)
    ax.set_xticks(np.arange(len(pairs)))
    ax.set_xticklabels([f"{p[0]}x{p[1]}" for p in pairs], rotation=20)
    ax.set_xlabel("ny_knots x nz_knots")
    ax.set_ylabel("fit RMS over kept native pixels [cm]")
    ax.set_title("Cathode spline fit: residual RMS vs knot count")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"\nwrote {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
