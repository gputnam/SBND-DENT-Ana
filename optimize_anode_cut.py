"""Optimise the anode-disagreement quality cut for the 2D double-anode
field-distortion analysis.

Figure of merit
---------------
Each double-anode track pierces two anodes; at each anode up to three
wire planes have an anode-piercing hit, each giving a view coordinate
s_p = what_p . (y, z). Three planes over-determine the anode (y, z): the
least-squares fit leaves a residual r_p per plane. The per-anode
disagreement is the RMS of those residuals, and the per-track

    FOM  =  max( RMS_resid(anode TPC0),  RMS_resid(anode TPC1) )   [cm]

A 2-plane anode fits exactly (RMS = 0), so the FOM is driven by the
worst over-determined anode. A small FOM means the three planes' anode
hits are mutually consistent; a large FOM flags a mis-identified anode
hit (a delta ray / noise hit picked up instead of the true pierce).

Cut
---
The FOM distribution is a resolution-limited core plus a junk tail. The
cut is placed by a robust core estimate:

    cut = median(FOM) + n_sigma * (1.4826 * MAD)

(MAD = median absolute deviation; 1.4826*MAD is the Gaussian-equivalent
sigma). n_sigma is a CLI knob (default 5); the script prints a scan over
n_sigma so the trade-off is visible, and draws the chosen cut on the
distribution.

Outputs
-------
  cut_tuning/anode_fom_distribution.{png,pdf}   linear + log distribution
  cut_tuning/anode_cut.json                     the cut value (read by
                                                plot_doubleanode_2d_voxels.py)

Usage
-----
    python optimize_anode_cut.py
        [--input /.../doubleanode_2d_df] [--wire-geom wire_geometry_sbnd.json]
        [--max-shards 2] [--n-sigma 5] [--anode-dt-window 40]
        [--out-dir cut_tuning]
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from plot_doubleanode_2d_voxels import load_wire_geom, reconstruct_anodes
from crtlib.constants import DATA_DIR

_REPO = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(DATA_DIR, "doubleanode_2d_df")


def shard_fom(path, geom, anode_dt_window):
    """Per-track anode-disagreement FOM for one parquet shard."""
    SLOPE, OFFSET, WHATY, WHATZ = geom
    df = pd.read_parquet(path, columns=["run", "subrun", "event", "trk_id",
                                        "plane", "tpc", "wire", "time"])
    tkey = df.groupby(["run", "subrun", "event", "trk_id"]).ngroup().to_numpy()
    if len(tkey) == 0:
        return np.zeros(0)
    ntrk = int(tkey.max()) + 1
    plane = df["plane"].to_numpy()
    tpc = df["tpc"].to_numpy()
    wire = df["wire"].to_numpy(np.float64)
    time = df["time"].to_numpy(np.float64)
    del df
    s = SLOPE[tpc, plane] * wire + OFFSET[tpc, plane]
    whaty = WHATY[tpc, plane]
    whatz = WHATZ[tpc, plane]
    _, _, fom, _ = reconstruct_anodes(tkey, tpc, plane, time, s,
                                      whaty, whatz, ntrk, anode_dt_window)
    return fom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT_DIR)
    ap.add_argument("--wire-geom",
                    default=os.path.join(_REPO, "config", "wire_geometry_sbnd.json"))
    ap.add_argument("--max-shards", type=int, default=2,
                    help="shards to sample for the FOM distribution "
                         "(default 2 -- already ~80k tracks)")
    ap.add_argument("--n-sigma", type=float, default=5.0,
                    help="cut = median + n_sigma * MAD-sigma (default 5)")
    ap.add_argument("--anode-dt-window", type=float, default=40.0)
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "config", "cut_tuning"))
    ap.add_argument("--out-config", default=None,
                    help="default <out-dir>/anode_cut.json")
    args = ap.parse_args()
    out_config = args.out_config or os.path.join(args.out_dir,
                                                 "anode_cut.json")

    if os.path.isdir(args.input):
        shards = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    else:
        shards = sorted(glob.glob(args.input))
    if not shards:
        print(f"No parquet shards at {args.input}", file=sys.stderr)
        sys.exit(1)
    if args.max_shards:
        shards = shards[:args.max_shards]
    print(f"reading {len(shards)} shard(s) from {args.input}", flush=True)

    geom = load_wire_geom(args.wire_geom)
    fom_all = []
    for path in tqdm(shards, unit="shard", file=sys.stderr):
        fom_all.append(shard_fom(path, geom, args.anode_dt_window))
    fom_all = np.concatenate(fom_all)
    n_tracks = fom_all.size

    # FOM == 0 exactly => the track had no over-determined (3-plane) anode,
    # so its disagreement is unmeasurable; exclude from the core estimate.
    meaningful = fom_all[fom_all > 0]
    n_meas = meaningful.size
    if n_meas < 100:
        print("too few tracks with a 3-plane anode", file=sys.stderr)
        sys.exit(1)

    median = float(np.median(meaningful))
    mad = float(np.median(np.abs(meaningful - median)))
    sigma = 1.4826 * mad
    cut = median + args.n_sigma * sigma

    print(f"\ntracks: {n_tracks:,} total, {n_meas:,} with a 3-plane anode "
          f"(FOM measurable)", flush=True)
    print(f"FOM core: median = {median:.4f} cm, "
          f"MAD-sigma = {sigma:.4f} cm", flush=True)
    pcts = [50, 90, 95, 99, 99.9]
    pv = np.percentile(meaningful, pcts)
    print("FOM percentiles: "
          + ", ".join(f"{p}%={v:.3f}" for p, v in zip(pcts, pv)), flush=True)
    print(f"\nn_sigma scan (cut = median + n*MAD-sigma):", flush=True)
    for n in (3, 4, 5, 6, 7, 8):
        c = median + n * sigma
        keep_m = float(np.mean(meaningful <= c))
        keep_a = float(np.mean(fom_all <= c))
        print(f"  n={n}: cut={c:.3f} cm  keep(3-plane)={keep_m*100:.2f}%  "
              f"keep(all)={keep_a*100:.2f}%", flush=True)

    keep_meas = float(np.mean(meaningful <= cut))
    keep_all = float(np.mean(fom_all <= cut))
    print(f"\n>>> chosen cut (n_sigma={args.n_sigma}): {cut:.3f} cm  "
          f"keep(3-plane)={keep_meas*100:.2f}%  "
          f"keep(all)={keep_all*100:.2f}%", flush=True)

    # ---- distribution plot: linear + log ----
    os.makedirs(args.out_dir, exist_ok=True)
    xmax = min(max(2.0 * cut, float(np.percentile(meaningful, 99.7))), 30.0)
    bins = np.linspace(0.0, xmax, 161)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, logy in zip(axes, (False, True)):
        ax.hist(meaningful, bins=bins, color="steelblue",
                edgecolor="white", linewidth=0.3)
        ax.axvline(cut, color="red", ls="--", lw=1.8,
                   label=f"cut = {cut:.3f} cm  "
                         f"(median + {args.n_sigma:g}$\\sigma_{{MAD}}$)")
        ax.axvline(median, color="green", ls=":", lw=1.2,
                   label=f"median = {median:.3f} cm")
        if logy:
            ax.set_yscale("log")
            ax.set_title("log scale")
        else:
            ax.set_title("linear scale")
        ax.set_xlabel("anode-disagreement FOM "
                      "[cm]  (max-anode RMS of 3-plane fit residuals)")
        ax.set_ylabel("tracks")
        ax.set_xlim(0, xmax)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(f"SBND 2D double-anode: anode-disagreement FOM "
                 f"({n_meas:,} tracks with a 3-plane anode;  "
                 f"keep {keep_meas*100:.1f}%)")
    fig.tight_layout()
    png = os.path.join(args.out_dir, "anode_fom_distribution.png")
    fig.savefig(png, dpi=120)
    fig.savefig(os.path.join(args.out_dir, "anode_fom_distribution.pdf"))
    plt.close(fig)
    print(f"\nwrote {png}", flush=True)

    # ---- cut config ----
    with open(out_config, "w") as fh:
        json.dump({
            "comment": "Anode-disagreement quality cut for the 2D "
                       "double-anode analysis. Discard a track if FOM > "
                       "cut_cm.",
            "fom": "per-track max over the two TPC anodes of the RMS of "
                   "the 3-plane anode-(y,z)-fit residuals (cm)",
            "method": "cut = median + n_sigma * (1.4826 * MAD)",
            "cut_cm": cut,
            "n_sigma": args.n_sigma,
            "median_cm": median,
            "mad_sigma_cm": sigma,
            "anode_dt_window_ticks": args.anode_dt_window,
            "n_tracks_sampled": int(n_tracks),
            "n_tracks_3plane_anode": int(n_meas),
            "keep_fraction_3plane": keep_meas,
            "keep_fraction_all": keep_all,
        }, fh, indent=2)
    print(f"wrote {out_config}", flush=True)


if __name__ == "__main__":
    main()
