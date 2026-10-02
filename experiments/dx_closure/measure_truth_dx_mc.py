#!/usr/bin/env python3
"""Direct truth measurement of the apparent drift-direction displacement in
the anode-muon-gun MC -- no chord, no inversion.

The double-anode diagnostics (diagnose_dx_offsets.py) show a coherent dx
excess over the dual-map truth that is linear in drift distance (~0 at each
anode, ~1 cm at the cathode, identical in all three wire planes) -- the
signature of an effective drift-velocity mismatch -- while the
cathode-crossing timing study (CATHODE_DRIFT_MC.md) measured full-drift
times consistent with the sim-nominal v. This script resolves the
contradiction from the MC truth itself, using the truth-matched hits in the
calibration ntuples (`trk/truth/truth.p.truehitsN.*`):

    p.x        true (energy-deposit) position
    p_scecorr.x  SCE-displaced position the simulation drifted from
    time       reconstructed hit time [ticks]
    tdrift     true drift time [us]

Per hit we form, and profile against true x:

    dx_spatial  = p_scecorr.x - p.x                (the as-applied SCE map)
    x_reco      = anode_x -/+ (time - t0) * 0.5us * v_reco
    dx_apparent = x_reco - p.x                     (what a perfect voxel map
                                                    should measure)
    dx_timing   = x_reco - p_scecorr.x             (drift-time part: linear
                                                    in drift distance <=> a
                                                    velocity mismatch)
    v_eff       = dist(p_scecorr.x -> anode)/tdrift  (the actual simulated
                                                      drift velocity)

Usage:
    python measure_truth_dx_mc.py --nfiles 100 --nproc 16 --out truth_dx_mc
"""

import argparse
import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

TICK_US = 0.5
V_RECO = 0.15627
T0 = 418.5
ANODE_X = (-200.0, 200.0)
NXBIN = 40                       # 10 cm columns over [-200, 200]

_ARGS = None


def _init(args):
    global _ARGS
    _ARGS = args


def read_file(url):
    """One ntuple -> per-hit arrays (tpc, plane, x_true, x_sce, time, tdrift).
    Returns dict of flat arrays or None on failure."""
    import uproot
    import awkward as ak
    cols = {}
    try:
        with uproot.open(url) as f:
            t = f["caloskim/TrackCaloSkim"]
            for p in (0, 1, 2):
                pre = f"truth.p.truehits{p}"
                arrs = t.arrays(
                    [f"{pre}.tpc", f"{pre}.p.x", f"{pre}.p_scecorr.x",
                     f"{pre}.time", f"{pre}.tdrift"],
                    library="ak")
                cols[p] = {k.split(".", 3)[-1]: ak.to_numpy(
                    ak.flatten(arrs[k], axis=None)) for k in arrs.fields}
    except Exception as e:
        print(f"skip {url.rsplit('/', 1)[-1][:50]}: {type(e).__name__} {e}",
              file=sys.stderr)
        return None
    tpc = np.concatenate([cols[p]["tpc"] for p in (0, 1, 2)])
    plane = np.concatenate([np.full(len(cols[p]["tpc"]), p, dtype=np.int8)
                            for p in (0, 1, 2)])
    return dict(tpc=tpc, plane=plane,
                x_true=np.concatenate([cols[p]["p.x"] for p in (0, 1, 2)]),
                x_sce=np.concatenate([cols[p]["p_scecorr.x"]
                                      for p in (0, 1, 2)]),
                time=np.concatenate([cols[p]["time"] for p in (0, 1, 2)]),
                tdrift=np.concatenate([cols[p]["tdrift"]
                                       for p in (0, 1, 2)]))


def profile(x, y, bins, rng):
    """Mean/err of y in bins of x."""
    idx = np.floor((x - rng[0]) / (rng[1] - rng[0]) * bins).astype(np.int64)
    ok = (idx >= 0) & (idx < bins) & np.isfinite(y)
    n = np.bincount(idx[ok], minlength=bins).astype(float)
    s = np.bincount(idx[ok], weights=y[ok], minlength=bins)
    s2 = np.bincount(idx[ok], weights=y[ok] ** 2, minlength=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / n, np.nan)
        rms = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mean ** 2, 0))
        err = rms / np.sqrt(np.maximum(n, 1))
    return mean, err, n


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filelist", default="filelist_mc_doubleanode_xroot.txt")
    ap.add_argument("--nfiles", type=int, default=100)
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--vreco", type=float, default=V_RECO)
    ap.add_argument("--t0", type=float, default=T0)
    ap.add_argument("--out", default="truth_dx_mc")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    urls = [l.strip() for l in open(args.filelist) if l.strip()][:args.nfiles]
    print(f"reading {len(urls)} MC ntuples ({args.nproc} workers)", flush=True)

    parts = []
    if args.nproc > 1:
        with Pool(args.nproc) as pool:
            for r in tqdm(pool.imap_unordered(read_file, urls),
                          total=len(urls), unit="file", file=sys.stderr):
                if r is not None:
                    parts.append(r)
    else:
        for u in tqdm(urls, unit="file", file=sys.stderr):
            r = read_file(u)
            if r is not None:
                parts.append(r)
    if not parts:
        sys.exit("no files read")
    D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n = len(D["tpc"])
    print(f"hits: {n:,}", flush=True)

    tpc = D["tpc"].astype(np.int64)
    anodex = np.where(tpc == 0, ANODE_X[0], ANODE_X[1])
    xdir = np.where(tpc == 0, 1.0, -1.0)
    # truehits 'time' is on a different clock epoch than the recob hits the
    # 2D pipeline reads (anode intercept ~3428 ticks, not 418.5; it also
    # carries a constant ~12 us transit lag). Calibrate the epoch per TPC
    # from near-anode hits so dx_timing is anchored to 0 at the anode --
    # only the DRIFT-DISTANCE DEPENDENCE of the timing residual is physical.
    drift0 = np.abs(D["x_sce"] - anodex)
    t0_nt = np.zeros(2)
    for t in (0, 1):
        m = (tpc == t) & (drift0 < 30) & np.isfinite(D["time"])
        t0_nt[t] = np.median(D["time"][m]
                             - drift0[m] / (TICK_US * args.vreco))
    print(f"empirical truehits time epoch per TPC [ticks]: {t0_nt}",
          flush=True)
    x_reco = anodex + xdir * (D["time"] - t0_nt[tpc]) * TICK_US * args.vreco
    dx_spatial = D["x_sce"] - D["x_true"]
    dx_apparent = x_reco - D["x_true"]
    dx_timing = x_reco - D["x_sce"]
    drift_dist = np.abs(D["x_sce"] - anodex)
    with np.errstate(invalid="ignore", divide="ignore"):
        v_eff = np.where(D["tdrift"] > 50, drift_dist / D["tdrift"], np.nan)

    # ---- printed summary: effective v per TPC -------------------------------
    print("\nper-TPC effective drift velocity from truth "
          "(dist(p_scecorr->anode)/tdrift, tdrift>300us):")
    for t in (0, 1):
        m = (tpc == t) & (D["tdrift"] > 300) & np.isfinite(v_eff)
        print(f"  TPC{t}: median v_eff = {np.median(v_eff[m]):.5f} cm/us "
              f"(reco assumed {args.vreco}); N={m.sum():,}", flush=True)
    # timing slope: dx_timing vs drift distance -> effective v of the full
    # chain (sim drift + hit-time reconstruction)
    print("\nfitted dx_timing = a + b * drift_dist  (per TPC, both = "
          "reco-time chain vs spatial truth):")
    for t in (0, 1):
        m = (tpc == t) & np.isfinite(dx_timing) & (drift_dist > 5) \
            & (drift_dist < 195) & (np.abs(dx_timing) < 5)
        if m.sum() < 100:
            print(f"  TPC{t}: too few hits in window ({m.sum()})", flush=True)
            continue
        A = np.stack([np.ones(m.sum()), drift_dist[m]], axis=1)
        coef, *_ = np.linalg.lstsq(A, np.abs(dx_timing[m])
                                   * np.sign(dx_timing[m]), rcond=None)
        sgn = 1.0 if t == 0 else -1.0
        v_imp = args.vreco / (1 + sgn * coef[1])
        print(f"  TPC{t}: a={coef[0]:+.3f} cm  b={coef[1]:+.5f} cm/cm "
              f"-> implied v_true = {v_imp:.5f} cm/us; N={m.sum():,}",
              flush=True)

    # ---- profiles vs true x ---------------------------------------------------
    rng = (-200.0, 200.0)
    edges = np.linspace(*rng, NXBIN + 1)
    cx = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for ax, (label, val) in zip(axes, (
            ("apparent: $x_{reco} - x_{true}$", dx_apparent),
            ("spatial: $x_{sce} - x_{true}$ (as-applied map)", dx_spatial),
            ("timing: $x_{reco} - x_{sce}$", dx_timing))):
        for t, color in ((0, "tab:blue"), (1, "tab:red")):
            m = tpc == t
            mean, err, cnt = profile(D["x_true"][m], val[m], NXBIN, rng)
            ax.errorbar(cx, mean, yerr=err, fmt="o-", ms=3, lw=1.2,
                        color=color, label=f"TPC{t}")
        ax.axhline(0, color="0.6", ls=":", lw=0.7)
        ax.axvline(0, color="0.85", lw=0.6)
        ax.set_xlabel("true X [cm]")
        ax.set_ylabel("[cm]")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"MC truth decomposition of the apparent drift-direction "
                 f"displacement (v_reco={args.vreco})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "dx_truth_decomposition.png"), dpi=120)
    fig.savefig(os.path.join(args.out, "dx_truth_decomposition.pdf"))
    plt.close(fig)

    # per-plane apparent dx (consistency with the voxel-map diagnostics)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    colors = ("tab:blue", "tab:orange", "tab:green")
    for t, ax in enumerate(axes):
        for p in range(3):
            m = (tpc == t) & (D["plane"] == p)
            mean, err, _ = profile(D["x_true"][m], dx_apparent[m], NXBIN, rng)
            ax.errorbar(cx, mean, yerr=err, fmt="o-", ms=3, lw=1,
                        color=colors[p], label=f"plane {p}")
        ax.axhline(0, color="0.6", ls=":", lw=0.7)
        ax.set_xlabel("true X [cm]")
        ax.set_title(f"TPC{t}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$x_{reco}-x_{true}$ [cm]")
    fig.suptitle("apparent dx per plane (truth-matched hits)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "dx_apparent_per_plane.png"), dpi=120)
    plt.close(fig)

    # v_eff profile vs x + histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for t, color in ((0, "tab:blue"), (1, "tab:red")):
        m = (tpc == t) & (D["tdrift"] > 300)
        mean, err, _ = profile(D["x_true"][m], v_eff[m], NXBIN, rng)
        axes[0].errorbar(cx, mean, yerr=err, fmt="o-", ms=3, lw=1.2,
                         color=color, label=f"TPC{t}")
        axes[1].hist(v_eff[m][np.isfinite(v_eff[m])], bins=120,
                     range=(0.150, 0.162), histtype="step", color=color,
                     label=f"TPC{t} median "
                           f"{np.nanmedian(v_eff[m]):.5f}")
    axes[0].axhline(args.vreco, color="k", ls="--", lw=1,
                    label=f"v_reco {args.vreco}")
    axes[0].axhline(0.15532, color="purple", ls=":", lw=1, label="0.15532")
    axes[0].set_xlabel("true X [cm]"); axes[0].set_ylabel("v_eff [cm/us]")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].axvline(args.vreco, color="k", ls="--", lw=1)
    axes[1].set_xlabel("v_eff = dist(x_sce->anode)/tdrift [cm/us]")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle("true simulated drift velocity (truth-matched hits, "
                 "tdrift > 300 us)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "v_eff_truth.png"), dpi=120)
    plt.close(fig)

    # ---- npz -------------------------------------------------------------------
    keep = {}
    for name, val in (("dx_apparent", dx_apparent), ("dx_spatial", dx_spatial),
                      ("dx_timing", dx_timing), ("v_eff", v_eff)):
        for t in (0, 1):
            m = tpc == t
            mean, err, cnt = profile(D["x_true"][m], val[m], NXBIN, rng)
            keep[f"{name}_T{t}_mean"] = mean
            keep[f"{name}_T{t}_err"] = err
            keep[f"{name}_T{t}_n"] = cnt
    np.savez_compressed(os.path.join(args.out, "truth_dx_profiles.npz"),
                        x_centers=cx, vreco=np.float32(args.vreco),
                        t0=np.float32(args.t0), **keep)
    print(f"\nwrote {args.out}/", flush=True)


if __name__ == "__main__":
    main()
