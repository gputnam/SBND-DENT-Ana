#!/usr/bin/env python3
"""Reco hit time vs TRUE arrival time, wire-by-wire, from the MC ntuples.

measure_truth_dx_mc.py established that the TRUE apparent drift-direction
displacement is ~0 (the spatial SCE dome toward the anode is cancelled by an
opposite drift-time-distortion dome), while the double-anode method measures
a ~3x dome of the timing component's sign. The remaining unverified link in
the chain is hit reconstruction: does the RECO peak time differ from the
TRUE mean arrival time in a drift-distance-dependent way?

Per track and plane, reco hits (`hitsP.h.{wire,tpc,time}`) are matched to
truth-matched hits (`truth.p.truehitsP.{wire,tpc,time}`) by (tpc, wire);
the truehits 'time' is the true mean arrival tick of the matched energy
deposits on that wire (verified: time = epoch + 2 ticks/us * tdrift). We
profile

    dt = time_reco - time_true   [ticks]   vs   true x of the deposit

per (tpc, plane). A flat dt = clock epoch difference (harmless). Any
x-dependence is a hit-reconstruction-induced apparent displacement
dx = -/+ dt * 0.5us * v -- exactly what the double-anode method would
measure on top of the (near-zero) true apparent displacement.

Usage:
    python compare_hit_times_mc.py --nfiles 50 --nproc 16 --out hit_times_mc
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
V = 0.15627
NXBIN = 40


def read_file(url):
    import uproot
    import awkward as ak
    out = {p: {k: [] for k in ("x_true", "tpc", "dt")} for p in (0, 1, 2)}
    try:
        with uproot.open(url) as f:
            t = f["caloskim/TrackCaloSkim"]
            br = []
            for p in (0, 1, 2):
                br += [f"hits{p}.h.wire", f"hits{p}.h.tpc",
                       f"hits{p}.h.time", f"hits{p}.oncalo",
                       f"truth.p.truehits{p}.wire",
                       f"truth.p.truehits{p}.tpc",
                       f"truth.p.truehits{p}.time",
                       f"truth.p.truehits{p}.p.x"]
            arr = t.arrays(br, library="ak")
            ntrk = len(arr[br[0]])
            for i in range(ntrk):
                for p in (0, 1, 2):
                    rw = np.asarray(arr[f"hits{p}.h.wire"][i], np.int64)
                    rt = np.asarray(arr[f"hits{p}.h.tpc"][i], np.int64)
                    rtime = np.asarray(arr[f"hits{p}.h.time"][i], np.float64)
                    onc = np.asarray(arr[f"hits{p}.oncalo"][i]).astype(bool)
                    tw = np.asarray(arr[f"truth.p.truehits{p}.wire"][i],
                                    np.int64)
                    tt = np.asarray(arr[f"truth.p.truehits{p}.tpc"][i],
                                    np.int64)
                    ttime = np.asarray(arr[f"truth.p.truehits{p}.time"][i],
                                       np.float64)
                    tx = np.asarray(arr[f"truth.p.truehits{p}.p.x"][i],
                                    np.float64)
                    if not len(rw) or not len(tw):
                        continue
                    # match by (tpc, wire); truehits are unique per wire
                    tkey = tt * 10000 + tw
                    rkey = rt * 10000 + rw
                    order = np.argsort(tkey)
                    pos = np.searchsorted(tkey[order], rkey)
                    pos = np.clip(pos, 0, len(order) - 1)
                    hit = order[pos]
                    ok = (tkey[hit] == rkey) & onc
                    out[p]["x_true"].append(tx[hit[ok]])
                    out[p]["tpc"].append(rt[ok])
                    out[p]["dt"].append(rtime[ok] - ttime[hit[ok]])
    except Exception as e:
        print(f"skip {url.rsplit('/', 1)[-1][:40]}: {type(e).__name__}",
              file=sys.stderr)
        return None
    return {p: {k: (np.concatenate(v) if v else np.empty(0))
                for k, v in d.items()} for p, d in out.items()}


def profile(x, y, bins, rng):
    idx = np.floor((x - rng[0]) / (rng[1] - rng[0]) * bins).astype(np.int64)
    ok = (idx >= 0) & (idx < bins) & np.isfinite(y)
    n = np.bincount(idx[ok], minlength=bins).astype(float)
    s = np.bincount(idx[ok], weights=y[ok], minlength=bins)
    s2 = np.bincount(idx[ok], weights=y[ok] ** 2, minlength=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / n, np.nan)
        rms = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mean ** 2, 0))
    return mean, rms / np.sqrt(np.maximum(n, 1)), n


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filelist", default="filelist_mc_doubleanode_xroot.txt")
    ap.add_argument("--nfiles", type=int, default=50)
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--out", default="hit_times_mc")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    urls = [l.strip() for l in open(args.filelist) if l.strip()][:args.nfiles]
    print(f"reading {len(urls)} ntuples", flush=True)
    parts = []
    with Pool(args.nproc) as pool:
        for r in tqdm(pool.imap_unordered(read_file, urls), total=len(urls),
                      unit="file", file=sys.stderr):
            if r is not None:
                parts.append(r)
    D = {p: {k: np.concatenate([q[p][k] for q in parts])
             for k in parts[0][p]} for p in (0, 1, 2)}
    ntot = sum(len(D[p]["dt"]) for p in (0, 1, 2))
    print(f"matched hits: {ntot:,}", flush=True)

    rng = (-200.0, 200.0)
    edges = np.linspace(*rng, NXBIN + 1)
    cx = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    colors = ("tab:blue", "tab:orange", "tab:green")
    keep = {}
    print("\nmedian dt (reco - true arrival) per (tpc, plane) [ticks]:")
    for t, ax in enumerate(axes):
        for p in (0, 1, 2):
            m = D[p]["tpc"] == t
            dt = D[p]["dt"][m]
            med = np.median(dt[np.isfinite(dt)])
            print(f"  TPC{t} plane{p}: {med:+.2f}", flush=True)
            # epoch-subtract with the per-(tpc,plane) median; the
            # x-DEPENDENCE is the physical content
            mean, err, n = profile(D[p]["x_true"][m], dt - med, NXBIN, rng)
            xdir = 1.0 if t == 0 else -1.0
            dx_equiv = xdir * mean * TICK_US * V
            ax.errorbar(cx, dx_equiv, yerr=err * TICK_US * V, fmt="o-",
                        ms=3, lw=1.1, color=colors[p], label=f"plane {p}")
            keep[f"dxequiv_T{t}_P{p}"] = dx_equiv
            keep[f"n_T{t}_P{p}"] = n
        ax.axhline(0, color="0.6", ls=":", lw=0.7)
        ax.set_xlabel("true X [cm]")
        ax.set_title(f"TPC{t}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("apparent dx from hit-time bias "
                       r"$\pm(t_{reco}-t_{true})\cdot 0.5\mu s\cdot v$ [cm]")
    fig.suptitle("hit-reconstruction time bias as an apparent drift-direction"
                 " displacement (median epoch removed per tpc/plane)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "hit_time_bias_dx.png"), dpi=120)
    fig.savefig(os.path.join(args.out, "hit_time_bias_dx.pdf"))
    plt.close(fig)
    np.savez_compressed(os.path.join(args.out, "hit_time_bias.npz"),
                        x_centers=cx, **keep)
    print(f"\nwrote {args.out}/", flush=True)


if __name__ == "__main__":
    main()
