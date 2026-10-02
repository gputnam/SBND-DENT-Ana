#!/usr/bin/env python3
"""Voxelized APPARENT-truth displacement map from the MC truehits.

The proper MC benchmark for the double-anode measurement is the apparent
displacement of reconstructed hits, not the SCE spatial map (whose dx layer
is nearly cancelled by the drift-time distortion -- see
measure_truth_dx_mc.py). This script voxelizes, on the standard 20 cm grid
(binned at TRUE position):

CONVENTIONS (verified in TrackCaloSkimmer_module.cc + the SBND sim chain):
`p` = the raw sim::IDE position = the DISPLACED (drift-frame) deposit
position (the g4 stage shifts deposits by GetPosOffsets; WireCell drifts
them at constant speed); `p_scecorr` = GetCalPosOffsets-corrected = the
TRUE position; truehits.time/tdrift are nominal-drift recomputations from
p.x (NOT simulated times). Therefore, binned at the TRUE (p_scecorr)
position, this script voxelizes TWO dx benchmarks and the applied dy/dz:

    delta_mu_all[...,0]  dx_apparent = x_reco(matched reco hit) - x_scecorr
                         (includes the hit-reco peak-time bias)
    dx_applied           = p.x - p_scecorr.x   (the as-applied SCE map --
                         the target the measured map should reproduce)
    delta_mu_all[...,1:] dy/dz_applied = p.{y,z} - p_scecorr.{y,z}

The reco-time epoch is anchored so the per-TPC anode columns average to the
418.5 production t0 (same anchoring as the pipeline's empirical anode
peak). Output npz carries `delta_mu_all` so score_dx_strategies.py /
compare_abc_maps-style tools can read it as a map.

Usage:
    python build_apparent_truth_map.py --nfiles 400 --nproc 16 \
        --out apparent_truth_map_mc.npz
"""

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
from tqdm.auto import tqdm

import plot_doubleanode_2d_voxels as P2

TICK_US = 0.5
V = 0.15627
T0 = 418.5


def read_file(url):
    import uproot
    import awkward as ak
    out = []
    try:
        with uproot.open(url) as f:
            t = f["caloskim/TrackCaloSkim"]
            br = []
            for p in (0, 1, 2):
                br += [f"hits{p}.h.wire", f"hits{p}.h.tpc",
                       f"hits{p}.h.time", f"hits{p}.oncalo",
                       f"truth.p.truehits{p}.wire",
                       f"truth.p.truehits{p}.tpc",
                       f"truth.p.truehits{p}.p.x",
                       f"truth.p.truehits{p}.p.y",
                       f"truth.p.truehits{p}.p.z",
                       f"truth.p.truehits{p}.p_scecorr.x",
                       f"truth.p.truehits{p}.p_scecorr.y",
                       f"truth.p.truehits{p}.p_scecorr.z"]
            arr = t.arrays(br, library="ak")
            ntrk = len(arr[br[0]])
            for i in range(ntrk):
                for p in (0, 1, 2):
                    tw = np.asarray(arr[f"truth.p.truehits{p}.wire"][i],
                                    np.int64)
                    if not len(tw):
                        continue
                    tt = np.asarray(arr[f"truth.p.truehits{p}.tpc"][i],
                                    np.int64)
                    g = {c: np.asarray(
                        arr[f"truth.p.truehits{p}.{c}"][i], np.float64)
                        for c in ("p.x", "p.y", "p.z", "p_scecorr.x",
                                  "p_scecorr.y", "p_scecorr.z")}
                    rw = np.asarray(arr[f"hits{p}.h.wire"][i], np.int64)
                    rt = np.asarray(arr[f"hits{p}.h.tpc"][i], np.int64)
                    rtime = np.asarray(arr[f"hits{p}.h.time"][i], np.float64)
                    onc = np.asarray(arr[f"hits{p}.oncalo"][i]).astype(bool)
                    tkey = tt * 10000 + tw
                    rkey = rt * 10000 + rw
                    order = np.argsort(tkey)
                    pos = np.clip(np.searchsorted(tkey[order], rkey), 0,
                                  len(order) - 1)
                    hit = order[pos]
                    ok = (tkey[hit] == rkey) & onc
                    if not ok.any():
                        continue
                    h = hit[ok]
                    out.append((tt[h], g["p.x"][h], g["p.y"][h],
                                g["p.z"][h], g["p_scecorr.x"][h],
                                g["p_scecorr.y"][h], g["p_scecorr.z"][h],
                                rtime[ok]))
    except Exception as e:
        print(f"skip: {type(e).__name__}", file=sys.stderr)
        return None
    if not out:
        return None
    return [np.concatenate([o[j] for o in out]) for j in range(8)]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filelist", default="filelist_mc_doubleanode_xroot.txt")
    ap.add_argument("--nfiles", type=int, default=400)
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--min-counts", type=int, default=30)
    ap.add_argument("--out", default="apparent_truth_map_mc.npz")
    args = ap.parse_args()

    urls = [l.strip() for l in open(args.filelist) if l.strip()][:args.nfiles]
    print(f"reading {len(urls)} ntuples", flush=True)
    parts = []
    with Pool(args.nproc) as pool:
        for r in tqdm(pool.imap_unordered(read_file, urls), total=len(urls),
                      unit="file", file=sys.stderr):
            if r is not None:
                parts.append(r)
    cols = [np.concatenate([p[j] for p in parts]) for j in range(8)]
    tpc, xt, yt, zt, xs, ys, zs, rtime = cols
    print(f"matched hits: {len(tpc):,}", flush=True)

    tpc = tpc.astype(np.int64)
    anodex = np.where(tpc == 0, -200.0, 200.0)
    xdir = np.where(tpc == 0, 1.0, -1.0)
    # NOTE on roles: xt/yt/zt hold `p` = the DISPLACED position; xs/ys/zs
    # hold `p_scecorr` = the TRUE position (see module docstring).
    # epoch: anchor reco times so hits whose TRUE position is near the
    # anode give dx_apparent ~ 0 (the displacement vanishes at the anode,
    # so this matches the pipeline's empirical-anode-peak t0 anchoring)
    drift0 = np.abs(xs - anodex)
    t0_eff = np.zeros(2)
    for t in (0, 1):
        m = (tpc == t) & (drift0 < 10)
        t0_eff[t] = np.median(rtime[m] - drift0[m] / (TICK_US * V))
    print(f"effective reco-time anode epoch per TPC: {t0_eff} ticks "
          f"(production t0 = 418.5)", flush=True)
    x_reco = anodex + xdir * (rtime - t0_eff[tpc]) * TICK_US * V

    dxa = x_reco - xs            # apparent (incl. hit-reco time bias)
    dx_applied = xt - xs         # the as-applied SCE map (p - p_scecorr)
    dya = yt - ys
    dza = zt - zs

    nx, ny, nz = P2.grid_shape(20.0)
    xe = np.linspace(*P2.X_RANGE, nx + 1)
    ye = np.linspace(*P2.Y_RANGE, ny + 1)
    ze = np.linspace(*P2.Z_RANGE, nz + 1)
    # bin at the TRUE (p_scecorr) position -- the forward-map convention
    ix = np.floor((xs - P2.X_RANGE[0]) / 20.0).astype(np.int64)
    iy = np.floor((ys - P2.Y_RANGE[0]) / 20.0).astype(np.int64)
    iz = np.floor((zs - P2.Z_RANGE[0]) / 20.0).astype(np.int64)
    ok = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0)
          & (iz < nz) & np.isfinite(dxa) & np.isfinite(dya)
          & np.isfinite(dza) & np.isfinite(dx_applied)
          & (np.abs(dxa) < 10))
    flat = ix[ok] * (ny * nz) + iy[ok] * nz + iz[ok]
    nvox = nx * ny * nz
    n = np.bincount(flat, minlength=nvox).astype(float)
    mu = np.full((nvox, 4), np.nan)
    for c, v in enumerate((dxa[ok], dya[ok], dza[ok], dx_applied[ok])):
        s = np.bincount(flat, weights=v, minlength=nvox)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu[:, c] = np.where(n >= args.min_counts, s / np.maximum(n, 1),
                                np.nan)
    mu = mu.reshape(nx, ny, nz, 4)
    # delta_applied = the as-applied SCE field (dx_applied, dy, dz): the
    # target the measured map should reproduce. delta_mu_all keeps the
    # apparent dx (incl. hit-time bias) for diagnostics.
    delta_applied = np.stack([mu[..., 3], mu[..., 1], mu[..., 2]], axis=-1)
    np.savez_compressed(args.out,
                        delta_mu_all=mu[..., :3].astype(np.float32),
                        delta_applied=delta_applied.astype(np.float32),
                        count_all=n.reshape(nx, ny, nz).astype(np.int64),
                        x_edges=xe, y_edges=ye, z_edges=ze,
                        t0_eff=t0_eff)
    nf = int(np.isfinite(mu[..., 0]).sum())
    print(f"apparent+applied truth maps: {nf:,} voxels -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
