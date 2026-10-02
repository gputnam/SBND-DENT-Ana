#!/usr/bin/env python3
"""theta_xw dependence of the reco-hit-time bias, and its rectification
into the double-anode dx.

theta_xw is the track angle in the (drift x, wire-coordinate s) projection,
measured from the drift axis:

    tan(theta_xw) = ds/dx        n_x = -sin(theta_xw)
    drift extent per wire = d_w / |tan(theta_xw)|

tau = (reco PeakTime - nominal-drift arrival time) is the
hit-reconstruction time bias. Drift-parallel tracks (small |theta_xw|)
integrate charge on each wire over a long window, so their reconstructed
peak times are biased in a drift-distance-dependent way; the per-voxel
3x3 LSQ rectifies the angle pattern into a fake dx dome (the measured
"factor ~2" over the as-applied SCE field). The bias goes away above
roughly tan(theta_xw) = 0.8 (|theta_xw| ~ 39 deg).

From truth-matched hits this script makes:

  * tau (x-equivalent, epoch-subtracted) vs |theta_xw| in bins of drift
    distance -> the angle dependence itself;
  * the |theta_xw| DISTRIBUTION of the selected tracks (one entry per
    track/TPC/plane), with the tan(theta_xw) = 0.8 cut line -> how much
    of the sample is in the biased region;
  * per (tpc, x) the PLAIN vs sin^2(theta_xw)-weighted mean of v*tau
    (the n_x^2 weighting of the dx LSQ);
  * optionally a per-hit time-correction LUT (--write-lut, binned in
    ds/dx for direct use by plot_doubleanode_2d_voxels --hit-time-corr).

Track theta_xw per plane is fit from the truehits themselves
(s = slope_cm*wire + offset vs x).

Usage:
    python angle_hit_time_bias.py --nfiles 200 --nproc 20 --out angle_bias_mc
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

import plot_doubleanode_2d_voxels as P2

TICK_US = 0.5
V = 0.15627
NXB = 20
NSB = 24          # signed ds/dx bins in [-3, 3] (LUT binning)
SMAX = 3.0
NTB = 18          # |theta_xw| bins in [0, 72] deg (plot binning)
THMAX = 72.0
TAN_CUT = 0.8     # bias goes away above tan(theta_xw) = 0.8
TH_CUT = np.degrees(np.arctan(TAN_CUT))

GEOM = P2.load_wire_geom(P2.WIRE_GEOM)


def read_file(url):
    import uproot
    import awkward as ak
    SLOPE, OFFSET, WHATY, WHATZ = GEOM
    cols = []
    tcols = []      # one entry per (track, tpc, plane): the track theta_xw
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
                       f"truth.p.truehits{p}.p.x",
                       f"truth.p.truehits{p}.p.y",
                       f"truth.p.truehits{p}.p.z"]
            arr = t.arrays(br, library="ak")
            ntrk = len(arr[br[0]])
            for i in range(ntrk):
                for p in (0, 1, 2):
                    tw = np.asarray(arr[f"truth.p.truehits{p}.wire"][i],
                                    np.int64)
                    if len(tw) < 10:
                        continue
                    tt = np.asarray(arr[f"truth.p.truehits{p}.tpc"][i],
                                    np.int64)
                    ttime = np.asarray(arr[f"truth.p.truehits{p}.time"][i],
                                       np.float64)
                    tx = np.asarray(arr[f"truth.p.truehits{p}.p.x"][i],
                                    np.float64)
                    tyy = np.asarray(arr[f"truth.p.truehits{p}.p.y"][i],
                                     np.float64)
                    tzz = np.asarray(arr[f"truth.p.truehits{p}.p.z"][i],
                                     np.float64)
                    rw = np.asarray(arr[f"hits{p}.h.wire"][i], np.int64)
                    rt = np.asarray(arr[f"hits{p}.h.tpc"][i], np.int64)
                    rtime = np.asarray(arr[f"hits{p}.h.time"][i], np.float64)
                    onc = np.asarray(arr[f"hits{p}.oncalo"][i]).astype(bool)
                    for tpc_ in (0, 1):
                        tm = tt == tpc_
                        if tm.sum() < 10:
                            continue
                        s = (SLOPE[tpc_, p] * tw[tm].astype(np.float64)
                             + OFFSET[tpc_, p])
                        xx = tx[tm]
                        if np.ptp(xx) < 30:
                            continue
                        A = np.stack([np.ones(len(xx)), xx], axis=1)
                        coef, *_ = np.linalg.lstsq(A, s, rcond=None)
                        slope_p = coef[1]
                        # track direction (ty, tz) -> along-wire angle
                        # |dw/dx| (wire dir = (-w_z, w_y) in (y, z))
                        ty_c, *_ = np.linalg.lstsq(A, tyy[tm], rcond=None)
                        tz_c, *_ = np.linalg.lstsq(A, tzz[tm], rcond=None)
                        dwdx_p = abs(-WHATZ[tpc_, p] * ty_c[1]
                                     + WHATY[tpc_, p] * tz_c[1])
                        tcols.append((tpc_, p, slope_p))
                        # match reco hits on this (tpc, plane)
                        rm = (rt == tpc_) & onc
                        if not rm.any():
                            continue
                        order = np.argsort(tw[tm])
                        tww = tw[tm][order]
                        pos = np.clip(np.searchsorted(tww, rw[rm]), 0,
                                      len(tww) - 1)
                        ok = tww[pos] == rw[rm]
                        if not ok.any():
                            continue
                        idx = np.nonzero(tm)[0][order][pos[ok]]
                        tau = rtime[rm][ok] - ttime[idx]
                        cols.append((np.full(ok.sum(), tpc_, np.int8),
                                     np.full(ok.sum(), p, np.int8),
                                     np.full(ok.sum(), slope_p),
                                     tx[idx], tau,
                                     np.full(ok.sum(), dwdx_p)))
    except Exception as e:
        print(f"skip: {type(e).__name__}", file=sys.stderr)
        return None
    if not cols:
        return None
    hits = [np.concatenate([c[j] for c in cols]) for j in range(6)]
    trks = [np.array([t[j] for t in tcols]) for j in range(3)]
    return hits, trks


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filelist", default="filelist_mc_doubleanode_xroot.txt")
    ap.add_argument("--nfiles", type=int, default=100)
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--out", default="angle_bias_mc")
    ap.add_argument("--write-lut", default=None,
                    help="also write a per-hit time-correction LUT npz: "
                         "mean tau [ticks] in (tpc, plane, |slope|, x) bins "
                         "for plot_doubleanode_2d_voxels --hit-time-corr")
    ap.add_argument("--lut-wslope-bins", type=int, default=0,
                    help="if > 0, write a 2-ANGLE LUT with this many "
                         "|dw/dx| (along-wire angle) bins in [0, 3] as an "
                         "extra axis: tau(tpc, plane, slope, |dw/dx|, x)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    urls = [l.strip() for l in open(args.filelist) if l.strip()][:args.nfiles]
    parts = []
    with Pool(args.nproc) as pool:
        for r in tqdm(pool.imap_unordered(read_file, urls), total=len(urls),
                      unit="file", file=sys.stderr):
            if r is not None:
                parts.append(r)
    tpc, plane, slope, xt, tau, dwdx = [
        np.concatenate([p[0][j] for p in parts]) for j in range(6)]
    ttpc, tplane, tslope = [
        np.concatenate([p[1][j] for p in parts]) for j in range(3)]
    tpc = tpc.astype(np.int64)
    print(f"matched hits: {len(tau):,}; track/plane entries: "
          f"{len(tslope):,}", flush=True)

    # epoch per (tpc, plane): median tau
    for t in (0, 1):
        for p in (0, 1, 2):
            m = (tpc == t) & (plane == p)
            tau[m] -= np.median(tau[m])
    xdir = np.where(tpc == 0, 1.0, -1.0)
    dxeq = xdir * tau * TICK_US * V              # x-equivalent bias [cm]
    theta = np.degrees(np.arctan(np.abs(slope)))      # |theta_xw| [deg]
    ttheta = np.degrees(np.arctan(np.abs(tslope)))
    w = np.sin(np.radians(theta)) ** 2           # n_x^2 = sin^2(theta_xw)

    xe = np.linspace(-200, 200, NXB + 1)
    cx = 0.5 * (xe[:-1] + xe[1:])
    ixb = np.clip(np.floor((xt + 200) / 400 * NXB).astype(np.int64), 0,
                  NXB - 1)

    # --- plain vs slope^2-weighted profiles per TPC ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    out = {}
    for t, ax in enumerate(axes):
        m = tpc == t
        n = np.bincount(ixb[m], minlength=NXB).astype(float)
        plain = np.bincount(ixb[m], weights=dxeq[m], minlength=NXB) \
            / np.maximum(n, 1)
        wsum = np.bincount(ixb[m], weights=w[m], minlength=NXB)
        wmean = np.bincount(ixb[m], weights=(w * dxeq)[m], minlength=NXB) \
            / np.maximum(wsum, 1e-9)
        ax.plot(cx, plain, "o-", ms=4, label="plain mean (what the "
                                             "angle-avg profile sees)")
        ax.plot(cx, wmean, "s-", ms=4, color="tab:red",
                label=r"$\sin^2\theta_{xw}$-weighted mean "
                      r"($n_x^2$, the dx LSQ weighting)")
        ax.axhline(0, color="0.6", ls=":", lw=0.7)
        ax.set_xlabel("true X [cm]")
        ax.set_title(f"TPC{t}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        out[f"plain_T{t}"] = plain
        out[f"wmean_T{t}"] = wmean
    axes[0].set_ylabel("hit-time bias as apparent dx [cm]")
    fig.suptitle(r"reco-hit-time bias: plain vs $\sin^2\theta_{xw}$-weighted"
                 " (the double-anode dx weighting)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "weighted_vs_plain.png"), dpi=120)
    plt.close(fig)

    # --- tau vs |theta_xw| in drift bins --------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    drift = np.abs(xt - np.where(tpc == 0, -200.0, 200.0))
    te = np.linspace(0.0, THMAX, NTB + 1)
    ct = 0.5 * (te[:-1] + te[1:])
    itb = np.clip(np.floor(theta / THMAX * NTB).astype(np.int64), 0, NTB - 1)
    for t in (0, 1):
        for p in (0, 1, 2):
            ax = axes[t, p]
            for dlo, dhi, col in ((0, 70, "tab:blue"),
                                  (70, 140, "tab:orange"),
                                  (140, 210, "tab:red")):
                m = (tpc == t) & (plane == p) & (drift >= dlo) \
                    & (drift < dhi)
                n = np.bincount(itb[m], minlength=NTB).astype(float)
                prof = np.bincount(itb[m], weights=dxeq[m], minlength=NTB) \
                    / np.maximum(n, 1)
                prof[n < 500] = np.nan
                ax.plot(ct, prof, "o-", ms=3.5, color=col,
                        label=f"drift {dlo}-{dhi} cm")
            ax.axvline(TH_CUT, color="0.3", ls="--", lw=1.2)
            ax.axhline(0, color="0.6", ls=":", lw=0.7)
            ax.set_title(f"TPC{t} plane{p}", fontsize=10)
            ax.grid(alpha=0.3)
            if t == 1:
                ax.set_xlabel(r"$|\theta_{xw}|$ [$^\circ$]")
            if p == 0:
                ax.set_ylabel("hit-time bias [cm]")
            ax.legend(fontsize=7)
    fig.suptitle(r"reco-hit-time bias vs $|\theta_{xw}|$, by drift distance")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "tau_vs_thetaxw.png"), dpi=120)
    fig.savefig(os.path.join(args.out, "tau_vs_thetaxw.pdf"))
    plt.close(fig)

    # --- |theta_xw| distribution of the selected tracks -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    colors = ("tab:blue", "tab:orange", "tab:green")
    frac = {}
    for t, ax in enumerate(axes):
        for p in (0, 1, 2):
            m = (ttpc == t) & (tplane == p)
            vals = ttheta[m]
            f_pass = float((vals > TH_CUT).mean()) if m.any() else np.nan
            frac[(t, p)] = f_pass
            ax.hist(vals, bins=np.linspace(0, 90, 91), histtype="step",
                    lw=1.4, color=colors[p],
                    label=f"plane {p}: {100*f_pass:.1f}% above cut")
        ax.axvline(TH_CUT, color="0.3", ls="--", lw=1.4)
        ax.set_xlabel(r"$|\theta_{xw}|$ [$^\circ$]")
        ax.set_title(f"TPC{t}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("tracks")
    fig.suptitle(r"$|\theta_{xw}|$ of the selected double-anode tracks "
                 "(one entry per track/TPC/plane)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "thetaxw_distribution.png"), dpi=120)
    fig.savefig(os.path.join(args.out, "thetaxw_distribution.pdf"))
    plt.close(fig)
    print("fraction of track/plane entries above the cut:")
    for t in (0, 1):
        print("  TPC%d: " % t + "  ".join(
            f"plane{p} {100*frac[(t,p)]:.1f}%" for p in (0, 1, 2)),
            flush=True)

    # LUT binning (signed ds/dx; kept for --hit-time-corr compatibility)
    se = np.linspace(-SMAX, SMAX, NSB + 1)
    isb = np.clip(np.floor((slope + SMAX) / (2 * SMAX) * NSB)
                  .astype(np.int64), 0, NSB - 1)

    np.savez_compressed(os.path.join(args.out, "angle_bias.npz"),
                        x_centers=cx, theta_centers=ct,
                        theta_cut_deg=np.float32(TH_CUT),
                        trk_theta=ttheta.astype(np.float32),
                        trk_tpc=ttpc.astype(np.int8),
                        trk_plane=tplane.astype(np.int8), **out)

    if args.write_lut:
        # mean tau [ticks] per (tpc, plane, slope[, |dw/dx|], x) bin; empty
        # bins fall back to the marginal over x, then 0.
        tau_ticks = tau                      # epoch already subtracted above
        nwb = args.lut_wslope_bins
        if nwb > 0:
            we = np.linspace(0.0, 3.0, nwb + 1)
            iwb = np.clip(np.floor(np.abs(dwdx) / 3.0 * nwb)
                          .astype(np.int64), 0, nwb - 1)
            flat = ((((tpc * 3 + plane.astype(np.int64)) * NSB + isb) * nwb
                     + iwb) * NXB + ixb)
            shape = (2, 3, NSB, nwb, NXB)
        else:
            flat = (((tpc * 3 + plane.astype(np.int64)) * NSB + isb) * NXB
                    + ixb)
            shape = (2, 3, NSB, NXB)
        nbin = int(np.prod(shape))
        s_ = np.bincount(flat, weights=tau_ticks, minlength=nbin)
        n_ = np.bincount(flat, minlength=nbin).astype(float)
        lut = (s_ / np.maximum(n_, 1)).reshape(shape)
        cntl = n_.reshape(shape)
        marg = np.where(cntl.sum(-1) > 0,
                        (lut * cntl).sum(-1) / np.maximum(cntl.sum(-1), 1),
                        0.0)
        lut = np.where(cntl >= 200, lut, marg[..., None])
        extra = dict(wslope_edges=we) if nwb > 0 else {}
        np.savez_compressed(args.write_lut, lut_ticks=lut, counts=cntl,
                            slope_edges=se, x_edges=xe, **extra)
        print(f"wrote {'2-angle ' if nwb else ''}hit-time LUT -> "
              f"{args.write_lut}", flush=True)
    print(f"wrote {args.out}/", flush=True)


if __name__ == "__main__":
    main()
