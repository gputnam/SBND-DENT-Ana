#!/usr/bin/env python3
"""Per-pixel histograms of the SUMMED drift-time span dt0+dt1 of
anode-cathode through-going tracks.

For an anode-to-anode track, dt_TPC = max_time - min_time is the drift-time
span of that TPC's segment. The SUM dt0+dt1 corresponds to the full
anode-to-anode drift span and is INDEPENDENT of the cathode position: a
shift of the cathode crossing point adds to one TPC's span exactly what it
removes from the other's. Mapping <dt0+dt1> per (y, z) pixel (pixel = the
cathode-crossing position) therefore isolates the total-span/velocity
observable and its spatial systematics (support frame, endpoint effects),
free of the cathode-shape signal that the per-TPC heatmaps carry.

Outputs (--out):
  dtsum_global.png            pooled dt0+dt1 histogram + mean/median/CB fit,
                              implied v under the 400 cm and 402.6 cm span
                              assumptions
  dtsum_mean_map.png          per-pixel mean dt0+dt1 heatmap [ticks]
  dtsum_count_map.png         tracks per pixel
  vimplied_map.png            per-pixel v = 400 cm / (0.5us * <dt0+dt1>)
  pixels/yNNN_zNNN.png        per-pixel dt0+dt1 histograms (gallery)
  dtsum_pixels.npz            per-pixel histograms + grids

Usage:
    python plot_cathode_dtsum.py --plane 2 --out cathode_dtsum_data
"""

import argparse
import os
import sys
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from scipy.optimize import curve_fit

DRIFT_DF = "/exp/sbnd/data/users/gputnam/efield-study/drift_df/drift_df_full_v2.parquet"
TICK_US = 0.5
SPAN_NOMINAL_CM = 400.0       # the assumption used for the production v_AC
SPAN_GDML_CM = 402.6          # 2 x 201.3 cm (first induction plane to cathode)
AC_CUT_TICKS = 2400.0         # per-TPC near-full-drift selection
Y_RANGE = (-200.0, 200.0)
Z_RANGE = (0.0, 500.0)


def crystal_ball(x, alpha, n, mu, sigma, N):
    z = (x - mu) / sigma
    out = np.where(z > -alpha, np.exp(-0.5 * z * z),
                   (n / abs(alpha)) ** n * np.exp(-0.5 * alpha * alpha)
                   * (n / abs(alpha) - abs(alpha) - z) ** -n)
    return N * out


def _pixel_fig(task):
    yc, zc, hist, centers, width, png = task
    h = hist.astype(float)
    tot = h.sum()
    if tot <= 0:
        return None
    mean = float((centers * h).sum() / tot)
    rms = float(np.sqrt(max((centers ** 2 * h).sum() / tot - mean ** 2, 0)))
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.bar(centers, h, width=width, color="steelblue", edgecolor="navy",
           linewidth=0.3)
    ax.axvline(mean, color="red", ls="--", lw=1.0, label=f"mean {mean:.1f}")
    ax.set_xlabel(r"$dt_0 + dt_1$ [ticks]")
    ax.set_ylabel(f"tracks / {width:g} ticks")
    ax.set_title(f"pixel Y={yc:+.0f} Z={zc:+.0f} cm\n"
                 f"N={int(tot):,}  mean={mean:.1f}  rms={rms:.1f} ticks",
                 fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(png, dpi=90)
    plt.close(fig)
    return png


def main():
    import multiprocessing as _mp
    try:
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DRIFT_DF)
    ap.add_argument("--plane", type=int, default=2,
                    help="wire plane (default collection)")
    ap.add_argument("--pixel-size", type=float, default=10.0)
    ap.add_argument("--win-lo", type=float, default=5000.0,
                    help="dt0+dt1 histogram window low [ticks]")
    ap.add_argument("--win-hi", type=float, default=5350.0)
    ap.add_argument("--bin", type=float, default=4.0)
    ap.add_argument("--min-counts", type=int, default=40)
    ap.add_argument("--out", default="cathode_dtsum_data")
    ap.add_argument("--no-gallery", action="store_true")
    ap.add_argument("--nproc", default="auto")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    p = args.plane

    cols = []
    for t in (0, 1):
        pfx = f"tpc{t}_p{p}"
        cols += [f"{pfx}_min_time", f"{pfx}_max_time",
                 f"{pfx}_max_y", f"{pfx}_max_z"]
    print(f"reading {args.input} (plane {p})", flush=True)
    df = pd.read_parquet(args.input, columns=cols)
    dt0 = (df[f"tpc0_p{p}_max_time"] - df[f"tpc0_p{p}_min_time"]).to_numpy()
    dt1 = (df[f"tpc1_p{p}_max_time"] - df[f"tpc1_p{p}_min_time"]).to_numpy()
    # cathode-crossing pixel: average the two TPCs' max-time (cathode-side)
    # endpoint positions
    yc = 0.5 * (df[f"tpc0_p{p}_max_y"] + df[f"tpc1_p{p}_max_y"]).to_numpy()
    zc = 0.5 * (df[f"tpc0_p{p}_max_z"] + df[f"tpc1_p{p}_max_z"]).to_numpy()
    del df

    ac = (np.isfinite(dt0) & np.isfinite(dt1)
          & (dt0 > AC_CUT_TICKS) & (dt1 > AC_CUT_TICKS))
    dtsum = dt0[ac] + dt1[ac]
    y, z = yc[ac], zc[ac]
    inw = ((dtsum >= args.win_lo) & (dtsum <= args.win_hi)
           & (y >= Y_RANGE[0]) & (y < Y_RANGE[1])
           & (z >= Z_RANGE[0]) & (z < Z_RANGE[1]))
    dtsum, y, z = dtsum[inw], y[inw], z[inw]
    print(f"anode-cathode through-going tracks: {ac.sum():,}; "
          f"in window/fiducial: {len(dtsum):,}", flush=True)

    # ---- global histogram -----------------------------------------------------
    edges = np.arange(args.win_lo, args.win_hi + args.bin, args.bin)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist, _ = np.histogram(dtsum, bins=edges)
    mean = dtsum.mean()
    med = np.median(dtsum)
    # CB fit of the global peak
    cb = None
    try:
        p0 = [1.5, 3.0, med, 12.0, hist.max()]
        popt, _ = curve_fit(crystal_ball, centers, hist.astype(float), p0=p0,
                            maxfev=20000)
        cb = popt
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.step(centers, hist, where="mid", color="navy", lw=1.2)
    ax.axvline(mean, color="red", ls="--", lw=1.2, label=f"mean {mean:.1f}")
    ax.axvline(med, color="darkorange", ls="--", lw=1.2,
               label=f"median {med:.1f}")
    if cb is not None:
        xs = np.linspace(args.win_lo, args.win_hi, 600)
        ax.plot(xs, crystal_ball(xs, *cb), "g-", lw=1.2,
                label=f"CB peak {cb[2]:.1f}")
    lines = []
    for nm, val in (("mean", mean), ("median", med),
                    ("CB", cb[2] if cb is not None else np.nan)):
        if np.isfinite(val):
            v400 = SPAN_NOMINAL_CM / (val * TICK_US)
            v4026 = SPAN_GDML_CM / (val * TICK_US)
            lines.append(f"{nm}: v(400cm)={v400:.5f}  "
                         f"v(402.6cm)={v4026:.5f} cm/us")
    ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes,
            va="top", fontsize=9, family="monospace")
    ax.set_xlabel(r"$dt_0 + dt_1$ [ticks]  (cathode-position independent)")
    ax.set_ylabel(f"tracks / {args.bin:g} ticks")
    ax.set_title(f"summed drift-time span, plane {p} "
                 f"({len(dtsum):,} AC-through tracks)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "dtsum_global.png"), dpi=120)
    plt.close(fig)
    print("\n".join(lines), flush=True)

    # ---- per-pixel histograms ---------------------------------------------------
    ny = int(round((Y_RANGE[1] - Y_RANGE[0]) / args.pixel_size))
    nz = int(round((Z_RANGE[1] - Z_RANGE[0]) / args.pixel_size))
    iy = np.clip(((y - Y_RANGE[0]) / args.pixel_size).astype(np.int64),
                 0, ny - 1)
    iz = np.clip(((z - Z_RANGE[0]) / args.pixel_size).astype(np.int64),
                 0, nz - 1)
    nb = len(edges) - 1
    ib = np.clip(((dtsum - args.win_lo) / args.bin).astype(np.int64),
                 0, nb - 1)
    flat = (iy * nz + iz) * nb + ib
    pix = np.bincount(flat, minlength=ny * nz * nb).reshape(ny, nz, nb)
    cnt = pix.sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        pmean = (pix * centers[None, None, :]).sum(axis=2) \
            / np.maximum(cnt, 1)
    pmean[cnt < args.min_counts] = np.nan
    vimp = SPAN_NOMINAL_CM / (pmean * TICK_US)

    yc_g = Y_RANGE[0] + (np.arange(ny) + 0.5) * args.pixel_size
    zc_g = Z_RANGE[0] + (np.arange(nz) + 0.5) * args.pixel_size
    extent = (Z_RANGE[0], Z_RANGE[1], Y_RANGE[0], Y_RANGE[1])
    for arr, label, name, cmap, vrng in (
            (pmean, r"$\langle dt_0+dt_1\rangle$ [ticks]", "dtsum_mean_map",
             "RdYlBu_r", (np.nanpercentile(pmean, 2),
                          np.nanpercentile(pmean, 98))),
            (cnt.astype(float), "tracks / pixel", "dtsum_count_map",
             "viridis", (0, np.nanpercentile(cnt, 99))),
            (vimp, "implied v (400 cm span) [cm/us]", "vimplied_map",
             "RdYlBu_r", (np.nanpercentile(vimp, 2),
                          np.nanpercentile(vimp, 98)))):
        fig, ax = plt.subplots(figsize=(10, 6.5))
        im = ax.imshow(arr, origin="lower", aspect="auto", extent=extent,
                       cmap=cmap, vmin=vrng[0], vmax=vrng[1],
                       interpolation="nearest")
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("Z [cm]")
        ax.set_ylabel("Y [cm]")
        ax.set_title(f"plane {p}: {label} per cathode-crossing pixel "
                     f"({args.pixel_size:g} cm)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"{name}.png"), dpi=120)
        plt.close(fig)

    np.savez_compressed(
        os.path.join(args.out, "dtsum_pixels.npz"),
        pix_hist=pix.astype(np.int32), edges=edges, y_centers=yc_g,
        z_centers=zc_g, mean=pmean, count=cnt,
        plane=np.int32(p), pixel_size=np.float32(args.pixel_size))

    # ---- gallery ------------------------------------------------------------------
    if not args.no_gallery:
        gdir = os.path.join(args.out, "pixels")
        os.makedirs(gdir, exist_ok=True)
        tasks = []
        for j in range(ny):
            for k in range(nz):
                if cnt[j, k] < args.min_counts:
                    continue
                tasks.append((yc_g[j], zc_g[k], pix[j, k], centers,
                              args.bin,
                              os.path.join(gdir, f"y{int(yc_g[j]):+04d}"
                                                 f"_z{int(zc_g[k]):+04d}.png")))
        nproc = cpu_count() if args.nproc == "auto" else int(args.nproc)
        print(f"rendering {len(tasks):,} per-pixel histograms", flush=True)
        with Pool(min(nproc, 32)) as pool:
            for _ in tqdm(pool.imap_unordered(_pixel_fig, tasks,
                                              chunksize=16),
                          total=len(tasks), unit="fig", file=sys.stderr):
                pass
    print(f"wrote {args.out}/", flush=True)


if __name__ == "__main__":
    main()
