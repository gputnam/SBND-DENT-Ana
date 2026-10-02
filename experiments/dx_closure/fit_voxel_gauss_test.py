"""Sample-voxel test: histogram dx/dy/dz per voxel and fit a Gaussian.

For each (subset, voxel-center) sample, walk the parquet shards once,
collect every hit that falls in that voxel, histogram dx, dy, dz, fit a
Gaussian and write a PNG with the histogram + fit overlay (and the
arithmetic mean for comparison).

Used to validate that a per-voxel Gaussian-fit μ is well-defined and to
see where it disagrees with the arithmetic mean.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

INPUT = "/exp/sbnd/data/users/gputnam/efield-study"
OUTDIR = "crtdisp_gauss_test"

# Sample voxels — vary X across the active volume in the side-CRT subset
# (which has the cleanest residuals), plus one top-only and one all-CRT
# at the geometric center for comparison.
#                  subset, x, y, z   (cm; voxel centre)
SAMPLES = [
    ("side", -150,    0,  250),
    ("side",  -50,    0,  250),
    ("side",    0,    0,  250),
    ("side",  +50,    0,  250),
    ("side", +150,    0,  250),
    ("side",    0, +100,  250),
    ("side",    0, -100,  250),
    ("side",    0,    0,  100),
    ("side",    0,    0,  400),
    ("top",     0,    0,  250),
    ("all",     0,    0,  250),
]


def gauss(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


def gauss_plus_const(x, A, mu, sigma, C):
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2)) + C


def fit_one(data, comp, sub, cx, cy, cz, outpath,
            hist_range=(-100.0, 100.0), bin_width=1.0):
    """Histogram one voxel's residuals; fit both Gauss and Gauss+const; plot."""
    if data.size == 0:
        return None
    edges = np.arange(hist_range[0], hist_range[1] + bin_width, bin_width)
    counts, _ = np.histogram(data, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    err = np.sqrt(np.maximum(counts.astype(float), 1.0))

    if counts.sum() < 30:
        return None

    peak_idx = int(np.argmax(counts))
    A0 = float(counts[peak_idx])
    mu0 = float(centers[peak_idx])
    w = counts.astype(float)
    mean_est = (centers * w).sum() / w.sum()
    var_est = ((centers - mean_est) ** 2 * w).sum() / w.sum()
    sigma0 = max(np.sqrt(var_est), 1.0)
    # estimate pedestal from the median count outside +/- 30 cm
    far_mask = np.abs(centers) > 30
    C0 = float(np.median(counts[far_mask])) if far_mask.any() else 0.0
    A0_pc = max(A0 - C0, 1.0)

    # --- bare Gaussian ---
    try:
        popt_g, pcov_g = curve_fit(
            gauss, centers, counts.astype(float),
            p0=[A0, mu0, sigma0], sigma=err, absolute_sigma=True,
            bounds=([1.0, hist_range[0], 0.1],
                    [1e10, hist_range[1], 200.0]),
            maxfev=8000,
        )
        gA, gmu, gsigma = popt_g
        gmu_err = float(np.sqrt(max(pcov_g[1, 1], 0)))
        gsigma_err = float(np.sqrt(max(pcov_g[2, 2], 0)))
        pred_g = gauss(centers, *popt_g)
        chi2_g = float(np.sum(((counts - pred_g) / err) ** 2))
        ndf_g = len(centers) - len(popt_g)
        ok_g = True
    except (RuntimeError, ValueError):
        gA, gmu, gsigma, gmu_err, gsigma_err = A0, mu0, sigma0, np.nan, np.nan
        chi2_g, ndf_g = np.nan, len(centers) - 3
        ok_g = False

    # --- Gauss + constant pedestal ---
    try:
        popt_pc, pcov_pc = curve_fit(
            gauss_plus_const, centers, counts.astype(float),
            p0=[A0_pc, mu0, sigma0, C0],
            sigma=err, absolute_sigma=True,
            bounds=([1.0, hist_range[0], 0.1, 0.0],
                    [1e10, hist_range[1], 200.0, 1e6]),
            maxfev=8000,
        )
        pcA, pcmu, pcsigma, pcC = popt_pc
        pcmu_err = float(np.sqrt(max(pcov_pc[1, 1], 0)))
        pcsigma_err = float(np.sqrt(max(pcov_pc[2, 2], 0)))
        pred_pc = gauss_plus_const(centers, *popt_pc)
        chi2_pc = float(np.sum(((counts - pred_pc) / err) ** 2))
        ndf_pc = len(centers) - len(popt_pc)
        ok_pc = True
    except (RuntimeError, ValueError):
        pcA, pcmu, pcsigma, pcC = A0_pc, mu0, sigma0, C0
        pcmu_err = pcsigma_err = np.nan
        chi2_pc, ndf_pc = np.nan, len(centers) - 4
        ok_pc = False

    arith_mean = float(np.mean(data))
    arith_med = float(np.median(data))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(centers, counts, yerr=err, fmt="o", ms=3, lw=0.6,
                color="black", label=f"data (N={int(counts.sum()):,})")
    xs = np.linspace(hist_range[0], hist_range[1], 800)
    ax.plot(xs, gauss(xs, gA, gmu, gsigma), color="red", lw=1.4,
            label=f"Gauss: $\\mu$={gmu:+.2f}, $\\sigma$={gsigma:.2f}, "
                  f"$\\chi^2$/ndf={chi2_g:.0f}/{ndf_g}")
    ax.plot(xs, gauss_plus_const(xs, pcA, pcmu, pcsigma, pcC),
            color="darkgreen", lw=1.4, ls="-",
            label=f"Gauss+C: $\\mu$={pcmu:+.2f}, $\\sigma$={pcsigma:.2f}, "
                  f"C={pcC:.1f}, $\\chi^2$/ndf={chi2_pc:.0f}/{ndf_pc}")
    ax.axhline(pcC, color="darkgreen", ls=":", lw=0.6)
    ax.axvline(0, color="grey", ls=":", lw=0.6)
    ax.axvline(arith_mean, color="blue", ls="-.", lw=0.6,
               label=f"arith mean = {arith_mean:+.2f}")
    ax.axvline(arith_med, color="orange", ls=":", lw=0.6,
               label=f"median = {arith_med:+.2f}")
    ax.set_xlabel(f"{comp} [cm]")
    ax.set_ylabel(f"hits / {bin_width:g} cm")
    ax.set_title(f"{sub}-CRT, voxel center = ({cx:+d}, {cy:+d}, {cz:+d}) cm  "
                 f"({comp})")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return {
        "subset": sub, "cx": cx, "cy": cy, "cz": cz, "comp": comp,
        "n": int(counts.sum()),
        "arith_mean": arith_mean, "median": arith_med,
        "gauss_mu": float(gmu), "gauss_mu_err": gmu_err,
        "gauss_sigma": float(gsigma), "gauss_chi2_ndf": chi2_g / max(1, ndf_g),
        "ok_gauss": bool(ok_g),
        "gpc_mu": float(pcmu), "gpc_mu_err": pcmu_err,
        "gpc_sigma": float(pcsigma), "gpc_C": float(pcC),
        "gpc_chi2_ndf": chi2_pc / max(1, ndf_pc),
        "ok_gpc": bool(ok_pc),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--voxel-size", type=float, default=20.0)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--hist-range", type=float, default=100.0,
                    help="symmetric histogram half-range in cm (default 100)")
    ap.add_argument("--bin-width", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    if args.max_shards:
        shards = shards[:args.max_shards]
    if not shards:
        raise SystemExit(f"no parquet shards under {args.input}")
    print(f"reading {len(shards)} shards from {args.input}", flush=True)

    half = args.voxel_size / 2.0
    print(f"voxel size = {args.voxel_size:g} cm  (half-width {half:g} cm)", flush=True)
    print(f"hist range = [{-args.hist_range:g}, {+args.hist_range:g}] cm  "
          f"bin = {args.bin_width:g} cm", flush=True)

    # accumulators per sample
    bufs = [{"sub": s[0], "cx": s[1], "cy": s[2], "cz": s[3],
             "dx": [], "dy": [], "dz": []} for s in SAMPLES]

    cols = ["hit_x", "hit_y", "hit_z", "dx", "dy", "dz",
            "crt_top_only", "crt_side_only"]
    for path in shards:
        df = pd.read_parquet(path, columns=cols)
        x = df["hit_x"].to_numpy()
        y = df["hit_y"].to_numpy()
        z = df["hit_z"].to_numpy()
        dx = df["dx"].to_numpy()
        dy = df["dy"].to_numpy()
        dz = df["dz"].to_numpy()
        top = df["crt_top_only"].to_numpy()
        side = df["crt_side_only"].to_numpy()

        for b in bufs:
            m = ((x >= b["cx"] - half) & (x < b["cx"] + half)
                 & (y >= b["cy"] - half) & (y < b["cy"] + half)
                 & (z >= b["cz"] - half) & (z < b["cz"] + half))
            if b["sub"] == "side":
                m = m & side
            elif b["sub"] == "top":
                m = m & top
            # "all" -> no extra filter
            if not m.any():
                continue
            b["dx"].append(dx[m])
            b["dy"].append(dy[m])
            b["dz"].append(dz[m])
        del df, x, y, z, dx, dy, dz, top, side
        print(f"  read {os.path.basename(path)}", flush=True)

    print("\nfitting samples", flush=True)
    rows = []
    hr = (-args.hist_range, +args.hist_range)
    for b in bufs:
        for comp in ("dx", "dy", "dz"):
            arrs = b[comp]
            if not arrs:
                print(f"  {b['sub']:>4s} ({b['cx']:+4d},{b['cy']:+4d},{b['cz']:+4d}) "
                      f"{comp}: NO HITS", flush=True)
                continue
            data = np.concatenate(arrs)
            tag = (f"{b['sub']}_x{int(b['cx']):+04d}_y{int(b['cy']):+04d}"
                   f"_z{int(b['cz']):+04d}_{comp}")
            outpath = os.path.join(args.outdir, f"{tag}.png")
            r = fit_one(data, comp, b["sub"], b["cx"], b["cy"], b["cz"],
                        outpath, hist_range=hr, bin_width=args.bin_width)
            if r is None:
                print(f"  {b['sub']:>4s} ({b['cx']:+4d},{b['cy']:+4d},"
                      f"{b['cz']:+4d}) {comp}: TOO FEW (N={data.size})",
                      flush=True)
                continue
            rows.append(r)
            print(f"  {b['sub']:>4s} ({b['cx']:+4d},{b['cy']:+4d},{b['cz']:+4d}) "
                  f"{comp}: N={r['n']:>6d}  "
                  f"arith={r['arith_mean']:+7.2f}  med={r['median']:+7.2f}  "
                  f"|  Gauss: mu={r['gauss_mu']:+6.2f} sig={r['gauss_sigma']:5.2f} "
                  f"chi2/ndf={r['gauss_chi2_ndf']:5.1f}  "
                  f"|  G+C: mu={r['gpc_mu']:+6.2f} sig={r['gpc_sigma']:5.2f} "
                  f"C={r['gpc_C']:5.1f} chi2/ndf={r['gpc_chi2_ndf']:5.1f}",
                  flush=True)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(args.outdir, "fit_summary.csv"), index=False)
        print(f"\nwrote {os.path.join(args.outdir, 'fit_summary.csv')}",
              flush=True)


if __name__ == "__main__":
    main()
