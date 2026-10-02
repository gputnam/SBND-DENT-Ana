#!/usr/bin/env python3
"""Drift-velocity determination via a Crystal Ball fit to the anode-cathode
tick-length distribution, per (TPC, plane) and TPC-combined.

Improves on the mean-of-<dt0+dt1> estimator (tail-sensitive, epoch- and
window-dependent) by fitting the PEAK of the dt distribution with a Crystal
Ball function -- the same robust mode the per-pixel cathode fits use.

For each (track, TPC, plane): dt = max_time - min_time [ticks] (anode-to-
cathode span of that TPC's segment). Selections:
  * anode-cathode through-going: dt0 > 2400 AND dt1 > 2400 ticks;
  * STABLE cathode regions only (cathode-side endpoint max_y/max_z away from
    the support-frame features at z~255, y~+-50/+-150 and the edges --
    same windows as plot_ac_ticklength.py).

Estimators reported per (TPC, plane):
  v_TPC = 200 cm / (mu_CB * 0.5 us/tick)        [single-TPC, L=200 cm]
and TPC-combined from the cathode-position-independent sum:
  v_sum = 400 cm / (mu_CB[dt0+dt1] * 0.5 us)    [full drift, L=400 cm]

For MC the measured v is compared to the WireCell truth (1.563 mm/us =
0.15630 cm/us, constant drift).

Usage:
    python fit_driftv_cb.py --input <drift_df.parquet> --out DIR \
        [--label TAG] [--truth 0.15630]
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

TICK_US = 0.5
L_TPC = 200.0           # cm, single-TPC anode-cathode (nominal)
L_SUM = 400.0           # cm, full anode-anode drift
AC_CUT = 2400.0         # ticks, per-TPC near-full-drift selection
ANODE_WIN = (410.0, 430.0)

Y_STABLE = [(-190.0, -160.0), (-140.0, -60.0), (-40.0, 40.0),
            (60.0, 140.0), (160.0, 190.0)]
Z_STABLE = [(30.0, 240.0), (270.0, 470.0)]


def crystal_ball(x, alpha, n, mu, sigma, N):
    z = (x - mu) / sigma
    a = abs(alpha)
    A = (n / a) ** n * np.exp(-0.5 * a * a)
    B = n / a - a
    return N * np.where(z > -alpha, np.exp(-0.5 * z * z),
                        A * np.maximum(B - z, 1e-9) ** (-n))


def in_windows(v, wins):
    m = np.zeros(v.shape, dtype=bool)
    for lo, hi in wins:
        m |= (v >= lo) & (v < hi)
    return m


def fit_cb(dt, half=90.0, binw=1.0):
    """CB fit of a dt sample. Returns (mu, mu_err, n_used, fit-params)."""
    dt = dt[np.isfinite(dt)]
    if len(dt) < 500:
        return np.nan, np.nan, len(dt), None
    lo, hi = np.percentile(dt, [1, 99])
    edges = np.arange(lo, hi + binw, binw)
    centers = 0.5 * (edges[:-1] + edges[1:])
    h, _ = np.histogram(dt, bins=edges)
    pk = centers[np.argmax(h)]
    fitm = (centers >= pk - half) & (centers <= pk + half)
    x, y = centers[fitm], h[fitm].astype(float)
    p0 = [1.5, 3.0, pk, 12.0, y.max()]
    bounds = ([0.1, 1.01, pk - half, 1.0, 1.0],
              [10.0, 200.0, pk + half, 60.0, 1e9])
    try:
        popt, pcov = curve_fit(crystal_ball, x, y, p0=p0, bounds=bounds,
                               maxfev=40000)
        return popt[2], float(np.sqrt(pcov[2, 2])), len(dt), popt
    except Exception:
        return pk, np.nan, len(dt), None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="driftv_cb")
    ap.add_argument("--label", default="")
    ap.add_argument("--truth", type=float, default=None,
                    help="true drift velocity for the comparison (MC)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cols = []
    for t in (0, 1):
        for p in (0, 1, 2):
            pfx = f"tpc{t}_p{p}"
            cols += [f"{pfx}_min_time", f"{pfx}_max_time",
                     f"{pfx}_max_y", f"{pfx}_max_z"]
    df = pd.read_parquet(args.input, columns=cols)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    rows = []
    v_tpc = {}
    for p in (0, 1, 2):
        dt0 = (df[f"tpc0_p{p}_max_time"] - df[f"tpc0_p{p}_min_time"]).to_numpy()
        dt1 = (df[f"tpc1_p{p}_max_time"] - df[f"tpc1_p{p}_min_time"]).to_numpy()
        ac = (np.isfinite(dt0) & np.isfinite(dt1)
              & (dt0 > AC_CUT) & (dt1 > AC_CUT))
        for t, dt in ((0, dt0), (1, dt1)):
            ymax = df[f"tpc{t}_p{p}_max_y"].to_numpy()
            zmax = df[f"tpc{t}_p{p}_max_z"].to_numpy()
            tmin = df[f"tpc{t}_p{p}_min_time"].to_numpy()
            sel = (ac & (tmin >= ANODE_WIN[0]) & (tmin <= ANODE_WIN[1])
                   & in_windows(ymax, Y_STABLE) & in_windows(zmax, Z_STABLE))
            mu, mu_err, n, popt = fit_cb(dt[sel])
            v = L_TPC / (mu * TICK_US)
            v_err = v * (mu_err / mu) if np.isfinite(mu_err) else np.nan
            v_tpc.setdefault(p, {})[t] = v
            rows.append(dict(plane=p, tpc=t, mu_ticks=mu, mu_err=mu_err,
                             n=n, v=v, v_err=v_err))
            ax = axes[t, p]
            d = dt[sel]
            d = d[(d > mu - 120) & (d < mu + 120)]
            edges = np.arange(mu - 120, mu + 120, 1.0)
            c = 0.5 * (edges[:-1] + edges[1:])
            ax.hist(d, bins=edges, color="steelblue", edgecolor="navy",
                    lw=0.2)
            if popt is not None:
                xs = np.linspace(mu - 90, mu + 90, 400)
                ax.plot(xs, crystal_ball(xs, *popt), "r-", lw=1.4)
            ax.axvline(mu, color="red", ls="--", lw=1.0)
            ax.set_xlim(mu - 120, mu + 120)
            ax.set_title(f"TPC{t} plane{p}"
                         + ("  (coll.)" if p == 2 else "  (ind.)")
                         + f": $\\mu$={mu:.1f}, v={v:.5f}", fontsize=9)
            if t == 1:
                ax.set_xlabel(r"$dt=t_{max}-t_{min}$ [ticks]")
            if p == 0:
                ax.set_ylabel("tracks / tick")
            ax.grid(alpha=0.3)
        # combined (sum) per plane
        ymax0 = df[f"tpc0_p{p}_max_y"].to_numpy()
        zmax0 = df[f"tpc0_p{p}_max_z"].to_numpy()
        ymax1 = df[f"tpc1_p{p}_max_y"].to_numpy()
        zmax1 = df[f"tpc1_p{p}_max_z"].to_numpy()
        tmin0 = df[f"tpc0_p{p}_min_time"].to_numpy()
        tmin1 = df[f"tpc1_p{p}_min_time"].to_numpy()
        sels = (ac & (tmin0 >= ANODE_WIN[0]) & (tmin0 <= ANODE_WIN[1])
                & (tmin1 >= ANODE_WIN[0]) & (tmin1 <= ANODE_WIN[1])
                & in_windows(ymax0, Y_STABLE) & in_windows(zmax0, Z_STABLE)
                & in_windows(ymax1, Y_STABLE) & in_windows(zmax1, Z_STABLE))
        mu, mu_err, n, _ = fit_cb((dt0 + dt1)[sels], half=140.0)
        vsum = L_SUM / (mu * TICK_US)
        rows.append(dict(plane=p, tpc=-1, mu_ticks=mu, mu_err=mu_err, n=n,
                         v=vsum, v_err=(vsum * mu_err / mu
                                        if np.isfinite(mu_err) else np.nan)))
    fig.suptitle(f"Crystal Ball drift-velocity fits, stable cathode regions "
                 f"{args.label}\n(single-TPC dt, L=200 cm)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "driftv_cb_fits.png"), dpi=120)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "driftv_cb.csv"), index=False)

    # ---- report ---------------------------------------------------------------
    print(f"\n=== Crystal Ball drift velocity {args.label} ===")
    print(f"{'plane':6s} {'TPC0 v':>9s} {'TPC1 v':>9s} {'TPC-avg':>9s} "
          f"{'v(sum/400)':>11s}" + ("  meas/true" if args.truth else ""))
    pl = res[res.tpc >= 0]
    for p in (0, 1, 2):
        v0 = v_tpc[p][0]
        v1 = v_tpc[p][1]
        vavg = 0.5 * (v0 + v1)
        vsum = res[(res.plane == p) & (res.tpc == -1)].v.iloc[0]
        extra = f"  {vsum/args.truth:.4f}" if args.truth else ""
        print(f"{p:<6d} {v0:>9.5f} {v1:>9.5f} {vavg:>9.5f} {vsum:>11.5f}"
              + extra)
    # 3-plane means
    m0 = np.mean([v_tpc[p][0] for p in (0, 1, 2)])
    m1 = np.mean([v_tpc[p][1] for p in (0, 1, 2)])
    msum = pl[pl.tpc == -1].v.mean() if False else \
        res[res.tpc == -1].v.mean()
    print(f"{'3-pl':6s} {m0:>9.5f} {m1:>9.5f} {0.5*(m0+m1):>9.5f} "
          f"{msum:>11.5f}"
          + (f"  {msum/args.truth:.4f}" if args.truth else ""))
    if args.truth:
        print(f"\n  true v = {args.truth:.5f} cm/us; "
              f"v(sum)/true = {msum/args.truth:.4f} "
              f"({100*(msum/args.truth-1):+.2f}%)")
    print(f"\nwrote {args.out}/driftv_cb_fits.png, driftv_cb.csv", flush=True)


if __name__ == "__main__":
    main()
