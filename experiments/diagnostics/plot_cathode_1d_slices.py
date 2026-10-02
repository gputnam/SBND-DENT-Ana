"""1-D cathode-tick slices vs Z, one figure per Y row.

For each native Y row (200), draws three error-bar lines vs detector Z:
TPC0 cathode tick (mu), TPC1 cathode tick (mu), and the per-pixel mean
(mu_TPC0 + mu_TPC1)/2 with sigma_err propagated in quadrature.

Two output subdirectories are written:
  no_spline/    raw measurements only.
  with_spline/  measurements + step2 cubic-B-spline overlay (converted
                from cm displacement back to ticks via
                mu_fit = mu_nominal + sign * x_fit_native / cm_per_tick).

Pixels whose per-TPC mu_err exceeds err_drop (the step2 cleaning
threshold, default 10 ticks) are masked out of both the line and the
error bar in that TPC; the mean is shown only where both TPCs survive.
"""

import argparse
import multiprocessing as mp
import os
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STEP1_DEFAULT = ("/exp/sbnd/data/users/gputnam/efield-study/"
                 "cathode_pipeline/step1_displacement")
STEP2_DEFAULT = ("/exp/sbnd/data/users/gputnam/efield-study/"
                 "cathode_pipeline/step2_spline")
OUTDIR_DEFAULT = "cathode_1d_slices"

C_TPC0 = "tab:blue"
C_TPC1 = "tab:red"
C_MEAN = "black"


def load_tpc(step1_dir, step2_dir, tpc):
    s1 = np.load(os.path.join(step1_dir, f"tpc{tpc}_displacement.npz"))
    s2 = np.load(os.path.join(step2_dir, f"tpc{tpc}_spline.npz"))

    mu = s1["mu"].astype(np.float64)
    mu_err = s1["mu_err"].astype(np.float64)
    n = s1["n_entries"]
    y = s1["y_centers"]
    z = s1["z_centers"]
    err_drop = float(s1["err_drop"])
    mu_nominal = float(s1["mu_nominal"])
    cm_per_tick = float(s1["cm_per_tick"])
    sign = float(s1["sign"])
    fit_rms_cm = float(s2["fit_rms_cm"])

    bad = (~np.isfinite(mu)) | (~np.isfinite(mu_err)) | (mu_err > err_drop) | (n <= 0)
    mu_m = np.where(bad, np.nan, mu)
    err_m = np.where(bad, np.nan, mu_err)

    # spline fit: x_fit_native is in cm displacement; convert back to ticks
    x_fit = s2["x_fit_native"].astype(np.float64)
    mu_fit = mu_nominal + sign * x_fit / cm_per_tick

    return dict(mu=mu_m, mu_err=err_m, mu_fit=mu_fit, y=y, z=z,
                mu_nominal=mu_nominal, sign=sign, cm_per_tick=cm_per_tick,
                err_drop=err_drop, fit_rms_cm=fit_rms_cm, tpc=tpc)


def _slice_along(tpc, idx, primary):
    """Return (primary_axis_values, mu_slice, err_slice, fit_slice)
    for either a Y-row (primary='Z') or a Z-column (primary='Y') slice."""
    if primary == "Z":
        return tpc["z"], tpc["mu"][idx], tpc["mu_err"][idx], tpc["mu_fit"][idx]
    elif primary == "Y":
        return (tpc["y"], tpc["mu"][:, idx], tpc["mu_err"][:, idx],
                tpc["mu_fit"][:, idx])
    raise ValueError(primary)


def _plot_one_slice(idx, primary, tpc0, tpc1, outdir_no, outdir_with,
                    held_value, ylim, fit_rms_label):
    """primary='Z': slice at fixed Y (idx into y axis); x-axis is Z.
       primary='Y': slice at fixed Z (idx into z axis); x-axis is Y."""
    x_axis, mu0, e0, fit0 = _slice_along(tpc0, idx, primary)
    _,       mu1, e1, fit1 = _slice_along(tpc1, idx, primary)

    both = np.isfinite(mu0) & np.isfinite(mu1)
    mu_mean = np.where(both, 0.5 * (mu0 + mu1), np.nan)
    err_mean = np.where(both, 0.5 * np.sqrt(e0 ** 2 + e1 ** 2), np.nan)
    fit_mean = 0.5 * (fit0 + fit1)

    if primary == "Z":
        xlabel = "Z [cm]"
        held_label = f"Y = {held_value:+.1f} cm"
        fname = f"Y_{held_value:+07.2f}cm.png"
    else:
        xlabel = "Y [cm]"
        held_label = f"Z = {held_value:+.1f} cm"
        fname = f"Z_{held_value:+07.2f}cm.png"

    for outdir, show_spline in ((outdir_no, False), (outdir_with, True)):
        fig, ax = plt.subplots(figsize=(11.5, 5.4))
        ax.errorbar(x_axis, mu0, yerr=e0, fmt="o", ms=2.5, lw=0.6,
                    color=C_TPC0, label="TPC0 (x<0)")
        ax.errorbar(x_axis, mu1, yerr=e1, fmt="o", ms=2.5, lw=0.6,
                    color=C_TPC1, label="TPC1 (x>0)")
        ax.errorbar(x_axis, mu_mean, yerr=err_mean, fmt="o", ms=2.5, lw=0.6,
                    color=C_MEAN, label="Mean (TPC0+TPC1)/2")
        if show_spline:
            ax.plot(x_axis, fit0, "-", color=C_TPC0, lw=1.3, alpha=0.85,
                    label="TPC0 spline")
            ax.plot(x_axis, fit1, "-", color=C_TPC1, lw=1.3, alpha=0.85,
                    label="TPC1 spline")
            ax.plot(x_axis, fit_mean, "-", color=C_MEAN, lw=1.3, alpha=0.85,
                    label="Mean spline")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("cathode tick  $\\mu$")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", ncol=2 if show_spline else 1,
                  fontsize=9)
        title = f"SBND cathode tick vs {primary}   {held_label}"
        if show_spline:
            title += f"   ({fit_rms_label})"
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=110)
        plt.close(fig)


def _worker(args_chunk, tpc0, tpc1, outdir_no, outdir_with, ylim,
            fit_rms_label):
    # args_chunk: list of (idx, primary, held_value)
    for idx, primary, held_value in args_chunk:
        _plot_one_slice(idx, primary, tpc0, tpc1, outdir_no, outdir_with,
                        held_value, ylim, fit_rms_label)
    return len(args_chunk)


def main():
    mp.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1-dir", default=STEP1_DEFAULT)
    ap.add_argument("--step2-dir", default=STEP2_DEFAULT)
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT)
    ap.add_argument("--nproc", type=int,
                    default=max(1, min(16, mp.cpu_count() // 2)))
    ap.add_argument("--ylim-pad-ticks", type=float, default=30.0,
                    help="symmetric pad above/below the measured-range "
                         "for the shared y-axis range")
    args = ap.parse_args()

    tpc0 = load_tpc(args.step1_dir, args.step2_dir, 0)
    tpc1 = load_tpc(args.step1_dir, args.step2_dir, 1)

    assert np.allclose(tpc0["y"], tpc1["y"]), "y_centers mismatch"
    assert np.allclose(tpc0["z"], tpc1["z"]), "z_centers mismatch"
    ny = tpc0["mu"].shape[0]
    nz = tpc0["mu"].shape[1]

    # global y-axis range: union of finite mu values, with a small pad
    all_mu = np.concatenate([tpc0["mu"].ravel(), tpc1["mu"].ravel()])
    finite = all_mu[np.isfinite(all_mu)]
    if finite.size:
        lo = float(np.nanpercentile(finite, 0.5))
        hi = float(np.nanpercentile(finite, 99.5))
        ylim = (lo - args.ylim_pad_ticks, hi + args.ylim_pad_ticks)
    else:
        ylim = None

    fit_rms_label = (f"step2 spline fit_rms = "
                     f"{tpc0['fit_rms_cm']:.3f} / {tpc1['fit_rms_cm']:.3f} cm "
                     f"(TPC0/TPC1)")

    # Build two slice families:
    #   by_Y/  : one plot per Y row, primary axis = Z (Z-scan)
    #   by_Z/  : one plot per Z column, primary axis = Y (Y-scan)
    families = [
        ("by_Y", "Z", [(iy, "Z", float(tpc0["y"][iy])) for iy in range(ny)]),
        ("by_Z", "Y", [(iz, "Y", float(tpc0["z"][iz])) for iz in range(nz)]),
    ]

    print(f"  ylim = {ylim}", flush=True)
    print(f"  {fit_rms_label}", flush=True)
    print(f"  nproc = {args.nproc}", flush=True)

    total = 0
    for famdir, primary, task_list in families:
        outdir_no = os.path.join(args.outdir, famdir, "no_spline")
        outdir_with = os.path.join(args.outdir, famdir, "with_spline")
        os.makedirs(outdir_no, exist_ok=True)
        os.makedirs(outdir_with, exist_ok=True)
        print(f"plotting {len(task_list)} slices -> {args.outdir}/{famdir}/"
              f"{{no_spline,with_spline}}/  (vs {primary})", flush=True)
        chunks = np.array_split(np.arange(len(task_list)), args.nproc)
        chunks = [[task_list[i] for i in c] for c in chunks if c.size]
        worker = partial(_worker, tpc0=tpc0, tpc1=tpc1,
                         outdir_no=outdir_no, outdir_with=outdir_with,
                         ylim=ylim, fit_rms_label=fit_rms_label)
        if args.nproc <= 1 or len(chunks) <= 1:
            for c in chunks:
                worker(c)
        else:
            with mp.Pool(args.nproc) as pool:
                for _ in pool.imap_unordered(worker, chunks):
                    pass
        total += 2 * len(task_list)

    print(f"\nwrote {total} PNGs under {args.outdir}/", flush=True)


if __name__ == "__main__":
    main()
