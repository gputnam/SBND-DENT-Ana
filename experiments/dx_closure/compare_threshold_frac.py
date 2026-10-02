"""Scan threshold-frac for the per-TPC anode-piercing cut.

Reuses find_peak_window() and stack_per_tpc_tmin() from
plot_track_cuts_per_tpc.py and reports, for each threshold_frac:

  - selected window [lo, hi]  (and width)
  - candidate count inside window
  - selected fraction of all per-TPC anode candidates

Output: stdout table + cut_tuning/threshold_frac_scan.png
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_track_cuts_per_tpc import (
    load_align_df, stack_per_tpc_tmin, find_peak_window,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet",
                    default="/exp/sbnd/data/users/gputnam/efield-study/align_df")
    ap.add_argument("--outdir", default="cut_tuning")
    ap.add_argument("--fracs", type=str, default="0.01,0.05,0.10,0.25")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_align_df(args.parquet)
    pt = stack_per_tpc_tmin(df)
    n_total = len(pt)
    print(f"  per-TPC anode-piercing candidates: {n_total:,}\n")

    fracs = [float(x) for x in args.fracs.split(",")]
    rows = []
    for f in fracs:
        peak, lo, hi = find_peak_window(
            pt["t_min"].values,
            search_lo=400, search_hi=440,
            bg_lo=600, bg_hi=1500,
            threshold_frac=f,
        )
        if peak is None:
            rows.append((f, None, None, None, 0, 0.0))
            continue
        n_in = int(((pt["t_min"] >= lo) & (pt["t_min"] <= hi)).sum())
        rows.append((f, peak, lo, hi, n_in, 100.0 * n_in / n_total))

    head = (f"{'thresh_frac':>12s}  {'peak':>6s}  {'lo':>6s}  "
            f"{'hi':>6s}  {'width':>6s}  {'survivors':>11s}  {'sel %':>7s}")
    print(head); print("-" * len(head))
    for f, peak, lo, hi, n_in, pct in rows:
        if peak is None:
            print(f"{f:>12.3f}  {'  --':>6s}  {'  --':>6s}  {'  --':>6s}  "
                  f"{'  --':>6s}  {n_in:>11,d}  {pct:>6.2f}%")
        else:
            print(f"{f:>12.3f}  {peak:>6.1f}  {lo:>6.1f}  {hi:>6.1f}  "
                  f"{hi-lo:>6.2f}  {n_in:>11,d}  {pct:>6.2f}%")

    h, edges = np.histogram(pt["t_min"], bins=4000, range=(0, 4000))
    c = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(c, h, where="mid", linewidth=1.0, color="0.25")
    ax.set_xlim(405, 440)
    ax.set_yscale("log")
    ax.set_xlabel("per-TPC tpc{0,1}_t_min, t0-corrected (ticks)")
    ax.set_ylabel("entries / 1-tick bin")
    palette = ["tab:red", "tab:orange", "tab:green", "tab:blue"]
    for (f, peak, lo, hi, n_in, pct), color in zip(rows, palette):
        if peak is None:
            continue
        ax.axvspan(lo, hi, color=color, alpha=0.15,
                   label=f"frac={f:g}: [{lo:.1f}, {hi:.1f}]  "
                         f"({pct:.1f}%, n={n_in:,})")
    if rows[0][1] is not None:
        ax.axvline(rows[0][1], color="k", ls=":", lw=0.7,
                   label=f"peak {rows[0][1]:.1f}")
    ax.set_title(f"Threshold-frac scan (per-TPC t_min, n={n_total:,})")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.92)
    fig.tight_layout()
    out = os.path.join(args.outdir, "threshold_frac_scan")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=140)
    plt.close(fig)
    print(f"\n  wrote {out}.png/.pdf")


if __name__ == "__main__":
    main()
