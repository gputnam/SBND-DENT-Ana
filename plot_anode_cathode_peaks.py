"""Plot the anode and cathode timing peaks for each SBND TPC.

For each TPC we combine track start (min) and end (max) hit times into one
distribution, then zoom in on the anode and cathode regions. Four plots are
produced:
    tpc0_anode_peak.png, tpc0_cathode_peak.png,
    tpc1_anode_peak.png, tpc1_cathode_peak.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT = "drift_df.parquet"

# Expected anode/cathode ticks for in-spill tracks, derived from sbndcode
# v10_14_02_02 detector configuration (no fitting).
#
#   Anode:
#     detectorclocks_sbnd.fcl  TriggerOffsetTPC = -205 us
#     ClockSpeedTPC = 2 MHz                      -> 500 ns/tick
#     => trigger sits at tick 205 us / 0.5 us = 410
#
#   Cathode = anode + drift_distance / v_drift / tick_period
#     drift distance (active) = 200 cm
#     E = 500 V/cm (detectorproperties_sbnd.fcl Efield[1])
#     T = 88.4 K   (sbnd_detproperties.Temperature)
#     v_drift = 0.15533 cm/us via Walkowiak (NIM A 449 (2000) 288),
#               same parameterization as LArSoft LArProperties
#     drift_time = 200 / 0.15533 = 1287.5 us = 2575.1 ticks
#     => cathode tick = 410 + 2575 = 2985
ANODE_TICK = 410
CATHODE_TICK = 2985

# Zoom windows (tick units) for each peak.
ANODE_WINDOW = (200, 700)
CATHODE_WINDOW = (2700, 3200)
BIN_WIDTH = 2.0  # ticks/bin (1 tick = 500 ns)


def combined_endpoint_times(df, tpc):
    a = df[f"tpc{tpc}_min_time"].dropna().to_numpy()
    b = df[f"tpc{tpc}_max_time"].dropna().to_numpy()
    return np.concatenate([a, b])


def plot_peak(times, window, expected_tick, title, outpath):
    lo, hi = window
    mask = (times >= lo) & (times < hi)
    sel = times[mask]

    bins = np.arange(lo, hi + BIN_WIDTH, BIN_WIDTH)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(sel, bins=bins, histtype="step", linewidth=1.2)
    ax.axvline(expected_tick, color="red", linestyle="--", linewidth=0.8,
               label=f"expected = {expected_tick} ticks")
    ax.set_xlabel("hit time [ticks, 500 ns/tick]")
    ax.set_ylabel(f"endpoints / {BIN_WIDTH:g} ticks")
    ax.set_title(title + f"  (N = {sel.size:,})")
    ax.set_xlim(lo, hi)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"wrote {outpath}  (entries in window = {sel.size:,})")


def main():
    df = pd.read_parquet(INPUT)
    print(f"loaded {len(df):,} tracks")

    for tpc in (0, 1):
        times = combined_endpoint_times(df, tpc)
        plot_peak(times, ANODE_WINDOW, ANODE_TICK,
                  f"SBND TPC{tpc} anode peak (track start + end hit times)",
                  f"tpc{tpc}_anode_peak.png")
        plot_peak(times, CATHODE_WINDOW, CATHODE_TICK,
                  f"SBND TPC{tpc} cathode peak (track start + end hit times)",
                  f"tpc{tpc}_cathode_peak.png")


if __name__ == "__main__":
    main()
