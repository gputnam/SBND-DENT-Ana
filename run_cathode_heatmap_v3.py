"""One-off wrapper: re-run plot_cathode_heatmap.main() against the
full-sample drift_df parquet with adjusted fit parameters.

Changes vs. the default plot_cathode_heatmap.py constants:
    DT_WINDOW : (2450, 2650)  ->  (2440, 2640)    # shifted 10 ticks earlier
    DT_BIN    :  2.0          ->   4.0            # doubled once
    MIN_COUNTS: 30            ->  10              # relaxed for marginal pixels

Output: /exp/sbnd/data/users/gputnam/efield-study/cathode_study_2cm_v3/
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plot_cathode_heatmap as P

# Patch module-level constants BEFORE main() runs argparse so the defaults
# pick them up. fit_pixel(), process_tpc() etc. dereference these at call
# time, so patching here is sufficient.
P.DT_WINDOW  = (2440.0, 2640.0)
P.DT_BIN     = 4.0
P.MIN_COUNTS = 10

sys.argv = [
    "plot_cathode_heatmap.py",
    "--input",  "/exp/sbnd/data/users/gputnam/efield-study/drift_df/drift_df_full_v2.parquet",
    "--outdir", "/exp/sbnd/data/users/gputnam/efield-study/cathode_study_2cm_v3",
    "--pixel-size", "2",
    "--mode", "combined",
    "--nproc", "auto",
    # explicit --min-counts so the CLI override matches the patched default
    "--min-counts", "10",
]

if __name__ == "__main__":
    P.main()
