"""AC-v3 epoch re-measurement: re-run plot_cathode_heatmap.main() against
the REBUILT drift_df parquet (drift_df_full_v3, the 258,225-file union
including the 2026-06-01 Run 2 crossingmuon expansion) with the same tuned
fit parameters as the v3 analysis (DT_WINDOW/DT_BIN/MIN_COUNTS).

Outputs (cathode_study_2cm_v4/, in the study dir):
  cathode_fit_results_tpc{0,1}.csv   per-2cm-pixel Crystal Ball dt fits
  per-pixel heatmaps (png/pdf)
  stdout: per-plane v_drift report (4 m / <dt0+dt1>) on the new epoch
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plot_cathode_heatmap as P

# same tuned constants as run_cathode_heatmap_v3.py
P.DT_WINDOW = (2440.0, 2640.0)
P.DT_BIN = 4.0
P.MIN_COUNTS = 10

sys.argv = [
    "plot_cathode_heatmap.py",
    "--input",
    "/exp/sbnd/data/users/gputnam/efield-study/drift_df/drift_df_full_v3.parquet",
    "--outdir", "cathode_study_2cm_v4",
    "--pixel-size", "2",
    "--mode", "combined",
    "--nproc", "auto",
    "--min-counts", "10",
]

if __name__ == "__main__":
    P.main()
