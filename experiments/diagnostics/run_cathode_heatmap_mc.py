"""MC analog of run_cathode_heatmap_v3.py: run plot_cathode_heatmap.main()
against the MC cathode-crossing drift_df with the same tuned fit parameters
(DT_WINDOW/DT_BIN/MIN_COUNTS) used for the data v3 analysis.

Prints the per-plane MC drift velocity (v = 4 m / <dt0+dt1>) and writes the
per-(y,z)-pixel cathode-tick heatmaps + cathode_fit_results_tpc{0,1}.csv to
cathode_study_mc/ (working dir).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import plot_cathode_heatmap as P

# same tuned constants as run_cathode_heatmap_v3.py (patched before main()):
P.DT_WINDOW = (2440.0, 2640.0)
P.DT_BIN = 4.0
P.MIN_COUNTS = 10

sys.argv = [
    "plot_cathode_heatmap.py",
    "--input", "/exp/sbnd/data/users/gputnam/efield-study/drift_df_mc.parquet",
    "--outdir", "cathode_study_mc",
    "--pixel-size", "2",
    "--mode", "combined",
    "--nproc", "auto",
    "--min-counts", "10",
]

if __name__ == "__main__":
    P.main()
