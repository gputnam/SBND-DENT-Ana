"""No-SCE MC analog of run_cathode_heatmap_mc.py: run plot_cathode_heatmap.main()
against the no-SCE cathode-crossing drift_df with the same tuned fit parameters
(DT_WINDOW/DT_BIN/MIN_COUNTS) used for the data v3 / MC analyses.

The no-SCE sample has no space-charge displacement, so the cathode is flat at
nominal x=0 and the drift velocity should be sim-nominal (~0.15627 cm/us).

Prints the per-plane drift velocity (v = 4 m / <dt0+dt1>) and writes the
per-(y,z)-pixel cathode-tick heatmaps + cathode_fit_results_tpc{0,1}.csv to
cathode_study_nosce/ (working dir).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import plot_cathode_heatmap as P

# same tuned constants as run_cathode_heatmap_mc.py (patched before main()):
P.DT_WINDOW = (2440.0, 2640.0)
P.DT_BIN = 4.0
P.MIN_COUNTS = 10

sys.argv = [
    "plot_cathode_heatmap.py",
    "--input", "/exp/sbnd/data/users/gputnam/efield-study/drift_df_nosce.parquet",
    "--outdir", "cathode_study_nosce",
    "--pixel-size", "2",
    "--mode", "combined",
    "--nproc", "auto",
    "--min-counts", "10",
]

if __name__ == "__main__":
    P.main()
