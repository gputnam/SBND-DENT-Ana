#!/bin/bash
# AC-v3 post-parquet workflow (run after drift_df_full_v3.parquet exists).
# Codifies the full anode+cathode measurement chain for this and future
# iterations:
#   1. cathode heatmap re-measurement, 2x2 cm pixels, per-TPC CB fits
#      (+ per-plane v_drift report)            -> cathode_study_2cm_v4/
#   2. per-pixel TPC0+TPC1 summed time-length histogram
#                                              -> cathode_study_2cm_v4/
#   3. per-TPC per-plane AC tick-length histograms in STABLE cathode
#      regions (linear y, peak +-30 ticks)     -> ac_ticklength_data_v3/
set -e
source setup.sh

PARQ=/exp/sbnd/data/users/gputnam/efield-study/drift_df/drift_df_full_v3.parquet

python run_cathode_heatmap_v4.py

python plot_pixel_dtsum_hist.py --fits-dir cathode_study_2cm_v4 \
  --out cathode_study_2cm_v4 --label "(v3 epoch)"

python plot_ac_ticklength.py --input $PARQ \
  --out ac_ticklength_data_v3 --label "(data, v3 epoch)"

# Crystal Ball drift-velocity determination, per (TPC, plane) + combined
python fit_driftv_cb.py --input $PARQ \
  --out driftv_cb_data_v3 --label "(data v3 epoch)"
python fit_driftv_cb.py \
  --input /exp/sbnd/data/users/gputnam/efield-study/drift_df_mc.parquet \
  --out driftv_cb_mc --label "(MC)" --truth 0.15630

echo AC_V3_WORKFLOW_DONE
