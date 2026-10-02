#!/bin/bash
# v11 cathode displacement -> spline -> v6-family ASCII, from the v11-epoch
# heatmap CSVs (cathode_study_2cm_v5). Mirrors the stepped v6 pipeline
# (WORKFLOW.md 4c) with _v11 output dirs. Produces the cathode ASCII
# deliverable (labelled v7 for this epoch).
set -e
source setup.sh

DATA=/exp/sbnd/data/users/gputnam/efield-study
CSV=$DATA/cathode_study_2cm_v5
S1=$DATA/cathode_pipeline/step1_displacement_v11
S2=$DATA/cathode_pipeline/step2_spline_v11
S3=$DATA/cathode_pipeline/step3_v11

python cathode_step1_tick_to_cm.py \
  --csv-dir "$CSV" --outdir "$S1" --v-drift 0.15542 --err-drop 10

python cathode_step2_spline.py \
  --step1-dir "$S1" --outdir "$S2" \
  --ny-knots 80 --nz-knots 120 --ridge-lambda 5.0

python cathode_step3_v6.py \
  --step1-dir "$S1" --step2-dir "$S2" --outdir "$S3" \
  --panel-half-thickness 1.5

# label this epoch's ASCII deliverable v7 (script hard-codes the v6 filename)
cp -f "$S3/sbnd_cathode_v6_drift.txt" "$S3/sbnd_cathode_v7_drift.txt"
echo "cathode ASCII deliverable: $S3/sbnd_cathode_v7_drift.txt"
echo CATHODE_PIPELINE_V11_DONE
