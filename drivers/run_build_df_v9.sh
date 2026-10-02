#!/bin/bash
# v9 doubleanode_2d_df rebuild over the expanded 258k-file union (xrootd).
set -euo pipefail
cd /exp/sbnd/app/users/gputnam/cathode-study
source setup.sh
export BEARER_TOKEN_FILE=/run/user/$(id -u)/bt_u$(id -u)
OUT=/exp/sbnd/data/users/gputnam/efield-study/doubleanode_2d_df_v9
echo "START $(date) -> $OUT"
python make_doubleanode_2d_df_sbnd.py "$OUT" filelist_drift_full.txt --nproc 64
echo "DONE $(date)"
