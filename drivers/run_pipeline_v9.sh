#!/bin/bash
# v9 field-distortion pipeline (downstream of the doubleanode_2d_df_v9 parquet).
# Mirrors the v7/v8 production commands with _v9 output names.
set -euo pipefail
cd /exp/sbnd/app/users/gputnam/cathode-study
source setup.sh
export BEARER_TOKEN_FILE=/run/user/$(id -u)/bt_u$(id -u)

DF=/exp/sbnd/data/users/gputnam/efield-study/doubleanode_2d_df_v9
SCE=sce_input_doubleanode_2d

step () { echo; echo "==================== $* ===================="; date; }

step "1/7 two-iteration spatial voxel map"
python plot_doubleanode_2d_voxels.py \
  --input "$DF" \
  --outdir doubleanode_2d_maps_20cm_v9 \
  --iterations 2 --iter-resid-cm 4.0 --voxel-size 20 --nproc auto

step "2/7 spatial spline + forward SCE ROOT"
python fit_doubleanode_2d_spline.py \
  --input doubleanode_2d_maps_20cm_v9/voxel_data.npz \
  --outdir doubleanode_2d_spline_20cm_v9 \
  --nx-knots 5 --ny-knots 6 --nz-knots 9 --ridge-lambda 1 \
  --write-sce-root --sce-root-name sbnd_sce_fwd_doubleanode_2d_v9.root

step "3/7 E-field voxel map"
python build_efield_2d_voxels.py \
  --input doubleanode_2d_maps_20cm_v9/voxel_data.npz \
  --spline-knots-json doubleanode_2d_spline_20cm_v9/spline_knots.json \
  --outdir efield_2d_maps_20cm_v9

step "4/7 E-field spline + E-field SCE ROOT"
python fit_efield_2d_spline.py \
  --input efield_2d_maps_20cm_v9/voxel_data.npz \
  --outdir efield_2d_spline_20cm_v9 \
  --nx-knots 8 --ny-knots 10 --nz-knots 16 --ridge-lambda 0.1 \
  --write-sce-root \
  --displacement-root "$SCE/sbnd_sce_fwd_doubleanode_2d_v9.root" \
  --sce-root-name sbnd_sce_efield_doubleanode_2d_v9.root

step "5/7 Gauss's-law check"
python check_gauss_2d.py \
  --input efield_2d_maps_20cm_v9/voxel_data.npz \
  --spline-npz efield_2d_spline_20cm_v9/spline_fit.npz \
  --outdir efield_2d_gauss_20cm_v9

step "6/7 assemble combined SCE ROOT (forward + derived backward + E-field)"
python build_sce_root.py \
  --forward-root "$SCE/sbnd_sce_fwd_doubleanode_2d_v9.root" \
  --efield-root  "$SCE/sbnd_sce_efield_doubleanode_2d_v9.root" \
  --out          "$SCE/sbnd_sce_doubleanode_2d_v9.root"

step "7/7 iter1-vs-iter2 before/after comparison"
python compare_iter1_iter2.py \
  --input doubleanode_2d_maps_20cm_v9/voxel_data.npz \
  --out compare_iter1_iter2_v9

step "PIPELINE COMPLETE"
