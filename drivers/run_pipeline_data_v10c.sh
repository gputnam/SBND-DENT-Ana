#!/bin/bash
# DATA spatial re-run, joint ramp + measured mean-cathode-offset
# (--x-ramp-cathode-offset). Applied correction stays the shared eps; c is
# measured per iteration (per-TPC drift-velocity-difference diagnostic).
# Full deliverable set: maps (+slice GIFs), spline, spline+voxel SCE ROOTs,
# compare vs dual, 1D profiles (voxel+spline). v9 shards ~ full-MC shard
# size -> --accum-nproc 1.
set -e
source setup.sh

PQ=/exp/sbnd/data/users/gputnam/efield-study/doubleanode_2d_df_v9
MAPS=doubleanode_2d_maps_20cm_v10c
SPL=doubleanode_2d_spline_20cm_v10c
DUAL=SCEoffsets_SBND_E500_dualmap_voxelTH3.root

python plot_doubleanode_2d_voxels.py \
  --input $PQ --outdir $MAPS \
  --vdrift 0.15532 --iterations 4 --iter-resid-cm 4.0 \
  --anode-extrapolate --x-ramp-correct hits --x-ramp-joint \
  --x-ramp-cathode-offset \
  --anode-peak-offset-json anode_hit_times_data/anode_peak_offset.json \
  --accum-nproc ${ACCUM_NPROC:-1} --nproc 24 --no-counts-maps

python fit_doubleanode_2d_spline.py \
  --input $MAPS/voxel_data.npz --outdir $SPL \
  --nx-knots 5 --ny-knots 6 --nz-knots 9 --ridge-lambda 1 \
  --write-sce-root \
  --sce-root-name sbnd_sce_fwd_spline_doubleanode_2d_v10c.root

python write_voxel_sce_root.py \
  --input $MAPS/voxel_data.npz \
  --sce-root-name sbnd_sce_fwd_voxel_doubleanode_2d_v10c.root

python make_voxel_map_gifs.py $MAPS

python compare_2d_vs_truth.py --input $MAPS/voxel_data.npz \
  --truth-root $DUAL --out compare_2d_vs_truth_v10c_dual
python plot_1d_profiles_vs_truth.py --input $MAPS/voxel_data.npz \
  --truth-root $DUAL --out profiles_1d_v10c_dual --label "data v10c"
python plot_1d_profiles_vs_truth.py --input $MAPS/voxel_data.npz \
  --use-spline $SPL/spline_fit.npz \
  --truth-root $DUAL --out profiles_1d_v10c_dual_spline \
  --label "data v10c (spline)"

echo DATA_V10C_DONE
