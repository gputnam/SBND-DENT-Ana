#!/bin/bash
# MC full-stats closure, joint ramp + measured mean-cathode-offset
# (--x-ramp-cathode-offset). Applied correction stays the shared eps; c is
# measured per iteration. Full deliverable set: maps (+slice GIFs), spline,
# spline+voxel SCE ROOTs, compare vs dual truth, 1D profiles (voxel+spline).
set -e
source setup.sh

PQ=/exp/sbnd/data/users/gputnam/efield-study/doubleanode_2d_df_mc_full
MAPS=doubleanode_2d_maps_20cm_mc_full_v10jc
SPL=doubleanode_2d_spline_20cm_mc_full_v10jc
DUAL=SCEoffsets_SBND_E500_dualmap_voxelTH3.root
SCEDIR=sce_input_doubleanode_2d

# spatial map: joint ramp + cathode-offset diagnostic; keep slice maps for GIFs
python plot_doubleanode_2d_voxels.py \
  --input $PQ --outdir $MAPS \
  --vdrift 0.15627 --iterations 4 --iter-resid-cm 4.0 \
  --anode-extrapolate --x-ramp-correct hits --x-ramp-joint \
  --x-ramp-cathode-offset \
  --accum-nproc ${ACCUM_NPROC:-2} --nproc 24 --no-counts-maps

# spline fit + spline SCE ROOT (also writes spline GIFs)
python fit_doubleanode_2d_spline.py \
  --input $MAPS/voxel_data.npz --outdir $SPL \
  --nx-knots 5 --ny-knots 6 --nz-knots 9 --ridge-lambda 1 \
  --write-sce-root \
  --sce-root-name sbnd_sce_fwd_spline_doubleanode_2d_mc_full_v10jc.root

# direct (voxel) measurement SCE ROOT
python write_voxel_sce_root.py \
  --input $MAPS/voxel_data.npz \
  --sce-root-name sbnd_sce_fwd_voxel_doubleanode_2d_mc_full_v10jc.root

# raw voxel-map GIFs (dx/dy/dz + stack)
python make_voxel_map_gifs.py $MAPS

# truth comparison + 1D profiles (voxel + spline)
python compare_2d_vs_truth.py --input $MAPS/voxel_data.npz \
  --truth-root $DUAL --out compare_2d_vs_truth_mc_full_v10jc_dual
python plot_1d_profiles_vs_truth.py --input $MAPS/voxel_data.npz \
  --truth-root $DUAL --out profiles_1d_mc_full_v10jc_dual \
  --label "MC joint+offset"
python plot_1d_profiles_vs_truth.py --input $MAPS/voxel_data.npz \
  --use-spline $SPL/spline_fit.npz \
  --truth-root $DUAL --out profiles_1d_mc_full_v10jc_dual_spline \
  --label "MC joint+offset (spline)"

echo MC_FULL_V10JC_DONE
