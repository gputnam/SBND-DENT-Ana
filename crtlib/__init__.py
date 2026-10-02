"""crtlib — shared library for the SBND field-distortion / cathode study.

Submodules (import what you need):

  crtlib.constants    SBND geometry / timing / drift constants, env-overridable
                      data locations (DATA_DIR, SCE_REFERENCE_ROOT), sentinels
  crtlib.io           file/parquet helpers, cut-config + wire-geometry loaders,
                      xroot prefix translation
  crtlib.ntuple       caloskim/crtana branch lists + per-cluster helpers
  crtlib.voxel_invert per-voxel scalar (rank-1) and rank-2 LSQ accumulators
  crtlib.spline       shared tensor-product cubic B-spline fit / eval / invert
  crtlib.plotting     LIBIMAGEQUANT GIF helpers + per-X-slice 3-pane figures

The deprecated CRT-alignment helper modules (align, jax_loss, wall, residuals,
rematch_run1) now live under experiments/crt_alignment/crtlib_legacy/.
"""
