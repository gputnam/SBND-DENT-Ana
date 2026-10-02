"""Write a zero-displacement 'truth' SCE ROOT for the no-SCE closure test.

The no-SCE MC sample has NO space-charge distortion, so the true forward
displacement is identically zero everywhere. This writes a TH3F file with
TrueFwd_Displacement_{X,Y,Z} = 0 on the standard 49x49x61 SCE grid (backward
+ E-field copied verbatim from the Gen1 reference, unused by the closure
plots) so compare_2d_vs_truth.py / plot_1d_profiles_vs_truth.py can be run
with --truth-root pointing here -> the 'truth' curves sit flat at 0 and the
residual = the measured map itself (the null-test quantity).
"""
import numpy as np
import build_sce_input as B

OUT = "zero_truth_voxelTH3.root"

x_edges = np.linspace(B.SCE_X_EDGE[0], B.SCE_X_EDGE[1], B.SCE_NX + 1)
y_edges = np.linspace(B.SCE_Y_EDGE[0], B.SCE_Y_EDGE[1], B.SCE_NY + 1)
z_edges = np.linspace(B.SCE_Z_EDGE[0], B.SCE_Z_EDGE[1], B.SCE_NZ + 1)

zeros = np.zeros((B.SCE_NX, B.SCE_NY, B.SCE_NZ), dtype=np.float32)

B.write_th3f_root(OUT, zeros, zeros, zeros, x_edges, y_edges, z_edges)
print(f"wrote {OUT} (TrueFwd_Displacement_* = 0 on the {B.SCE_NX}x{B.SCE_NY}x{B.SCE_NZ} SCE grid)")
