"""Solve a dumped per-voxel accumulator with the OLD (production isotropic,
ridge~0) and NEW (anisotropic x-ridge) inversion, writing two
voxel_data.npz-compatible maps so the existing comparison/profile scripts can
consume them. Used to make the 4-version (v_drift x regularization) study.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import crtlib.voxel_invert as VI
import explore_inversion as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accum", required=True)
    ap.add_argument("--out-old", required=True)
    ap.add_argument("--out-new", required=True)
    ap.add_argument("--x-ridge", type=float, default=0.1)
    ap.add_argument("--min-counts", type=int, default=30)
    ap.add_argument("--cond-max", type=float, default=1000.0)
    args = ap.parse_args()

    d = np.load(args.accum)
    acc = {"nnT": d["nnT"].astype(np.float64),
           "sum_nm": d["sum_nm"].astype(np.float64),
           "sum_m2": d["sum_m2"].astype(np.float64),
           "n": d["n"].astype(np.int64)}

    # OLD: exactly the production inverter (isotropic ridge 1e-6).
    res = VI.invert_scalar_accumulator(acc, min_n=args.min_counts,
                                       cond_max=args.cond_max)
    delta_old = res["delta_mu"]
    cov, cond, cnt = res["delta_cov"], res["cond"], res["n"]
    sh = cnt.shape

    # NEW: anisotropic x-ridge R = diag(lamx*n, 0, 0).
    M, v, n, _, _, _, _ = E.build_MV(args.accum)
    ex = np.zeros((M.shape[0], 3, 3)); ex[:, 0, 0] = n
    delta_new = E.solve(M, v, args.x_ridge * ex).reshape(*sh, 3)
    valid = np.isfinite(delta_old[..., 0])          # same kept voxel set
    delta_new = np.where(valid[..., None], delta_new,
                         np.nan).astype(np.float32)

    common = dict(delta_cov_all=cov, cond_all=cond, count_all=cnt,
                  x_edges=d["x_edges"], y_edges=d["y_edges"],
                  z_edges=d["z_edges"])
    np.savez_compressed(args.out_old, delta_mu_all=delta_old, **common)
    np.savez_compressed(args.out_new, delta_mu_all=delta_new, **common)
    print(f"wrote {args.out_old}  and  {args.out_new}  "
          f"(x-ridge lamx={args.x_ridge})", flush=True)


if __name__ == "__main__":
    main()
