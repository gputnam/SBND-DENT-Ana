"""Assemble the complete SBND SpaceCharge (SCE) ROOT input from the v7
(iter2) double-anode maps.

Combines, into one Voxelized_TH3 file consumable by SpaceChargeServiceSBND,
all three distortion layers built from the v7 two-iteration maps:

  * TrueFwd_Displacement_{X,Y,Z}  -- copied from the iter2 spatial SCE
    ROOT (--forward-root): forward displacement = the v7 spline fit.
  * TrueBkwd_Displacement_{X,Y,Z} -- NEWLY DERIVED here, by numerically
    inverting the forward map (B(r) = -F(r + B(r)), fixed-point), instead
    of copying the 2020 Gen1 reference.
  * True_ElecField_{X,Y,Z,Mag}    -- copied from the iter2 E-field SCE
    ROOT (--efield-root): the v6 E-field layer derived from the iter2
    spatial spline.

Storage conventions (verified against the Gen1 reference SCEoffsets file
and SpaceChargeSBND.cxx):

  * GetPosOffsets flips TrueFwd_Displacement_X for x<0 (TPC0). The forward
    histogram is therefore stored with X = -Delta_x in TPC0, +Delta_x in
    TPC1 (already done upstream by fit_doubleanode_2d_spline.py).
  * The backward histogram is stored in the SAME convention as the
    forward histogram (same per-TPC X flip). The reference file does this
    too: un-flipping both and round-tripping forward o backward closes to
    ~0.005 cm, vs ~0.6 cm if the backward X is left un-flipped.

The inversion is therefore done in *physical* coordinates: un-flip the
forward histogram to F, invert to physical B, re-apply the same flip to B.

Usage:
    python build_sce_root.py \\
        --forward-root sce_input_doubleanode_2d/sbnd_sce_fwd_doubleanode_2d_iter2.root \\
        --efield-root  sce_input_doubleanode_2d/sbnd_sce_efield_doubleanode_2d_iter2.root \\
        --out          sce_input_doubleanode_2d/sbnd_sce_doubleanode_2d_iter2.root
"""
import argparse
import os

import numpy as np
import uproot
from scipy.interpolate import RegularGridInterpolator

import build_sce_input as B

COMPONENTS = ("X", "Y", "Z")


def _read_th3(path, names):
    """Read the named TH3 histograms from `path` as float64 arrays."""
    with uproot.open(path) as f:
        keys = {k.split(";")[0] for k in f.keys()}
        out = []
        for n in names:
            if n not in keys:
                raise KeyError(f"{n} missing from {path}")
            out.append(f[n].values().astype(np.float64))
    return out


def roundtrip_closure(fwd_phys, bkwd_phys, sx, sy, sz):
    """forward o backward closure on the SCE grid, well inside the volume.

    Returns the per-voxel |r'' - r| (worst axis), where r -> r+F(r) ->
    +B(r'). A correct inverse pair closes to ~mm."""
    def interp(arr):
        return RegularGridInterpolator((sx, sy, sz), arr, method="linear",
                                       bounds_error=False, fill_value=None)
    if_ = [interp(fwd_phys[..., c]) for c in range(3)]
    ib_ = [interp(bkwd_phys[..., c]) for c in range(3)]
    lo = np.array([sx[0], sy[0], sz[0]])
    hi = np.array([sx[-1], sy[-1], sz[-1]])

    def step(pts, ii):
        clamped = np.clip(pts, lo, hi)
        return pts + np.stack([ii[c](clamped) for c in range(3)], axis=1)

    x, y, z = np.meshgrid(sx, sy, sz, indexing="ij")
    r = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    rt = step(step(r, if_), ib_)
    d = np.abs(rt - r).max(axis=1)
    inside = ((np.abs(r[:, 0]) > 20) & (np.abs(r[:, 0]) < 190)
              & (np.abs(r[:, 1]) < 190) & (r[:, 2] > 20) & (r[:, 2] < 480))
    return d[inside]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--forward-root",
        default="sce_input_doubleanode_2d/sbnd_sce_fwd_doubleanode_2d_iter2.root",
        help="SCE ROOT supplying TrueFwd_Displacement_*")
    ap.add_argument(
        "--efield-root",
        default="sce_input_doubleanode_2d/sbnd_sce_efield_doubleanode_2d_iter2.root",
        help="SCE ROOT supplying True_ElecField_*")
    ap.add_argument(
        "--out",
        default="sce_input_doubleanode_2d/sbnd_sce_doubleanode_2d_iter2.root")
    ap.add_argument("--bkwd-iterations", type=int, default=10,
                    help="fixed-point iterations for the backward inversion")
    ap.add_argument("--reference-root", default=B.REFERENCE_SCE_ROOT_DEFAULT,
                    help="Gen1 reference SCE ROOT (binning sanity check)")
    args = ap.parse_args()

    sx, sy, sz = B.sce_bin_centers()
    sign = np.where(sx < 0.0, -1.0, 1.0)[:, None, None]   # loader's corr

    # ---- forward: copy histograms, un-flip to physical F -------------
    fwd_hist = np.stack(_read_th3(
        args.forward_root,
        [f"TrueFwd_Displacement_{c}" for c in COMPONENTS]), axis=-1)
    print(f"forward displacement read from {args.forward_root}", flush=True)
    fwd_phys = fwd_hist.copy()
    fwd_phys[..., 0] *= sign[..., 0]               # X: undo per-TPC flip
    for c, comp in enumerate(COMPONENTS):
        g = fwd_hist[..., c]
        print(f"  TrueFwd_{comp}: histogram range "
              f"[{g.min():+.3f}, {g.max():+.3f}] cm", flush=True)

    # ---- backward: numerically invert the forward map ---------------
    print(f"\ninverting forward map ({args.bkwd_iterations} iterations) ...",
          flush=True)
    bkwd_phys = B.invert_displacement_grid(
        fwd_phys, sx, sy, sz, n_iter=args.bkwd_iterations)
    bkwd_hist = bkwd_phys.copy()
    bkwd_hist[..., 0] *= sign[..., 0]              # X: re-apply per-TPC flip
    for c, comp in enumerate(COMPONENTS):
        g = bkwd_hist[..., c]
        print(f"  TrueBkwd_{comp}: histogram range "
              f"[{g.min():+.3f}, {g.max():+.3f}] cm", flush=True)

    # ---- round-trip closure + reference-convention cross-check ------
    d = roundtrip_closure(fwd_phys, bkwd_phys, sx, sy, sz)
    print(f"\nforward o backward closure (well inside volume): "
          f"median {np.median(d):.4f}  RMS {np.sqrt((d ** 2).mean()):.4f}  "
          f"p95 {np.percentile(d, 95):.4f}  max {d.max():.4f} cm", flush=True)
    for tag, m in (("TPC0", sx < 0), ("TPC1", sx > 0)):
        a = fwd_hist[m, :, :, 0].ravel()
        b = bkwd_hist[m, :, :, 0].ravel()
        print(f"  {tag}  corr(TrueFwd_X, TrueBkwd_X) = "
              f"{np.corrcoef(a, b)[0, 1]:+.3f}  (reference: -0.998)",
              flush=True)

    # ---- E-field: copy histograms -----------------------------------
    efx, efy, efz, efmag = _read_th3(
        args.efield_root,
        [f"True_ElecField_{c}" for c in COMPONENTS] + ["True_ElecField_Mag"])
    print(f"\nE-field read from {args.efield_root}", flush=True)
    for comp, g in zip(("X", "Y", "Z", "Mag"), (efx, efy, efz, efmag)):
        print(f"  True_ElecField_{comp}: range "
              f"[{g.min():+.4f}, {g.max():+.4f}]", flush=True)

    # ---- write the combined SCE ROOT --------------------------------
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    xe = np.linspace(*B.SCE_X_EDGE, B.SCE_NX + 1)
    ye = np.linspace(*B.SCE_Y_EDGE, B.SCE_NY + 1)
    ze = np.linspace(*B.SCE_Z_EDGE, B.SCE_NZ + 1)
    print(flush=True)
    B.write_th3f_root(
        args.out, fwd_hist[..., 0], fwd_hist[..., 1], fwd_hist[..., 2],
        xe, ye, ze, reference_root=args.reference_root,
        backward=(bkwd_hist[..., 0], bkwd_hist[..., 1], bkwd_hist[..., 2]),
        efield=(efx, efy, efz), efield_mag=efmag)


if __name__ == "__main__":
    main()
