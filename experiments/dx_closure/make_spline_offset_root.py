"""Write a ROOT file of the per-voxel offset (spline - measured) for the
2D double-anode *spatial* field-distortion map (v7 / iter2).

For every voxel of the 20 cm map (20 x 20 x 25), the offset is

    offset = spline(voxel center) - measured_delta

where `measured_delta` is the LSQ-inverted voxel value (`delta_mu_all`) and
`spline` is the fitted cubic B-spline evaluated at that voxel center
(`spline_mu`, written by fit_doubleanode_2d_spline.py). Voxels with a
missing measurement -- either never inverted (NaN delta_mu) or dropped by
the spline's quality cuts (cond > 100 or worst-axis resolution > 2 cm),
i.e. the complement of the spline `keep_mask` -- are filled with the
sentinel value -1000.

Three TH3F histograms (voxel-grid binning, X-fastest ROOT layout) are
written:
    SplineMinusMeasured_X / _Y / _Z   [cm]

Usage:
    python make_spline_offset_root.py
    python make_spline_offset_root.py --maps DIR --spline DIR --out FILE
"""

import argparse
import os

import numpy as np
import uproot

import build_sce_input as B

MAPS_DEFAULT = "doubleanode_2d_maps_20cm_iter2"
SPLINE_DEFAULT = "doubleanode_2d_spline_20cm_iter2"
OUT_DEFAULT = "spline_offset_doubleanode_2d_iter2.root"
SENTINEL = -1000.0
COMPONENTS = ("X", "Y", "Z")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default=MAPS_DEFAULT,
                    help="dir with voxel_data.npz (the measured map)")
    ap.add_argument("--spline", default=SPLINE_DEFAULT,
                    help="dir with spline_fit.npz (spline_mu + keep_mask)")
    ap.add_argument("--out", default=OUT_DEFAULT,
                    help="output ROOT path")
    ap.add_argument("--sentinel", type=float, default=SENTINEL)
    args = ap.parse_args()

    mp = np.load(os.path.join(args.maps, "voxel_data.npz"))
    sp = np.load(os.path.join(args.spline, "spline_fit.npz"))

    measured = mp["delta_mu_all"].astype(np.float64)   # (nx,ny,nz,3)
    spline = sp["spline_mu"].astype(np.float64)        # (nx,ny,nz,3)
    keep = sp["keep_mask"].astype(bool)                # (nx,ny,nz)
    x_edges = mp["x_edges"]; y_edges = mp["y_edges"]; z_edges = mp["z_edges"]

    if measured.shape != spline.shape:
        raise RuntimeError(f"shape mismatch: measured {measured.shape} "
                           f"vs spline {spline.shape}")

    offset = spline - measured                         # (nx,ny,nz,3)
    # sentinel where there is no usable measurement (uninverted OR cut-failed)
    fill = ~keep
    offset[fill, :] = args.sentinel
    # belt-and-suspenders: any residual non-finite (e.g. spline NaN) -> sentinel
    offset[~np.isfinite(offset)] = args.sentinel

    n_real = int(keep.sum())
    n_sent = int(fill.sum())
    print(f"voxels: {keep.size}  measured(kept): {n_real}  "
          f"sentinel: {n_sent}", flush=True)
    for ci, comp in enumerate(COMPONENTS):
        vals = offset[keep, ci]
        print(f"  d{comp.lower()} offset (kept voxels): "
              f"range [{vals.min():+.3f}, {vals.max():+.3f}] cm  "
              f"mean {vals.mean():+.4f}  RMS {np.sqrt((vals**2).mean()):.4f}",
              flush=True)

    out_arrays = {
        f"SplineMinusMeasured_{c}": offset[..., ci].astype(np.float32)
        for ci, c in enumerate(COMPONENTS)
    }
    with uproot.recreate(args.out) as f:
        for name, arr in out_arrays.items():
            f[name] = B._build_th3f(arr, x_edges, y_edges, z_edges, name)
    print(f"\nwrote {args.out} "
          f"({', '.join(out_arrays)}; sentinel = {args.sentinel:g})",
          flush=True)


if __name__ == "__main__":
    main()
