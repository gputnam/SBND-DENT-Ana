#!/usr/bin/env python3
"""Export the v3 cathode distortion measurement to a ROOT file.

Reads the per-2x2cm2-pixel Crystal Ball cathode-tick fits
(cathode_fit_results_tpc{0,1}.csv of the v3 analysis epoch,
drift_df_full_v2) and writes TH2D maps per TPC:

  cathode_tick_tpc{0,1}       CB peak mu of the per-pixel dt distribution
                              [ticks]; the cathode drift-time length map
  cathode_tick_err_tpc{0,1}   CB mu fit uncertainty [ticks]
  nentries_tpc{0,1}           tracks entering the pixel fit
  dtsum                       per-pixel mu_TPC0 + mu_TPC1 [ticks]
                              (cathode-position-independent total span)

Axes: X = detector Z [0, 500] cm, Y = detector Y [-200, 200] cm, 2 cm bins.
Pixels with no fit (absent row or non-finite mu) carry the sentinel value
-1000. No quality cuts are applied -- the err/nentries maps let consumers
cut as needed.

Usage:
    python export_cathode_v3_root.py \
        --fits-dir /exp/sbnd/data/users/gputnam/efield-study/cathode_study_2cm_v3 \
        --out cathode_study_2cm_v3.root
"""

import argparse

import numpy as np
import pandas as pd
import uproot

SENTINEL = -1000.0
Y_RANGE = (-200.0, 200.0)
Z_RANGE = (0.0, 500.0)
PIX = 2.0


def grids_from_csv(path):
    df = pd.read_csv(path)
    ny = int(round((Y_RANGE[1] - Y_RANGE[0]) / PIX))
    nz = int(round((Z_RANGE[1] - Z_RANGE[0]) / PIX))
    mu = np.full((nz, ny), SENTINEL)
    err = np.full((nz, ny), SENTINEL)
    nent = np.zeros((nz, ny))
    iy = np.round((df.y_lo.to_numpy() - Y_RANGE[0]) / PIX).astype(int)
    iz = np.round((df.z_lo.to_numpy() - Z_RANGE[0]) / PIX).astype(int)
    ok = ((iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
          & np.isfinite(df.mu.to_numpy()))
    mu[iz[ok], iy[ok]] = df.mu.to_numpy()[ok]
    err[iz[ok], iy[ok]] = df.mu_err.to_numpy()[ok]
    nent[iz[ok], iy[ok]] = df.n_entries.to_numpy()[ok]
    return mu, err, nent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fits-dir",
                    default="/exp/sbnd/data/users/gputnam/efield-study/"
                            "cathode_study_2cm_v3")
    ap.add_argument("--out", default="cathode_study_2cm_v3.root")
    args = ap.parse_args()

    ny = int(round((Y_RANGE[1] - Y_RANGE[0]) / PIX))
    nz = int(round((Z_RANGE[1] - Z_RANGE[0]) / PIX))
    z_edges = np.linspace(*Z_RANGE, nz + 1)
    y_edges = np.linspace(*Y_RANGE, ny + 1)

    maps = {}
    for t in (0, 1):
        mu, err, nent = grids_from_csv(
            f"{args.fits_dir}/cathode_fit_results_tpc{t}.csv")
        maps[t] = (mu, err, nent)

    # per-pixel summed span: only where BOTH TPCs have a fit
    both = (maps[0][0] != SENTINEL) & (maps[1][0] != SENTINEL)
    dtsum = np.where(both, maps[0][0] + maps[1][0], SENTINEL)

    with uproot.recreate(args.out) as f:
        for t in (0, 1):
            mu, err, nent = maps[t]
            f[f"cathode_tick_tpc{t}"] = (mu, z_edges, y_edges)
            f[f"cathode_tick_err_tpc{t}"] = (err, z_edges, y_edges)
            f[f"nentries_tpc{t}"] = (nent, z_edges, y_edges)
            nfit = int((mu != SENTINEL).sum())
            print(f"TPC{t}: {nfit:,}/{ny*nz:,} pixels with fits "
                  f"({100.0*nfit/(ny*nz):.1f}%)")
        f["dtsum"] = (dtsum, z_edges, y_edges)
        print(f"dtsum: {int((dtsum != SENTINEL).sum()):,} matched pixels")
    print(f"wrote {args.out} (axes: X=Z [0,500] cm, Y=Y [-200,200] cm, "
          f"2 cm bins; sentinel {SENTINEL:g})")


if __name__ == "__main__":
    main()
