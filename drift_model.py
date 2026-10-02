#!/usr/bin/env python3
"""SBND-adapted drift-velocity model v(E) and its inverse E(v).

The MicroBooNE space-charge calibration (arXiv:2008.09765, section 4.1 and
table 1) uses a fifth-order polynomial fit of the LAr drift velocity vs.
electric field,

    v(E) = p1 E + p2 E^2 + p3 E^3 + p4 E^4 + p5 E^5      (p0 forced to 0)

fit to ICARUS T600 measurements plus the MicroBooNE nominal-field drift
velocity, all at an operating temperature of 89 K. E is in kV/cm, v in mm/us.

SBND runs at 88.4 K, so the MicroBooNE curve does not pass through SBND's
own measured drift velocity at the nominal 500 V/cm field. Following the
prescription for this study, the MicroBooNE model is *adapted* to SBND by a
single multiplicative scale chosen so the scaled curve reproduces the SBND
measurement exactly at the SBND nominal field:

    FUDGE = v_SBND(500 V/cm) / v_microboone(500 V/cm)
    v(E)  = FUDGE * v_microboone(E)

with v_SBND(500 V/cm) = 0.15532 cm/us -- the 3-plane cathode-crossing
measurement (drift_df_full_v2), the value the v5 spatial-distortion map was
built with. This is the v(E) the MicroBooNE E-field-from-spatial-distortion
technique needs; build_efield_2d_voxels.py inverts it per voxel.

Run:
    python drift_model.py            # print the scaling + sanity numbers
"""

import argparse

import numpy as np
from scipy.optimize import brentq

# MicroBooNE arXiv:2008.09765 table 1 -- fifth-order polynomial v(E),
# E in kV/cm, v in mm/us, fit at 89 K. p0 forced to zero in the fit.
MB_COEFFS = (0.0, 5.534, -6.531, 3.208, 0.3897, -0.5562)   # p0 .. p5

# SBND nominal operating point.
SBND_E0_KVCM = 0.5          # nominal drift field, kV/cm  (= 500 V/cm)
SBND_E0_VCM  = 500.0        # nominal drift field, V/cm
SBND_V0_CMUS = 0.15532      # measured SBND drift velocity at E0, cm/us
                            # (drift_df_full_v2 3-plane cathode-crossing mean;
                            #  the VDRIFT_DEFAULT the v5 spatial map was built
                            #  with -- see plot_doubleanode_2d_voxels.py).

# Inversion bracket: +/-100% around the nominal field covers any physical
# voxel; v_of_E is monotonically increasing across it (checked in main()).
E_LO_KVCM = 0.1
E_HI_KVCM = 1.0


def v_microboone(efield_kVcm):
    """Unscaled MicroBooNE drift-velocity model, cm/us. E in kV/cm."""
    e = np.asarray(efield_kVcm, dtype=float)
    v_mm_us = np.polyval(MB_COEFFS[::-1], e)    # mm/us
    return v_mm_us / 10.0                       # cm/us


# Single multiplicative scale adapting the 89 K MicroBooNE curve to the
# SBND 88.4 K measurement at the SBND nominal field.
FUDGE = SBND_V0_CMUS / float(v_microboone(SBND_E0_KVCM))


def v_of_E(efield_kVcm):
    """SBND-adapted drift velocity, cm/us. v_of_E(0.5) == 0.15532 exactly."""
    return FUDGE * v_microboone(efield_kVcm)


def E_of_v(v_cmus, e_lo=E_LO_KVCM, e_hi=E_HI_KVCM):
    """Invert v_of_E: electric field (kV/cm) giving drift speed v (cm/us).

    Scalar. Raises ValueError if v is outside [v_of_E(e_lo), v_of_E(e_hi)].
    """
    return brentq(lambda E: v_of_E(E) - v_cmus, e_lo, e_hi,
                  xtol=1e-10, rtol=1e-12)


def E_of_v_array(v, e_lo=E_LO_KVCM, e_hi=E_HI_KVCM):
    """Vectorized E_of_v over an arbitrary-shape array, returning kV/cm.

    NaN is returned where v is non-finite, non-positive, or outside the
    invertible range [v_of_E(e_lo), v_of_E(e_hi)] -- a voxel driven there
    by an unphysically large distortion gradient is correctly dropped.
    """
    v = np.asarray(v, dtype=float)
    out = np.full(v.shape, np.nan, dtype=float)
    v_min = v_of_E(e_lo)
    v_max = v_of_E(e_hi)
    for i, vv in enumerate(v.ravel()):
        if np.isfinite(vv) and v_min < vv < v_max:
            out.flat[i] = brentq(lambda E: v_of_E(E) - vv, e_lo, e_hi,
                                 xtol=1e-10, rtol=1e-12)
    return out


def ve_log_slope(E0=SBND_E0_KVCM, h=1.0e-4):
    """Logarithmic slope alpha = (E0/v0) dv/dE of the scaled model at E0.

    The linearization constant relating a fractional drift-field change to a
    fractional drift-speed change; in the small-distortion limit the E-field
    inversion reduces to dEx/E0 ~ -gx/alpha (gx = d(Delta_x)/dx)."""
    v0 = v_of_E(E0)
    dvdE = (v_of_E(E0 + h) - v_of_E(E0 - h)) / (2.0 * h)
    return (E0 / v0) * dvdE


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    v_mb0 = float(v_microboone(SBND_E0_KVCM))
    print("MicroBooNE drift-velocity model (arXiv:2008.09765 table 1)")
    print(f"  coefficients p0..p5 = {MB_COEFFS}")
    print(f"  v_microboone(500 V/cm)      = {v_mb0:.6f} cm/us")
    print(f"  SBND measured v(500 V/cm)   = {SBND_V0_CMUS:.6f} cm/us")
    print(f"  FUDGE = v_SBND / v_uBooNE   = {FUDGE:.6f}")
    print()
    print("SBND-adapted model checks:")
    print(f"  v_of_E(0.5 kV/cm)           = {v_of_E(SBND_E0_KVCM):.6f} cm/us"
          f"   (target {SBND_V0_CMUS:.6f})")
    print(f"  E_of_v(0.15532 cm/us)       = {E_of_v(SBND_V0_CMUS):.6f} kV/cm"
          f"   (target {SBND_E0_KVCM:.6f})")
    print(f"  ve_log_slope alpha          = {ve_log_slope():.6f}")

    # round-trip and monotonicity
    max_rt = 0.0
    for E in np.arange(0.2, 0.95, 0.1):
        max_rt = max(max_rt, abs(E_of_v(v_of_E(E)) - E))
    print(f"  max round-trip |E_of_v(v_of_E(E)) - E|, E in [0.2,0.9] "
          f"= {max_rt:.2e} kV/cm")
    Es = np.linspace(E_LO_KVCM, E_HI_KVCM, 400)
    vs = v_of_E(Es)
    mono = bool(np.all(np.diff(vs) > 0.0))
    print(f"  v_of_E monotonically increasing on [{E_LO_KVCM}, {E_HI_KVCM}] "
          f"kV/cm: {mono}")
    print(f"  invertible v range          = [{v_of_E(E_LO_KVCM):.5f}, "
          f"{v_of_E(E_HI_KVCM):.5f}] cm/us")


if __name__ == "__main__":
    main()
