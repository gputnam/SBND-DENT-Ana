#!/usr/bin/env python3
"""Compute the LAr drift velocity that LArSoft's
``DetectorPropertiesService::DriftVelocity()`` returns for a given electric
field and temperature.

The implementation is a verbatim port of
LArSoft/lardataalg ``DetectorPropertiesStandard::DriftVelocity`` (the default
``fUseIcarusMicrobooneDriftModel = false`` path), with the same constants and
the same smooth ICARUS + Walkowiak transition.

Defaults are the SBND simulation values from sbndcode v10_20_05
(``fcl/detectorproperties_sbnd.fcl``):

    Efield[0]   = 0.5 kV/cm   (drift region)
    Temperature = 88.4 K

Run:
    python print_drift_velocity.py                      # SBND defaults
    python print_drift_velocity.py --efield 0.5         # custom E
    python print_drift_velocity.py --efield 0.5 --temp 87.0
"""

import argparse
import math
import sys

# SBND simulation defaults (sbndcode v10_20_05 / detectorproperties_sbnd.fcl)
SBND_EFIELD_KV_CM = 0.5
SBND_TEMPERATURE_K = 88.4
DRIFT_VEL_FUDGE_FACTOR = 1.0  # LArSoft default; SBND does not override


def drift_velocity(efield, temperature, fudge=DRIFT_VEL_FUDGE_FACTOR):
    """Drift velocity in cm/us for a given E field (kV/cm) and temperature (K).

    Faithful port of LArSoft/lardataalg
    DetectorPropertiesStandard::DriftVelocity, default
    (fUseIcarusMicrobooneDriftModel = false) path.
    """
    if efield > 4.0:
        print(f"WARNING: E-field {efield} kV/cm is outside the parameterization "
              "range; result may not be accurate.", file=sys.stderr)
    if temperature < 87.0 or temperature > 94.0:
        print(f"WARNING: Temperature {temperature} K is outside the "
              "parameterization range (87-94 K); result may not be accurate.",
              file=sys.stderr)

    tshift = -87.203 + temperature
    xFit = 0.0938163 - 0.0052563 * tshift - 0.0001470 * tshift * tshift
    uFit = (5.18406 + 0.01448 * tshift
            - 0.003497 * tshift * tshift
            - 0.000516 * tshift * tshift * tshift)

    # ICARUS parameter set (T0 = 105.749 K)
    P1 = -0.04640
    P2 = 0.01712
    P3 = 1.88125
    P4 = 0.99408
    P5 = 0.01172
    P6 = 4.20214
    T0 = 105.749

    # Walkowiak parameter set (T0W = 90.371 K)
    P1W = -0.01481
    P2W = -0.0075
    P3W = 0.141
    P4W = 12.4
    P5W = 1.627
    P6W = 0.317
    T0W = 90.371

    def icarus_branch(E):
        return ((P1 * (temperature - T0) + 1.0)
                * (P3 * E * math.log(1.0 + P4 / E) + P5 * E**P6)
                + P2 * (temperature - T0))

    def walkowiak_branch(E):
        return ((P1W * (temperature - T0W) + 1.0)
                * (P3W * E * math.log(1.0 + P4W / E) + P5W * E**P6W)
                + P2W * (temperature - T0W))

    if efield < xFit:
        vd = efield * uFit
    elif efield < 0.619:
        vd = icarus_branch(efield)
    elif efield < 0.699:
        # 12.5-weighted linear blend across the transition window
        vd = (12.5 * (efield - 0.619) * walkowiak_branch(efield)
              + 12.5 * (0.699 - efield) * icarus_branch(efield))
    else:
        vd = walkowiak_branch(efield)

    # Walkowiak's fits are in mm/us; LArSoft divides by 10 -> cm/us
    return vd * fudge / 10.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--efield", type=float, default=SBND_EFIELD_KV_CM,
                    help=f"electric field in kV/cm (default: {SBND_EFIELD_KV_CM})")
    ap.add_argument("--temp", "--temperature", type=float,
                    default=SBND_TEMPERATURE_K, dest="temp",
                    help=f"LAr temperature in K (default: {SBND_TEMPERATURE_K})")
    ap.add_argument("--fudge", type=float, default=DRIFT_VEL_FUDGE_FACTOR,
                    help=f"DriftVelFudgeFactor (default: {DRIFT_VEL_FUDGE_FACTOR})")
    args = ap.parse_args()

    v = drift_velocity(args.efield, args.temp, args.fudge)
    is_sbnd_default = (args.efield == SBND_EFIELD_KV_CM
                       and args.temp == SBND_TEMPERATURE_K
                       and args.fudge == DRIFT_VEL_FUDGE_FACTOR)
    print(f"E-field      = {args.efield:.4f} kV/cm")
    print(f"Temperature  = {args.temp:.4f} K")
    if args.fudge != 1.0:
        print(f"Fudge factor = {args.fudge:.4f}")
    print(f"v_drift      = {v:.5f} cm/us")
    print(f"             = {v * 10:.4f} mm/us")
    print(f"drift time over 200 cm = {200/v:.2f} us")
    if is_sbnd_default:
        print(f"\n(SBND nominal sim values: sbndcode v10_20_05 "
              f"detectorproperties_sbnd.fcl)")


if __name__ == "__main__":
    main()
