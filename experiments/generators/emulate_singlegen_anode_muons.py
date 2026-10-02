#!/usr/bin/env python3
"""Emulate the SingleGen particle gun configured in singlegen_anode_muons.fcl
and measure empirically how many of its muons project to cross BOTH SBND
anodes.

This re-implements LArSoft SingleGen's exact sampling (larsim
SingleGen_module.cc) in numpy:

  * momentum   p   : uniform  -> P0 + SigmaP*(2u-1)        ; Gaussian -> N(P0,SigmaP)
  * position x,y,z : uniform  -> C0 + Sigma*(2u-1)         ; Gaussian -> N(C0,Sigma)
  * angle  ThetaXZ : uniform  -> Theta0XZ + SigmaThetaXZ*(2u-1)
  * angle  ThetaYZ : uniform  -> uniform in SIN(theta) between the clamped
                                 [Theta0YZ +/- SigmaThetaYZ] bounds
  * direction      : dx = cos(thyz) sin(thxz)
                     dy = sin(thyz)
                     dz = cos(thyz) cos(thxz)

ParticleSelectionMode "all" with an N-entry PDG list produces N muons per
event. SingleGen treats every numeric parameter as a per-particle VECTOR
(PadOutVectors pads length-1 vectors to the PDG length), so a single gun can
fire muons in BOTH drift directions by giving X0 / Theta0XZ full-length
vectors. This script honours that: it builds per-particle parameter arrays,
groups the emulated muons by event, and reports the per-muon crossing rate,
the +x / -x split, and the per-event distribution.

Each muon's straight line is projected to the anode planes x = -200 and
x = +200; it "impinges both anodes" if both crossings land in the active
footprint (y in [-200,200], z in [0,500]).

Usage:
    source setup.sh
    python emulate_singlegen_anode_muons.py
"""

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crtlib.constants import ANODE_X_NOMINAL

XA0, XA1 = ANODE_X_NOMINAL[0], ANODE_X_NOMINAL[1]   # -200, +200
YMIN, YMAX = -200.0, 200.0
ZMIN, ZMAX = 0.0, 500.0
_DIST = {"uniform": 0, "gaussian": 1, "histograms": 2}   # SingleGen kUNIF/kGAUS/kHIST


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fcl", default=os.path.join(here, "singlegen_anode_muons.fcl"),
                   help="SingleGen fcl to emulate")
    p.add_argument("--corsika", default=os.path.join(here, "corsika_anode_muons.txt"),
                   help="CORSIKA reference list for the coverage comparison")
    p.add_argument("--nevents", type=int, default=20000,
                   help="events to sample for the crossing statistics")
    p.add_argument("--ntable", type=int, default=6,
                   help="events to print in the detailed trajectory table")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------
# fcl parsing -- pull the SingleGen 'physics.producers.generator.*' settings
# --------------------------------------------------------------------------
def parse_fcl(path):
    cfg = {}
    pat = re.compile(r"^\s*physics\.producers\.generator\.(\w+)\s*:\s*(.+?)\s*$")
    for line in open(path):
        line = line.split("#")[0]
        m = pat.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if val.startswith("["):
            toks = val.strip("[]").replace(",", " ").split()
            cfg[key] = [_coerce(t) for t in toks]
        else:
            cfg[key] = _coerce(val)
    return cfg


def _coerce(tok):
    tok = tok.strip().strip('"')
    low = tok.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return float(tok) if ("." in tok or "e" in low) else int(tok)
    except ValueError:
        return tok


def _vec(cfg, key, default, mpe):
    """Per-particle array of length mpe (SingleGen PadOutVectors semantics):
    a length-1 vector is padded out, a length-mpe vector is used as-is."""
    v = cfg.get(key, default)
    v = v if isinstance(v, list) else [v]
    arr = np.asarray(v, dtype=float)
    if arr.size == 1:
        return np.repeat(arr, mpe)
    if arr.size != mpe:
        raise ValueError("parameter %s has length %d, expected 1 or %d"
                         % (key, arr.size, mpe))
    return arr


def _dist(cfg, key):
    v = cfg.get(key, "uniform")
    return _DIST[v.lower()] if isinstance(v, str) else int(v)


def muons_per_event(cfg):
    """Number of muons SingleGen makes per event for this configuration."""
    pdg = cfg.get("PDG", [13])
    pdg = pdg if isinstance(pdg, list) else [pdg]
    mode = str(cfg.get("ParticleSelectionMode", "all")).lower()
    return (len(pdg) if mode == "all" else 1), pdg, mode


# --------------------------------------------------------------------------
# SingleGen emulation
# --------------------------------------------------------------------------
def sample_singlegen(cfg, mpe, n_ev, rng):
    """Return per-muon kinematics arrays (p, x, y, z, dx, dy, dz), length
    n_ev*mpe, reproducing SingleGen's sampling. Muon j of every event uses
    per-particle vector index j (X0[j], Theta0XZ[j], ...)."""
    n = n_ev * mpe
    u = lambda: rng.random(n)
    box = lambda c, s: c + s * (2.0 * u() - 1.0)
    # per-particle vector -> per-muon array (index j repeats every event)
    per_muon = lambda key, d=0.0: np.tile(_vec(cfg, key, d, mpe), n_ev)

    P0, SigP = per_muon("P0", 6.0), per_muon("SigmaP", 0.0)
    p = rng.normal(P0, SigP) if _dist(cfg, "PDist") == 1 else box(P0, SigP)

    X0, Y0, Z0 = per_muon("X0"), per_muon("Y0"), per_muon("Z0")
    SX, SY, SZ = per_muon("SigmaX"), per_muon("SigmaY"), per_muon("SigmaZ")
    if _dist(cfg, "PosDist") == 1:
        x, y, z = rng.normal(X0, SX), rng.normal(Y0, SY), rng.normal(Z0, SZ)
    else:
        x, y, z = box(X0, SX), box(Y0, SY), box(Z0, SZ)

    TXZ, STXZ = per_muon("Theta0XZ"), per_muon("SigmaThetaXZ")
    TYZ, STYZ = per_muon("Theta0YZ"), per_muon("SigmaThetaYZ")
    if _dist(cfg, "AngleDist") == 1:                       # Gaussian
        thxz = rng.normal(TXZ, STXZ)
        thyz = rng.normal(TYZ, STYZ)
    else:                                                  # uniform (SingleGen rule)
        thxz = box(TXZ, STXZ)
        tyz0 = np.arcsin(np.sin(np.radians(TYZ)))
        lo = np.maximum(tyz0 - np.radians(np.abs(STYZ)), -np.pi / 2)
        hi = np.minimum(tyz0 + np.radians(np.abs(STYZ)),  np.pi / 2)
        sin_thyz = np.sin(lo) + u() * (np.sin(hi) - np.sin(lo))
        thyz = np.degrees(np.arcsin(sin_thyz))

    txr, tyr = np.radians(thxz), np.radians(thyz)
    dx = np.cos(tyr) * np.sin(txr)
    dy = np.sin(tyr)
    dz = np.cos(tyr) * np.cos(txr)
    return p, x, y, z, dx, dy, dz


def project(x, y, z, dx, dy, dz, xa):
    """(y,z) where the straight line through (x,y,z)+t*d hits plane x=xa."""
    t = (xa - x) / dx
    return y + dy * t, z + dz * t


def in_footprint(yy, zz):
    return (yy >= YMIN) & (yy <= YMAX) & (zz >= ZMIN) & (zz <= ZMAX)


# --------------------------------------------------------------------------
# CORSIKA reference
# --------------------------------------------------------------------------
def read_corsika(path):
    """Return SingleGen-convention arrays for ALL double-anode crossers.
    ThetaXZ is folded to the +x sense (atan2(|px|,pz)) so the +x and -x
    gun muons can be compared against the whole CORSIKA sample at once."""
    if not os.path.exists(path):
        return None
    rows = []
    lines = [l for l in open(path).read().split("\n") if l]
    for i in range(0, len(lines), 2):
        f = lines[i + 1].split()
        rows.append([float(v) for v in f[6:9]] + [float(f[9])]
                    + [float(v) for v in f[11:14]])
    a = np.asarray(rows)                       # px,py,pz,E,x,y,z
    px, py, pz, E, x, y, z = (a[:, k] for k in range(7))
    p = np.sqrt(px**2 + py**2 + pz**2)
    thxz = np.degrees(np.arctan2(np.abs(px), pz))     # folded to +x sense
    thyz = np.degrees(np.arcsin(py / p))
    return dict(p=p, thxz=thxz, thyz=thyz, y=y, z=z)


def rng_str(v):
    q = np.percentile(v, [5, 95])
    return "[%8.1f, %8.1f]  median %8.1f" % (q[0], q[1], np.median(v))


# --------------------------------------------------------------------------
def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if not os.path.exists(args.fcl):
        sys.exit("ERROR: fcl not found: %s" % args.fcl)
    cfg = parse_fcl(args.fcl)
    mpe, pdg_list, mode = muons_per_event(cfg)
    print("Emulating SingleGen config from: %s" % args.fcl)
    print("  ParticleSelectionMode=%s  PDG=%s  ->  %d muon(s)/event\n"
          % (mode, pdg_list, mpe))

    # ---- sample n_events * mpe muons ------------------------------------
    n_ev = args.nevents
    n_mu = n_ev * mpe
    p, x, y, z, dx, dy, dz = sample_singlegen(cfg, mpe, n_ev, rng)
    if mode == "all":
        pdg = np.tile(pdg_list, n_ev)
    else:
        pdg = rng.choice(pdg_list, n_mu)

    y0, z0 = project(x, y, z, dx, dy, dz, XA0)
    y1, z1 = project(x, y, z, dx, dy, dz, XA1)
    cross = in_footprint(y0, z0) & in_footprint(y1, z1)
    pos = dx > 0
    neg = ~pos

    # ---- per-muon crossing rate + direction split -----------------------
    print("=== anode-crossing verification (%d events x %d muons = %d muons) ==="
          % (n_ev, mpe, n_mu))
    print("  per-muon crossing rate : %.2f%%   (all directions)"
          % (100.0 * cross.mean()))
    print("  direction split        : +x %.2f%%   /   -x %.2f%%"
          % (100.0 * pos.mean(), 100.0 * neg.mean()))
    print("    +x  TPC0->TPC1 : %9d muons   crossing rate %.2f%%"
          % (int(pos.sum()), 100.0 * cross[pos].mean()))
    print("    -x  TPC1->TPC0 : %9d muons   crossing rate %.2f%%"
          % (int(neg.sum()), 100.0 * cross[neg].mean()))

    # ---- per-event distribution -----------------------------------------
    per_ev = cross.reshape(n_ev, mpe).sum(axis=1)
    print("  mean crossers / event  : %.3f" % per_ev.mean())
    print("  events with >= 1 cross : %.2f%%" % (100.0 * (per_ev >= 1).mean()))
    print("  crossers/event   count   fraction")
    for k in range(mpe + 1):
        nk = int((per_ev == k).sum())
        print("    %d              %8d   %6.2f%%" % (k, nk, 100.0 * nk / n_ev))

    # ---- detailed per-event trajectory table ----------------------------
    print("\n=== %d sampled events (%d muons each) ===" % (args.ntable, mpe))
    for e in range(min(args.ntable, n_ev)):
        sl = slice(e * mpe, (e + 1) * mpe)
        thxz_e = np.degrees(np.arctan2(dx[sl], dz[sl]))
        thyz_e = np.degrees(np.arcsin(np.clip(dy[sl], -1.0, 1.0)))
        nc = int(cross[sl].sum())
        print("  event %d:  %d/%d muons impinge both anodes" % (e + 1, nc, mpe))
        for j in range(mpe):
            g = e * mpe + j
            res = "PASS" if cross[g] else "fail"
            d = "+x" if dx[g] > 0 else "-x"
            print("    pdg %4d  %s  p %6.2f  thXZ %7.1f  thYZ %6.1f | "
                  "start(%7.1f,%7.1f,%7.1f) -> anode0(%6.1f,%6.1f) "
                  "anode1(%6.1f,%6.1f)  %s"
                  % (pdg[g], d, p[g], thxz_e[j], thyz_e[j], x[g], y[g], z[g],
                     y0[g], z0[g], y1[g], z1[g], res))

    # ---- coverage comparison vs CORSIKA ---------------------------------
    ref = read_corsika(args.corsika)
    print("\n=== coverage: SingleGen gun  vs  CORSIKA double-anode crossers ===")
    print("  (ThetaXZ* folded to the +x sense: atan2(|dx|,dz); ranges are [p5, p95])")
    thxz_s = np.degrees(np.arctan2(np.abs(dx), dz))
    thyz_s = np.degrees(np.arcsin(np.clip(dy, -1.0, 1.0)))
    rows = [("ThetaXZ* [deg]", thxz_s, "thxz"),
            ("ThetaYZ [deg]", thyz_s, "thyz"),
            ("momentum [GeV]", p, "p"),
            ("start y [cm]", y, "y"),
            ("start z [cm]", z, "z")]
    for label, gun_v, refkey in rows:
        line = "  %-15s gun %s" % (label, rng_str(gun_v))
        if ref is not None:
            line += "   corsika %s" % rng_str(ref[refkey])
        print(line)


if __name__ == "__main__":
    main()
