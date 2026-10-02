#!/usr/bin/env python3
"""Build a TextFileGen HEPEVT muon list of CORSIKA cosmic muons that cross
both SBND anodes.

What it does
------------
1. Reads the CORSIKA cosmic-ray shower SQLite databases on cvmfs
   (``/cvmfs/sbnd.osgstorage.org/.../CORSIKA/standard/``). Each ``.db`` has a
   ``particles(shower, pdg, px, py, pz, x, z, t, e)`` table -- momenta in
   GeV/c, ``e`` the TOTAL energy in GeV.
2. Keeps muons (pdg +/-13) above a minimum total energy.
3. Converts each muon's momentum from the CORSIKA frame to the SBND detector
   frame.  CORSIKAGen's convention is
       (p_x_drift, p_y_vert, p_z_beam) = (pz_db, py_db, -px_db)
   The db x,z positions are km-scale CORSIKA observation-plane coordinates and
   are DISCARDED -- only the direction and energy of each muon are physical
   input here.
4. Geometric selection: a muon's direction qualifies if a start point exists
   for which the straight line crosses BOTH anode planes (x = -200 and
   x = +200) inside the active footprint (y in [-200,200], z in [0,500]).
5. For each event, draws a random qualifying muon, samples a valid entry
   point on its first anode, backs the start point up just outside the TPC,
   and writes one HEPEVT event (one muon) to the output text file.

One random muon per event
-------------------------
LArSoft's TextFileGen reads events strictly sequentially -- it has no
random-pick feature.  The randomisation is therefore done here, at build
time: this script draws N muons (with replacement, fixed --seed) from the
qualifying pool and writes them in random order.  The companion fcl
(run_corsika_anode_muons_gen.fcl) then just processes events 1..N in order,
so each generated event gets a distinct random muon.

Usage
-----
    source setup.sh
    python make_corsika_anode_muons.py            # CMC mix, 10000 events, 2 GeV

See ``--help`` for all options.
"""

import argparse
import glob
import os
import sqlite3
import sys
import time

import numpy as np

# crtlib is a sibling package; make sure it imports regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crtlib.constants import ANODE_X_NOMINAL

# --- SBND geometry (detector frame: X = drift, Y = vertical, Z = beam) -------
XA0 = ANODE_X_NOMINAL[0]          # TPC0 anode, x = -200 cm
XA1 = ANODE_X_NOMINAL[1]          # TPC1 anode, x = +200 cm
DRIFT_SPAN = XA1 - XA0            # 400 cm anode-to-anode
YMIN, YMAX = -200.0, 200.0        # active vertical extent (cm); not in crtlib
ZMIN, ZMAX = 0.0, 500.0           # active beam extent (cm); not in crtlib

MUON_MASS_GEV = 0.105658          # PDG muon mass
DX_EPS = 1e-6                     # reject muons with |dx| below this (no anode-to-anode crossing)

DB_DIR_DEFAULT = ("/cvmfs/sbnd.osgstorage.org/pnfs/fnal.gov/usr/sbnd/"
                  "persistent/stash/CORSIKA/standard")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-dir", default=DB_DIR_DEFAULT,
                   help="directory of CORSIKA .db files (default: cvmfs path)")
    p.add_argument("--primary", choices=["proton", "all"], default="all",
                   help="'proton' = p_showers only; 'all' = full CMC mix "
                        "(p,He,N,Mg,Fe). default: all")
    p.add_argument("--max-files", type=int, default=0,
                   help="cap on number of db files read (0 = all). speed knob.")
    p.add_argument("-o", "--output", default="corsika_anode_muons.txt",
                   help="output HEPEVT text file (default: corsika_anode_muons.txt)")
    p.add_argument("-n", "--nevents", type=int, default=10000,
                   help="number of events (muons) to write (default: 10000)")
    p.add_argument("--min-energy", type=float, default=2.0,
                   help="minimum muon TOTAL energy in GeV (default: 2.0)")
    p.add_argument("--seed", type=int, default=0,
                   help="random seed (default: 0)")
    p.add_argument("--backup-margin", type=float, default=5.0,
                   help="distance (cm) to back the start point up along -d so "
                        "the muon starts just outside the TPC (default: 5.0)")
    return p.parse_args()


def select_db_files(db_dir, primary, max_files):
    """Return a sorted list of CORSIKA db file paths for the chosen species."""
    pattern = "p_showers_*.db" if primary == "proton" else "*_showers_*.db"
    files = sorted(glob.glob(os.path.join(db_dir, pattern)))
    if not files:
        sys.exit("ERROR: no db files matching %r in %s" % (pattern, db_dir))
    if max_files > 0:
        files = files[:max_files]
    return files


def read_muons(db_path):
    """Read all muon rows from one db. Returns (pdg, px_db, py_db, pz_db, e)
    as numpy arrays, or None if the file could not be read."""
    try:
        uri = "file:%s?mode=ro" % db_path
        con = sqlite3.connect(uri, uri=True)
        rows = con.execute(
            "SELECT pdg, px, py, pz, e FROM particles WHERE pdg IN (13, -13)"
        ).fetchall()
        con.close()
    except sqlite3.Error as exc:
        print("  WARNING: could not read %s (%s)" % (db_path, exc))
        return None
    if not rows:
        return (np.empty(0, np.int64),) + tuple(np.empty(0) for _ in range(4))
    a = np.asarray(rows, dtype=np.float64)
    return (a[:, 0].astype(np.int64), a[:, 1], a[:, 2], a[:, 3], a[:, 4])


def to_detector_frame(px_db, py_db, pz_db):
    """CORSIKA -> SBND detector frame (momentum only): see CORSIKAGen_module."""
    return pz_db.copy(), py_db.copy(), -px_db


def select_geometric(px, py, pz):
    """Vectorised anode-crossing selection.

    Returns (mask, info) where mask[i] is True if muon i's direction admits a
    start point crossing both anode planes inside the active footprint, and
    info holds per-muon arrays used to place the muon:
      dx, dy, dz   - unit direction in the detector frame
      x_entry      - x of the anode the muon enters (-200 or +200)
      y_lo, y_hi   - valid sampling interval for the entry y
      z_lo, z_hi   - valid sampling interval for the entry z
    """
    pmag = np.sqrt(px * px + py * py + pz * pz)
    good_p = pmag > 0
    dx = np.where(good_p, px / np.where(good_p, pmag, 1.0), 0.0)
    dy = np.where(good_p, py / np.where(good_p, pmag, 1.0), 0.0)
    dz = np.where(good_p, pz / np.where(good_p, pmag, 1.0), 0.0)

    ok_dx = np.abs(dx) > DX_EPS
    # entry anode and signed x-travel from entry to exit anode
    x_entry = np.where(dx > 0, XA0, XA1)
    sx = np.where(dx > 0, DRIFT_SPAN, -DRIFT_SPAN)

    dx_safe = np.where(ok_dx, dx, 1.0)          # avoid 0-division; masked out below
    dY = sx * dy / dx_safe                      # y change across the drift gap
    dZ = sx * dz / dx_safe                      # z change across the drift gap

    height = YMAX - YMIN                        # 400
    length = ZMAX - ZMIN                        # 500
    mask = ok_dx & (np.abs(dY) <= height) & (np.abs(dZ) <= length)

    y_lo = np.maximum(YMIN, YMIN - dY)
    y_hi = np.minimum(YMAX, YMAX - dY)
    z_lo = np.maximum(ZMIN, ZMIN - dZ)
    z_hi = np.minimum(ZMAX, ZMAX - dZ)

    info = dict(dx=dx, dy=dy, dz=dz, x_entry=x_entry.astype(np.float64),
                y_lo=y_lo, y_hi=y_hi, z_lo=z_lo, z_hi=z_hi)
    return mask, info


def build_pool(db_files, min_energy):
    """Scan all db files; return the qualifying-muon pool and counters."""
    cols = {k: [] for k in ("pdg", "px", "py", "pz", "e",
                            "dx", "dy", "dz", "x_entry",
                            "y_lo", "y_hi", "z_lo", "z_hi")}
    n_scanned = n_after_e = n_files = 0
    t0 = time.time()
    for i, path in enumerate(db_files, 1):
        out = read_muons(path)
        if out is None:
            continue
        n_files += 1
        pdg, px_db, py_db, pz_db, e = out
        n_scanned += pdg.size
        if pdg.size == 0:
            continue

        ecut = e >= min_energy
        n_after_e += int(ecut.sum())
        if not ecut.any():
            continue
        pdg, px_db, py_db, pz_db, e = (pdg[ecut], px_db[ecut], py_db[ecut],
                                       pz_db[ecut], e[ecut])

        px, py, pz = to_detector_frame(px_db, py_db, pz_db)
        mask, info = select_geometric(px, py, pz)
        if not mask.any():
            continue

        cols["pdg"].append(pdg[mask])
        cols["px"].append(px[mask])
        cols["py"].append(py[mask])
        cols["pz"].append(pz[mask])
        cols["e"].append(e[mask])
        for k in ("dx", "dy", "dz", "x_entry", "y_lo", "y_hi", "z_lo", "z_hi"):
            cols[k].append(info[k][mask])

        if i % 25 == 0 or i == len(db_files):
            print("  scanned %d/%d db files (%.0f s elapsed)"
                  % (i, len(db_files), time.time() - t0))

    pool = {k: (np.concatenate(v) if v else np.empty(0))
            for k, v in cols.items()}
    pool["pdg"] = pool["pdg"].astype(np.int64)
    counters = dict(n_files=n_files, n_scanned=n_scanned, n_after_e=n_after_e,
                    n_pool=pool["pdg"].size)
    return pool, counters


def sample_events(pool, n, rng):
    """Draw n events from the pool (with replacement). Returns a dict of
    per-event arrays: pdg, px, py, pz, e, and start position x, y, z."""
    size = pool["pdg"].size
    if size == 0:
        sys.exit("ERROR: no muons passed the geometric selection.")
    if n > size:
        print("  NOTE: requested %d events > pool of %d muons -- sampling "
              "with replacement (some muons repeat)." % (n, size))

    idx = rng.integers(0, size, n)
    y0 = rng.uniform(pool["y_lo"][idx], pool["y_hi"][idx])
    z0 = rng.uniform(pool["z_lo"][idx], pool["z_hi"][idx])

    return dict(
        pdg=pool["pdg"][idx], px=pool["px"][idx], py=pool["py"][idx],
        pz=pool["pz"][idx], e=pool["e"][idx],
        x=pool["x_entry"][idx], y=y0, z=z0,
        dx=pool["dx"][idx], dy=pool["dy"][idx], dz=pool["dz"][idx])


def write_hepevt(path, ev, backup_margin):
    """Write the HEPEVT text file: N events, one muon per event."""
    # back the start point up along -d so the muon begins just outside the TPC
    x = ev["x"] - backup_margin * ev["dx"]
    y = ev["y"] - backup_margin * ev["dy"]
    z = ev["z"] - backup_margin * ev["dz"]
    n = ev["pdg"].size
    with open(path, "w") as fh:
        for i in range(n):
            fh.write("%d 1\n" % (i + 1))
            # status pdg m1 m2 d1 d2  px py pz E mass  x y z t
            fh.write("1 %d 0 0 0 0 %.6f %.6f %.6f %.6f %.6f "
                     "%.4f %.4f %.4f 0.0\n"
                     % (ev["pdg"][i], ev["px"][i], ev["py"][i], ev["pz"][i],
                        ev["e"][i], MUON_MASS_GEV, x[i], y[i], z[i]))
    return n


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    db_files = select_db_files(args.db_dir, args.primary, args.max_files)
    print("Reading %d CORSIKA db file(s) from %s" % (len(db_files), args.db_dir))
    print("  primary=%s  min muon energy=%.2f GeV" % (args.primary, args.min_energy))

    pool, c = build_pool(db_files, args.min_energy)

    ev = sample_events(pool, args.nevents, rng)
    out_path = os.path.abspath(args.output)
    n_written = write_hepevt(out_path, ev, args.backup_margin)

    print("\n=== summary ===")
    print("  db files read .................. %d" % c["n_files"])
    print("  muon rows scanned .............. %d" % c["n_scanned"])
    print("  muons with E >= %.2f GeV ........ %d" % (args.min_energy, c["n_after_e"]))
    print("  muons passing anode geometry ... %d" % c["n_pool"])
    print("  events written ................. %d" % n_written)
    print("  output ......................... %s" % out_path)
    print("  random seed .................... %d" % args.seed)
    print("  backup margin .................. %.1f cm" % args.backup_margin)
    if c["n_after_e"]:
        print("  geometric pass fraction ........ %.1f%%"
              % (100.0 * c["n_pool"] / c["n_after_e"]))


if __name__ == "__main__":
    main()
