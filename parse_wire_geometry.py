"""Parse SBND-geometry.txt (LArSoft `DumpGeometry` output) into the
per-(tpc, plane) wire->position constants used by the 2D double-anode
field-distortion analysis.

The LArSoft geometry service resolves SBND's wrapped induction wires
into proper geo::Wire objects, so each `W:` index in the dump is exactly
the ntuple `hits<p>.h.wire` value -- no channel-map reconstruction
needed. For every plane the dump gives:

  * "direction of increasing wire number: (dx,dy,dz)"  -> w-hat
  * "wire pitch of P cm"
  * every wire's centre (x, y, z)

We define the per-plane "view coordinate"

  s_view(y, z) = what_y * y + what_z * z          (what = w-hat in y-z)

and fit s_view(centre) vs wire index W to get

  s_view = slope_cm * W + offset_cm               (slope_cm = +pitch)

so a hit's integer `wire` becomes a view coordinate, and 3 planes'
view coordinates at a point can be inverted for (y, z).

The dump's (C, T, P) map directly to ntuple (cryo, tpc, plane).

Generate SBND-geometry.txt with:
    lar -c dump_wire_geom.fcl          (inside the SL7 container, see
                                        the run recipe in this repo)

Usage:
    python parse_wire_geometry.py [SBND-geometry.txt]
        [--out wire_geometry_sbnd.json] [--verify-file FILE.root]
"""

import argparse
import json
import re

import numpy as np

VEC_RE = r"\(\s*([-0-9.eE]+)\s*,\s*([-0-9.eE]+)\s*,\s*([-0-9.eE]+)\s*\)"
PLANE_RE = re.compile(r"plane C:(\d+) T:(\d+) P:(\d+) at " + VEC_RE
                      + r" cm, theta:\s*([-0-9.eE]+) rad")
INFO_RE = re.compile(r"has (\d+) wires measuring (\w+) with a wire pitch "
                     r"of ([0-9.]+) cm")
WDIR_RE = re.compile(r"direction of increasing wire number:\s*" + VEC_RE)
WIRE_RE = re.compile(r"C:(\d+) T:(\d+) P:(\d+) W:(\d+) wire from")
CTR_RE = re.compile(r"center at\s*" + VEC_RE + r" cm")


def parse_dump(path):
    planes = {}          # (c,t,p) -> dict
    wires = {}           # (c,t,p) -> list of (W, cx, cy, cz)
    cur = None
    pending = None
    for raw in open(path):
        s = raw.strip()
        m = PLANE_RE.search(s)
        if m:
            c, t, p = int(m[1]), int(m[2]), int(m[3])
            cur = (c, t, p)
            planes[cur] = dict(theta_rad=float(m[7]),
                               center=(float(m[4]), float(m[5]),
                                       float(m[6])))
            wires[cur] = []
            continue
        if cur is not None:
            m = INFO_RE.search(s)
            if m:
                planes[cur]["nwires"] = int(m[1])
                planes[cur]["view"] = m[2]
                planes[cur]["pitch_cm"] = float(m[3])
                continue
            m = WDIR_RE.search(s)
            if m:
                planes[cur]["wdir"] = (float(m[1]), float(m[2]),
                                       float(m[3]))
                continue
        m = WIRE_RE.match(s)
        if m:
            pending = (int(m[1]), int(m[2]), int(m[3]), int(m[4]))
            continue
        if pending is not None:
            m = CTR_RE.match(s)
            if m:
                c, t, p, W = pending
                wires[(c, t, p)].append((W, float(m[1]), float(m[2]),
                                         float(m[3])))
                pending = None
    return planes, wires


def fit_plane(pl, wb):
    """wb: (N,4) array W, cx, cy, cz. Returns fit dict."""
    W = wb[:, 0]
    cy, cz = wb[:, 2], wb[:, 3]
    wd = pl["wdir"]
    what = np.array([wd[1], wd[2]])          # y-z part of w-hat
    what = what / np.linalg.norm(what)
    s = cy * what[0] + cz * what[1]          # view coord per wire (cm)
    A = np.stack([W, np.ones_like(W)], axis=1)
    (slope, offset), *_ = np.linalg.lstsq(A, s, rcond=None)
    rms = float((s - A @ np.array([slope, offset])).std())
    return dict(what_y=float(what[0]), what_z=float(what[1]),
                slope_cm=float(slope), offset_cm=float(offset),
                pitch_cm=float(pl["pitch_cm"]),
                theta_deg=float(np.degrees(pl["theta_rad"])),
                view=pl["view"], nwires=int(pl["nwires"]),
                wire_min=int(W.min()), wire_max=int(W.max()),
                fit_rms_cm=rms,
                plane_center=list(pl["center"]))


def verify(final, verify_file):
    """Cross-check: predicted view coord from ntuple wire vs from sp."""
    import uproot
    import awkward as ak
    br = [f"hits{p}.h.{f}" for p in (0, 1, 2)
          for f in ("wire", "tpc", "sp.y", "sp.z", "hasSP")]
    t = uproot.open(verify_file, timeout=120)["caloskim/TrackCaloSkim"]
    a = t.arrays(br, library="ak")
    print("\n=== ntuple cross-check (wire vs space-point) ===")
    for p in (0, 1, 2):
        wire = ak.flatten(a[f"hits{p}.h.wire"]).to_numpy().astype(float)
        tpc = ak.flatten(a[f"hits{p}.h.tpc"]).to_numpy()
        spy = ak.flatten(a[f"hits{p}.h.sp.y"]).to_numpy()
        spz = ak.flatten(a[f"hits{p}.h.sp.z"]).to_numpy()
        has = ak.flatten(a[f"hits{p}.h.hasSP"]).to_numpy().astype(bool)
        good = has & np.isfinite(spy) & np.isfinite(spz)
        for tv in (0, 1):
            g = final.get(f"tpc{tv}_plane{p}")
            if g is None:
                continue
            m = good & (tpc == tv)
            if m.sum() < 50:
                continue
            s_geom = g["slope_cm"] * wire[m] + g["offset_cm"]
            s_sp = g["what_y"] * spy[m] + g["what_z"] * spz[m]
            d = s_geom - s_sp
            print(f"  tpc{tv}_plane{p}: N={m.sum():6d}  "
                  f"<s_geom-s_sp>={d.mean():+.3f} cm  rms={d.std():.3f} cm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", nargs="?", default="SBND-geometry.txt")
    ap.add_argument("--out", default="wire_geometry_sbnd.json")
    ap.add_argument("--verify-file", default=None)
    args = ap.parse_args()

    planes, wires = parse_dump(args.dump)
    print(f"parsed {len(planes)} planes from {args.dump}")

    final = {}
    print(f"\n{'key':14s} {'view':5s} {'theta':>7s} {'pitch':>7s} "
          f"{'slope':>9s} {'offset':>10s} {'nwire':>6s} {'fit_rms':>9s}")
    for key in sorted(planes):
        c, t, p = key
        wb = np.array(wires[key], dtype=float)
        g = fit_plane(planes[key], wb)
        final[f"tpc{t}_plane{p}"] = dict(g, cryo=c, tpc=t, plane=p)
        print(f"tpc{t}_plane{p}     {g['view']:5s} {g['theta_deg']:7.2f} "
              f"{g['pitch_cm']:7.3f} {g['slope_cm']:+9.5f} "
              f"{g['offset_cm']:+10.4f} {g['nwires']:6d} "
              f"{g['fit_rms_cm']:9.2e}")

    if args.verify_file:
        verify(final, args.verify_file)

    with open(args.out, "w") as fh:
        json.dump({"comment": "SBND wire geometry from LArSoft DumpGeometry "
                              "(sbnd_v02_06). s_view = slope_cm*wire + "
                              "offset_cm = what_y*y + what_z*z.",
                   "source": args.dump, "planes": final}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
