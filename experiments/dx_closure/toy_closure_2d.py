#!/usr/bin/env python3
"""Toy closure test of the 2D double-anode pipeline with a KNOWN field.

The MC diagnostics leave a puzzle: per-hit times and the as-applied spatial
field are clean in truth (measure_truth_dx_mc.py), yet the chord method
measures a dx dome ~2-3x the as-applied one (diagnose_dx_offsets.py), in all
three planes identically. This toy discriminates "pipeline math/geometry
artifact" from "real per-hit content": it generates synthetic hits from an
exactly known displacement field on anode-to-anode straight tracks, writes
them as a doubleanode_2d_df-format parquet, and the REAL pipeline
(plot_doubleanode_2d_voxels.shard_hits + inversion) is then run on it
unchanged. If the inverted map reproduces the input field (slope ~1, right
sign), the pipeline is exact and the data effect is upstream (hit
reconstruction); if it shows the amplified dome, the pipeline itself is at
fault and the toy gives a controlled testbed to find it.

Hit model (mirrors the truth decomposition findings):
  deposit at true r on the track; displaced by the field Delta(r); drifts at
  the SAME velocity used in reco (no v mismatch);
    s_meas = what_y (y+Dy) + what_z (z+Dz) + noise
    x_meas = x_true + Dx,    time = t0 + |x_meas - anode|/(0.5us * v)
    tpc    = side of x_meas; wire quantized to the real 0.3 cm pitch.

The field is the dual truth map read as PHYSICAL displacement
(dx_phys = -raw stored X, dy/dz = raw; matching the as-applied truehits
profiles), or an analytic dome with --field analytic.

Usage:
    python toy_closure_2d.py generate --out toy_closure --ntracks 30000
    python plot_doubleanode_2d_voxels.py --input toy_closure/parquet \
        --outdir toy_closure/maps --vdrift 0.15627 --iterations 1 \
        --accum-nproc 4 --nproc 16 --no-counts-maps
    python toy_closure_2d.py compare --out toy_closure \
        --maps toy_closure/maps/voxel_data.npz
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import build_sce_input as B
import plot_doubleanode_2d_voxels as P2

TICK_US = 0.5
T0 = 418.5
V = 0.15627
COMPONENTS = ("dx", "dy", "dz")


# ----------------------------------------------------------------------------
# truth fields
# ----------------------------------------------------------------------------
class DualMapField:
    """Physical displacement field from the dual truth map:
    dx_phys = -(raw stored X), dy/dz = raw stored (matches the as-applied
    truehits profiles measured by measure_truth_dx_mc.py)."""

    def __init__(self, root):
        import uproot
        sx, sy, sz = B.sce_bin_centers()
        self.interp = {}
        with uproot.open(root) as f:
            for c, comp in enumerate("XYZ"):
                for side, tag in ((0, "E"), (1, "W")):
                    g = f[f"TrueFwd_Displacement_{comp}_{tag}"]\
                        .values().astype(np.float64)
                    if c == 0:
                        g = -g
                    self.interp[(c, side)] = RegularGridInterpolator(
                        (sx, sy, sz), g, method="linear",
                        bounds_error=False, fill_value=0.0)

    def __call__(self, x, y, z):
        pts = np.stack([x, y, z], axis=1)
        side = (x >= 0).astype(np.int64)
        out = np.zeros((len(x), 3))
        for c in range(3):
            for s in (0, 1):
                m = side == s
                if m.any():
                    out[m, c] = self.interp[(c, s)](pts[m])
        return out


class AnalyticField:
    """Separable dome: zero at anodes and cathode for dx, toward-anode sign;
    dy/dz domes peaked mid-drift."""

    def __init__(self, ax=0.6, ay=1.5, az=1.0):
        self.ax, self.ay, self.az = ax, ay, az

    def __call__(self, x, y, z):
        u = np.clip(np.abs(x) / 200.0, 0, 1)        # 0 cathode, 1 anode
        dome = np.sin(np.pi * u)
        w = 1.0 - 0.4 * ((y / 200.0) ** 2 + ((z - 250.0) / 250.0) ** 2)
        out = np.zeros((len(x), 3))
        out[:, 0] = -np.sign(x) * self.ax * dome * w    # toward the anode
        out[:, 1] = self.ay * dome * w * (1 - 0.5 * np.abs(y) / 200.0)
        out[:, 2] = self.az * dome * w * (z / 500.0 - 0.5)
        return out


def make_field(args):
    if args.field == "dual":
        return DualMapField(args.truth_root)
    return AnalyticField()


# ----------------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------------
def generate(args):
    rng = np.random.default_rng(args.seed)
    field = make_field(args)
    geom = P2.load_wire_geom(args.wire_geom)
    SLOPE, OFFSET, WHATY, WHATZ = geom

    pq_dir = os.path.join(args.out, "parquet")
    os.makedirs(pq_dir, exist_ok=True)
    n_per_shard = args.ntracks // args.nshards

    # truth at the standard voxel-grid centers, for the closure comparison
    nx, ny, nz = P2.grid_shape(20.0)
    xe = np.linspace(*P2.X_RANGE, nx + 1)
    ye = np.linspace(*P2.Y_RANGE, ny + 1)
    ze = np.linspace(*P2.Z_RANGE, nz + 1)
    vx = 0.5 * (xe[:-1] + xe[1:])
    vy = 0.5 * (ye[:-1] + ye[1:])
    vz = 0.5 * (ze[:-1] + ze[1:])
    X, Y, Z = np.meshgrid(vx, vy, vz, indexing="ij")
    tg = field(X.ravel(), Y.ravel(), Z.ravel()).reshape(nx, ny, nz, 3)
    np.savez_compressed(os.path.join(args.out, "truth_field.npz"),
                        truth=tg.astype(np.float32), x_edges=xe, y_edges=ye,
                        z_edges=ze, field=args.field)

    x_step = args.x_step
    xs0 = np.arange(-200.0 + x_step / 2, 200.0, x_step)   # sample points
    for ishard in range(args.nshards):
        rows = []
        for itrk in range(n_per_shard):
            y0, y1 = rng.uniform(-180, 180, 2)
            z0, z1 = rng.uniform(20, 480, 2)
            u = (xs0 + 200.0) / 400.0
            yt = y0 + u * (y1 - y0)
            zt = z0 + u * (z1 - z0)
            ok = (np.abs(yt) < 199) & (zt > 1) & (zt < 499)
            xt, yt, zt = xs0[ok], yt[ok], zt[ok]
            d = field(xt, yt, zt)
            x_m = xt + d[:, 0]
            tpc = (x_m >= 0).astype(np.int64)
            anodex = np.where(tpc == 0, -200.0, 200.0)
            time = T0 + np.abs(x_m - anodex) / (TICK_US * args.vdrift)
            time = time + rng.normal(0, args.tick_noise, len(time))
            for p in (0, 1, 2):
                wy = WHATY[tpc, p]; wz = WHATZ[tpc, p]
                s = (wy * (yt + d[:, 1]) + wz * (zt + d[:, 2])
                     + rng.normal(0, args.s_noise, len(xt)))
                wire = np.round((s - OFFSET[tpc, p]) / SLOPE[tpc, p])
                nw = 1984 if p == 2 else 1664
                wok = (wire >= 0) & (wire < nw)
                rows.append(pd.DataFrame({
                    "run": 1, "subrun": ishard, "event": itrk, "trk_id": 0,
                    "plane": np.int32(p), "tpc": tpc[wok].astype(np.int32),
                    "wire": wire[wok].astype(np.int32),
                    "time": time[wok].astype(np.float32)}))
        df = pd.concat(rows, ignore_index=True)
        path = os.path.join(pq_dir, f"toy_part{ishard:03d}.parquet")
        df.to_parquet(path, index=False)
        print(f"shard {ishard}: {len(df):,} hits -> {path}", flush=True)
    print(f"\ntruth field + {args.nshards} shards under {args.out}/",
          flush=True)


# ----------------------------------------------------------------------------
# comparison
# ----------------------------------------------------------------------------
def compare(args):
    t = np.load(os.path.join(args.out, "truth_field.npz"))
    truth = t["truth"].astype(np.float64)
    d = np.load(args.maps)
    meas = d["delta_mu_all"].astype(np.float64)
    cnt = d["count_all"]
    vx = B.voxel_centers(d["x_edges"])
    keep = np.isfinite(meas).all(-1) & np.isfinite(truth).all(-1) \
        & (cnt >= 30)
    print(f"voxels compared: {keep.sum()}")
    print(f"{'comp':4s} {'slope':>8s} {'rms':>8s}   (slope = "
          f"<m.t>/<t.t> over |t|>0.1)")
    for c, comp in enumerate(COMPONENTS):
        mm = meas[..., c][keep]; tt = truth[..., c][keep]
        sig = np.abs(tt) > 0.1
        slope = (mm[sig] * tt[sig]).sum() / (tt[sig] ** 2).sum() \
            if sig.sum() > 5 else np.nan
        rms = np.sqrt(np.mean((mm - tt) ** 2))
        print(f"{comp:4s} {slope:>8.3f} {rms:>8.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    for c, (ax, comp) in enumerate(zip(axes, COMPONENTS)):
        mp = np.nanmean(np.where(keep, meas[..., c], np.nan), axis=(1, 2))
        tp = np.nanmean(np.where(keep, truth[..., c], np.nan), axis=(1, 2))
        ax.plot(vx, mp, "o-", ms=4, label="pipeline output")
        ax.plot(vx, tp, "r-", lw=2, label="input field")
        ax.axhline(0, color="0.6", ls=":", lw=0.7)
        ax.set_xlabel("X [cm]"); ax.set_ylabel(f"$\\Delta_{{{comp[1]}}}$ [cm]")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("toy closure: pipeline output vs known input field")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "closure_profiles.png"), dpi=120)
    plt.close(fig)
    print(f"wrote {args.out}/closure_profiles.png", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("generate", "compare"))
    ap.add_argument("--out", default="toy_closure")
    ap.add_argument("--field", choices=("dual", "analytic"), default="dual")
    ap.add_argument("--truth-root",
                    default="SCEoffsets_SBND_E500_dualmap_voxelTH3.root")
    ap.add_argument("--wire-geom", default=P2.WIRE_GEOM)
    ap.add_argument("--ntracks", type=int, default=30000)
    ap.add_argument("--nshards", type=int, default=4)
    ap.add_argument("--x-step", type=float, default=1.2)
    ap.add_argument("--vdrift", type=float, default=V)
    ap.add_argument("--s-noise", type=float, default=0.3,
                    help="gaussian noise on the measured view coord [cm]")
    ap.add_argument("--tick-noise", type=float, default=1.0,
                    help="gaussian noise on the hit time [ticks]")
    ap.add_argument("--seed", type=int, default=20260610)
    ap.add_argument("--maps", default="toy_closure/maps/voxel_data.npz")
    args = ap.parse_args()
    if args.mode == "generate":
        generate(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
