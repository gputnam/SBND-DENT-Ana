#!/usr/bin/env python3
"""Per-voxel histograms of the 2D double-anode voxel-LSQ fit offsets.

Each oncalo hit measures a scalar `m = n_hat . Delta` -- the field
distortion projected onto `n_hat`, the in-plane axis perpendicular to the
track. `plot_doubleanode_2d_voxels.py` inverts all hits in a voxel to the
3-D distortion `Delta_hat`. This script forms, per hit, the offset between
that measured projection and the inverted distortion projected onto the
SAME perpendicular axis -- i.e. the per-hit residual of the voxel LSQ fit --
and writes, per voxel, ONE linear-y histogram figure each for:

  * offset_hist/    the scalar offset            r   = m - n_hat . Delta_hat
  * offset_hist_x/  its drift (x) component      r_x = r * n_hat_x
                    = the apparent-minus-expected track-position offset
                      projected onto the drift axis.

The `doubleanode_2d_df` parquet is re-read once (the per-hit `m` / `n_hat`
are not stored in `voxel_data.npz`), reusing
`plot_doubleanode_2d_voxels.shard_hits` so the geometry is bit-identical to
the map build; `Delta_hat` is taken from the existing `voxel_data.npz`
(no re-inversion). The non-npz build cuts (`--max-r-cm`, `--anode-dt-window`,
anode-disagreement FOM) default to the production builder values and MUST
match the run that produced the npz.

Usage:
    python plot_voxel_offset_hist.py
    python plot_voxel_offset_hist.py --bin-cm 1.0
"""

import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import plot_doubleanode_2d_voxels as P2

MAPS_DEFAULT = "doubleanode_2d_maps_20cm/voxel_data.npz"
OUTDIR_DEFAULT = "doubleanode_2d_maps_20cm"

# worker globals (set by _init)
_DELTA = None          # (nx,ny,nz,3) inverted Delta_hat
_HIST = None           # ((nb_r,lo_r,w_r), (nb_x,lo_x,w_x))


def _init(geom, params, delta, hist_params):
    """Pool initializer: set both plot_doubleanode_2d_voxels' geometry
    globals (for shard_hits) and this module's offset-histogram globals."""
    P2._worker_init(geom, params)
    global _DELTA, _HIST
    _DELTA = delta
    _HIST = hist_params


def _bin(values, base, nbins, lo, width, nvox):
    """Per-voxel histogram of `values` keyed by flat voxel index `base`."""
    ib = np.floor((values - lo) / width).astype(np.int64)
    inr = np.isfinite(values) & (ib >= 0) & (ib < nbins)
    flat = base[inr] * nbins + ib[inr]
    cnt = np.bincount(flat, minlength=nvox * nbins)
    return cnt.reshape(-1, nbins), int(inr.sum())


def offset_shard(path):
    """Re-process one shard -> per-voxel histograms of the LSQ-fit offset
    r = m - n_hat.Delta_hat and its drift component r_x = r * n_hat_x.

    Hits in voxels with no inverted Delta_hat (NaN) give a NaN offset and
    are dropped; hits whose offset falls outside the histogram range are
    counted in `stats` but not binned."""
    ix, iy, iz, mk, nk, _atick, stats = P2.shard_hits(path)
    nx, ny, nz = _DELTA.shape[:3]
    nvox = nx * ny * nz
    (nb_r, lo_r, w_r), (nb_x, lo_x, w_x) = _HIST
    hist_r = np.zeros((nx, ny, nz, nb_r), dtype=np.int64)
    hist_x = np.zeros((nx, ny, nz, nb_x), dtype=np.int64)
    stats["n_offset"] = 0
    if stats["n_hits"]:
        proj = np.einsum("hc,hc->h", nk, _DELTA[ix, iy, iz])   # n_hat.Delta
        r = mk - proj                                          # scalar offset
        rx = r * nk[:, 0]                                      # drift (x) part
        base = ix * (ny * nz) + iy * nz + iz
        cr, n_in = _bin(r, base, nb_r, lo_r, w_r, nvox)
        hist_r += cr.reshape(nx, ny, nz, nb_r)
        cx, _ = _bin(rx, base, nb_x, lo_x, w_x, nvox)
        hist_x += cx.reshape(nx, ny, nz, nb_x)
        stats["n_offset"] = n_in
    return hist_r, hist_x, stats


def _fig_worker(task):
    """Render one per-voxel offset histogram (linear y-axis)."""
    xc, yc, zc, hist, centers, width, dvec, xlabel, png = task
    h = hist.astype(float)
    tot = h.sum()
    if tot <= 0:
        return None
    mean = float((centers * h).sum() / tot)
    rms = float(np.sqrt(max((centers ** 2 * h).sum() / tot - mean ** 2, 0.0)))
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.bar(centers, h, width=width, color="steelblue",
           edgecolor="navy", linewidth=0.3)
    ax.axvline(0.0, color="0.4", ls=":", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"hits / {width:g} cm")
    ax.set_ylim(bottom=0.0)
    ax.set_title(f"voxel  X={xc:+.0f}  Y={yc:+.0f}  Z={zc:+.0f} cm\n"
                 f"N={int(tot):,}   mean={mean:+.2f}  rms={rms:.2f} cm",
                 fontsize=9)
    ax.text(0.02, 0.97,
            r"$\hat{\Delta}$=(" + f"{dvec[0]:+.1f}, {dvec[1]:+.1f}, "
            f"{dvec[2]:+.1f}) cm", transform=ax.transAxes, fontsize=7,
            va="top", ha="left")
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(png, dpi=90)
    plt.close(fig)
    return png


def make_set(hist, edges, outdir, xlabel, gtitle, delta, grid, nproc):
    """Write one histogram set: offset_hist.npz, offset_distribution.png,
    and one per-voxel figure per inverted voxel. Returns the figure count."""
    vx_c, vy_c, vz_c, x_edges, y_edges, z_edges = grid
    nx, ny, nz = delta.shape[:3]
    nbins = hist.shape[-1]
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = float(edges[1] - edges[0])
    os.makedirs(outdir, exist_ok=True)

    np.savez_compressed(
        os.path.join(outdir, "offset_hist.npz"),
        offset_hist=hist.astype(np.int32), offset_edges=edges,
        x_edges=x_edges, y_edges=y_edges, z_edges=z_edges,
        delta_mu_all=delta.astype(np.float32), bin_cm=np.float32(width))

    # global aggregate
    tot = hist.reshape(-1, nbins).sum(axis=0).astype(float)
    N = max(tot.sum(), 1.0)
    g_mean = float((centers * tot).sum() / N)
    g_rms = float(np.sqrt(max((centers ** 2 * tot).sum() / N - g_mean ** 2,
                              0.0)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(centers, tot, width=width, color="steelblue",
           edgecolor="navy", linewidth=0.4)
    ax.axvline(0.0, color="0.4", ls=":", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"hits / {width:g} cm")
    ax.set_ylim(bottom=0.0)
    ax.set_title(f"{gtitle}, all voxels\n"
                 f"N={int(tot.sum()):,}   mean={g_mean:+.3f}  "
                 f"rms={g_rms:.3f} cm")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "offset_distribution.png"), dpi=120)
    plt.close(fig)

    # one figure per inverted voxel
    tasks = []
    for ix in range(nx):
        sub = os.path.join(outdir, f"x{int(round(vx_c[ix])):+04d}")
        made = False
        for iy in range(ny):
            for iz in range(nz):
                if not np.isfinite(delta[ix, iy, iz, 0]):
                    continue
                if hist[ix, iy, iz].sum() <= 0:
                    continue
                if not made:
                    os.makedirs(sub, exist_ok=True)
                    made = True
                png = os.path.join(
                    sub, f"y{int(round(vy_c[iy])):+04d}"
                         f"_z{int(round(vz_c[iz])):+04d}.png")
                tasks.append((vx_c[ix], vy_c[iy], vz_c[iz],
                              hist[ix, iy, iz], centers, width,
                              delta[ix, iy, iz], xlabel, png))
    if nproc > 1:
        with Pool(nproc) as pool:
            for _ in tqdm(pool.imap_unordered(_fig_worker, tasks,
                                              chunksize=16),
                          total=len(tasks), unit="fig", file=sys.stderr):
                pass
    else:
        for t in tasks:
            _fig_worker(t)
    return len(tasks)


def edges_for(range_cm, bin_cm):
    nbins = int(round(2 * range_cm / bin_cm))
    return np.linspace(-range_cm, range_cm, nbins + 1)


def main():
    # Use "spawn", not "fork", for the worker Pools. This script uses
    # matplotlib/numpy in the parent and creates several Pools in sequence
    # (accumulation + one per histogram set); a fork-started Pool inherits
    # the parent's held locks and deadlocks on teardown of the second Pool
    # (same failure mode patched in bootstrap_wall_fit.py -- see CLAUDE.md).
    import multiprocessing as _mp
    try:
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=P2.INPUT_DIR,
                    help="doubleanode_2d_df parquet dir (re-read for m/n_hat)")
    ap.add_argument("--maps", default=MAPS_DEFAULT,
                    help="voxel_data.npz supplying the inverted Delta_hat")
    ap.add_argument("--outdir", default=OUTDIR_DEFAULT,
                    help="parent dir; sets go in offset_hist/ + offset_hist_x/")
    ap.add_argument("--wire-geom", default=P2.WIRE_GEOM)
    ap.add_argument("--anode-cut-config", default=P2.ANODE_CUT_CONFIG)
    ap.add_argument("--max-r-cm", type=float, default=25.0,
                    help="per-hit |m| cut -- must match the map build")
    ap.add_argument("--anode-dt-window", type=float, default=40.0,
                    help="anode-hit tick window -- must match the map build")
    ap.add_argument("--max-anode-fom-cm", type=float, default=None,
                    help="anode-disagreement cut; default from "
                         "--anode-cut-config (must match the map build)")
    ap.add_argument("--bin-cm", type=float, default=0.625,
                    help="histogram bin width [cm] (default 0.625 -> 80 bins "
                         "over the scalar-offset range)")
    ap.add_argument("--range-cm", type=float, default=25.0,
                    help="scalar-offset histogram covers [-range, +range] cm")
    ap.add_argument("--range-cm-x", type=float, default=10.0,
                    help="drift-component histogram half-range cm (r_x is "
                         "smaller than r, so a tighter range)")
    ap.add_argument("--accum-nproc", type=int, default=4,
                    help="parquet shards processed in parallel")
    ap.add_argument("--nproc", default="auto",
                    help="workers for the per-voxel figures")
    ap.add_argument("--max-shards", type=int, default=None)
    args = ap.parse_args()

    # ---- inverted distortion + grid from the map npz ----
    d = np.load(args.maps)
    delta = d["delta_mu_all"].astype(np.float64)        # (nx,ny,nz,3)
    nx, ny, nz, _ = delta.shape
    x_edges, y_edges, z_edges = d["x_edges"], d["y_edges"], d["z_edges"]
    vx_c = 0.5 * (x_edges[:-1] + x_edges[1:])
    vy_c = 0.5 * (y_edges[:-1] + y_edges[1:])
    vz_c = 0.5 * (z_edges[:-1] + z_edges[1:])
    grid = (vx_c, vy_c, vz_c, x_edges, y_edges, z_edges)
    size = float(d["voxel_size"])
    vdrift = float(d["vdrift"])
    t0 = float(d["anode_t0"])
    n_inverted = int(np.isfinite(delta[..., 0]).sum())

    # ---- histogram binning (shared bin width; per-set range) ----
    edges_r = edges_for(args.range_cm, args.bin_cm)
    edges_x = edges_for(args.range_cm_x, args.bin_cm)
    hist_params = ((len(edges_r) - 1, edges_r[0], args.bin_cm),
                   (len(edges_x) - 1, edges_x[0], args.bin_cm))

    # ---- build cuts (must match the map build) ----
    max_fom = args.max_anode_fom_cm
    if max_fom is None:
        try:
            max_fom = float(json.load(open(args.anode_cut_config))["cut_cm"])
        except Exception:
            max_fom = 5.0
    geom = P2.load_wire_geom(args.wire_geom)
    params = dict(size=size, t0=t0, vdrift=vdrift, max_r_cm=args.max_r_cm,
                  max_anode_fom=max_fom, anode_dt_window=args.anode_dt_window)

    if os.path.isdir(args.input):
        shards = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    else:
        shards = sorted(glob.glob(args.input))
    if args.max_shards:
        shards = shards[:args.max_shards]
    if not shards:
        print(f"no parquet shards at {args.input}", file=sys.stderr)
        sys.exit(1)

    print("offset-histogram build:", flush=True)
    print(f"  r   = m - n_hat . Delta_hat        ({len(edges_r)-1} bins, "
          f"+/-{args.range_cm:g} cm)", flush=True)
    print(f"  r_x = r * n_hat_x  (drift comp.)   ({len(edges_x)-1} bins, "
          f"+/-{args.range_cm_x:g} cm)", flush=True)
    print(f"  maps      : {args.maps}  ({n_inverted:,} inverted voxels)",
          flush=True)
    print(f"  parquet   : {len(shards)} shards from {args.input}", flush=True)
    print(f"  build cuts: vdrift={vdrift} t0={t0} max_r={args.max_r_cm} "
          f"anode_fom={max_fom:.3f} dt_window={args.anode_dt_window}",
          flush=True)

    # ---- re-read shards -> per-voxel offset histograms ----
    hist_r = np.zeros((nx, ny, nz, len(edges_r) - 1), dtype=np.int64)
    hist_x = np.zeros((nx, ny, nz, len(edges_x) - 1), dtype=np.int64)
    n_hits = n_off = 0
    initargs = (geom, params, delta, hist_params)
    if args.accum_nproc > 1:
        with Pool(args.accum_nproc, initializer=_init,
                  initargs=initargs) as pool:
            for hr, hx, st in tqdm(pool.imap_unordered(offset_shard, shards),
                                   total=len(shards), unit="shard",
                                   file=sys.stderr):
                hist_r += hr; hist_x += hx
                n_hits += st["n_hits"]; n_off += st["n_offset"]
    else:
        _init(*initargs)
        for path in tqdm(shards, unit="shard", file=sys.stderr):
            hr, hx, st = offset_shard(path)
            hist_r += hr; hist_x += hx
            n_hits += st["n_hits"]; n_off += st["n_offset"]
    print(f"\nhits processed: {n_hits:,}; scalar offsets in range: "
          f"{n_off:,} ({100.0*n_off/max(1,n_hits):.1f}%)", flush=True)

    nproc = cpu_count() if args.nproc == "auto" else int(args.nproc)

    # ---- set 1: scalar offset r ----
    out_r = os.path.join(args.outdir, "offset_hist")
    print(f"\n[offset_hist]  scalar offset r -> {out_r}/", flush=True)
    n1 = make_set(hist_r, edges_r, out_r,
                  r"offset  $r = m - \hat{n}\cdot\hat{\Delta}$  [cm]",
                  "SBND 2D double-anode: per-hit LSQ-fit offset",
                  delta, grid, nproc)

    # ---- set 2: drift (x) component r_x ----
    out_x = os.path.join(args.outdir, "offset_hist_x")
    print(f"\n[offset_hist_x]  drift component r_x -> {out_x}/", flush=True)
    n2 = make_set(hist_x, edges_x, out_x,
                  r"drift offset  $r_x=(m-\hat{n}\cdot\hat{\Delta})\,"
                  r"\hat{n}_x$  [cm]",
                  "SBND 2D double-anode: per-hit LSQ-fit offset, "
                  "drift (x) component",
                  delta, grid, nproc)

    print(f"\nwrote {n1:,} figures to {out_r}/ and {n2:,} to {out_x}/ "
          f"(+ offset_hist.npz + offset_distribution.png each)", flush=True)


if __name__ == "__main__":
    main()
