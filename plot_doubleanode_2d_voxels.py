"""Voxel-binned field-distortion maps from the 2D double-anode method.

CRT-free and SPACE-POINT-free. The input parquet (built by
`make_doubleanode_2d_df_sbnd.py`) carries, per oncalo hit on a
double-anode-piercing track, only the raw detector quantities
(plane, tpc, wire, tick). This script:

  1. tick  -> drift-x   :  x = anode_x[tpc] -/+ (tick - t0)*0.5us*v_drift
  2. wire  -> view coord:  s_view = slope_cm*wire + offset_cm
                                  = what_y*y + what_z*z
     (geometry from wire_geometry_sbnd.json, LArSoft DumpGeometry).
  3. Per track, per TPC: reconstruct the anode-pierce (y, z) by matching
     the >=2 earliest-tick hits across the wire planes
     (least squares of  what_p . (y,z) = s_anode_p).
  4. The 3D reference line joins the two anode-pierce points; the 3D
     voxel of every hit is the interpolation of that line at the hit's
     drift-x.
  5. Each hit's residual from the line, in its plane's 2D (x, s_view)
     frame, is the perpendicular distance m = n_hat . Delta -- a scalar
     projection of the 3-D distortion onto a known unit direction.
  6. Per voxel, the MicroBooNE-style least-squares stack of all hits /
     planes / tracks recovers the 3-D distortion Delta-hat
     (crtlib.voxel_invert.invert_scalar_accumulator).

Three wire planes per track => three 2D lines => three n_hat directions
per voxel, which (with the spread of track angles) make the per-voxel
3x3 system full rank.

Output layout (under --outdir) mirrors plot_doubleanode_voxels.py so the
comparison plotters read it unchanged:
    all/{dx,dy,dz}/{png,pdf}/x{NNN}.{png,pdf}     Delta-hat maps
    sigma/all/{dx,dy,dz}/{png,pdf}/x{NNN}.*       LSQ sigma maps
    cond/all/{png,pdf}/x{NNN}.*                   condition-number maps
    counts/all/{png,pdf}/x{NNN}.*                 hit counts
    residual_distribution.png                     global m histogram
    voxel_data.npz                                LSQ data + npz schema
"""

import argparse
import glob
import json
import os
import sys
import warnings
from multiprocessing import Pool, cpu_count

try:
    os.nice(10)
except (AttributeError, PermissionError):
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from crtlib import voxel_invert
from crtlib.constants import DATA_DIR
from crtlib.io import load_wire_geometry

warnings.filterwarnings("ignore", category=RuntimeWarning)

_REPO = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(DATA_DIR, "doubleanode_2d_df")
OUTDIR = "doubleanode_2d_maps"
WIRE_GEOM = os.path.join(_REPO, "config", "wire_geometry_sbnd.json")
CUT_CONFIG = os.path.join(_REPO, "config", "cut_tuning", "cut_config_per_tpc.json")
ANODE_CUT_CONFIG = os.path.join(_REPO, "config", "cut_tuning", "anode_cut.json")

VOXEL_SIZE = 20.0
X_RANGE = (-200.0, 200.0)
Y_RANGE = (-200.0, 200.0)
Z_RANGE = (0.0, 500.0)

TICK_US = 0.5                       # 2 MHz drift clock -> 500 ns/tick
VDRIFT_DEFAULT = 0.15532            # cm/us, 3-plane mean (cathode stage)
ANODE_T0_DEFAULT = 418.5            # empirical anode-peak tick
ANODE_X = {0: -200.0, 1: 200.0}

# residual histogram (per voxel, of the scalar m = n_hat.Delta)
RESID_HALF = 50.0
RESID_BIN = 0.5

# anode-hit tick diagnostic: histogram of (anode-hit tick - collection
# anode tick T_c) per (tpc, plane). Shows the per-plane anode-tick bump
# and the pre-anode noise contamination of the induction planes.
ATICK_HALF = 120.0
ATICK_BIN = 1.0

SUBSETS = ("all",)
COMPONENTS = ("dx", "dy", "dz")

DEFAULT_VRANGE = 5.0
DEFAULT_SIGMA_VRANGE = 5.0
DEFAULT_MIN_COUNTS = 30

HEATMAP_CMAP = matplotlib.colormaps["RdYlBu_r"].copy()
HEATMAP_CMAP.set_bad("white")
SIGMA_CMAP = matplotlib.colormaps["viridis"].copy()
SIGMA_CMAP.set_bad("white")


def grid_shape(size):
    nx = int(round((X_RANGE[1] - X_RANGE[0]) / size))
    ny = int(round((Y_RANGE[1] - Y_RANGE[0]) / size))
    nz = int(round((Z_RANGE[1] - Z_RANGE[0]) / size))
    return nx, ny, nz


def resid_setup():
    n_bins = int(round(2 * RESID_HALF / RESID_BIN))
    edges = np.linspace(-RESID_HALF, RESID_HALF, n_bins + 1)
    return n_bins, edges


def atick_setup():
    n = int(round(2 * ATICK_HALF / ATICK_BIN))
    edges = np.linspace(-ATICK_HALF, ATICK_HALF, n + 1)
    return n, edges


def load_wire_geom(path):
    """Return SLOPE, OFFSET, WHATY, WHATZ as (2,3) arrays [tpc, plane].
    Thin wrapper over crtlib.io.load_wire_geometry (kept for back-compat)."""
    return load_wire_geometry(path)


# ---- worker globals (set by _worker_init) -----------------------------------
_GEOM = None
_PARAMS = None


def _worker_init(geom, params):
    global _GEOM, _PARAMS
    _GEOM = geom
    _PARAMS = params


def reconstruct_anodes(tkey, tpc, plane, time, s, whaty, whatz, ntrk,
                       anode_dt_window, x=None, peak_offset=None):
    """Per-track anode-pierce (y, z) reconstruction from the 3 wire planes.

    The collection plane (2) has a sharp anode-tick peak so its earliest
    hit is the anode pierce; the induction planes' earliest hit is often
    pre-anode noise, so their anode hit is the hit closest in tick to the
    collection-plane anode tick T_c (unified rule: minimum |time - T_c|).
    Each TPC anode (y, z) is the least-squares solution of the
    over-determined system  what_p . (y, z) = s_p  over its >= 2 planes.

    If per-hit drift coordinates `x` are supplied (--anode-extrapolate),
    each selected anode hit's view coordinate is extrapolated from its own
    drift-x to the anode plane along the current track direction,
    s(anode) = s(x_hit) - (x_hit - x_anode) * d s/dx, and the (y, z) fit is
    iterated twice. Without it the anode hits (which sit up to the
    dt-window inside the detector) are implicitly assigned to x = +/-200,
    biasing the chord by (inset * track slope) -- a fake odd-linear dx
    (demonstrated in toy_closure_2d.py).

    Returns
    -------
    YA, ZA : (ntrk, 2) anode (y, z) per TPC; NaN if < 2 planes available.
    fom    : (ntrk,) anode-disagreement figure of merit (cm) -- the larger
             of the two anodes' RMS of the 3-plane anode-fit residuals.
             A 2-plane anode fits exactly and contributes 0; FOM is 0 for
             a track with no 3-plane anode.
    atick  : dict of (2, 3, n_at) tick-offset histograms ('selected' =
             closest-to-T_c hit, 'earliest' = naive earliest hit).
    """
    n_at, _ = atick_setup()
    adf = pd.DataFrame({"tkey": tkey, "tpc": tpc, "plane": plane,
                        "time": time, "s": s, "wy": whaty, "wz": whatz})
    if x is not None:
        adf["x"] = x
    col = adf[adf.plane == 2]
    tc = col.groupby(["tkey", "tpc"])["time"].min().reset_index()
    Tc = np.full((ntrk, 2), np.nan)
    Tc[tc.tkey.to_numpy(), tc.tpc.to_numpy()] = tc.time.to_numpy()
    # Per-plane anode reference: T_c (the collection-plane anode tick) plus a
    # per-(tpc, plane) offset = that plane's anode peak minus collection's.
    # Each plane then selects the hit closest to ITS OWN anode peak rather
    # than the collection peak (the induction responses peak a few ticks off
    # collection -- see plot_anode_hit_times_data.py). peak_offset is a (2,3)
    # array; None reproduces the collection-peak selection.
    ref = Tc[tkey, tpc]
    if peak_offset is not None:
        ref = ref + np.asarray(peak_offset)[tpc, plane]
    adf["dt"] = np.abs(adf["time"].to_numpy() - ref)
    # tracks with no collection hits in a TPC have NaN Tc -> NaN dt; drop
    # those rows before the idxmin selection (no-op for the production df,
    # whose builder guarantees collection hits in both TPCs)
    sel = adf[np.isfinite(adf["dt"].to_numpy())]
    aidx = sel.groupby(["tkey", "tpc", "plane"])["dt"].idxmin().to_numpy()
    an_all = adf.loc[aidx]                        # closest-to-Tc per plane
    an = an_all[an_all["dt"] < anode_dt_window]   # windowed -> used
    eidx = adf.groupby(["tkey", "tpc", "plane"])["time"].idxmin().to_numpy()
    earliest = adf.loc[eidx]                      # naive earliest hit

    # anode-tick diagnostic: histogram (anode tick - T_c) per (tpc, plane).
    atick = {}
    for tag, rows in (("selected", an_all), ("earliest", earliest)):
        dts = (rows["time"].to_numpy()
               - Tc[rows["tkey"].to_numpy(), rows["tpc"].to_numpy()])
        ib = np.floor((dts + ATICK_HALF) / ATICK_BIN).astype(np.int64)
        inr = np.isfinite(dts) & (ib >= 0) & (ib < n_at)
        rtpc = rows["tpc"].to_numpy().astype(np.int64)[inr]
        rpln = rows["plane"].to_numpy().astype(np.int64)[inr]
        fl = rtpc * (3 * n_at) + rpln * n_at + ib[inr]
        atick[tag] = np.bincount(
            fl, minlength=2 * 3 * n_at).reshape(2, 3, n_at).astype(np.int64)

    # per-(tkey,tpc) 2x2 normal equations for what.(y,z) = s
    def fit_yz(s_fit):
        a = an.assign(s_fit=s_fit)
        a = a.assign(wy2=a.wy * a.wy, wz2=a.wz * a.wz, wyz=a.wy * a.wz,
                     wys=a.wy * a.s_fit, wzs=a.wz * a.s_fit)
        ag = a.groupby(["tkey", "tpc"]).agg(
            A00=("wy2", "sum"), A11=("wz2", "sum"), A01=("wyz", "sum"),
            b0=("wys", "sum"), b1=("wzs", "sum"),
            npl=("s_fit", "count")).reset_index()
        det = ag.A00 * ag.A11 - ag.A01 ** 2
        ya_v = (ag.A11 * ag.b0 - ag.A01 * ag.b1) / det
        za_v = (ag.A00 * ag.b1 - ag.A01 * ag.b0) / det
        ok = (ag.npl >= 2) & (det > 1e-9)
        YA_ = np.full((ntrk, 2), np.nan)
        ZA_ = np.full((ntrk, 2), np.nan)
        tk_, tp_ = ag.tkey.to_numpy(), ag.tpc.to_numpy()
        YA_[tk_[ok], tp_[ok]] = ya_v[ok]
        ZA_[tk_[ok], tp_[ok]] = za_v[ok]
        return YA_, ZA_

    s_use = an.s.to_numpy()
    YA, ZA = fit_yz(s_use)

    if x is not None:
        # extrapolate each anode hit's s from its own drift-x to the anode
        # plane along the current chord direction, refit (2 iterations).
        an_tk = an.tkey.to_numpy()
        an_tpc = an.tpc.to_numpy()
        an_x = an.x.to_numpy()
        anode_of = np.where(an_tpc == 0, ANODE_X[0], ANODE_X[1])
        for _ in range(2):
            ty = (YA[:, 1] - YA[:, 0]) / (ANODE_X[1] - ANODE_X[0])
            tz = (ZA[:, 1] - ZA[:, 0]) / (ANODE_X[1] - ANODE_X[0])
            dsdx = (an.wy.to_numpy() * ty[an_tk]
                    + an.wz.to_numpy() * tz[an_tk])
            corr = (an_x - anode_of) * np.nan_to_num(dsdx)
            s_use = an.s.to_numpy() - corr
            YA, ZA = fit_yz(s_use)

    # anode-disagreement FOM: per (tkey,tpc) RMS of the 3-plane fit
    # residuals; per-track FOM = max over the two anodes.
    an_pred = (an.wy.to_numpy() * YA[an.tkey.to_numpy(), an.tpc.to_numpy()]
               + an.wz.to_numpy() * ZA[an.tkey.to_numpy(), an.tpc.to_numpy()])
    an = an.assign(r2=(an_pred - s_use) ** 2)
    rg = an.groupby(["tkey", "tpc"]).agg(sr2=("r2", "sum"),
                                         nr=("r2", "count")).reset_index()
    fom_tt = np.zeros((ntrk, 2))
    fom_tt[rg.tkey.to_numpy(), rg.tpc.to_numpy()] = np.sqrt(
        rg.sr2.to_numpy() / rg.nr.to_numpy())
    fom = fom_tt.max(axis=1)
    return YA, ZA, fom, atick


def shard_hits(path, extra=False):
    """Process one parquet shard into per-hit voxel indices, scalar
    measurement m = n_hat . Delta, and measurement direction n_hat.

    The per-hit geometry/residual computation shared by accumulate_shard
    (the voxel inversion) and plot_voxel_offset_hist.py (the post-inversion
    offset histogram). Returns (ix, iy, iz, m, nhat, atick, stats): ix/iy/iz
    and m are (N,), nhat is (N, 3), all after the per-hit keep mask; atick is
    the anode-tick diagnostic dict; stats carries the track/hit counts.

    With extra=True an additional dict is appended to the return tuple with
    per-hit (kept-hit) context for diagnostics: plane, tpc, x (drift coord),
    slope (chord d s_view/dx), and tkey (shard-local track index).
    """
    SLOPE, OFFSET, WHATY, WHATZ = _GEOM
    P = _PARAMS
    size = P["size"]
    nx, ny, nz = grid_shape(size)

    df = pd.read_parquet(path, columns=["run", "subrun", "event", "trk_id",
                                        "plane", "tpc", "wire", "time"])
    tkey = df.groupby(["run", "subrun", "event", "trk_id"]).ngroup().to_numpy()
    ntrk = int(tkey.max()) + 1 if len(tkey) else 0
    plane = df["plane"].to_numpy()
    tpc = df["tpc"].to_numpy()
    wire = df["wire"].to_numpy(np.float64)
    time = df["time"].to_numpy(np.float64)
    del df

    n_at, _ = atick_setup()
    empty_atick = {"selected": np.zeros((2, 3, n_at), dtype=np.int64),
                   "earliest": np.zeros((2, 3, n_at), dtype=np.int64)}
    stats = dict(n_tracks=ntrk, n_tracks_used=0, n_hits=0, path=path)
    if ntrk == 0:
        empty_i = np.empty(0, dtype=np.int64)
        out = (empty_i, empty_i, empty_i, np.empty(0), np.empty((0, 3)),
               empty_atick, stats)
        if extra:
            out = out + (dict(plane=np.empty(0, np.int8),
                              tpc=np.empty(0, np.int8),
                              x=np.empty(0), slope=np.empty(0),
                              tkey=empty_i),)
        return out

    # --- per-hit geometry ---------------------------------------------------
    slope_g = SLOPE[tpc, plane]
    off_g = OFFSET[tpc, plane]
    whaty = WHATY[tpc, plane]
    whatz = WHATZ[tpc, plane]
    s = slope_g * wire + off_g
    # optional per-(tpc, plane) tick offset calibration (--plane-t0-json):
    # applied to the raw times so the anode selection, the anode (y,z) fit
    # and the drift coordinate all see the corrected clock.
    plane_t0 = P.get("plane_t0")
    if plane_t0 is not None:
        time = time - np.asarray(plane_t0)[tpc, plane]
    anodex = np.where(tpc == 0, ANODE_X[0], ANODE_X[1])
    xdir = np.where(tpc == 0, 1.0, -1.0)
    x = anodex + xdir * (time - P["t0"]) * TICK_US * P["vdrift"]
    # in-situ ramp correction (--x-ramp-correct hits): rescale each hit's
    # drift distance by 1/(1+eps_tpc) -- equivalent to reconstructing at the
    # double-anode effective velocity v_DA = v/(1+eps). eps is fitted from
    # the previous iteration's map (fit_dx_ramp; anode+cathode anchored).
    eps_r = P.get("x_ramp_eps")
    if eps_r is not None:
        eps_h = np.where(tpc == 0, eps_r[0], eps_r[1])
        x = anodex + (x - anodex) / (1.0 + eps_h)

    # --- anode-pierce reconstruction + disagreement FOM --------------------
    # forensic override: build the chord from a SIBLING parquet (same tracks
    # and wires, different hit times) -- e.g. measure reco-time hits against
    # the true-time chord to isolate the chord machinery's contribution.
    if P.get("chord_source_dir"):
        cpath = os.path.join(P["chord_source_dir"], os.path.basename(path))
        cdf = pd.read_parquet(cpath, columns=["run", "subrun", "event",
                                              "trk_id", "plane", "tpc",
                                              "wire", "time"])
        ctkey = cdf.groupby(["run", "subrun", "event",
                             "trk_id"]).ngroup().to_numpy()
        cplane = cdf["plane"].to_numpy()
        ctpc = cdf["tpc"].to_numpy()
        cwire = cdf["wire"].to_numpy(np.float64)
        ctime = cdf["time"].to_numpy(np.float64)
        cs = SLOPE[ctpc, cplane] * cwire + OFFSET[ctpc, cplane]
        cwy = WHATY[ctpc, cplane]
        cwz = WHATZ[ctpc, cplane]
        canodex = np.where(ctpc == 0, ANODE_X[0], ANODE_X[1])
        cxdir = np.where(ctpc == 0, 1.0, -1.0)
        cx = canodex + cxdir * (ctime - P["t0"]) * TICK_US * P["vdrift"]
        YA, ZA, fom, atick = reconstruct_anodes(
            ctkey, ctpc, cplane, ctime, cs, cwy, cwz,
            int(ctkey.max()) + 1 if len(ctkey) else 0,
            P["anode_dt_window"],
            x=(cx if P.get("anode_extrapolate") else None),
            peak_offset=P.get("anode_peak_offset"))
    else:
        YA, ZA, fom, atick = reconstruct_anodes(
            tkey, tpc, plane, time, s, whaty, whatz, ntrk,
            P["anode_dt_window"],
            x=(x if P.get("anode_extrapolate") else None),
        peak_offset=P.get("anode_peak_offset"))
    trk_ok = (np.isfinite(YA[:, 0]) & np.isfinite(YA[:, 1])
              & np.isfinite(ZA[:, 0]) & np.isfinite(ZA[:, 1])
              & (fom <= P["max_anode_fom"]))
    stats["n_tracks_used"] = int(trk_ok.sum())

    # --- per-hit residual + measurement direction ---------------------------
    A0y = YA[tkey, 0]; A0z = ZA[tkey, 0]
    A1y = YA[tkey, 1]; A1z = ZA[tkey, 1]
    dy = A1y - A0y; dz = A1z - A0z
    u = (x - ANODE_X[0]) / (ANODE_X[1] - ANODE_X[0])
    yL = A0y + u * dy
    zL = A0z + u * dz
    sL = whaty * yL + whatz * zL
    ds = s - sL
    slope = (whaty * dy + whatz * dz) / (ANODE_X[1] - ANODE_X[0])
    denom = np.sqrt(1.0 + slope * slope)
    m = ds / denom
    nhat = np.stack([-slope / denom, whaty / denom, whatz / denom], axis=1)

    # --- optional per-hit time correction (--hit-time-corr) -----------------
    # The reco PeakTime carries a track-angle- and drift-dependent bias
    # (angle_hit_time_bias.py); the per-voxel LSQ converts that angular
    # pattern into a fake dx dome. Correct each hit's time with the
    # tau(tpc, plane, |slope|, x) LUT -- slope needs the chord, so this runs
    # after the first chord pass, then the anode fit, chord and residuals
    # are rebuilt with the corrected times.
    htc = P.get("hit_time_corr")
    if htc is not None:
        lut, s_edges, lx_edges, w_edges = htc
        nsb, nxb = lut.shape[2], lut.shape[-1]
        # generic binning: works for signed ([-S, S]) or unsigned ([0, S])
        # slope edges
        sl_in = slope if s_edges[0] < 0 else np.abs(slope)
        isb = np.clip(((sl_in - s_edges[0])
                       / (s_edges[-1] - s_edges[0]) * nsb).astype(np.int64),
                      0, nsb - 1)
        ixl = np.clip(((x - lx_edges[0]) / (lx_edges[-1] - lx_edges[0])
                       * nxb).astype(np.int64), 0, nxb - 1)
        if lut.ndim == 5:
            # 2-angle LUT: second axis = |dw/dx|, the along-wire angle
            # (wire dir = (-w_z, w_y) in (y,z); track dir = (dy,dz)/400)
            nwb = lut.shape[3]
            ty = dy / (ANODE_X[1] - ANODE_X[0])
            tz = dz / (ANODE_X[1] - ANODE_X[0])
            dwdx = np.abs(-whatz * ty + whaty * tz)
            iwb = np.clip(((dwdx - w_edges[0])
                           / (w_edges[-1] - w_edges[0]) * nwb)
                          .astype(np.int64), 0, nwb - 1)
            tau = lut[tpc, plane, isb, iwb, ixl]
        else:
            tau = lut[tpc, plane, isb, ixl]
        tau[~np.isfinite(slope)] = 0.0
        time = time - tau
        x = anodex + xdir * (time - P["t0"]) * TICK_US * P["vdrift"]
        YA, ZA, fom, atick = reconstruct_anodes(
            tkey, tpc, plane, time, s, whaty, whatz, ntrk,
            P["anode_dt_window"],
            x=(x if P.get("anode_extrapolate") else None),
        peak_offset=P.get("anode_peak_offset"))
        trk_ok = (np.isfinite(YA[:, 0]) & np.isfinite(YA[:, 1])
                  & np.isfinite(ZA[:, 0]) & np.isfinite(ZA[:, 1])
                  & (fom <= P["max_anode_fom"]))
        stats["n_tracks_used"] = int(trk_ok.sum())
        A0y = YA[tkey, 0]; A0z = ZA[tkey, 0]
        A1y = YA[tkey, 1]; A1z = ZA[tkey, 1]
        dy = A1y - A0y; dz = A1z - A0z
        u = (x - ANODE_X[0]) / (ANODE_X[1] - ANODE_X[0])
        yL = A0y + u * dy
        zL = A0z + u * dz
        sL = whaty * yL + whatz * zL
        ds = s - sL
        slope = (whaty * dy + whatz * dz) / (ANODE_X[1] - ANODE_X[0])
        denom = np.sqrt(1.0 + slope * slope)
        m = ds / denom
        nhat = np.stack([-slope / denom, whaty / denom, whatz / denom],
                        axis=1)

    # --- optional map-feedback position correction (--iter-correct-positions)
    # On iterations >= 2, bin each hit at its estimated TRUE position instead
    # of its measured one (MicroBooNE laser technique, arXiv:1910.01430
    # sec 5.1.2): correct the drift coordinate by the previous-iteration
    # Delta_x looked up at the hit's current position estimate (2-step
    # fixed point), and place the hit on the (undistorted) anode-anode chord
    # at the corrected x. m and n_hat are unchanged -- the chord between the
    # two (undisplaced) anode points IS the true track, so only the binning
    # location moves. The resulting map is indexed by true position (the
    # forward-map convention of the SCE truth).
    cmap = P.get("correct_map")
    if cmap is not None:
        relax = P.get("iter_relax", 1.0)
        xc = x
        for _ in range(2):
            uc = (xc - ANODE_X[0]) / (ANODE_X[1] - ANODE_X[0])
            yLc = A0y + uc * dy
            zLc = A0z + uc * dz
            jx = np.floor((xc - X_RANGE[0]) / size).astype(np.int64)
            jy = np.floor((yLc - Y_RANGE[0]) / size).astype(np.int64)
            jz = np.floor((zLc - Z_RANGE[0]) / size).astype(np.int64)
            inb = ((jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
                   & (jz >= 0) & (jz < nz))
            dxp = np.zeros_like(x)
            dxp[inb] = np.nan_to_num(cmap[jx[inb], jy[inb], jz[inb], 0])
            xc = x - relax * dxp
        u = (xc - ANODE_X[0]) / (ANODE_X[1] - ANODE_X[0])
        x = xc
        yL = A0y + u * dy
        zL = A0z + u * dz

    ix = np.floor((x - X_RANGE[0]) / size).astype(np.int64)
    iy = np.floor((yL - Y_RANGE[0]) / size).astype(np.int64)
    iz = np.floor((zL - Z_RANGE[0]) / size).astype(np.int64)
    keep = (trk_ok[tkey]
            & np.isfinite(m) & np.isfinite(yL) & np.isfinite(zL)
            & (np.abs(m) <= P["max_r_cm"])
            & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            & (iz >= 0) & (iz < nz))
    ix, iy, iz = ix[keep], iy[keep], iz[keep]
    mk = m[keep]
    nk = nhat[keep]
    stats["n_hits"] = int(keep.sum())
    out = (ix, iy, iz, mk, nk, atick, stats)
    if extra:
        out = out + (dict(plane=plane[keep].astype(np.int8),
                          tpc=tpc[keep].astype(np.int8),
                          x=x[keep], slope=slope[keep],
                          tkey=tkey[keep]),)
    return out


def _iter_keep_mask(ix, iy, iz, mk, nk, delta_prev, iter_resid_cm):
    """Keep mask for iterations >= 2: drop hits whose residual against the
    previous-iteration fit exceeds iter_resid_cm.

    Each hit measures m = n_hat . Delta; the previous iteration inverted its
    voxel to Delta_prev. r = m - n_hat . Delta_prev is the hit's offset from
    that fit (the same quantity plot_voxel_offset_hist.py histograms). Hits in
    voxels with no previous fit (NaN delta_prev -> NaN proj) are always kept --
    there is nothing to reject against, so the voxel is left untouched.
    """
    proj = np.einsum("hc,hc->h", nk, delta_prev[ix, iy, iz])
    return ~np.isfinite(proj) | (np.abs(mk - proj) <= iter_resid_cm)


def _detrend_tracks(m, ext, mode):
    """Subtract, per (track, tpc, plane), the mean ('mean') or a linear-in-x
    fit ('linear') of the m residuals. Targets track-correlated reco-time
    structure (coherent offsets/drifts along a track that no per-hit binned
    correction captures). Returns the detrended copy of m."""
    tkey = ext["tkey"].astype(np.int64)
    gkey = tkey * 6 + ext["tpc"].astype(np.int64) * 3 \
        + ext["plane"].astype(np.int64)
    ngrp = 6 * (int(tkey.max()) + 1) if len(tkey) else 0
    if not ngrp:
        return m
    x = ext["x"]
    okm = np.isfinite(m)
    n_g = np.bincount(gkey, weights=okm.astype(np.float64), minlength=ngrp)
    sm = np.bincount(gkey, weights=np.where(okm, m, 0.0), minlength=ngrp)
    if mode == "mean":
        with np.errstate(invalid="ignore", divide="ignore"):
            a_g = np.where(n_g >= 3, sm / np.maximum(n_g, 1), 0.0)
        return m - a_g[gkey]
    if mode == "linear":
        xz = np.where(okm, x, 0.0)
        sx = np.bincount(gkey, weights=xz, minlength=ngrp)
        sxx = np.bincount(gkey, weights=xz * xz, minlength=ngrp)
        sxm = np.bincount(gkey, weights=np.where(okm, x * m, 0.0),
                          minlength=ngrp)
        det_g = n_g * sxx - sx * sx
        with np.errstate(invalid="ignore", divide="ignore"):
            b_g = np.where((n_g >= 3) & (det_g > 1e-6),
                           (n_g * sxm - sx * sm) / det_g, 0.0)
            a_g = np.where(n_g >= 3,
                           (sm - b_g * sx) / np.maximum(n_g, 1), 0.0)
        return m - (a_g[gkey] + b_g[gkey] * x)
    raise ValueError(f"track_detrend={mode}")


def accumulate_shard(path):
    """Process one parquet shard -> (scalar accumulator, x-cut accumulator
    or None, resid hist, stats).

    For iterations >= 2 (_PARAMS carries a 'delta_prev' map) the per-hit
    residual filter is applied before accumulation, so both the LSQ stack and
    the residual histogram reflect only the kept hits.

    With _PARAMS['min_abs_slope'] set, a SECOND accumulator is filled with
    only the hits passing |d s_view/dx| >= cut (equivalently
    tan(theta_xw) >= cut): the X-component of the per-voxel solve is later
    taken from this high-angle subset (where the reco-hit-time bias
    vanishes), while dy/dz keep the full statistics.
    """
    nx, ny, nz = grid_shape(_PARAMS["size"])
    n_bins, _ = resid_setup()
    dtr = _PARAMS.get("track_detrend")
    if dtr:
        ix, iy, iz, mk, nk, atick, stats, ext = shard_hits(path, extra=True)
        mk_det = _detrend_tracks(mk, ext, dtr)
        # with the X-only cut active, the detrend applies ONLY to the
        # X-subset accumulator (dy/dz keep the raw residuals); standalone
        # it replaces m everywhere (exploratory -- the per-track mean of
        # n.Delta is real signal, quantified by the A-variant control).
        if not _PARAMS.get("min_abs_slope"):
            mk = mk_det
    else:
        ix, iy, iz, mk, nk, atick, stats = shard_hits(path)
        mk_det = mk
    stats["n_hits_dropped"] = 0

    delta_prev = _PARAMS.get("delta_prev")
    if delta_prev is not None and stats["n_hits"]:
        keep = _iter_keep_mask(ix, iy, iz, mk, nk, delta_prev,
                               _PARAMS["iter_resid_cm"])
        stats["n_hits_dropped"] = int((~keep).sum())
        ix, iy, iz = ix[keep], iy[keep], iz[keep]
        mk, nk = mk[keep], nk[keep]
        mk_det = mk_det[keep] if dtr else mk
        stats["n_hits"] = int(keep.sum())

    acc = voxel_invert.make_scalar_accumulator(nx, ny, nz)
    s_cut = _PARAMS.get("min_abs_slope")
    acc_x = (voxel_invert.make_scalar_accumulator(nx, ny, nz)
             if s_cut else None)
    resid_hist = np.zeros((nx, ny, nz, n_bins), dtype=np.int64)
    if stats["n_hits"]:
        voxel_invert.add_scalar_hits(acc, ix, iy, iz, mk, nk)
        if s_cut:
            # |slope| >= cut  <=>  |n_x| >= cut/sqrt(1+cut^2)
            nx_min = s_cut / np.sqrt(1.0 + s_cut * s_cut)
            mx = np.abs(nk[:, 0]) >= nx_min
            stats["n_hits_xcut"] = int(mx.sum())
            if mx.any():
                voxel_invert.add_scalar_hits(acc_x, ix[mx], iy[mx], iz[mx],
                                             mk_det[mx], nk[mx])
        # residual histogram per voxel (raw measurement m, pre-inversion)
        ib = np.floor((mk - (-RESID_HALF)) / RESID_BIN).astype(np.int64)
        inr = (ib >= 0) & (ib < n_bins)
        flat = (ix[inr] * (ny * nz) + iy[inr] * nz + iz[inr]) * n_bins + ib[inr]
        cnt = np.bincount(flat, minlength=nx * ny * nz * n_bins)
        resid_hist += cnt.reshape(nx, ny, nz, n_bins).astype(np.int64)
    return acc, acc_x, resid_hist, atick, stats


def x_label(ix, size):
    lo = X_RANGE[0] + ix * size
    return f"x{int(round(lo)):+04d}"


def _plot_heat(args):
    (comp, ix, grid_slice, count_slice, vrange, min_counts, cmap, label,
     title, png_path, pdf_path, x_lo, x_hi, size, symmetric) = args
    grid = grid_slice.astype(float).copy()
    grid[count_slice < min_counts] = np.nan
    valid = grid[np.isfinite(grid)]
    if valid.size == 0:
        return 0
    fig, ax = plt.subplots(figsize=(9, 6))
    kw = dict(origin="lower", aspect="equal",
              extent=(Z_RANGE[0], Z_RANGE[1], Y_RANGE[0], Y_RANGE[1]),
              cmap=cmap, interpolation="nearest")
    if symmetric:
        kw.update(vmin=-vrange, vmax=vrange)
    else:
        kw.update(vmin=0, vmax=vrange)
    im = ax.imshow(grid, **kw)
    fig.colorbar(im, ax=ax, extend="both" if symmetric else "max", label=label)
    ax.set_xlabel("Z [cm]")
    ax.set_ylabel("Y [cm]")
    ax.set_title(f"{title}\nX in [{x_lo:+.0f}, {x_hi:+.0f}] cm, "
                 f"valid voxels = {valid.size}, voxel = {size:g}$^3$ cm$^3$")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    fig.savefig(pdf_path)
    plt.close(fig)
    return int(valid.size)


def _plot_count(args):
    (ix, count_slice, png_path, pdf_path, x_lo, x_hi, size) = args
    grid = count_slice.astype(float).copy()
    grid[grid == 0] = np.nan
    valid = grid[np.isfinite(grid)]
    if valid.size == 0:
        return 0
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid, origin="lower", aspect="equal",
                   extent=(Z_RANGE[0], Z_RANGE[1], Y_RANGE[0], Y_RANGE[1]),
                   cmap="viridis", interpolation="nearest",
                   norm=matplotlib.colors.LogNorm(vmin=1,
                                                  vmax=max(1, valid.max())))
    fig.colorbar(im, ax=ax, label="hits per voxel")
    ax.set_xlabel("Z [cm]")
    ax.set_ylabel("Y [cm]")
    ax.set_title(f"SBND 2D double-anode voxel hit counts\n"
                 f"X in [{x_lo:+.0f}, {x_hi:+.0f}] cm, "
                 f"populated voxels = {valid.size}, voxel = {size:g}$^3$ cm$^3$")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    fig.savefig(pdf_path)
    plt.close(fig)
    return int(valid.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=INPUT_DIR)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--wire-geom", default=WIRE_GEOM)
    ap.add_argument("--cut-config", default=CUT_CONFIG)
    ap.add_argument("--voxel-size", type=float, default=VOXEL_SIZE)
    ap.add_argument("--vdrift", type=float, default=VDRIFT_DEFAULT,
                    help="drift velocity cm/us (default %(default)s, "
                         "3-plane cathode-stage mean)")
    ap.add_argument("--anode-t0", type=float, default=None,
                    help="anode-peak tick (default: anode_tick_peak from "
                         "the cut config, else %(default)s)")
    ap.add_argument("--min-counts", type=int, default=DEFAULT_MIN_COUNTS)
    ap.add_argument("--max-r-cm", type=float, default=25.0,
                    help="reject hits with |residual| > this before LSQ")
    ap.add_argument("--anode-cut-config", default=ANODE_CUT_CONFIG,
                    help="JSON with the optimized anode-disagreement cut "
                         "(written by optimize_anode_cut.py)")
    ap.add_argument("--max-anode-fom-cm", type=float, default=None,
                    help="anode-disagreement FOM cut in cm; default reads "
                         "cut_cm from --anode-cut-config, else falls back "
                         "to 5.0")
    ap.add_argument("--anode-dt-window", type=float, default=40.0,
                    help="an induction anode hit must be within this many "
                         "ticks of the per-plane anode peak")
    ap.add_argument("--anode-peak-offset-json", default=None,
                    help="JSON with per-(tpc,plane) anode-peak offsets "
                         "(plane peak minus collection peak, ticks): "
                         '{"offsets": [[a,b,c],[d,e,f]]}. Each plane then '
                         "selects the hit closest to its OWN anode peak "
                         "(T_c + offset) instead of the collection peak.")
    ap.add_argument("--cond-max", type=float,
                    default=voxel_invert.DEFAULT_COND_MAX)
    ap.add_argument("--vrange", type=float, default=DEFAULT_VRANGE)
    ap.add_argument("--sigma-vrange", type=float, default=DEFAULT_SIGMA_VRANGE)
    ap.add_argument("--nproc", default="auto")
    ap.add_argument("--accum-nproc", type=int, default=1,
                    help="parallel shards during accumulation (memory-heavy; "
                         "default 1 = serial)")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=1,
                    help="outlier-rejection passes; iteration 1 runs as "
                         "normal, iterations >= 2 re-read the shards and "
                         "drop hits whose residual vs the previous fit "
                         "exceeds --iter-resid-cm, then re-invert")
    ap.add_argument("--iter-resid-cm", type=float, default=4.0,
                    help="for iterations >= 2, drop a hit if "
                         "|m - n_hat . Delta_prev| exceeds this (cm)")
    ap.add_argument("--iter-correct-positions", action="store_true",
                    help="for iterations >= 2, also bin hits at their "
                         "estimated TRUE position (drift coordinate "
                         "corrected by the previous-iteration Delta_x; "
                         "MicroBooNE laser-style map feedback)")
    ap.add_argument("--iter-relax", type=float, default=1.0,
                    help="relaxation factor of the position correction")
    ap.add_argument("--plane-t0-json", default=None,
                    help="JSON with per-(tpc,plane) tick offsets to subtract "
                         "from the raw hit times, e.g. "
                         '{"offsets": [[a,b,c],[d,e,f]]} (tpc-major)')
    ap.add_argument("--anode-extrapolate", action="store_true",
                    help="extrapolate the selected anode hits' view coords "
                         "from their own drift-x to the anode plane before "
                         "the (y,z) fit (removes the anode-hit inset chord "
                         "bias; see toy_closure_2d.py)")
    ap.add_argument("--chord-source-dir", default=None,
                    help="FORENSIC: build each track's chord from the "
                         "same-named shard in this directory instead of the "
                         "input shard (e.g. true-time chord for reco-time "
                         "hits)")
    ap.add_argument("--hit-time-corr", default=None,
                    help="npz LUT of per-hit time corrections "
                         "tau(tpc, plane, |slope|, x) [ticks] "
                         "(angle_hit_time_bias.py --write-lut)")
    ap.add_argument("--min-abs-slope", type=float, default=None,
                    help="X-ONLY angular selection: the dx component of "
                         "each voxel is solved from only the hits with "
                         "|d s_view/dx| = tan(theta_xw) >= this (where the "
                         "reco-hit-time bias vanishes; see angle_bias_mc/), "
                         "conditional on the all-hits dy/dz. dy/dz keep "
                         "full statistics.")
    ap.add_argument("--track-detrend", choices=("mean", "linear"),
                    default=None,
                    help="subtract per-(track,tpc,plane) residual trend "
                         "before accumulation (targets track-correlated "
                         "reco-time structure). With --min-abs-slope it "
                         "applies ONLY to the X-subset accumulator; "
                         "standalone it applies to everything "
                         "(exploratory -- can absorb real signal).")
    ap.add_argument("--x-ramp-correct", choices=("off", "map", "hits"),
                    default="off",
                    help="in-situ linear-ramp (v_AC vs v_DA) dx correction "
                         "(fit_dx_ramp): 'map' subtracts the fitted ramp "
                         "from the final dx layer; 'hits' refits eps each "
                         "iteration and reconstructs subsequent iterations' "
                         "drift coordinates at v_eff = v/(1+eps) per TPC "
                         "(needs --iterations >= 2)")
    ap.add_argument("--x-ramp-joint", action="store_true",
                    help="fit a single TPC-AVERAGED ramp eps (shared across "
                         "both TPCs) instead of per-TPC; the common eps is "
                         "the detector artifact, leaving a genuine per-TPC "
                         "drift-velocity difference in the map. Use for data.")
    ap.add_argument("--x-ramp-cathode-offset", action="store_true",
                    help="add the shared mean-cathode-offset term c*E(d) to "
                         "the joint ramp fit (fit_dx_ramp --cathode-offset). "
                         "The APPLIED correction stays the shared eps; c is "
                         "MEASURED per iteration and stored (the per-TPC "
                         "drift-velocity-difference diagnostic). Implies "
                         "--x-ramp-joint.")
    ap.add_argument("--no-counts-maps", action="store_true")
    ap.add_argument("--no-slice-maps", action="store_true",
                    help="skip all per-X-slice heatmap rendering (campaign "
                         "fast mode; npz + summary plots only)")
    ap.add_argument("--no-save-npz", action="store_true")
    ap.add_argument("--dump-accumulator", default=None,
                    help="also write the raw per-voxel normal equations "
                         "(nnT=M, sum_nm=v, sum_m2, n) of the FINAL iteration "
                         "to this npz, for offline inversion/regularization "
                         "studies. Additive; does not change normal outputs.")
    args = ap.parse_args()
    if args.x_ramp_cathode_offset:
        args.x_ramp_joint = True

    if os.path.isdir(args.input):
        shards = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    else:
        shards = sorted(glob.glob(args.input))
    if not shards:
        print(f"No parquet shards at {args.input}", file=sys.stderr)
        sys.exit(1)
    if args.max_shards:
        shards = shards[:args.max_shards]

    t0 = args.anode_t0
    if t0 is None:
        try:
            t0 = float(json.load(open(args.cut_config))["anode_tick_peak"])
        except Exception:
            t0 = ANODE_T0_DEFAULT
    nproc = cpu_count() if args.nproc == "auto" else int(args.nproc)
    size = args.voxel_size
    nx, ny, nz = grid_shape(size)
    n_bins, resid_edges = resid_setup()
    n_at, atick_edges = atick_setup()

    print(f"reading {len(shards)} shards from {args.input}", flush=True)
    print(f"voxel grid: {nx} x {ny} x {nz} = {nx*ny*nz:,} voxels", flush=True)
    print(f"anode t0 = {t0:.2f} ticks, v_drift = {args.vdrift} cm/us", flush=True)

    max_anode_fom = args.max_anode_fom_cm
    if max_anode_fom is None:
        try:
            max_anode_fom = float(
                json.load(open(args.anode_cut_config))["cut_cm"])
            print(f"anode-disagreement cut: {max_anode_fom:.3f} cm "
                  f"(from {args.anode_cut_config})", flush=True)
        except Exception:
            max_anode_fom = 5.0
            print(f"anode-disagreement cut: {max_anode_fom:.3f} cm "
                  f"(fallback default -- {args.anode_cut_config} not found)",
                  flush=True)

    geom = load_wire_geom(args.wire_geom)
    params = dict(size=size, t0=t0, vdrift=args.vdrift,
                  max_r_cm=args.max_r_cm,
                  max_anode_fom=max_anode_fom,
                  anode_dt_window=args.anode_dt_window)
    if args.plane_t0_json:
        pt0 = np.asarray(json.load(open(args.plane_t0_json))["offsets"],
                         dtype=np.float64)
        assert pt0.shape == (2, 3), "plane-t0 offsets must be (2 tpc, 3 plane)"
        params["plane_t0"] = pt0
        print(f"per-plane t0 offsets [ticks]: {pt0.tolist()}", flush=True)
    if args.anode_peak_offset_json:
        apo = np.asarray(json.load(open(args.anode_peak_offset_json))["offsets"],
                         dtype=np.float64)
        assert apo.shape == (2, 3), "anode-peak offsets must be (2 tpc, 3 plane)"
        params["anode_peak_offset"] = apo
        print(f"per-plane anode-peak offsets [ticks]: {apo.tolist()} "
              f"(select closest to T_c + offset)", flush=True)
    if args.anode_extrapolate:
        params["anode_extrapolate"] = True
        print("anode-hit extrapolation to the anode plane: ON", flush=True)
    if args.chord_source_dir:
        params["chord_source_dir"] = args.chord_source_dir
        print(f"FORENSIC chord source: {args.chord_source_dir}", flush=True)
    if args.min_abs_slope:
        params["min_abs_slope"] = args.min_abs_slope
        print(f"X-only angular selection: dx from hits with tan(theta_xw) "
              f"= |ds/dx| >= {args.min_abs_slope} (dy/dz from all hits)",
              flush=True)
    if args.track_detrend:
        params["track_detrend"] = args.track_detrend
        scope = ("X-subset accumulator only" if args.min_abs_slope
                 else "ALL accumulated hits (exploratory)")
        print(f"per-track residual detrend: {args.track_detrend} "
              f"({scope})", flush=True)
    if args.hit_time_corr:
        hl = np.load(args.hit_time_corr)
        lut_arr = hl["lut_ticks"].astype(np.float64)
        params["hit_time_corr"] = (lut_arr, hl["slope_edges"], hl["x_edges"],
                                   hl["wslope_edges"] if "wslope_edges" in hl
                                   else None)
        print(f"per-hit time correction LUT: {args.hit_time_corr} "
              f"({lut_arr.ndim - 3}-angle, range {lut_arr.min():+.1f}.."
              f"{lut_arr.max():+.1f} ticks)", flush=True)

    # ---- iteration loop: iteration 1 accumulates every hit; iterations >= 2
    # re-read the shards dropping hits whose residual vs the previous fit
    # exceeds --iter-resid-cm, then re-invert. The accumulator stores only
    # moments (not hits), so each iteration is a fresh pass over the shards.
    saved = {}                       # per-iteration outputs kept for the npz
    delta_mu = None                  # iteration-(it-1) map, fed forward
    ramp_eps_tot = np.zeros(2)       # cumulative hits-mode ramp correction
    ramp_cathode_offset = []         # measured mean cathode offset c per iter
    if args.x_ramp_correct == "hits" and args.iterations < 2:
        print("WARNING: --x-ramp-correct hits needs --iterations >= 2; "
              "no ramp correction will be applied", flush=True)

    for it in range(1, args.iterations + 1):
        iter_params = dict(params)
        if it >= 2:
            iter_params["delta_prev"] = delta_mu
            iter_params["iter_resid_cm"] = args.iter_resid_cm
            if args.iter_correct_positions:
                iter_params["correct_map"] = delta_mu
                iter_params["iter_relax"] = args.iter_relax
            if args.x_ramp_correct == "hits":
                iter_params["x_ramp_eps"] = ramp_eps_tot.copy()

        acc = voxel_invert.make_scalar_accumulator(nx, ny, nz)
        acc_x = (voxel_invert.make_scalar_accumulator(nx, ny, nz)
                 if args.min_abs_slope else None)
        resid_hist = np.zeros((nx, ny, nz, n_bins), dtype=np.int64)
        atick = {"selected": np.zeros((2, 3, n_at), dtype=np.int64),
                 "earliest": np.zeros((2, 3, n_at), dtype=np.int64)}
        n_tracks = n_used = n_hits = n_dropped = n_xcut = 0

        def _consume(a, ax_, rh, at, st):
            nonlocal n_tracks, n_used, n_hits, n_dropped, n_xcut
            voxel_invert.combine_scalar_accumulators(acc, a)
            if acc_x is not None and ax_ is not None:
                voxel_invert.combine_scalar_accumulators(acc_x, ax_)
            resid_hist[...] += rh
            for k in atick:
                atick[k] += at[k]
            n_tracks += st["n_tracks"]; n_used += st["n_tracks_used"]
            n_hits += st["n_hits"]; n_dropped += st.get("n_hits_dropped", 0)
            n_xcut += st.get("n_hits_xcut", 0)

        if args.accum_nproc > 1:
            with Pool(args.accum_nproc, initializer=_worker_init,
                      initargs=(geom, iter_params)) as pool:
                for out in tqdm(pool.imap_unordered(accumulate_shard, shards),
                                total=len(shards), unit="shard",
                                file=sys.stderr):
                    _consume(*out)
        else:
            _worker_init(geom, iter_params)
            for path in tqdm(shards, unit="shard", file=sys.stderr):
                _consume(*accumulate_shard(path))

        tag = f"[iter {it}/{args.iterations}]"
        print(f"\n{tag} tracks: {n_tracks:,} total, {n_used:,} used "
              f"({100.0*n_used/max(1,n_tracks):.1f}%)", flush=True)
        if it >= 2:
            n_seen = n_hits + n_dropped
            print(f"{tag} hits dropped by |r| > {args.iter_resid_cm:g} cm: "
                  f"{n_dropped:,} ({100.0*n_dropped/max(1,n_seen):.2f}% of "
                  f"{n_seen:,})", flush=True)
        print(f"{tag} hits accumulated: {n_hits:,}", flush=True)
        if acc_x is not None:
            print(f"{tag} hits passing |ds/dx| >= {args.min_abs_slope:g} "
                  f"(X-component subset): {n_xcut:,} "
                  f"({100.0*n_xcut/max(1,n_hits):.1f}%)", flush=True)
        nvox_pop = int((acc["n"] >= args.min_counts).sum())
        print(f"{tag} voxels with >= {args.min_counts} hits: "
              f"{nvox_pop:,} / {nx*ny*nz:,}", flush=True)

        res = voxel_invert.invert_scalar_accumulator(
            acc, min_n=args.min_counts, cond_max=args.cond_max)
        delta_mu = res["delta_mu"]
        delta_cov = res["delta_cov"]
        cond = res["cond"]
        count = res["n"]
        n_inv = int(np.isfinite(delta_mu[..., 0]).sum())
        print(f"{tag} voxels inverted: {n_inv:,} "
              f"({nvox_pop - n_inv:,} rejected by cond > {args.cond_max:.0f})",
              flush=True)

        # --- X-only angular selection: replace the dx component with the
        # conditional LSQ solution from the high-theta_xw subset, given the
        # all-hits (dy, dz):  dx = (v_x' - M_xy' dy - M_xz' dz) / M_xx'
        # (primes = cut accumulator). dy/dz keep full statistics; only hits
        # with tan(theta_xw) >= cut inform dx (flat reco-time bias region).
        if acc_x is not None:
            Tx = acc_x["nnT"]
            vx_ = acc_x["sum_nm"][..., 0]
            n_x_cnt = acc_x["n"]
            dy_ = delta_mu[..., 1]
            dz_ = delta_mu[..., 2]
            Mxx = Tx[..., 0]; Mxy = Tx[..., 3]; Mxz = Tx[..., 4]
            okx = (np.isfinite(dy_) & (n_x_cnt >= args.min_counts)
                   & (Mxx > 1e-6))
            with np.errstate(invalid="ignore", divide="ignore"):
                dx_new = np.where(
                    okx, (vx_ - Mxy * dy_ - Mxz * dz_) / Mxx, np.nan)
                # conditional-LSQ variance of dx from the cut subset
                rss = (acc_x["sum_m2"]
                       - 2 * (dx_new * vx_ + dy_ * acc_x["sum_nm"][..., 1]
                              + dz_ * acc_x["sum_nm"][..., 2])
                       + (dx_new ** 2 * Mxx + dy_ ** 2 * Tx[..., 1]
                          + dz_ ** 2 * Tx[..., 2]
                          + 2 * (dx_new * dy_ * Mxy + dx_new * dz_ * Mxz
                                 + dy_ * dz_ * Tx[..., 5])))
                var_x = np.where(okx, np.maximum(rss, 0.0)
                                 / np.maximum(n_x_cnt - 1, 1) / Mxx, np.nan)
            delta_mu[..., 0] = dx_new
            delta_cov[..., 0] = var_x
            n_invx = int(np.isfinite(dx_new).sum())
            print(f"{tag} X-component voxels from the theta_xw subset: "
                  f"{n_invx:,} (>= {args.min_counts} cut hits)", flush=True)

        if it == 1 and args.iterations >= 2:
            saved["delta_mu_all_iter1"] = delta_mu.copy()
            saved["count_all_iter1"] = count.astype(np.int64)
        # keep every iteration's map for the per-iteration ramp-calibration
        # 1D profile plots (make_iter_profiles.py)
        if args.iterations >= 2:
            saved[f"delta_mu_all_iter{it}"] = delta_mu.copy()

        # hits-mode ramp correction: fit eps on this iteration's map and
        # fold it into the cumulative drift rescaling for the next pass
        if args.x_ramp_correct == "hits" and it < args.iterations:
            import fit_dx_ramp as FR
            md = dict(delta_mu_all=delta_mu, delta_cov_all=delta_cov,
                      cond_all=cond, count_all=count,
                      x_edges=np.linspace(*X_RANGE, nx + 1),
                      y_edges=np.linspace(*Y_RANGE, ny + 1),
                      z_edges=np.linspace(*Z_RANGE, nz + 1),
                      vdrift=np.float32(args.vdrift))
            rr = FR.fit_ramp_npz(md, min_counts=args.min_counts,
                                 joint=args.x_ramp_joint,
                                 cathode_offset=args.x_ramp_cathode_offset)
            eps_new = np.nan_to_num(np.array(
                [rr["fits"][0].get("eps", np.nan),
                 rr["fits"][1].get("eps", np.nan)]))
            ramp_eps_tot = (1.0 + ramp_eps_tot) * (1.0 + eps_new) - 1.0
            print(f"{tag} ramp fit: eps = {eps_new[0]*100:+.3f} / "
                  f"{eps_new[1]*100:+.3f} cm/m; cumulative "
                  f"{ramp_eps_tot[0]*100:+.3f} / {ramp_eps_tot[1]*100:+.3f} "
                  f"-> next pass at v_eff = "
                  f"{args.vdrift/(1+ramp_eps_tot[0]):.5f} / "
                  f"{args.vdrift/(1+ramp_eps_tot[1]):.5f} cm/us", flush=True)
            if args.x_ramp_cathode_offset:
                c = float(rr["fits"][0].get("cathode_offset", np.nan))
                ce = float(rr["fits"][0].get("cathode_offset_err", np.nan))
                ramp_cathode_offset.append(c)
                v0 = args.vdrift / (1 + ramp_eps_tot[0] + c / 200.0)
                v1 = args.vdrift / (1 + ramp_eps_tot[0] - c / 200.0)
                print(f"{tag} mean cathode offset c = {c:+.4f} ± {ce:.4f} cm "
                      f"(measured, NOT applied); implied per-TPC v_DA = "
                      f"{v0:.5f} / {v1:.5f} cm/us", flush=True)
            os.makedirs(args.outdir, exist_ok=True)
            FR.plot_fit(rr,
                        f"x-ramp fit, iteration {it} (eps "
                        f"{eps_new[0]*100:+.3f}/{eps_new[1]*100:+.3f} cm/m; "
                        f"cumulative {ramp_eps_tot[0]*100:+.3f}/"
                        f"{ramp_eps_tot[1]*100:+.3f})",
                        os.path.join(args.outdir,
                                     f"ramp_fit_iter{it:02d}.png"))

    # ---- ramp correction bookkeeping / map-mode application -----------------
    ramp_eps_applied = np.full(2, np.nan)
    ramp_eps_resid = np.full(2, np.nan)
    if args.x_ramp_correct != "off":
        import fit_dx_ramp as FR
        md = dict(delta_mu_all=delta_mu, delta_cov_all=delta_cov,
                  cond_all=cond, count_all=count,
                  x_edges=np.linspace(*X_RANGE, nx + 1),
                  y_edges=np.linspace(*Y_RANGE, ny + 1),
                  z_edges=np.linspace(*Z_RANGE, nz + 1),
                  vdrift=np.float32(args.vdrift))
        rr = FR.fit_ramp_npz(md, min_counts=args.min_counts,
                             joint=args.x_ramp_joint,
                             cathode_offset=args.x_ramp_cathode_offset)
        eps_fin = np.array([rr["fits"][0].get("eps", np.nan),
                            rr["fits"][1].get("eps", np.nan)])
        os.makedirs(args.outdir, exist_ok=True)
        FR.plot_fit(rr,
                    f"x-ramp fit, FINAL map ({args.x_ramp_correct} mode; "
                    f"eps {eps_fin[0]*100:+.3f}/{eps_fin[1]*100:+.3f} cm/m)",
                    os.path.join(args.outdir, "ramp_fit_final.png"))
        if args.x_ramp_cathode_offset:
            c = float(rr["fits"][0].get("cathode_offset", np.nan))
            ce = float(rr["fits"][0].get("cathode_offset_err", np.nan))
            ramp_cathode_offset.append(c)
            print(f"[x-ramp] FINAL mean cathode offset c = {c:+.4f} ± "
                  f"{ce:.4f} cm (measured diagnostic; correction stays "
                  f"shared eps)", flush=True)
        if args.x_ramp_correct == "map":
            ramp_eps_applied = eps_fin
            vxc = X_RANGE[0] + (np.arange(nx) + 0.5) * size
            rampv = FR.ramp_grid(vxc, np.nan_to_num(eps_fin))
            delta_mu[..., 0] = delta_mu[..., 0] - rampv[:, None, None]
            print(f"[x-ramp map] subtracted eps = {eps_fin[0]*100:+.3f} / "
                  f"{eps_fin[1]*100:+.3f} cm/m from the dx layer "
                  f"(v_DA = {args.vdrift/(1+eps_fin[0]):.5f} / "
                  f"{args.vdrift/(1+eps_fin[1]):.5f} cm/us)", flush=True)
        else:
            ramp_eps_applied = ramp_eps_tot
            ramp_eps_resid = eps_fin
            print(f"[x-ramp hits] applied cumulative eps = "
                  f"{ramp_eps_tot[0]*100:+.3f} / {ramp_eps_tot[1]*100:+.3f} "
                  f"cm/m; residual ramp of the final map = "
                  f"{eps_fin[0]*100:+.3f} / {eps_fin[1]*100:+.3f} cm/m",
                  flush=True)

    # Optional: dump the raw per-voxel normal equations of the final
    # iteration so inversion/regularization strategies can be explored
    # offline (M and v are the sufficient statistics for any linear solve).
    if args.dump_accumulator:
        dpar = os.path.dirname(args.dump_accumulator)
        if dpar:
            os.makedirs(dpar, exist_ok=True)
        dump = dict(
            nnT=acc["nnT"], sum_nm=acc["sum_nm"], sum_m2=acc["sum_m2"],
            n=acc["n"],
            x_edges=np.linspace(*X_RANGE, nx + 1),
            y_edges=np.linspace(*Y_RANGE, ny + 1),
            z_edges=np.linspace(*Z_RANGE, nz + 1),
            voxel_size=np.float32(size), vdrift=np.float32(args.vdrift))
        if acc_x is not None:
            dump.update(nnT_x=acc_x["nnT"], sum_nm_x=acc_x["sum_nm"],
                        sum_m2_x=acc_x["sum_m2"], n_x=acc_x["n"],
                        min_abs_slope_x=np.float32(args.min_abs_slope))
        np.savez_compressed(args.dump_accumulator, **dump)
        print(f"dumped accumulator -> {args.dump_accumulator}"
              + (" (+ X-cut accumulator)" if acc_x is not None else ""),
              flush=True)

    # delta_mu/delta_cov/cond/count/resid_hist/atick now hold the FINAL
    # iteration; everything below operates on that map unchanged.
    mu = {c: delta_mu[..., i] for i, c in enumerate(COMPONENTS)}
    sigma = {c: np.sqrt(np.maximum(0.0, delta_cov[..., i]))
             for i, c in enumerate(COMPONENTS)}

    os.makedirs(args.outdir, exist_ok=True)

    # ---- Delta-hat maps ----
    tasks = []
    plot_comps = () if args.no_slice_maps else COMPONENTS
    for comp in plot_comps:
        pdir = os.path.join(args.outdir, "all", comp, "png")
        qdir = os.path.join(args.outdir, "all", comp, "pdf")
        os.makedirs(pdir, exist_ok=True); os.makedirs(qdir, exist_ok=True)
        for ix in range(nx):
            x_lo = X_RANGE[0] + ix * size
            tag = x_label(ix, size)
            tasks.append((comp, ix, mu[comp][ix], count[ix], args.vrange,
                          args.min_counts, HEATMAP_CMAP,
                          f"LSQ $\\hat{{\\Delta}}_{{{comp[1]}}}$ [cm]",
                          f"SBND 2D double-anode $\\hat{{\\Delta}}_{{{comp[1]}}}$",
                          os.path.join(pdir, f"{tag}.png"),
                          os.path.join(qdir, f"{tag}.pdf"),
                          x_lo, x_lo + size, size, True))
    # ---- sigma maps ----
    for comp in plot_comps:
        pdir = os.path.join(args.outdir, "sigma", "all", comp, "png")
        qdir = os.path.join(args.outdir, "sigma", "all", comp, "pdf")
        os.makedirs(pdir, exist_ok=True); os.makedirs(qdir, exist_ok=True)
        for ix in range(nx):
            x_lo = X_RANGE[0] + ix * size
            tag = x_label(ix, size)
            tasks.append((comp, ix, sigma[comp][ix], count[ix],
                          args.sigma_vrange, args.min_counts, SIGMA_CMAP,
                          f"LSQ $\\hat{{\\sigma}}_{{{comp[1]}}}$ [cm]",
                          f"SBND 2D double-anode $\\hat{{\\sigma}}_{{{comp[1]}}}$",
                          os.path.join(pdir, f"{tag}.png"),
                          os.path.join(qdir, f"{tag}.pdf"),
                          x_lo, x_lo + size, size, False))
    # ---- condition-number maps ----
    pdir = os.path.join(args.outdir, "cond", "all", "png")
    qdir = os.path.join(args.outdir, "cond", "all", "pdf")
    os.makedirs(pdir, exist_ok=True); os.makedirs(qdir, exist_ok=True)
    for ix in range(nx) if not args.no_slice_maps else ():
        x_lo = X_RANGE[0] + ix * size
        tag = x_label(ix, size)
        tasks.append(("cond", ix, cond[ix], count[ix], args.cond_max,
                      args.min_counts, SIGMA_CMAP, "LSQ matrix cond. number",
                      "SBND 2D double-anode LSQ condition number",
                      os.path.join(pdir, f"{tag}.png"),
                      os.path.join(qdir, f"{tag}.pdf"),
                      x_lo, x_lo + size, size, False))

    print(f"\nplotting {len(tasks)} map slices on {nproc} workers", flush=True)
    if nproc > 1:
        with Pool(nproc) as pool:
            for _ in tqdm(pool.imap_unordered(_plot_heat, tasks, chunksize=4),
                          total=len(tasks), unit="map", file=sys.stderr):
                pass
    else:
        for t in tqdm(tasks, unit="map", file=sys.stderr):
            _plot_heat(t)

    # ---- count maps ----
    if not args.no_counts_maps:
        pdir = os.path.join(args.outdir, "counts", "all", "png")
        qdir = os.path.join(args.outdir, "counts", "all", "pdf")
        os.makedirs(pdir, exist_ok=True); os.makedirs(qdir, exist_ok=True)
        ctasks = []
        for ix in range(nx):
            x_lo = X_RANGE[0] + ix * size
            tag = x_label(ix, size)
            ctasks.append((ix, count[ix],
                           os.path.join(pdir, f"{tag}.png"),
                           os.path.join(qdir, f"{tag}.pdf"),
                           x_lo, x_lo + size, size))
        if nproc > 1:
            with Pool(nproc) as pool:
                for _ in tqdm(pool.imap_unordered(_plot_count, ctasks,
                                                  chunksize=4),
                              total=len(ctasks), unit="map", file=sys.stderr):
                    pass
        else:
            for t in ctasks:
                _plot_count(t)

    # ---- global residual distribution ----
    centers = 0.5 * (resid_edges[:-1] + resid_edges[1:])
    tot = resid_hist.reshape(-1, n_bins).sum(axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(centers, tot, where="mid", color="navy")
    ax.set_xlabel("per-hit residual $m = \\hat{n}\\cdot\\Delta$ [cm]")
    ax.set_ylabel(f"hits / {RESID_BIN:g} cm")
    ax.set_yscale("log")
    ax.axvline(0, color="grey", ls=":", lw=0.6)
    ax.set_title(f"SBND 2D double-anode per-hit residual ({tot.sum():,.0f} hits)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "residual_distribution.png"), dpi=120)
    plt.close(fig)

    # ---- anode-hit tick diagnostic (per tpc, per plane) ----
    ac = 0.5 * (atick_edges[:-1] + atick_edges[1:])
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for t in (0, 1):
        for p in (0, 1, 2):
            ax = axes[t, p]
            he = atick["earliest"][t, p].astype(float)
            hs = atick["selected"][t, p].astype(float)
            ax.step(ac, he, where="mid", color="0.6", lw=1.0,
                    label=f"earliest hit (N={int(he.sum()):,})")
            ax.step(ac, hs, where="mid", color="navy", lw=1.2,
                    label=f"selected anode (N={int(hs.sum()):,})")
            ax.axvspan(-args.anode_dt_window, args.anode_dt_window,
                       color="gold", alpha=0.15)
            ax.axvline(0, color="grey", ls=":", lw=0.6)
            ax.set_yscale("log")
            ax.set_ylim(0.5, None)
            ax.set_title(f"TPC{t} plane{p}"
                         + ("  (collection)" if p == 2 else "  (induction)"))
            ax.grid(alpha=0.3)
            if t == 1:
                ax.set_xlabel("anode-hit tick - collection anode $T_c$")
            if p == 0:
                ax.set_ylabel("tracks")
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("SBND 2D double-anode: per-plane anode-hit tick relative to "
                 "the collection-plane anode (gold = selection window)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "anode_tick_diagnostic.png"), dpi=120)
    fig.savefig(os.path.join(args.outdir, "anode_tick_diagnostic.pdf"))
    plt.close(fig)

    # ---- npz ----
    if not args.no_save_npz:
        npz = os.path.join(args.outdir, "voxel_data.npz")
        save = {
            "voxel_size": np.float32(size),
            "x_edges": np.linspace(*X_RANGE, nx + 1),
            "y_edges": np.linspace(*Y_RANGE, ny + 1),
            "z_edges": np.linspace(*Z_RANGE, nz + 1),
            "resid_edges": resid_edges,
            "atick_edges": atick_edges,
            "min_counts": np.int32(args.min_counts),
            "vdrift": np.float32(args.vdrift),
            "anode_t0": np.float32(t0),
            "n_iterations": np.int32(args.iterations),
            "iter_resid_cm": np.float32(args.iter_resid_cm),
            "iter_correct_positions": np.bool_(args.iter_correct_positions),
            "min_abs_slope_x": np.float32(args.min_abs_slope or 0.0),
            "x_ramp_mode": np.bytes_(args.x_ramp_correct.encode()),
            "x_ramp_eps": ramp_eps_applied.astype(np.float32),
            "x_ramp_eps_resid": ramp_eps_resid.astype(np.float32),
            "x_ramp_cathode_offset": np.asarray(ramp_cathode_offset,
                                                dtype=np.float32),
            "count_all": count.astype(np.int64),
            "delta_mu_all": delta_mu,
            "delta_cov_all": delta_cov,
            "cond_all": cond,
            "hist_all_resid": resid_hist.astype(np.int32),
            "anode_dtick_selected": atick["selected"],
            "anode_dtick_earliest": atick["earliest"],
        }
        if acc_x is not None:
            save["count_x_all"] = acc_x["n"].astype(np.int64)
        for i, c in enumerate(COMPONENTS):
            save[f"mu_all_{c}"] = mu[c]
            save[f"sigma_all_{c}"] = sigma[c]
        # for a multi-iteration run, also keep iteration 1 (pre-rejection) so
        # the before/after comparison can be made; delta_mu_all stays the
        # FINAL map. Absent for a 1-iteration run (iteration 1 == final).
        save.update(saved)
        np.savez_compressed(npz, **save)
        print(f"\nsaved {npz} ({os.path.getsize(npz)/1e6:.1f} MB)", flush=True)

    print(f"\nwrote {args.outdir}/", flush=True)


if __name__ == "__main__":
    main()
