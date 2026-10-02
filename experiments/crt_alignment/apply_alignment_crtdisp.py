"""Apply a CRT alignment correction to crtdisp_df rows.

The crtdisp_df has columns:
  hit_x, hit_y, hit_z          — TPC space-point position
  crt_start_x/y/z, crt_end_x/y/z — CRT track endpoints (= the two CRT space-points)
  crt_x/y/z                     — foot of perpendicular on the line through start..end
  dx, dy, dz                    — hit_xyz - crt_xyz (the residual that becomes
                                  field-distortion when CRT alignment is correct)

This module re-derives crt_x/y/z and dx/dy/dz under a supplied alignment
correction (per-wall or per-module) and an optional TPC1 anode shift.

Inputs to ``apply_corrections``:
  start_xyz, end_xyz: (N, 3) arrays of CRT space-point positions
  hit_xyz:            (N, 3) array of TPC-hit positions
  alignment:          dict from ``load_alignment(csv_path)``

Returns:
  new_dxyz: (N, 3) array of corrected residual components
  (the voxel position is the original hit_xyz, unchanged)

Coordinate convention notes:
  - SBND central cathode at x=0; TPC0 anode at x=-200, TPC1 anode at x=+200.
  - We apply the TPC1 anode rigid-body correction to TPC1 hits ONLY (hit_x>0).
  - TPC0 anode is pinned (gauge fix), so TPC0 hits are not shifted.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def _rodrigues_batch(dR, v):
    """Rotate (N, 3) vectors v by axis-angle (N, 3) dR. Smooth at |dR|→0."""
    ang2 = np.sum(dR ** 2, axis=1, keepdims=True)
    small = ang2 < 1e-10
    safe_ang2 = np.where(small, 1.0, ang2)
    safe_ang = np.sqrt(safe_ang2)
    s_full = np.sin(safe_ang) / safe_ang
    c_full = (1.0 - np.cos(safe_ang)) / safe_ang2
    s = np.where(small, 1.0 - ang2 / 6.0, s_full)
    c = np.where(small, 0.5 - ang2 / 24.0, c_full)
    cross_dR_v = np.cross(dR, v)
    return v + s * cross_dR_v + c * np.cross(dR, cross_dR_v)


def load_alignment(csv_path):
    """Load an align_crt.py results CSV (per-wall or per-module).

    Returns dict with:
      mode:    'wall' or 'module'
      ids:     (M,) integer ids (FEB MAC for module mode, tagger for wall mode)
      centroids: (M, 3) rotation pivots
      dT:      (M, 3) translations (cm)
      dR:      (M, 3) axis-angle rotations (rad)
      kdtree:  cKDTree over centroids for nearest-neighbour lookup
      anode_centroids: (2, 3)
      anode_dT:        (2, 3) — TPC0 is zero (pinned), TPC1 fitted
      anode_dR:        (2, 3) — same
    """
    df = pd.read_csv(csv_path)
    if "centroid_x" not in df.columns:
        raise ValueError(
            f"{csv_path} doesn't have centroid_x/y/z columns — "
            "re-run align_crt.py with the version that saves centroids.")
    is_wall  = (df["kind"] == "wall").any()
    is_mod   = (df["kind"] == "module").any()
    if is_wall and is_mod:
        raise ValueError(f"{csv_path} contains both 'wall' and 'module' rows; "
                         "expected only one.")
    if is_wall:
        mode = "wall"
        units = df[df["kind"] == "wall"].copy()
    elif is_mod:
        mode = "module"
        units = df[df["kind"] == "module"].copy()
    else:
        raise ValueError(f"{csv_path} has no wall/module rows.")
    ids = units["module_id"].to_numpy(np.int64)
    centroids = units[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(np.float64)
    dT = units[["dx", "dy", "dz"]].to_numpy(np.float64)
    dR = units[["dthx", "dthy", "dthz"]].to_numpy(np.float64)

    anode = df[df["kind"] == "anode"].copy().sort_values("module_id")
    if len(anode) != 2:
        raise ValueError(f"{csv_path} should have exactly 2 anode rows, "
                         f"got {len(anode)}.")
    anode_centroids = anode[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(np.float64)
    anode_dT = anode[["dx", "dy", "dz"]].to_numpy(np.float64)
    anode_dR = anode[["dthx", "dthy", "dthz"]].to_numpy(np.float64)

    kdtree = cKDTree(centroids)
    return {
        "mode": mode,
        "ids": ids,
        "centroids": centroids,
        "dT": dT, "dR": dR,
        "kdtree": kdtree,
        "anode_centroids": anode_centroids,
        "anode_dT": anode_dT, "anode_dR": anode_dR,
        "csv_path": csv_path,
    }


def _apply_unit_transform(points, alignment):
    """Apply per-row module/wall correction to a (N, 3) point array.

    For each point we find the nearest centroid (its assumed unit), apply
    the rigid-body transform: P' = R(δθ) (P - C) + C + δT.
    """
    _, idx = alignment["kdtree"].query(points, k=1)
    c = alignment["centroids"][idx]
    dT = alignment["dT"][idx]
    dR = alignment["dR"][idx]
    return _rodrigues_batch(dR, points - c) + c + dT


def _apply_anode_transform(points, hit_x, alignment):
    """Apply the TPC1 anode rigid-body transform to TPC1 hits only.
    TPC1 selection: hit_x > 0 (central cathode at 0; TPC1 anode at +200)."""
    is_tpc1 = hit_x > 0
    if not is_tpc1.any():
        return points
    out = points.copy()
    sub = points[is_tpc1]
    c = alignment["anode_centroids"][1]
    dT = alignment["anode_dT"][1]
    dR = alignment["anode_dR"][1]
    out[is_tpc1] = _rodrigues_batch(np.broadcast_to(dR, sub.shape).copy(),
                                    sub - c) + c + dT
    return out


def apply_corrections(start_xyz, end_xyz, hit_xyz, alignment,
                      apply_anode_to_tpc1=True):
    """Recompute (dx, dy, dz) for each crtdisp_df row under the alignment.

    Returns (dx, dy, dz) arrays of shape (N,) each. The voxel position
    is the ORIGINAL hit_xyz which the caller continues to use; only the
    residual is recomputed.
    """
    start_xyz = np.asarray(start_xyz, dtype=np.float64)
    end_xyz   = np.asarray(end_xyz, dtype=np.float64)
    hit_xyz   = np.asarray(hit_xyz, dtype=np.float64)

    # 1. Move CRT endpoints by their per-unit alignment correction.
    P1 = _apply_unit_transform(start_xyz, alignment)
    P2 = _apply_unit_transform(end_xyz, alignment)

    # 2. (Optional) move TPC1 hit positions by the TPC1 anode rigid transform.
    #    We use the ORIGINAL hit_x to decide which TPC the hit is in (the
    #    decision is geometric, not affected by the anode shift).
    if apply_anode_to_tpc1:
        hit_for_resid = _apply_anode_transform(hit_xyz, hit_xyz[:, 0],
                                               alignment)
    else:
        hit_for_resid = hit_xyz

    # 3. Foot of perpendicular from corrected_hit to the line through (P1, P2).
    line = P2 - P1
    L2 = np.sum(line * line, axis=1, keepdims=True)
    L2 = np.maximum(L2, 1e-9)
    delta = hit_for_resid - P1
    t = np.sum(delta * line, axis=1, keepdims=True) / L2
    foot = P1 + t * line

    d = hit_for_resid - foot
    return d[:, 0], d[:, 1], d[:, 2]
