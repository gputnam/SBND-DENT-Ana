"""File / parquet / cut-config I/O helpers."""

import glob
import json
import os

import pandas as pd

from crtlib.constants import XROOT_PREFIX

# Module-level toggle. Scripts may flip this from --no-xroot.
USE_XROOT = True


def pnfs_to_xroot(path, use_xroot=None):
    """Translate /pnfs/... → xroot://... unless `scratch` is in the path.

    If `use_xroot` is None, defer to the module-level USE_XROOT toggle.
    """
    if use_xroot is None:
        use_xroot = USE_XROOT
    if use_xroot and path.startswith("/pnfs") and "scratch" not in path:
        return XROOT_PREFIX + path[len("/pnfs"):]
    return path


def set_use_xroot(flag):
    """Update the module-level USE_XROOT toggle."""
    global USE_XROOT
    USE_XROOT = bool(flag)


def load_align_df(input_path):
    """Load the per-track align_df from a parquet file or shard directory."""
    if os.path.isdir(input_path):
        shards = sorted(glob.glob(os.path.join(input_path, "*.parquet")))
        if not shards:
            raise FileNotFoundError(f"no parquet shards in {input_path}")
        return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    return pd.read_parquet(input_path)


def load_cut_config(path):
    """Load the per-TPC anode-piercing cut window from cut_config_per_tpc.json.

    Returns dict with keys ``tmin_lo``, ``tmin_hi``. Falls back to a legacy
    envelope-based config (``intime_anode_crosser`` block) if the per-TPC
    keys are absent.
    """
    with open(path) as f:
        cfg = json.load(f)
    if "tpc_t_min_lo" in cfg:
        return {"tmin_lo": float(cfg["tpc_t_min_lo"]),
                "tmin_hi": float(cfg["tpc_t_min_hi"])}
    return {"tmin_lo": float(cfg["intime_anode_crosser"]["t_min_p2_lo"]),
            "tmin_hi": float(cfg["intime_anode_crosser"]["t_min_p2_hi"])}


def load_wire_geometry(path):
    """Load per-(tpc,plane) wire->position constants from a
    ``wire_geometry_sbnd.json`` produced by ``parse_wire_geometry.py``.

    Returns ``(slope, offset, what_y, what_z)``, each a ``(2, 3)`` float
    array indexed ``[tpc, plane]``. A hit's view coordinate is
    ``s = slope*wire + offset = what_y*y + what_z*z``.
    """
    import numpy as np
    with open(path) as f:
        planes = json.load(f)["planes"]
    slope = np.zeros((2, 3)); offset = np.zeros((2, 3))
    what_y = np.zeros((2, 3)); what_z = np.zeros((2, 3))
    for t in (0, 1):
        for p in (0, 1, 2):
            g = planes[f"tpc{t}_plane{p}"]
            slope[t, p] = g["slope_cm"]
            offset[t, p] = g["offset_cm"]
            what_y[t, p] = g["what_y"]
            what_z[t, p] = g["what_z"]
    return slope, offset, what_y, what_z


def collect_filelist(inputs, maxfiles=None):
    """Resolve a list of paths/filelists into a flat list of ROOT file paths.

    `inputs` may contain literal paths, glob patterns, or ``*.txt`` files
    (one path per line, blank lines ignored). The result is optionally
    truncated to the first `maxfiles` entries.
    """
    paths = []
    for inp in inputs:
        if inp.endswith(".txt") or inp.endswith(".list"):
            with open(inp) as f:
                paths.extend([ln.strip() for ln in f if ln.strip()])
        elif "*" in inp or "?" in inp:
            paths.extend(sorted(glob.glob(inp)))
        else:
            paths.append(inp)
    if maxfiles:
        paths = paths[:maxfiles]
    return paths
