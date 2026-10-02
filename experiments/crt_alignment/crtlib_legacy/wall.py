"""Per-wall basis vectors derived from tagger IDs."""

import numpy as np

from crtlib.constants import WALL_NORMAL_AXIS


def basis_per_wall(tagger_ids):
    """Return ``(normals, e1s, e2s)`` of shape (n_walls, 3) each.

    ``e1, e2`` form an orthonormal in-plane basis of each wall, oriented
    by global axis ordering (so all walls with normal=ŷ share the same
    in-plane axes (x̂, ẑ), and so on)."""
    n = len(tagger_ids)
    normals = np.zeros((n, 3))
    e1s     = np.zeros((n, 3))
    e2s     = np.zeros((n, 3))
    for i, tag in enumerate(tagger_ids):
        ax = WALL_NORMAL_AXIS[int(tag)]
        normals[i, ax] = 1.0
        if ax == 0:
            e1s[i] = (0, 1, 0); e2s[i] = (0, 0, 1)
        elif ax == 1:
            e1s[i] = (1, 0, 0); e2s[i] = (0, 0, 1)
        else:
            e1s[i] = (1, 0, 0); e2s[i] = (0, 1, 0)
    return normals, e1s, e2s


def aggregate_modules_to_walls(taggers, mod_to_wall_idx_map=None):
    """Map per-module tagger IDs to per-wall indices.

    Returns ``(unique_walls, wall_to_idx, mod_to_wall)`` where
    ``mod_to_wall[i]`` is the wall-row index for module index i.
    """
    unique_walls = np.array(sorted(set(int(t) for t in taggers)),
                            dtype=np.int32)
    wall_to_idx = {int(w): i for i, w in enumerate(unique_walls)}
    mod_to_wall = np.array([wall_to_idx[int(t)] for t in taggers],
                           dtype=np.int32)
    return unique_walls, wall_to_idx, mod_to_wall
