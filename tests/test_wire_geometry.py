"""Unit tests for the wire-geometry loader and the committed geometry JSON."""
import os

import numpy as np

from crtlib.io import load_wire_geometry


def _geom_path(repo_root):
    return os.path.join(repo_root, "config", "wire_geometry_sbnd.json")


def test_loader_shapes(repo_root):
    slope, offset, what_y, what_z = load_wire_geometry(_geom_path(repo_root))
    for a in (slope, offset, what_y, what_z):
        assert a.shape == (2, 3)


def test_view_vectors_unit_norm(repo_root):
    _, _, what_y, what_z = load_wire_geometry(_geom_path(repo_root))
    norm = np.hypot(what_y, what_z)
    # (what_y, what_z) is a unit view direction for each (tpc, plane).
    assert np.allclose(norm, 1.0, atol=1e-6)


def test_collection_plane_is_vertical(repo_root):
    # Plane 2 (collection) wires measure Z only -> what_y ~ 0, |what_z| ~ 1.
    _, _, what_y, what_z = load_wire_geometry(_geom_path(repo_root))
    assert np.all(np.abs(what_y[:, 2]) < 1e-6)
    assert np.allclose(np.abs(what_z[:, 2]), 1.0, atol=1e-6)


def test_induction_planes_pm60_degrees(repo_root):
    # Induction planes sit at +/-60 deg: |what_y| = sin(60) ~ 0.866.
    _, _, what_y, _ = load_wire_geometry(_geom_path(repo_root))
    for p in (0, 1):
        assert np.allclose(np.abs(what_y[:, p]), np.sin(np.radians(60)), atol=1e-3)


def test_matches_delegating_wrapper(repo_root):
    import plot_doubleanode_2d_voxels as P2
    a = load_wire_geometry(_geom_path(repo_root))
    b = P2.load_wire_geom(P2.WIRE_GEOM)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
