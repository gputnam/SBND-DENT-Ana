"""Extract per-CRT-module world-frame centroids from a SBND GDML.

The SBND geometry hierarchy nests volCRTModuleN inside a tagger volume
(volTaggerTop{High,Low}, volTaggerSouth/North/East/West, volTaggerBot)
inside the cryostat / detector hall / world. Each module's world
centroid is the chain of placement transforms applied to (0, 0, 0) in
the module's local frame.

We parse the GDML XML directly (no ROOT / sbndcode build needed), build
a position+rotation lookup, walk the volume tree from the world volume,
and emit one row per volCRTModuleN with its world-frame centroid
(plus tagger label for sanity).

Output: crt_module_gdml_centroids.csv
  module_idx, module_name, tagger, world_x, world_y, world_z

Usage:
  python extract_gdml_crt_centroids.py \
      /exp/sbnd/app/users/gputnam/sbndcode/build_slf7.x86_64/sbndcode/gdml/sbnd_v02_00.gdml \
      [--out crt_module_gdml_centroids.csv]
"""

import argparse
import os
import re
import xml.etree.ElementTree as ET

import numpy as np


# GDML rotations: per spec the matrix is R = Rx(ax)·Ry(ay)·Rz(az)
# (extrinsic XYZ). LArSoft / sbndcode follow this convention.
def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _angle_to_rad(val, unit):
    if unit is None or unit == "rad":
        return float(val)
    if unit == "deg":
        return float(val) * np.pi / 180.0
    raise ValueError(f"unknown rotation unit {unit!r}")


def _len_to_cm(val, unit):
    if unit is None or unit == "cm":
        return float(val)
    if unit == "mm":
        return float(val) / 10.0
    if unit == "m":
        return float(val) * 100.0
    raise ValueError(f"unknown length unit {unit!r}")


def parse_gdml_definitions(root):
    """Build dictionaries of named <position> and <rotation> nodes (plus
    inline Identity for unrotated placements). All values returned in cm
    / radians.
    """
    positions = {}
    rotations = {}
    for define in root.iter():
        if define.tag != "define":
            continue
        for child in define:
            if child.tag == "position":
                name = child.attrib["name"]
                u = child.attrib.get("unit", "cm")
                x = _len_to_cm(child.attrib.get("x", 0), u)
                y = _len_to_cm(child.attrib.get("y", 0), u)
                z = _len_to_cm(child.attrib.get("z", 0), u)
                positions[name] = np.array([x, y, z])
            elif child.tag == "rotation":
                name = child.attrib["name"]
                u = child.attrib.get("unit", "deg")
                ax = _angle_to_rad(child.attrib.get("x", 0), u)
                ay = _angle_to_rad(child.attrib.get("y", 0), u)
                az = _angle_to_rad(child.attrib.get("z", 0), u)
                rotations[name] = _Rx(ax) @ _Ry(ay) @ _Rz(az)
    return positions, rotations


def _read_position_from_node(node, default_unit="cm"):
    u = node.attrib.get("unit", default_unit)
    x = _len_to_cm(node.attrib.get("x", 0), u)
    y = _len_to_cm(node.attrib.get("y", 0), u)
    z = _len_to_cm(node.attrib.get("z", 0), u)
    return np.array([x, y, z])


def _read_rotation_from_node(node):
    u = node.attrib.get("unit", "deg")
    ax = _angle_to_rad(node.attrib.get("x", 0), u)
    ay = _angle_to_rad(node.attrib.get("y", 0), u)
    az = _angle_to_rad(node.attrib.get("z", 0), u)
    return _Rx(ax) @ _Ry(ay) @ _Rz(az)


def parse_gdml_volumes(root):
    """Build a dict {volume_name: [(child_name, position, rotation), ...]}.

    Each entry is the list of physvol placements inside the given volume.
    Inline <position>/<rotation> are resolved against the named pools.
    """
    positions, rotations = parse_gdml_definitions(root)

    volumes = {}  # name -> list of (child_volname, R, t)
    for structure in root.iter():
        if structure.tag != "structure":
            continue
        for vol in structure:
            if vol.tag != "volume":
                continue
            vol_name = vol.attrib["name"]
            children = []
            for pv in vol.findall("physvol"):
                vref = pv.find("volumeref")
                if vref is None:
                    continue
                child_name = vref.attrib.get("ref")
                # Resolve placement transform.
                t = np.zeros(3)
                R = np.eye(3)
                for sub in pv:
                    if sub.tag == "position":
                        t = _read_position_from_node(sub)
                    elif sub.tag == "positionref":
                        t = positions[sub.attrib["ref"]]
                    elif sub.tag == "rotation":
                        R = _read_rotation_from_node(sub)
                    elif sub.tag == "rotationref":
                        R = rotations[sub.attrib["ref"]]
                children.append((child_name, R, t))
            volumes[vol_name] = children
    return volumes


def find_world_volume(root):
    """The world volume is the volumeref inside <setup>'s <world>."""
    for setup in root.iter():
        if setup.tag == "setup":
            world = setup.find("world")
            if world is not None and "ref" in world.attrib:
                return world.attrib["ref"]
    # Fallback: the volume with no parent.
    return None


# Regex: volCRTModule<digits>.
MODULE_RE = re.compile(r"^volCRTModule(\d+)$")
# Regex: volTagger<wallname>.
TAGGER_RE = re.compile(r"^volTagger(.+)$")


def walk_for_modules(volumes, vol_name, R_world, t_world,
                     enclosing_tagger, results):
    """Recursively walk volumes from `vol_name` outwards-in.

    R_world, t_world is the current world-frame transform of `vol_name`'s
    local frame. enclosing_tagger is the most recent tagger label seen
    in the chain (or None).
    """
    m = MODULE_RE.match(vol_name)
    if m:
        # Module's local origin (0,0,0) -> world frame is just t_world.
        results.append({
            "module_idx": int(m.group(1)),
            "module_name": vol_name,
            "tagger": enclosing_tagger or "",
            "world_x": float(t_world[0]),
            "world_y": float(t_world[1]),
            "world_z": float(t_world[2]),
        })
        # Don't descend further; CRTModule's children are strip arrays.
        return

    tag = TAGGER_RE.match(vol_name)
    if tag:
        enclosing_tagger = tag.group(1)

    for (child_name, R_p, t_p) in volumes.get(vol_name, []):
        # Compose: world transform of child = world transform of parent
        # composed with the placement transform (R_p, t_p).
        # If P_parent = R_world · P_local_parent + t_world
        # and P_local_parent = R_p · P_local_child + t_p
        # then P_world = R_world · (R_p · P_local_child + t_p) + t_world
        #              = (R_world @ R_p) · P_local_child + (R_world @ t_p + t_world)
        R_child = R_world @ R_p
        t_child = R_world @ t_p + t_world
        walk_for_modules(volumes, child_name, R_child, t_child,
                         enclosing_tagger, results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gdml")
    ap.add_argument("--out", default="crt_module_gdml_centroids.csv")
    args = ap.parse_args()

    print(f"Parsing {args.gdml} ...", flush=True)
    tree = ET.parse(args.gdml)
    root = tree.getroot()

    volumes = parse_gdml_volumes(root)
    print(f"  {len(volumes)} volumes parsed", flush=True)

    world = find_world_volume(root)
    if world is None:
        # Find the volume with no parent (i.e. not referenced as a child)
        all_vols = set(volumes.keys())
        children = {c for ch in volumes.values() for (c, _, _) in ch}
        no_parents = all_vols - children
        if len(no_parents) == 1:
            world = list(no_parents)[0]
        else:
            raise RuntimeError(f"could not find world volume; "
                               f"unparented = {no_parents}")
    print(f"  world volume: {world}", flush=True)

    results = []
    walk_for_modules(volumes, world, np.eye(3), np.zeros(3), None, results)
    print(f"  found {len(results)} CRT modules", flush=True)

    if not results:
        return

    import pandas as pd
    df = pd.DataFrame(results).sort_values("module_idx").reset_index(drop=True)
    df.to_csv(args.out, index=False)
    print(f"  wrote {args.out}", flush=True)
    print("\nper-tagger module count:")
    print(df["tagger"].value_counts().to_string())
    print("\nworld_x range:", df["world_x"].min(), "to", df["world_x"].max())
    print("world_y range:", df["world_y"].min(), "to", df["world_y"].max())
    print("world_z range:", df["world_z"].min(), "to", df["world_z"].max())


if __name__ == "__main__":
    main()
