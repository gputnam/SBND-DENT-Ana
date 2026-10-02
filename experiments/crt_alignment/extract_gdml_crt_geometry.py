"""Extract per-CRT-module box GEOMETRY (centroid + dimensions + orientation)
from a SBND GDML, plus the TPC-active footprint at the anode planes.

Adds, on top of `extract_gdml_crt_centroids.py`:
  - module box dimensions (`box_x`, `box_y`, `box_z`, full extents in cm)
  - module orientation in the world frame as a 3×3 rotation matrix
    (flattened into nine columns `R00..R22` where row i is the world-frame
     direction of local axis i)
  - TPC active-volume footprint at each anode (Y, Z extents)

Output:
  crt_module_gdml_geometry.csv     one row per volCRTModule
  anode_gdml_geometry.csv          one row per anode (TPC0, TPC1)

Usage:
  python extract_gdml_crt_geometry.py \
      /exp/sbnd/app/users/gputnam/sbndcode/build_slf7.x86_64/sbndcode/gdml/sbnd_v02_00.gdml
"""

import argparse
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

# Reuse helpers from the existing extractor (parsers + walker).
from extract_gdml_crt_centroids import (
    parse_gdml_volumes, parse_gdml_definitions, find_world_volume,
    _len_to_cm, MODULE_RE, TAGGER_RE,
)


def parse_box_solids(root):
    """Return {solid_name: (x, y, z) full extents in cm} for each <box>."""
    boxes = {}
    for solids in root.iter():
        if solids.tag != "solids":
            continue
        for box in solids.findall("box"):
            name = box.attrib["name"]
            u = box.attrib.get("lunit") or box.attrib.get("unit") or "cm"
            x = _len_to_cm(box.attrib.get("x", 0), u)
            y = _len_to_cm(box.attrib.get("y", 0), u)
            z = _len_to_cm(box.attrib.get("z", 0), u)
            boxes[name] = (x, y, z)
    return boxes


def parse_volume_solids(root):
    """Return {volume_name: solidref} so we can join volumes -> box."""
    out = {}
    for structure in root.iter():
        if structure.tag != "structure":
            continue
        for vol in structure:
            if vol.tag != "volume":
                continue
            sr = vol.find("solidref")
            if sr is not None and "ref" in sr.attrib:
                out[vol.attrib["name"]] = sr.attrib["ref"]
    return out


def walk_for_modules_with_R(volumes, vol_name, R_world, t_world,
                             enclosing_tagger, results):
    """Same as the centroid walker but ALSO records R_world per module."""
    m = MODULE_RE.match(vol_name)
    if m:
        results.append({
            "module_idx": int(m.group(1)),
            "module_name": vol_name,
            "tagger": enclosing_tagger or "",
            "world_x": float(t_world[0]),
            "world_y": float(t_world[1]),
            "world_z": float(t_world[2]),
            "R": R_world.copy(),
        })
        return
    tag = TAGGER_RE.match(vol_name)
    if tag:
        enclosing_tagger = tag.group(1)
    for (child_name, R_p, t_p) in volumes.get(vol_name, []):
        R_child = R_world @ R_p
        t_child = R_world @ t_p + t_world
        walk_for_modules_with_R(volumes, child_name, R_child, t_child,
                                 enclosing_tagger, results)


def walk_for_named_volume(volumes, vol_name, R_world, t_world, target_name):
    """Find one named volume's world-frame placement(s).

    Returns list of (R_world, t_world) for every occurrence of
    `target_name` reachable from `vol_name`.
    """
    out = []
    if vol_name == target_name:
        out.append((R_world.copy(), t_world.copy()))
    for (child_name, R_p, t_p) in volumes.get(vol_name, []):
        R_child = R_world @ R_p
        t_child = R_world @ t_p + t_world
        out.extend(walk_for_named_volume(volumes, child_name,
                                          R_child, t_child, target_name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gdml")
    ap.add_argument("--out-modules",
                    default="crt_module_gdml_geometry.csv")
    ap.add_argument("--out-anodes",
                    default="anode_gdml_geometry.csv")
    args = ap.parse_args()

    print(f"Parsing {args.gdml} ...", flush=True)
    tree = ET.parse(args.gdml)
    root = tree.getroot()

    boxes = parse_box_solids(root)
    print(f"  {len(boxes):,} <box> solids parsed", flush=True)

    vol_solid = parse_volume_solids(root)
    print(f"  {len(vol_solid):,} volume->solid bindings", flush=True)

    volumes = parse_gdml_volumes(root)
    world = find_world_volume(root)
    if world is None:
        all_vols = set(volumes); kids = {c for ch in volumes.values() for (c,_,_) in ch}
        world = list(all_vols - kids)[0]
    print(f"  world volume: {world}", flush=True)

    # ----- CRT modules -----
    rows = []
    walk_for_modules_with_R(volumes, world, np.eye(3), np.zeros(3), None, rows)
    print(f"  found {len(rows)} CRT modules", flush=True)

    out = []
    missing_solid = 0
    for r in rows:
        sname = vol_solid.get(r["module_name"])
        if sname is None or sname not in boxes:
            missing_solid += 1
            bx, by, bz = (np.nan, np.nan, np.nan)
        else:
            bx, by, bz = boxes[sname]
        R = r["R"]
        out.append({
            "module_idx": r["module_idx"],
            "module_name": r["module_name"],
            "tagger": r["tagger"],
            "world_x": r["world_x"],
            "world_y": r["world_y"],
            "world_z": r["world_z"],
            "box_x": bx, "box_y": by, "box_z": bz,
            "R00": R[0,0], "R01": R[0,1], "R02": R[0,2],
            "R10": R[1,0], "R11": R[1,1], "R12": R[1,2],
            "R20": R[2,0], "R21": R[2,1], "R22": R[2,2],
        })
    if missing_solid:
        print(f"  WARNING: {missing_solid} modules with no resolvable box solid",
              flush=True)

    df = pd.DataFrame(out).sort_values("module_idx").reset_index(drop=True)
    df.to_csv(args.out_modules, index=False)
    print(f"  wrote {args.out_modules}", flush=True)

    # ----- TPC active footprint -----
    placements = walk_for_named_volume(volumes, world, np.eye(3),
                                        np.zeros(3), "volTPCActive")
    print(f"  found {len(placements)} TPCActive placements", flush=True)
    active_box = boxes.get("boxTPCActive", (np.nan, np.nan, np.nan))
    print(f"  boxTPCActive (cm) = {active_box}", flush=True)
    anode_rows = []
    for i, (R, t) in enumerate(placements):
        # In the local frame the box runs from -x/2 to +x/2 (drift), -y/2 to +y/2,
        # -z/2 to +z/2. The anode of this TPC sits at the +x/2 face in the local
        # frame for an unrotated TPC, OR at -x/2 for a Y180-rotated TPC. We
        # resolve the world-frame anode position by mapping (±x/2, 0, 0) and
        # picking the one with larger |x| in world coordinates.
        ax_local_pos = np.array([active_box[0]/2, 0.0, 0.0])
        ax_local_neg = -ax_local_pos
        wx_pos = R @ ax_local_pos + t
        wx_neg = R @ ax_local_neg + t
        anode_world = wx_pos if abs(wx_pos[0]) > abs(wx_neg[0]) else wx_neg
        anode_rows.append({
            "anode_idx": i,
            "anode_world_x": float(anode_world[0]),
            "anode_world_y": float(anode_world[1]),
            "anode_world_z": float(anode_world[2]),
            "active_y_full_cm": float(active_box[1]),
            "active_z_full_cm": float(active_box[2]),
            "centroid_world_x": float(t[0]),
            "centroid_world_y": float(t[1]),
            "centroid_world_z": float(t[2]),
            "R00": R[0,0], "R01": R[0,1], "R02": R[0,2],
            "R10": R[1,0], "R11": R[1,1], "R12": R[1,2],
            "R20": R[2,0], "R21": R[2,1], "R22": R[2,2],
        })
    pd.DataFrame(anode_rows).to_csv(args.out_anodes, index=False)
    print(f"  wrote {args.out_anodes}", flush=True)

    # Sanity print.
    print("\nper-tagger module count:")
    print(df["tagger"].value_counts().to_string())
    print("\nbox sizes (full extents, cm) seen by tagger:")
    print(df.groupby("tagger")[["box_x","box_y","box_z"]].first().to_string())


if __name__ == "__main__":
    main()
