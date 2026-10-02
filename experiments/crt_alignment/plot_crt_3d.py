"""3D figure of the SBND anode + CRT module assembly, optionally with
per-module / per-wall alignment corrections applied.

Reads:
  crt_module_gdml_geometry.csv      (from extract_gdml_crt_geometry.py)
  anode_gdml_geometry.csv
  --align-csv FILE                  optional crt_alignment_results.csv

Produces:
  --outdir DIR/proj_<name>.{png,pdf} for each projection.

Projections:
  x       — view down +x (Y, Z plane visible)
  y       — view down +y (X, Z)
  z       — view down +z (X, Y)
  iso1    — elev=25°, azim=45°
  iso2    — elev=25°, azim=-45°
  iso3    — elev=15°, azim=120°

Each CRT module is drawn as an oriented rectangular slab; each anode as a
flat plane slab thickness 1 cm. Per-tagger colour, alpha=0.4. Modules
with alignment correction get their (T, R) replaced by the corrected
pose: P_world(corrected) = R_world · P_local + T_world  →  R'·(P-c) + c + dT
with c = pre-alignment world centroid, R' = R_dR · R, T' as appropriate.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from crtlib.constants import TAG_NAME

PROJS = {
    "x":    dict(elev=0,   azim=0,   title="View along +x (Y-Z plane)"),
    "y":    dict(elev=0,   azim=-90, title="View along +y (X-Z plane)"),
    "z":    dict(elev=90,  azim=-90, title="View along +z (X-Y plane)"),
    "iso1": dict(elev=25,  azim=45,  title="Isometric 1"),
    "iso2": dict(elev=25,  azim=-45, title="Isometric 2"),
    "iso3": dict(elev=15,  azim=120, title="Isometric 3"),
}

WALL_COLORS = {
    "Bot":     "#bf6f4a",
    "South":   "#4a8fbf",
    "North":   "#3568a4",
    "West":    "#5cb85c",
    "East":    "#2c8c2c",
    "TopLow":  "#bfa54a",
    "TopHigh": "#bf8a3c",
    "":        "#999999",
}


def rodrigues(dR):
    """3-vector axis-angle (length=angle) → 3×3 rotation matrix."""
    angle = np.linalg.norm(dR)
    if angle < 1e-12:
        return np.eye(3)
    axis = dR / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def box_corners(centroid, R, half_x, half_y, half_z):
    """Return (8, 3) world-frame corner positions of a box at `centroid`,
    oriented by `R` (local→world), with half-extents (half_x, half_y, half_z)."""
    sx = np.array([-1, +1, +1, -1, -1, +1, +1, -1]) * half_x
    sy = np.array([-1, -1, +1, +1, -1, -1, +1, +1]) * half_y
    sz = np.array([-1, -1, -1, -1, +1, +1, +1, +1]) * half_z
    local = np.stack([sx, sy, sz], axis=1)
    return (R @ local.T).T + centroid


_BOX_FACES = [
    (0, 1, 2, 3),   # -z face
    (4, 5, 6, 7),   # +z face
    (0, 1, 5, 4),   # -y face
    (3, 2, 6, 7),   # +y face
    (0, 3, 7, 4),   # -x face
    (1, 2, 6, 5),   # +x face
]


def add_box(ax, corners, color, alpha=0.4, edge="black"):
    polys = [corners[list(f)] for f in _BOX_FACES]
    pc = Poly3DCollection(polys, facecolor=color, edgecolor=edge,
                          linewidths=0.3, alpha=alpha)
    ax.add_collection3d(pc)


def add_anode_plane(ax, anode_x, y_half, z_half, z_centre, color, alpha=0.18):
    """Anode = thin slab at x = anode_x, ±y_half × [z_centre - z_half, z_centre + z_half]."""
    thick = 0.5  # cm
    ymin, ymax = -y_half, +y_half
    zmin, zmax = z_centre - z_half, z_centre + z_half
    xmin, xmax = anode_x - thick / 2, anode_x + thick / 2
    verts = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin],
        [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax],
        [xmax, ymax, zmax], [xmin, ymax, zmax],
    ])
    add_box(ax, verts, color=color, alpha=alpha, edge=color)


def load_alignment(csv_path):
    """Load a crt_alignment_results.csv. Returns:
      - per_module: dict mod_id -> (dT (3,), dR (3,))
      - per_wall:   dict wall_id -> (dT, dR)
      - anode1:     (dT, dR) or None
    """
    if csv_path is None or not os.path.exists(csv_path):
        return {}, {}, None
    df = pd.read_csv(csv_path)
    per_module, per_wall, anode1 = {}, {}, None
    for _, r in df.iterrows():
        dT = np.array([r["dx"], r["dy"], r["dz"]])
        dR = np.array([r["dthx"], r["dthy"], r["dthz"]])
        if r["kind"] == "module":
            per_module[int(r["module_id"])] = (dT, dR)
        elif r["kind"] == "wall":
            per_wall[int(r["module_id"])] = (dT, dR)
        elif r["kind"] == "anode":
            if int(r["module_id"]) == 1:
                anode1 = (dT, dR)
    return per_module, per_wall, anode1


def apply_alignment(centroid, R_local2world, dT, dR):
    """Compose: rotation by dR about centroid, then translation dT.

       P_corrected = R_dR · (P_local2world - centroid) + centroid + dT
    """
    R_dR = rodrigues(dR)
    R_new = R_dR @ R_local2world
    c_new = centroid + dT  # rotation about centroid leaves centroid fixed; only dT shifts it
    return c_new, R_new


def make_figure(modules, anodes, per_module, per_wall, anode1,
                proj, outdir, mod_to_wall=None):
    cfg = PROJS[proj]
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=cfg["elev"], azim=cfg["azim"])

    # CRT modules.
    for _, r in modules.iterrows():
        c = np.array([r["world_x"], r["world_y"], r["world_z"]])
        R = np.array([[r["R00"], r["R01"], r["R02"]],
                      [r["R10"], r["R11"], r["R12"]],
                      [r["R20"], r["R21"], r["R22"]]])
        hx, hy, hz = r["box_x"]/2, r["box_y"]/2, r["box_z"]/2
        if not (np.isfinite(hx) and np.isfinite(hy) and np.isfinite(hz)):
            continue
        # Apply alignment if available — module-level first, fall back to
        # the matching wall correction.
        mod_id = int(r["module_idx"])
        # The alignment CSV uses FEB MAC IDs as module_id, NOT the GDML
        # module_idx. We can't reliably join. Take a best-effort match:
        # if the per_module dict is non-empty AND mod_id present, use it.
        # Otherwise check per_wall by tagger name.
        dT, dR = None, None
        if mod_id in per_module:
            dT, dR = per_module[mod_id]
        elif per_wall:
            # Wall fit uses 0..6 tagger IDs.
            tag_id_map = {v: k for k, v in TAG_NAME.items()}
            tag_id = tag_id_map.get(r["tagger"], None)
            if tag_id is not None and tag_id in per_wall:
                dT, dR = per_wall[tag_id]
        if dT is not None:
            c, R = apply_alignment(c, R, dT, dR)
        corners = box_corners(c, R, hx, hy, hz)
        col = WALL_COLORS.get(r["tagger"], "#999")
        add_box(ax, corners, color=col, alpha=0.45)

    # Anode planes.
    for _, r in anodes.iterrows():
        idx = int(r["anode_idx"])
        anode_x = r["anode_world_x"]
        c = np.array([anode_x, r["anode_world_y"], r["anode_world_z"]])
        R = np.eye(3)
        if idx == 1 and anode1 is not None:
            dT, dR = anode1
            c, R = apply_alignment(c, R, dT, dR)
            anode_x = c[0]
        add_anode_plane(ax, anode_x,
                        y_half=r["active_y_full_cm"]/2,
                        z_half=r["active_z_full_cm"]/2,
                        z_centre=c[2],
                        color="#999999", alpha=0.18)

    # Cosmetics.
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)"); ax.set_zlabel("z (cm)")
    # Bounding box from data extent.
    xs = list(modules["world_x"]) + list(anodes["anode_world_x"])
    ys = list(modules["world_y"]) + [-r["active_y_full_cm"]/2 for _, r in anodes.iterrows()] \
                                  + [+r["active_y_full_cm"]/2 for _, r in anodes.iterrows()]
    zs = list(modules["world_z"]) + [r["centroid_world_z"] - r["active_z_full_cm"]/2 for _, r in anodes.iterrows()] \
                                  + [r["centroid_world_z"] + r["active_z_full_cm"]/2 for _, r in anodes.iterrows()]
    pad = 50
    ax.set_xlim(min(xs)-pad, max(xs)+pad)
    ax.set_ylim(min(ys)-pad, max(ys)+pad)
    ax.set_zlim(min(zs)-pad, max(zs)+pad)
    ax.set_box_aspect((max(xs)-min(xs)+2*pad,
                       max(ys)-min(ys)+2*pad,
                       max(zs)-min(zs)+2*pad))
    ax.set_title(f"{cfg['title']}  (proj={proj})")

    # Legend.
    handles = [plt.Rectangle((0,0), 1, 1, color=c, alpha=0.6, label=n)
               for n, c in WALL_COLORS.items() if n]
    handles.append(plt.Rectangle((0,0), 1, 1, color="#999", alpha=0.4, label="Anode planes"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    base = os.path.join(outdir, f"proj_{proj}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=140)
    plt.close(fig)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules-csv", default="crt_module_gdml_geometry.csv")
    ap.add_argument("--anodes-csv",  default="anode_gdml_geometry.csv")
    ap.add_argument("--align-csv",   default=None,
                    help="crt_alignment_results.csv to apply; default: nominal")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--projs", default="x,y,z,iso1,iso2,iso3",
                    help="comma list of projections to render")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    modules = pd.read_csv(args.modules_csv)
    anodes  = pd.read_csv(args.anodes_csv)
    print(f"  modules: {len(modules)},  anodes: {len(anodes)}")

    per_module, per_wall, anode1 = load_alignment(args.align_csv)
    if args.align_csv:
        print(f"  alignment: {len(per_module)} per-module, "
              f"{len(per_wall)} per-wall, anode1={'yes' if anode1 else 'no'}")
    else:
        print(f"  no alignment applied (nominal GDML)")

    for proj in args.projs.split(","):
        proj = proj.strip()
        if proj not in PROJS:
            print(f"  WARN: unknown proj '{proj}' (skipped)")
            continue
        base = make_figure(modules, anodes, per_module, per_wall, anode1,
                            proj, args.outdir)
        print(f"  wrote {base}.png/.pdf")


if __name__ == "__main__":
    main()
