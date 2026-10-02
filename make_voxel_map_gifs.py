#!/usr/bin/env python3
"""Animate a voxel map's per-X-slice PNGs into dx/dy/dz GIFs + a stacked
dxyz GIF. The voxelizer writes static per-X-slice frames under
<maps>/all/{dx,dy,dz}/png/x*.png; this turns them into GIFs (frames ordered
by X = -200 -> +180 cm), using the LIBIMAGEQUANT quantizer per CLAUDE.md.

Usage: python make_voxel_map_gifs.py <maps_dir> [<maps_dir> ...]
"""
import glob
import os
import re
import sys

from PIL import Image

import stitch_gifs as S


def xval(p):
    return int(re.search(r"x([+-]\d+)", os.path.basename(p)).group(1))


def build_gif(comp_dir, out):
    frames = sorted(glob.glob(os.path.join(comp_dir, "png", "x*.png")),
                    key=xval)
    if not frames:
        return None
    imgs = [S.quantize(Image.open(p).convert("RGB")) for p in frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=250, loop=0, optimize=False, disposal=2)
    print(f"  {out}: {len(imgs)} frames, X {xval(frames[0])}..{xval(frames[-1])} cm, "
          f"{os.path.getsize(out)/1e6:.2f} MB", flush=True)
    return out


def main():
    for maps in sys.argv[1:]:
        base = os.path.join(maps, "all")
        print(f"[{maps}] voxel-map GIFs:", flush=True)
        paths = {}
        for comp in ("dx", "dy", "dz"):
            out = os.path.join(base, comp, f"{comp}.gif")
            p = build_gif(os.path.join(base, comp), out)
            if p:
                paths[comp] = p
        if len(paths) == 3:
            S.stitch_one(paths["dx"], paths["dy"], paths["dz"],
                         os.path.join(base, "dxyz_stack.gif"),
                         f"{os.path.basename(maps)} raw voxel maps vs X")
        else:
            print(f"  WARNING: missing slice PNGs in {base} "
                  "(run the voxelizer WITHOUT --no-slice-maps)", flush=True)


if __name__ == "__main__":
    main()
