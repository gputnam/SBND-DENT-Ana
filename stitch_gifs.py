"""Vertically stitch three GIFs (dx, dy, dz) into a single combined GIF.

Opens the three input GIFs, unwraps every frame to full-size RGB,
vertically stacks the corresponding frame triplet (dx top, dy middle,
dz bottom), and writes out a single quantized animated GIF.

    python stitch_gifs.py dx.gif dy.gif dz.gif dxyz_stack.gif
"""

import argparse
import os

from PIL import Image, ImageSequence

# Shared palette quantizer (consolidated into crtlib.plotting).
from crtlib.plotting import quantize_rgb as quantize


def read_frames(path):
    """Return (list_of_RGB_frames, per_frame_duration_ms)."""
    img = Image.open(path)
    frames = []
    duration = img.info.get("duration", 200)
    for fr in ImageSequence.Iterator(img):
        frames.append(fr.convert("RGB").copy())
    return frames, duration


def stitch_one(dx_path, dy_path, dz_path, out_path, label):
    print(f"\n[{label}]", flush=True)
    print(f"  dx: {dx_path}")
    print(f"  dy: {dy_path}")
    print(f"  dz: {dz_path}")

    dx_frames, dur_x = read_frames(dx_path)
    dy_frames, dur_y = read_frames(dy_path)
    dz_frames, dur_z = read_frames(dz_path)
    n = min(len(dx_frames), len(dy_frames), len(dz_frames))
    if len(dx_frames) != len(dy_frames) or len(dy_frames) != len(dz_frames):
        print(f"  WARNING: frame counts differ "
              f"(dx={len(dx_frames)}, dy={len(dy_frames)}, dz={len(dz_frames)}); "
              f"using min = {n}", flush=True)
    duration = max(dur_x, dur_y, dur_z)
    # equal widths assumed (these GIFs come from matplotlib with fixed
    # figsize). If not equal, pad with white to the max width.
    max_w = max(dx_frames[0].size[0], dy_frames[0].size[0], dz_frames[0].size[0])

    def pad(im):
        if im.size[0] == max_w:
            return im
        new = Image.new("RGB", (max_w, im.size[1]), color=(255, 255, 255))
        new.paste(im, ((max_w - im.size[0]) // 2, 0))
        return new

    combined = []
    for i in range(n):
        a, b, c = pad(dx_frames[i]), pad(dy_frames[i]), pad(dz_frames[i])
        h = a.size[1] + b.size[1] + c.size[1]
        out = Image.new("RGB", (max_w, h), color=(255, 255, 255))
        out.paste(a, (0, 0))
        out.paste(b, (0, a.size[1]))
        out.paste(c, (0, a.size[1] + b.size[1]))
        combined.append(quantize(out))

    print(f"  -> writing {n} frames at {duration} ms each, "
          f"frame size {combined[0].size}", flush=True)
    combined[0].save(out_path, save_all=True,
                     append_images=combined[1:],
                     duration=duration, loop=0,
                     optimize=False, disposal=2)
    sz = os.path.getsize(out_path) / 1e6
    print(f"  wrote {out_path} ({sz:.2f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dx_gif")
    ap.add_argument("dy_gif")
    ap.add_argument("dz_gif")
    ap.add_argument("out_gif")
    ap.add_argument("--label", default="dx/dy/dz stack")
    a = ap.parse_args()
    stitch_one(a.dx_gif, a.dy_gif, a.dz_gif, a.out_gif, a.label)


if __name__ == "__main__":
    main()
