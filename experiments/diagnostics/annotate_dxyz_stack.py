"""Make an annotated copy of dxyz_stack.gif with two place-name
annotations that fade in/out over a few frames.

  "Sparkstown" — above the -dx feature in the dx fit pane (top panel,
                 middle pane). Target frame: x = +75 cm.
  "Shortsville" — below the +dy feature in the dy fit pane (middle
                  panel, middle pane). Target frame: x = -50 cm.

Each label uses a 9-frame fade schedule centred on its target frame
(opacity: 0.1, 0.3, 0.6, 0.85, 1.0, 0.85, 0.6, 0.3, 0.1).
"""

import os
from PIL import Image, ImageDraw, ImageFont, ImageSequence

SRC = "/exp/sbnd/data/users/gputnam/efield-study/sce_input_doubleanode/fit_diagnostics/dxyz_stack.gif"
DST = "/exp/sbnd/data/users/gputnam/efield-study/sce_input_doubleanode/fit_diagnostics/dxyz_stack_annotated.gif"
# Display-weight italic — dramatic, dynamic, slightly playful for
# overlaying town names on top of a serious-looking voxel plot.
FONT_PATH = "/usr/share/fonts/julietaula-montserrat/Montserrat-BlackItalic.otf"

# Stitched GIF layout (verified): 1980 x 1650; three panels stacked
# vertically, each 1980 x 550. Each panel has three panes: left =
# measured, middle = fit, right = fit - measured. The plot AXES inside
# the middle pane span approximately:
#   x pixel: 720 .. 1240   (z phys 0 .. 500 cm)
#   y pixel (within panel): 485 .. 45  (y phys -200 .. +200 cm,
#                                       image-y inverted)
PANEL_HEIGHT = 550
AXES = {
    "x_left_px":   720,
    "x_right_px":  1240,
    "y_bottom_px": 485,
    "y_top_px":    45,
}


def fit_pane_coords(panel_idx, z_phys, y_phys):
    """Convert (z, y) cm in a panel's fit pane to (x_px, y_px) of the
    stitched GIF. panel_idx: 0=top dx, 1=middle dy, 2=bottom dz."""
    x_l, x_r = AXES["x_left_px"], AXES["x_right_px"]
    y_b, y_t = AXES["y_bottom_px"], AXES["y_top_px"]
    x_px = x_l + (z_phys / 500.0) * (x_r - x_l)
    y_in_panel = y_b - ((y_phys + 200.0) / 400.0) * (y_b - y_t)
    y_px = panel_idx * PANEL_HEIGHT + y_in_panel
    return int(round(x_px)), int(round(y_px))


# Frame indices: 49 X-slices at 8.333 cm steps starting at x = -200.
def frame_for_x(x_phys):
    return int(round((x_phys + 200.0) / 8.3333333))


def frame_x(idx):
    return -200.0 + idx * 8.3333333


SPARKSTOWN_FRAME  = frame_for_x(+50.0)   # = 30 (x = +50 cm)
SHORTSVILLE_FRAME = frame_for_x(-100.0)  # = 12 (x = -100 cm)

# Fade schedule: 9 frames centred on target.
FADE_OPACITIES = [0.10, 0.30, 0.60, 0.85, 1.00, 0.85, 0.60, 0.30, 0.10]


def opacity_at(frame_idx, target_frame):
    di = frame_idx - target_frame
    pos = di + 4   # target frame is index 4 of the 9-entry schedule
    if 0 <= pos < len(FADE_OPACITIES):
        return FADE_OPACITIES[pos]
    return 0.0


# Annotation positions. "Above the -dx feature" at x=+75 frame: the
# blue blob lives around (z, y) ~ (50, -180) in the dx fit pane; placing
# the label centred at (z, y) = (70, -110) sits clearly above and to
# the right of the feature. "Below the +dy feature" at x=-50 frame:
# the red blob sits at (z, y) ~ (360, +175); the label at (350, +115)
# is just below it.
ANNOTATIONS = [
    {
        "label": "Sparkstown",
        "target_frame": SPARKSTOWN_FRAME,
        "panel_idx": 0,           # dx panel (top)
        "z_phys":  70.0,
        "y_phys": -110.0,
        "anchor": "mb",           # middle-bottom; the label sits ABOVE
                                  # the feature, anchored at its bottom edge
        "fill_rgb":   (255,  20, 147),  # deep pink / hot magenta
        "outline_rgb": ( 50,   0,  30),
    },
    {
        "label": "Shortsville",
        "target_frame": SHORTSVILLE_FRAME,
        "panel_idx": 1,           # dy panel (middle)
        "z_phys": 350.0,
        "y_phys": 115.0,
        "anchor": "mt",           # middle-top; sits BELOW the feature
        "fill_rgb":   (  0, 230, 230),  # bright cyan / electric aqua
        "outline_rgb": (  0,  40,  60),
    },
]


def render_text_with_alpha(base_rgba, text, x, y, anchor, alpha, font,
                           fill_rgb, outline_rgb):
    """Composite `text` onto `base_rgba` at (x, y) with global alpha in
    [0, 1]. Coloured fill with a thicker dark outline for legibility
    over the matplotlib heatmap (which can be any colour from blue to
    red, so a single contrasting colour isn't enough — the outline
    handles the worst-case backgrounds)."""
    overlay = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    a = int(round(255 * max(0.0, min(1.0, alpha))))
    a_out = int(a * 0.95)
    # 3-pixel halo (16-direction) for thick, readable outline
    for r in (3, 2):
        for ox, oy in [(-r, 0), (r, 0), (0, -r), (0, r),
                       (-r, -r), (r, r), (-r, r), (r, -r)]:
            d.text((x + ox, y + oy), text, font=font,
                   fill=(*outline_rgb, a_out), anchor=anchor)
    d.text((x, y), text, font=font, fill=(*fill_rgb, a), anchor=anchor)
    return Image.alpha_composite(base_rgba, overlay)


def quantize(rgb):
    method = getattr(Image, "LIBIMAGEQUANT", None)
    if method is not None:
        try:
            return rgb.quantize(colors=255, method=method,
                                dither=Image.FLOYDSTEINBERG)
        except ValueError:
            pass
    return rgb.quantize(colors=255, dither=Image.FLOYDSTEINBERG)


def main():
    src = Image.open(SRC)
    duration = src.info.get("duration", 180)
    n = src.n_frames
    print(f"src: {SRC}\n  size {src.size}  frames {n}  duration {duration} ms")
    print(f"Sparkstown  target frame: {SPARKSTOWN_FRAME} "
          f"(x={frame_x(SPARKSTOWN_FRAME):+.0f} cm)")
    print(f"Shortsville target frame: {SHORTSVILLE_FRAME} "
          f"(x={frame_x(SHORTSVILLE_FRAME):+.0f} cm)")

    font = ImageFont.truetype(FONT_PATH, size=44)

    out_frames = []
    annotated_frame_set = set()
    for ann in ANNOTATIONS:
        for offset in range(-4, 5):
            annotated_frame_set.add(ann["target_frame"] + offset)

    for i, fr in enumerate(ImageSequence.Iterator(src)):
        rgba = fr.convert("RGBA").copy()
        if i in annotated_frame_set:
            for ann in ANNOTATIONS:
                a = opacity_at(i, ann["target_frame"])
                if a <= 0:
                    continue
                x_px, y_px = fit_pane_coords(ann["panel_idx"],
                                              ann["z_phys"], ann["y_phys"])
                rgba = render_text_with_alpha(
                    rgba, ann["label"], x_px, y_px,
                    anchor=ann["anchor"], alpha=a, font=font,
                    fill_rgb=ann["fill_rgb"],
                    outline_rgb=ann["outline_rgb"])
        out_frames.append(quantize(rgba.convert("RGB")))
        if i % 10 == 0:
            print(f"  frame {i:2d} processed", flush=True)

    print(f"writing {DST}", flush=True)
    out_frames[0].save(DST, save_all=True, append_images=out_frames[1:],
                       duration=duration, loop=0,
                       optimize=False, disposal=2)
    sz = os.path.getsize(DST) / 1e6
    print(f"  wrote {DST} ({sz:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
