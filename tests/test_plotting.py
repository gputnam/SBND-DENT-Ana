"""Unit tests for the shared plotting / GIF helpers."""
import os

import numpy as np
from PIL import Image

from crtlib import plotting as P


def test_quantize_rgb_returns_palette_image():
    img = Image.fromarray((np.random.default_rng(0)
                           .integers(0, 255, (16, 16, 3))).astype("uint8"), "RGB")
    q = P.quantize_rgb(img)
    assert q.mode == "P"


def test_make_gif(tmp_path):
    pngs = []
    for i in range(3):
        a = np.full((12, 12, 3), i * 60, dtype="uint8")
        p = str(tmp_path / f"f{i}.png")
        Image.fromarray(a, "RGB").save(p)
        pngs.append(p)
    gif = str(tmp_path / "out.gif")
    P.make_gif(pngs, gif, duration_ms=100)
    assert os.path.exists(gif) and os.path.getsize(gif) > 0


def test_make_gif_empty_is_noop(tmp_path):
    gif = str(tmp_path / "empty.gif")
    P.make_gif([], gif)
    assert not os.path.exists(gif)


def test_plot_xslice_3pane(tmp_path):
    xcen = np.array([-150.0, 150.0])
    g = np.random.default_rng(1).standard_normal((2, 8, 10))
    paths = P.plot_xslice_3pane(xcen, [g, g, g * 0],
                                ["a", "b", "diff"], str(tmp_path), vrange=3.0)
    assert len(paths) == 2
    assert all(os.path.exists(p) for p in paths)
