"""Shared plotting / GIF helpers for the distortion pipelines.

Consolidates the LIBIMAGEQUANT palette-quantization and animated-GIF
assembly that were copied between ``build_sce_input`` and ``stitch_gifs``,
plus a reusable per-X-slice 3-pane heatmap figure generator.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def quantize_rgb(rgb_img, colors=255):
    """Quantize an RGB PIL image to a <=256-colour palette.

    Prefers the LIBIMAGEQUANT quantizer with Floyd-Steinberg dither (per the
    CLAUDE.md GIF note -- it preserves light pastel heatmap shades the default
    median-cut collapses), falling back to the default adaptive palette when
    PIL was built without LIBIMAGEQUANT.
    """
    method = getattr(Image, "LIBIMAGEQUANT", None)
    if method is not None:
        try:
            return rgb_img.quantize(colors=colors, method=method,
                                    dither=Image.FLOYDSTEINBERG)
        except ValueError:
            pass
    return rgb_img.quantize(colors=colors, dither=Image.FLOYDSTEINBERG)


def make_gif(png_paths, gif_path, duration_ms=200):
    """Assemble an animated GIF from a list of PNG paths, quantized with
    :func:`quantize_rgb`. No-op if ``png_paths`` is empty."""
    frames = [quantize_rgb(Image.open(p).convert("RGB")) for p in png_paths]
    if not frames:
        return
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=False, disposal=2)


def plot_xslice_3pane(x_centers, grids, titles, outdir, *, vrange,
                      extent=(0, 500, -200, 200), xlabel="Z [cm]",
                      ylabel="Y [cm]", cmap_name="RdYlBu_r",
                      resid_vrange=None, dpi=110, fname_fmt="x{:+04d}.png"):
    """One 3-pane heatmap figure per X-slice.

    Parameters
    ----------
    x_centers : (nx,) X coordinates, one figure each.
    grids     : 3 arrays of shape (nx, ny, nz) -- [panel0, panel1, panel2].
                A panel that is all-NaN for a slice is drawn as an empty axis.
    titles    : 3 panel titles.
    outdir    : directory for the PNGs (created if absent).
    vrange    : colour half-range for the first two panels.
    resid_vrange : half-range for the 3rd (residual/diff) panel; default
                ``max(0.5*vrange, 0.5)``.

    Returns the list of written PNG paths, in X order.
    """
    os.makedirs(outdir, exist_ok=True)
    cmap = matplotlib.colormaps[cmap_name].copy()
    cmap.set_bad("white")
    if resid_vrange is None:
        resid_vrange = max(0.5 * vrange, 0.5)
    vrs = [vrange, vrange, resid_vrange]
    paths = []
    for ix, xc in enumerate(x_centers):
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        for ax, grid, title, vr in zip(axs, grids, titles, vrs):
            slc = None if grid is None else grid[ix]
            if slc is None or not np.any(np.isfinite(slc)):
                ax.set_axis_off(); ax.set_title(title); continue
            im = ax.imshow(slc, origin="lower", aspect="equal", extent=extent,
                           cmap=cmap, vmin=-vr, vmax=vr, interpolation="nearest")
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"X = {xc:+.1f} cm", fontsize=12)
        fig.tight_layout()
        fname = os.path.join(outdir, fname_fmt.format(int(round(xc))))
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
        paths.append(fname)
    return paths
