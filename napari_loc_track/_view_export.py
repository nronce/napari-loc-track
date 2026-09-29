"""The image layers on screen, composed as napari shows them, at their own resolution.

A screenshot has the screen's resolution and the window's decorations; this
resamples every visible image layer - at the time point on the slider - onto one
grid in world coordinates (nm here), at the finest pixel among them, and blends
them as napari does: each through its own contrast limits, gamma and colormap,
with its opacity and blending mode, bottom to top on black.

Out come the blended RGB (for a PNG, or an RGB TIFF) with the scale bar burned
in, and each layer's values on the common grid with its lookup table and display
range - an ImageJ composite TIFF opens with the same colours and contrast, every
channel still editable.

No Qt: the widget hands over the layers and the slider position.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _render as smlm_render

# Largest composite, in output pixels: past this the grid is coarsened.
MAX_PIXELS = 60_000_000


@dataclass
class Channel:
    name: str
    values: np.ndarray              # (H, W) float32, NaN outside the layer
    lut: np.ndarray                 # (256, 3) uint8
    display_range: tuple            # (low, high) contrast limits
    blending: str
    opacity: float


@dataclass
class Composite:
    rgb: np.ndarray                 # (H, W, 3) float in [0, 1]
    channels: list
    pixel_nm: float
    origin_nm: tuple                # world (y, x) of the centre of pixel (0, 0)
    notes: list = field(default_factory=list)

    def rgb8(self):
        return (np.clip(self.rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def _linear_2d(layer):
    """The 2x2 linear part and offset of a layer's data -> world map, last two axes."""
    affine = np.asarray(layer._data_to_world.affine_matrix, dtype=float)
    n = affine.shape[0] - 1
    return affine[n - 2:n, n - 2:n], affine[n - 2:n, n], affine


def layer_pixel_nm(layer):
    """World units per data pixel of a layer, along its image axes."""
    linear, _offset, _a = _linear_2d(layer)
    return float(math.sqrt(abs(np.linalg.det(linear))))


def _plane(layer, world_point):
    """The 2D data plane a layer shows at a world point (the slider position)."""
    data = layer.data
    if getattr(layer, "multiscale", False):
        data = data[0]
    image_axes = 3 if getattr(layer, "rgb", False) else 2
    leading = layer.ndim - 2
    if leading <= 0:
        return np.asarray(data)
    point = np.asarray(world_point, dtype=float)[-layer.ndim:]
    data_point = np.asarray(layer.world_to_data(point), dtype=float)
    index = tuple(int(np.clip(round(v), 0, s - 1))
                  for v, s in zip(data_point[:leading], data.shape[:leading]))
    plane = np.asarray(data[index])
    return plane if plane.ndim == image_axes else plane.reshape(plane.shape[-image_axes:])


def _sample(plane, layer, yy, xx, order):
    """The plane at world (yy, xx): NaN where the layer does not reach."""
    from scipy import ndimage

    linear, offset, _a = _linear_2d(layer)
    inverse = np.linalg.inv(linear)
    world = np.stack([yy.ravel(), xx.ravel()])
    data = inverse @ (world - offset[:, None])
    if plane.ndim == 3:        # RGB(A): each colour on its own
        return np.stack([ndimage.map_coordinates(plane[..., c].astype(np.float32), data,
                                                 order=order, mode="constant", cval=np.nan)
                         .reshape(yy.shape) for c in range(plane.shape[-1])], axis=-1)
    return ndimage.map_coordinates(np.asarray(plane, np.float32), data, order=order,
                                   mode="constant", cval=np.nan).reshape(yy.shape)


def _lut(layer):
    return (np.clip(layer.colormap.map(np.linspace(0, 1, 256))[:, :3], 0, 1) * 255
            + 0.5).astype(np.uint8)


def view_box(viewer, layers):
    """(y0, x0, y1, x1) in world units: what the canvas shows, within the layers."""
    union = layers_box(layers)
    size = getattr(viewer, "_canvas_size", None)
    zoom = float(getattr(viewer.camera, "zoom", 0) or 0)
    if not size or zoom <= 0:
        return union
    cy, cx = (float(v) for v in viewer.camera.center[-2:])
    hh, hw = size[0] / zoom / 2.0, size[1] / zoom / 2.0
    box = (max(cy - hh, union[0]), max(cx - hw, union[1]),
           min(cy + hh, union[2]), min(cx + hw, union[3]))
    return box if box[2] > box[0] and box[3] > box[1] else union


def layers_box(layers):
    """(y0, x0, y1, x1) world extent of the layers' image planes."""
    boxes = []
    for layer in layers:
        # napari's extent runs from the first pixel's centre to the last one's
        extent = np.asarray(layer.extent.world, dtype=float)[:, -2:]
        half = layer_pixel_nm(layer) / 2.0
        boxes.append((extent[0, 0] - half, extent[0, 1] - half,
                      extent[1, 0] + half, extent[1, 1] + half))
    b = np.array(boxes)
    return (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())


def compose(layers, world_point, box, pixel_nm=None, max_pixels=MAX_PIXELS):
    """The visible image layers, bottom to top, blended onto one grid over box."""
    layers = [layer for layer in layers if layer.visible and layer.opacity > 0]
    if not layers:
        raise ValueError("no visible image layer to save")
    notes = []
    finest = min(layer_pixel_nm(layer) for layer in layers)
    pixel = float(pixel_nm) if pixel_nm else finest
    y0, x0, y1, x1 = box
    h = max(1, int(round((y1 - y0) / pixel)))
    w = max(1, int(round((x1 - x0) / pixel)))
    if h * w > max_pixels:
        factor = math.sqrt(h * w / max_pixels)
        pixel *= factor
        h, w = max(1, int(round((y1 - y0) / pixel))), max(1, int(round((x1 - x0) / pixel)))
        notes.append(f"coarsened to {pixel:.1f} nm/px to stay under {max_pixels / 1e6:.0f} MP")
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    yy = y0 + (yy + 0.5) * pixel
    xx = x0 + (xx + 0.5) * pixel
    out = np.zeros((h, w, 3), np.float32)
    channels = []
    for layer in layers:
        plane = _plane(layer, world_point)
        order = 0 if str(getattr(layer, "interpolation2d", "nearest")) == "nearest" else 1
        values = _sample(plane, layer, yy, xx, order)
        lo, hi = (float(v) for v in layer.contrast_limits)
        span = hi - lo if hi != lo else 1.0
        gamma = float(getattr(layer, "gamma", 1.0))
        inside = np.isfinite(values) if values.ndim == 2 else np.isfinite(values).all(axis=-1)
        if values.ndim == 3:
            rgba = np.clip((np.nan_to_num(values) - lo) / span, 0, 1) ** gamma
            color = rgba[..., :3]
            alpha = rgba[..., 3] if rgba.shape[-1] == 4 else np.ones((h, w), np.float32)
        else:
            v = np.clip((np.nan_to_num(values, nan=lo) - lo) / span, 0, 1) ** gamma
            mapped = layer.colormap.map(v.ravel()).reshape(h, w, 4)
            color, alpha = mapped[..., :3], mapped[..., 3]
            channels.append(Channel(layer.name, values.astype(np.float32), _lut(layer),
                                    (lo, hi), str(layer.blending), float(layer.opacity)))
        a = (alpha * float(layer.opacity) * inside)[..., None].astype(np.float32)
        blending = str(layer.blending)
        if blending == "additive":
            out = out + color * a
        elif blending == "minimum":
            out = np.where(inside[..., None], np.minimum(out, color), out)
        elif blending == "opaque":
            out = np.where(inside[..., None], color, out)
        else:                                   # translucent, translucent_no_depth
            out = out * (1 - a) + color * a
    return Composite(np.clip(out, 0, 1), channels, pixel, (y0 + pixel / 2, x0 + pixel / 2), notes)


def burn_scale_bar(rgb8, pixel_nm, length_nm=None, color="white", position="bottom right"):
    """A scale bar with its label, into an (H, W, 3) uint8 image, in place.
    Returns the bar's length in nm, or None if it would not fit."""
    h, w = rgb8.shape[:2]
    width_nm = w * pixel_nm
    length_nm = float(length_nm or smlm_render.nice_scale_length(width_nm))
    length_px = int(round(length_nm / pixel_nm))
    if length_px < 2 or length_px > 0.9 * w:
        return None
    thickness = max(2, int(round(min(h, w) * 0.008)))
    label_height = max(8, int(round(min(h, w) * 0.03)))
    label = smlm_render.compose_text(smlm_render.glyph_atlas(label_height),
                                     smlm_render.format_length(length_nm))
    mask = smlm_render.scale_bar_mask(length_px, thickness, label)
    smlm_render.burn_text(rgb8, mask, color=color, position=position)
    return length_nm


def save_png(path, rgb8, pixel_nm):
    from PIL import Image

    dpi = 25.4e6 / pixel_nm                        # pixels per inch at the sample
    Image.fromarray(rgb8).save(path, dpi=(dpi, dpi))
    return Path(path)


def save_rgb_tiff(path, rgb8, pixel_nm, description=""):
    import tifffile

    tifffile.imwrite(path, rgb8, photometric="rgb", imagej=True,
                     resolution=(1000.0 / pixel_nm, 1000.0 / pixel_nm),
                     metadata={"unit": "micron", "Info": description})
    return Path(path)


def save_channels_tiff(path, composite, description=""):
    """Each layer a channel, float32, with its LUT and display range: ImageJ opens
    it as a composite in the same colours and contrast."""
    import tifffile

    if not composite.channels:
        raise ValueError("no single-channel image layer to write as a channel")
    stack = np.stack([np.nan_to_num(c.values, nan=0.0) for c in composite.channels]).astype(np.float32)
    luts = [np.ascontiguousarray(c.lut.T) for c in composite.channels]         # (3, 256)
    ranges = [v for c in composite.channels for v in c.display_range]
    labels = [c.name for c in composite.channels]
    tifffile.imwrite(path, stack, imagej=True,
                     resolution=(1000.0 / composite.pixel_nm, 1000.0 / composite.pixel_nm),
                     metadata={"axes": "CYX", "mode": "composite", "unit": "micron",
                               "LUTs": luts, "Ranges": ranges, "Labels": labels,
                               "Info": description})
    return Path(path)
