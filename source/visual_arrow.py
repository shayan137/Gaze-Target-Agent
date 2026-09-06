"""Draws the gaze-target visual prompt onto an image: arrow, dot, or heatmap overlay."""
import math

import numpy as np
from PIL import Image, ImageDraw

COLORS = {
    "red": (255, 0, 0),
    "yellow": (255, 220, 0),
    "green": (0, 200, 0),
    "blue": (0, 0, 255),
}


def resolve_color(value):
    if isinstance(value, str):
        return COLORS[value.lower()]
    return tuple(value)


def head_bbox_center(head_bbox_px):
    x1, y1, x2, y2 = head_bbox_px
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def draw_arrow(image: Image.Image, start_xy, end_xy, color, width: int = 4) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    x0, y0 = start_xy
    x1, y1 = end_xy
    draw.line([(x0, y0), (x1, y1)], fill=color, width=width)

    length = math.hypot(x1 - x0, y1 - y0)
    if length >= 8:
        angle = math.atan2(y1 - y0, x1 - x0)
        head_len = max(12, width * 3)
        head_angle = math.radians(25)
        for sign in (1, -1):
            hx = x1 - head_len * math.cos(angle - sign * head_angle)
            hy = y1 - head_len * math.sin(angle - sign * head_angle)
            draw.line([(x1, y1), (hx, hy)], fill=color, width=width)
    return img


def draw_dot(image: Image.Image, xy, color, radius: int = 6) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    x, y = xy
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
    return img


def draw_heatmap_overlay(image: Image.Image, heatmap, color, alpha: float = 0.5) -> Image.Image:
    hm = heatmap.detach().cpu().numpy() if hasattr(heatmap, "detach") else np.asarray(heatmap)
    hm = hm - hm.min()
    if hm.max() > 0:
        hm = hm / hm.max()

    hm_img = Image.fromarray((hm * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    hm_arr = np.asarray(hm_img).astype(np.float32) / 255.0

    base = image.convert("RGB")
    overlay = Image.new("RGB", image.size, color)
    alpha_arr = (hm_arr * alpha * 255).astype(np.uint8)
    alpha_img = Image.fromarray(alpha_arr, mode="L")
    return Image.composite(overlay, base, alpha_img)


def draw_gaze_prompt(
    image: Image.Image,
    head_bbox_px,
    gaze_xy_px,
    style: str = "arrow",
    color="blue",
    arrow_width: int = 4,
    dot_radius: int = 6,
    heatmap_alpha: float = 0.5,
    heatmap=None,
) -> Image.Image:
    rgb = resolve_color(color)
    if style == "arrow":
        return draw_arrow(image, head_bbox_center(head_bbox_px), gaze_xy_px, rgb, arrow_width)
    if style == "dot":
        return draw_dot(image, gaze_xy_px, rgb, dot_radius)
    if style == "heatmap":
        if heatmap is None:
            raise ValueError("style='heatmap' requires a heatmap tensor")
        return draw_heatmap_overlay(image, heatmap, rgb, heatmap_alpha)
    raise ValueError(f"unknown style: {style!r} (expected arrow, dot, or heatmap)")
