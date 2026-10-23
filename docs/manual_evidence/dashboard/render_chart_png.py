"""Rasterise one inline SVG chart of the generated dashboard into a PNG.

This is *evidence tooling*, not part of the dashboard and not needed to use it. The
sandbox used to verify this change has no browser and no libcairo, so the charts that the
dashboard embeds were extracted from the generated ``index.html`` and rasterised with
Pillow using the DejaVu fonts that ship with Matplotlib. A browser renders the same SVG
directly; nothing in these PNGs was drawn by hand.

Usage::

    .venv/bin/python docs/manual_evidence/dashboard/render_chart_png.py chart.svg

Writes ``chart.png`` next to the input. Requires Pillow, which Matplotlib already
installs. Only the small subset of SVG that ``dashboard/charts.py`` emits is supported.
"""
from __future__ import annotations

import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import matplotlib

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts/ttf"
REGULAR = FONT_DIR / "DejaVuSans.ttf"
BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

_font_cache: dict = {}


def font(size: float, bold: bool = False):
    key = (round(size * 4), bold)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(str(BOLD if bold else REGULAR), max(1, int(round(size * 4))))
    return _font_cache[key]


def parse_color(value, opacity=1.0):
    if not value or value == "none":
        return None
    value = value.strip()
    m = re.match(r"#([0-9a-fA-F]{6})", value)
    if m:
        rgb = tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4))
        return rgb + (int(round(255 * opacity)),)
    return None


def render(path: Path, out: Path, scale: float = 2.0, background=(255, 255, 255)):
    tree = ET.parse(path)
    root = tree.getroot()
    vb = [float(v) for v in root.get("viewBox").split()]
    width, height = vb[2], vb[3]
    img = Image.new("RGBA", (int(width * scale), int(height * scale)), background + (255,))
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    def num(el, name, default=0.0):
        value = el.get(name)
        return float(value) if value not in (None, "") else default

    def draw_text(el, inherited):
        text = "".join(el.itertext())
        if not text.strip():
            return
        x, y = num(el, "x"), num(el, "y")
        size = num(el, "font-size", inherited.get("font-size", 12)) * scale
        bold = (el.get("font-weight") or inherited.get("font-weight") or "400") in ("600", "700", "bold")
        f = font(size / 4.0, bold)
        anchor = el.get("text-anchor") or inherited.get("text-anchor") or "start"
        fill = parse_color(el.get("fill") or inherited.get("fill") or "#000000") or (0, 0, 0, 255)
        bbox = draw.textbbox((0, 0), text, font=f)
        w = bbox[2] - bbox[0]
        px, py = x * scale, y * scale
        if anchor == "middle":
            px -= w / 2
        elif anchor == "end":
            px -= w
        draw.text((px, py), text, font=f, fill=fill, anchor="ls")

    def walk(el, inherited):
        tag = el.tag.split("}")[-1]
        attrs = dict(inherited)
        for key in ("fill", "font-size", "font-weight", "text-anchor"):
            if el.get(key):
                attrs[key] = el.get(key)
        if tag in ("svg", "g", "defs", "title", "desc"):
            if tag == "title" or tag == "desc":
                return
            if tag != "defs":
                for child in el:
                    walk(child, attrs)
            return
        if tag == "rect":
            x, y = num(el, "x"), num(el, "y")
            w, h = num(el, "width"), num(el, "height")
            rx = num(el, "rx")
            fill_raw = el.get("fill")
            fill_op = float(el.get("fill-opacity", 1) or 1)
            stroke = parse_color(el.get("stroke"), float(el.get("stroke-opacity", 1) or 1))
            sw = num(el, "stroke-width", 1)
            if fill_raw and fill_raw.startswith("url(#switch-hatch"):
                box = [x * scale, y * scale, (x + w) * scale, (y + h) * scale]
                draw.rectangle(box, fill=(238, 241, 244, 255))
                step = 6 * scale
                x0 = box[0] - (box[3] - box[1])
                while x0 < box[2]:
                    draw.line([(x0, box[3]), (x0 + (box[3] - box[1]), box[1])], fill=(185, 196, 209, 255), width=max(1, int(1.6 * scale)))
                    x0 += step
                if stroke:
                    draw.rectangle(box, outline=stroke, width=max(1, int(sw * scale)))
                return
            fill = parse_color(fill_raw, fill_op) if fill_raw else None
            box = [x * scale, y * scale, (x + w) * scale, (y + h) * scale]
            if fill or stroke:
                if rx:
                    draw.rounded_rectangle(box, radius=rx * scale, fill=fill, outline=stroke, width=max(1, int(sw * scale)) if stroke else 0)
                else:
                    draw.rectangle(box, fill=fill, outline=stroke, width=max(1, int(sw * scale)) if stroke else 0)
            return
        if tag == "line":
            stroke = parse_color(el.get("stroke") or "#000") or (0, 0, 0, 255)
            sw = num(el, "stroke-width", 1)
            points = [(num(el, "x1") * scale, num(el, "y1") * scale), (num(el, "x2") * scale, num(el, "y2") * scale)]
            dash = el.get("stroke-dasharray")
            if dash:
                pattern = [float(v) * scale for v in dash.replace(",", " ").split()]
                draw_dashed(draw, points, pattern, stroke, max(1, int(sw * scale)))
            else:
                draw.line(points, fill=stroke, width=max(1, int(sw * scale)))
            return
        if tag == "circle":
            cx, cy, r = num(el, "cx") * scale, num(el, "cy") * scale, num(el, "r") * scale
            fill = parse_color(el.get("fill")) if el.get("fill") else None
            stroke = parse_color(el.get("stroke")) if el.get("stroke") else None
            sw = max(1, int(num(el, "stroke-width", 1) * scale))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=stroke, width=sw if stroke else 0)
            return
        if tag == "path":
            d = el.get("d", "")
            nums = [float(v) for v in re.findall(r"-?\d+\.?\d*", d)]
            stroke = parse_color(el.get("stroke") or "#000") or (0, 0, 0, 255)
            sw = max(1, int(num(el, "stroke-width", 1) * scale))
            # supports "M x y l dx dy l dx dy" (chevrons) and "M x y l ..."
            if len(nums) >= 6 and "l" in d:
                x0, y0 = nums[0] * scale, nums[1] * scale
                x1, y1 = x0 + nums[2] * scale, y0 + nums[3] * scale
                x2, y2 = x1 + nums[4] * scale, y1 + nums[5] * scale
                draw.line([(x0, y0), (x1, y1), (x2, y2)], fill=stroke, width=sw, joint="curve")
            return
        if tag == "text":
            draw_text(el, attrs)
            return
        for child in el:
            walk(child, attrs)

    walk(root, {})
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(img, overlay).convert("RGB").save(out)
    return out


def draw_dashed(draw, points, pattern, fill, width):
    (x1, y1), (x2, y2) = points
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    pos, index, on = 0.0, 0, True
    while pos < length:
        seg = min(pattern[index % len(pattern)], length - pos)
        if on:
            draw.line([(x1 + ux * pos, y1 + uy * pos), (x1 + ux * (pos + seg), y1 + uy * (pos + seg))], fill=fill, width=width)
        pos += seg
        index += 1
        on = not on


if __name__ == "__main__":
    for source in sys.argv[1:]:
        src = Path(source)
        target = src.with_suffix(".png")
        render(src, target)
        print(target)
