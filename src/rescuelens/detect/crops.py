"""Build the images we show to Claude (and to the UI) for each candidate."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from rescuelens.detect.detector import Detection

OUTLINE = (255, 0, 255)  # magenta: rare in natural terrain, referenced by the prompt


def crop_box(
    det: Detection, img_w: int, img_h: int, *, context: float = 4.0, min_side: int = 96
) -> tuple[int, int, int, int]:
    """Square window centred on the detection, `context` times its longest side, clamped.

    The window is shifted (not shrunk) when it would cross the frame edge.
    """
    side = int(max(det.w, det.h) * context)
    side = max(min_side, min(side, img_w, img_h))
    cx, cy = det.center
    left = round(cx - side / 2)
    top = round(cy - side / 2)
    left = min(max(left, 0), img_w - side)
    top = min(max(top, 0), img_h - side)
    return left, top, left + side, top + side


def _jpeg(img: Image.Image, quality: int = 88) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def make_crop(image: Image.Image, det: Detection, *, out_size: int = 384) -> bytes:
    """Close-up around the detection with a thin outline around the detector box."""
    left, top, right, bottom = crop_box(det, image.width, image.height)
    crop = image.crop((left, top, right, bottom))
    scale = out_size / crop.width
    crop = crop.resize((out_size, out_size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(crop)
    pad = 3
    draw.rectangle(
        (
            (det.x - left) * scale - pad,
            (det.y - top) * scale - pad,
            (det.x + det.w - left) * scale + pad,
            (det.y + det.h - top) * scale + pad,
        ),
        outline=OUTLINE,
        width=1,
    )
    return _jpeg(crop)


def make_thumbnail(image: Image.Image, *, max_side: int = 768) -> tuple[Image.Image, float]:
    """Downscale the full frame once per frame; reused for every candidate in it."""
    scale = max_side / max(image.width, image.height)
    thumb = image.resize(
        (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
    )
    return thumb, scale


def make_context(thumb: Image.Image, scale: float, det: Detection, img_w: int, img_h: int) -> bytes:
    """Downscaled full frame with the candidate region outlined."""
    ctx = thumb.copy()
    left, top, right, bottom = crop_box(det, img_w, img_h)
    ImageDraw.Draw(ctx).rectangle(
        (left * scale, top * scale, right * scale, bottom * scale), outline=OUTLINE, width=2
    )
    return _jpeg(ctx, quality=80)
