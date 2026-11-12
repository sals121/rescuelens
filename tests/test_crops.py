import io

import pytest
from PIL import Image

from rescuelens.detect.crops import crop_box, make_context, make_crop, make_thumbnail
from rescuelens.detect.detector import Detection


@pytest.mark.parametrize(
    ("det", "expected"),
    [
        # centred: 4x the longest side (50) = 200 px square around the centre (1025, 525)
        (Detection(1000, 500, 50, 50, 0.9), (925, 425, 1125, 625)),
        # near the top-left corner: shifted inside the frame, not shrunk
        (Detection(5, 5, 50, 50, 0.9), (0, 0, 200, 200)),
        # near the bottom-right corner
        (Detection(3990, 2990, 10, 10, 0.9), (3904, 2904, 4000, 3000)),
        # tiny box: minimum side of 96 px
        (Detection(2000, 1500, 4, 4, 0.9), (1954, 1454, 2050, 1550)),
    ],
)
def test_crop_box(det: Detection, expected: tuple[int, int, int, int]) -> None:
    assert crop_box(det, 4000, 3000) == expected


def test_crop_box_never_exceeds_frame() -> None:
    det = Detection(0, 0, 900, 900, 0.9)  # 4x would be 3600 > height
    left, top, right, bottom = crop_box(det, 4000, 3000)
    assert (left, top) == (0, 0)
    assert right - left == bottom - top == 3000


def test_make_crop_and_context_are_jpegs() -> None:
    img = Image.new("RGB", (4000, 3000), "darkgreen")
    det = Detection(1800, 1400, 30, 60, 0.5)
    crop = Image.open(io.BytesIO(make_crop(img, det)))
    assert crop.format == "JPEG" and crop.size == (384, 384)

    thumb, scale = make_thumbnail(img)
    assert thumb.size == (768, 576) and scale == pytest.approx(0.192)
    ctx = Image.open(io.BytesIO(make_context(thumb, scale, det, 4000, 3000)))
    assert ctx.size == (768, 576)
    # the thumbnail itself must not be modified by drawing the outline
    assert thumb.getpixel((350, 270)) == img.getpixel((0, 0))
