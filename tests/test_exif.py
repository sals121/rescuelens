from pathlib import Path

import pytest
from PIL import ExifTags, Image
from PIL.TiffImagePlugin import IFDRational

from rescuelens.ingest.exif import dms_to_decimal, parse_gps_ifd, read_gps

G = ExifTags.GPS


def _write_jpeg_with_gps(path: Path, gps: dict[int, object]) -> None:
    exif = Image.Exif()
    exif[ExifTags.IFD.GPSInfo] = gps
    Image.new("RGB", (32, 24), "green").save(path, exif=exif)


def _dms(d: int, m: int, s: float) -> tuple[IFDRational, IFDRational, IFDRational]:
    return IFDRational(d, 1), IFDRational(m, 1), IFDRational(round(s * 100), 100)


def test_dms_to_decimal_hemispheres() -> None:
    assert dms_to_decimal((42, 38, 24.0), "N") == pytest.approx(42.64)
    assert dms_to_decimal((0, 2, 31.2), "W") == pytest.approx(-0.042)
    assert dms_to_decimal(((42, 1), (30, 1), (0, 1)), "S") == pytest.approx(-42.5)


def test_read_gps_roundtrip(tmp_path: Path) -> None:
    img = tmp_path / "drone.jpg"
    _write_jpeg_with_gps(
        img,
        {
            G.GPSLatitudeRef: "N",
            G.GPSLatitude: _dms(42, 38, 36.6),
            G.GPSLongitudeRef: "W",
            G.GPSLongitude: _dms(0, 2, 31.2),
            G.GPSAltitudeRef: b"\x00",
            G.GPSAltitude: IFDRational(18505, 10),
        },
    )
    fix = read_gps(img)
    assert fix is not None
    assert fix.lat == pytest.approx(42.643500, abs=1e-6)
    assert fix.lon == pytest.approx(-0.042, abs=1e-6)
    assert fix.alt_m == pytest.approx(1850.5)


def test_read_gps_absent(tmp_path: Path) -> None:
    img = tmp_path / "plain.jpg"
    Image.new("RGB", (8, 8)).save(img)
    assert read_gps(img) is None


def test_read_gps_unreadable_file(tmp_path: Path) -> None:
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    assert read_gps(bad) is None


@pytest.mark.parametrize(
    "gps",
    [
        {G.GPSLatitudeRef: "N", G.GPSLatitude: (42, 0, 0)},  # no longitude
        {
            G.GPSLatitudeRef: "N",
            G.GPSLatitude: (95, 0, 0),  # out of range
            G.GPSLongitudeRef: "E",
            G.GPSLongitude: (1, 0, 0),
        },
        {
            G.GPSLatitudeRef: "N",
            G.GPSLatitude: ((42, 0), (0, 1), (0, 1)),  # zero denominator
            G.GPSLongitudeRef: "E",
            G.GPSLongitude: (1, 0, 0),
        },
    ],
)
def test_parse_gps_invalid(gps: dict[int, object]) -> None:
    assert parse_gps_ifd(gps) is None
