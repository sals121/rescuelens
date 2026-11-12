"""Resumable, checksum-verified download and safe extraction of the HERIDAL archive."""

from __future__ import annotations

import hashlib
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.error import URLError

from tqdm import tqdm

from rescuelens.logging import get_logger

log = get_logger(__name__)

# Aalto University mirror of HERIDAL in Pascal VOC format (Zenodo record 5662351, CC BY 3.0).
HERIDAL_URL = "https://zenodo.org/records/5662351/files/heridal_keras_retinanet_voc.zip?download=1"
HERIDAL_MD5 = "c2b625ae986898a870b9d8a907aefe45"
HERIDAL_SIZE = 8_311_735_812
_CHUNK = 8 * 1024 * 1024


class ChecksumError(RuntimeError):
    pass


def md5sum(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        while block := f.read(_CHUNK):
            h.update(block)
    return h.hexdigest()


def download(
    url: str,
    dest: Path,
    *,
    expected_md5: str,
    expected_size: int | None = None,
    retries: int = 20,
) -> Path:
    """Download ``url`` to ``dest``, resuming partial files, then verify the md5.

    Zenodo drops long connections now and then, so every failure resumes from the bytes
    already on disk (HTTP Range) instead of starting over.
    """
    if not url.startswith("https://"):
        raise ValueError("only https URLs are allowed")
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if expected_size is not None and have >= expected_size:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})  # noqa: S310 - https enforced above
        try:
            with (
                urllib.request.urlopen(req, timeout=60) as resp,  # noqa: S310
                dest.open("ab" if have and resp.status == 206 else "wb") as out,
                tqdm(
                    total=expected_size,
                    initial=have if resp.status == 206 else 0,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                ) as bar,
            ):
                while block := resp.read(_CHUNK):
                    out.write(block)
                    bar.update(len(block))
            if expected_size is None:
                break
        except (URLError, TimeoutError, ConnectionError) as exc:
            log.warning("download_interrupted", attempt=attempt, error=str(exc))
            time.sleep(min(30, 2 * attempt))
    else:
        raise RuntimeError(f"download failed after {retries} attempts: {url}")

    log.info("verifying_md5", file=str(dest))
    got = md5sum(dest)
    if got != expected_md5:
        raise ChecksumError(f"{dest.name}: md5 {got} != expected {expected_md5}")
    return dest


def safe_extract(archive: Path, out_dir: Path) -> int:
    """Extract a zip, skipping macOS metadata and refusing paths that escape ``out_dir``."""
    out_dir = out_dir.resolve()
    n = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or name.startswith("__MACOSX/") or name.endswith(".DS_Store"):
                continue
            target = (out_dir / name).resolve()
            if not target.is_relative_to(out_dir):
                raise ValueError(f"unsafe path in archive: {name}")
            zf.extract(info, out_dir)
            n += 1
    return n
