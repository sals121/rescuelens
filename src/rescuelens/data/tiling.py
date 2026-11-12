"""Cut large frames into overlapping tiles, the same way SAHI slices them at inference."""

from __future__ import annotations

from rescuelens.data.voc import Box


def tile_origins(length: int, tile: int, overlap: float) -> list[int]:
    """Start offsets along one axis.

    Mirrors ``sahi.slicing.get_slice_bboxes``: stride ``tile * (1 - overlap)``, and the last
    tile is shifted back so it ends exactly at the border (no padding, no partial tiles).
    """
    if tile <= 0 or not 0.0 <= overlap < 1.0:
        raise ValueError("tile must be > 0 and 0 <= overlap < 1")
    if length <= tile:
        return [0]
    stride = int(tile * (1.0 - overlap))
    origins: list[int] = []
    pos = 0
    while pos < length:
        end = pos + tile
        if end > length:
            origins.append(max(0, length - tile))
            break
        origins.append(pos)
        if end == length:
            break
        pos += stride
    return origins


def tile_grid(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int]]:
    """(x, y) origins of every tile, row by row."""
    return [
        (x, y)
        for y in tile_origins(height, tile, overlap)
        for x in tile_origins(width, tile, overlap)
    ]


def boxes_in_tile(
    boxes: list[Box], x0: int, y0: int, tile: int, min_visible: float = 0.5
) -> tuple[list[Box], int]:
    """Clip boxes to a tile and shift them to tile coordinates.

    A box is kept when at least ``min_visible`` of its area falls inside the tile.
    Returns the kept boxes and how many boxes were only partially visible and dropped
    (those people still appear complete in a neighbouring overlapping tile).
    """
    kept: list[Box] = []
    dropped = 0
    for b in boxes:
        ix0, iy0 = max(b.xmin, x0), max(b.ymin, y0)
        ix1, iy1 = min(b.xmax, x0 + tile), min(b.ymax, y0 + tile)
        inter = Box(ix0, iy0, ix1, iy1)
        if inter.w <= 0 or inter.h <= 0:
            continue
        if inter.area / b.area >= min_visible:
            kept.append(Box(ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0))
        else:
            dropped += 1
    return kept, dropped
