"""Content-addressed on-disk cache for LLM verdicts, so re-runs cost nothing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from rescuelens.logging import get_logger
from rescuelens.verify.schema import Verification

log = get_logger(__name__)


class CachedVerdict(BaseModel):
    verification: Verification
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int


def cache_key(*, model: str, prompt_version: str, images: list[bytes], text: str) -> str:
    """Hash everything that influences the answer: model, prompt, image bytes and text."""
    h = hashlib.sha256()
    for part in (model.encode(), prompt_version.encode(), text.encode(), *images):
        h.update(len(part).to_bytes(8, "big"))
        h.update(part)
    return h.hexdigest()


class VerdictCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> CachedVerdict | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return CachedVerdict.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            log.warning("cache_entry_corrupt", path=str(path), error=str(exc))
            return None

    def put(self, key: str, value: CachedVerdict) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(value.model_dump_json(), encoding="utf-8")
        tmp.replace(path)  # atomic rename on POSIX and Windows
