"""On-disk response cache keyed by the request hash (R6)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tcc_rca.llm.types import ProviderResult


class DiskCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> ProviderResult | None:
        path = self._path(key)
        if not path.is_file():
            return None
        return ProviderResult.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, key: str, result: ProviderResult) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(result.model_dump(mode="json"), ensure_ascii=False), "utf-8")
        tmp.replace(path)  # atomic: a crash never leaves a half-written entry
