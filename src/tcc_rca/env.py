"""Capture of the execution environment: versions, git commit, hardware, dataset hash."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_info(cwd: Path | None = None) -> dict[str, Any]:
    """Commit hash and whether the working tree has uncommitted changes."""
    root = cwd or Path.cwd()
    commit = _git(["rev-parse", "HEAD"], root)
    status = _git(["status", "--porcelain"], root)
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_hash(manifest: Path) -> str | None:
    """Hash of the versioned data manifest, or None while the manifest does not exist."""
    return sha256_file(manifest) if manifest.is_file() else None


def hash_config(resolved: dict[str, Any]) -> str:
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _physical_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def capture_environment(cwd: Path | None = None) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": _physical_memory_bytes(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "git": git_info(cwd),
    }
