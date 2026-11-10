"""Versioned data manifest (path, size, SHA-256) and local integrity verification (R1)."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from tcc_rca.env import sha256_file

HEADER = ["path", "size_bytes", "sha256"]
_IGNORED_NAMES = {".DS_Store"}


@dataclass(frozen=True)
class ManifestEntry:
    path: str  # relative to the data root, POSIX separators
    size_bytes: int
    sha256: str


@dataclass
class VerificationReport:
    missing: list[str] = field(default_factory=list)
    size_mismatch: list[str] = field(default_factory=list)
    checksum_mismatch: list[str] = field(default_factory=list)
    unlisted: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.size_mismatch or self.checksum_mismatch)


def list_data_files(data_root: Path, systems: Iterable[str]) -> list[Path]:
    """Data files (relative to ``data_root``) of the given systems, sorted, junk excluded."""
    found: list[Path] = []
    for system in systems:
        base = data_root / system
        if not base.is_dir():
            raise FileNotFoundError(f"system directory not found: {base}")
        found.extend(
            p.relative_to(data_root)
            for p in base.rglob("*")
            if p.is_file() and p.name not in _IGNORED_NAMES
        )
    return sorted(found)


def build_manifest(
    data_root: Path, systems: Iterable[str], workers: int = 4
) -> list[ManifestEntry]:
    files = list_data_files(data_root, systems)

    def entry(rel: Path) -> ManifestEntry:
        full = data_root / rel
        return ManifestEntry(rel.as_posix(), full.stat().st_size, sha256_file(full))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(entry, files))


def write_manifest(entries: Iterable[ManifestEntry], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HEADER)
        for e in sorted(entries, key=lambda e: e.path):
            writer.writerow([e.path, e.size_bytes, e.sha256])


def read_manifest(path: Path) -> list[ManifestEntry]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != HEADER:
            raise ValueError(f"{path}: unexpected header {reader.fieldnames}")
        return [ManifestEntry(r["path"], int(r["size_bytes"]), r["sha256"]) for r in reader]


def verify_manifest(
    data_root: Path, entries: list[ManifestEntry], check_hashes: bool = True, workers: int = 4
) -> VerificationReport:
    """Compare local files against the manifest. Sizes first; checksums only if sizes match."""
    if not entries:
        raise ValueError("manifest has no entries; run `tcc-rca manifest` first")
    report = VerificationReport()
    to_hash: list[ManifestEntry] = []
    for e in entries:
        full = data_root / e.path
        if not full.is_file():
            report.missing.append(e.path)
        elif full.stat().st_size != e.size_bytes:
            report.size_mismatch.append(e.path)
        else:
            to_hash.append(e)
    if check_hashes:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            digests = list(pool.map(lambda e: sha256_file(data_root / e.path), to_hash))
        report.checksum_mismatch = [
            e.path for e, d in zip(to_hash, digests, strict=True) if d != e.sha256
        ]
    systems = {e.path.split("/", 1)[0] for e in entries}
    listed = {e.path for e in entries}
    report.unlisted = [
        p.as_posix() for p in list_data_files(data_root, systems) if p.as_posix() not in listed
    ]
    return report
