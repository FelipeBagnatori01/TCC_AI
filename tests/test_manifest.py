from __future__ import annotations

from pathlib import Path

import pytest

from synthetic import make_system
from tcc_rca.ingestion.manifest import (
    build_manifest,
    read_manifest,
    verify_manifest,
    write_manifest,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    make_system(tmp_path)
    (tmp_path / "Telecom" / ".DS_Store").write_bytes(b"junk")
    return tmp_path


def test_manifest_round_trip_and_ignores_junk(root: Path, tmp_path: Path) -> None:
    entries = build_manifest(root, ["Telecom"])
    assert all(".DS_Store" not in e.path for e in entries)
    assert any(e.path == "Telecom/record.csv" for e in entries)
    out = tmp_path / "MANIFEST.csv"
    write_manifest(entries, out)
    assert read_manifest(out) == sorted(entries, key=lambda e: e.path)


def test_untouched_data_verifies_ok(root: Path) -> None:
    report = verify_manifest(root, build_manifest(root, ["Telecom"]))
    assert report.ok
    assert report.unlisted == []


def test_same_size_corruption_is_caught_by_checksum_only(root: Path) -> None:
    entries = build_manifest(root, ["Telecom"])
    record = root / "Telecom" / "record.csv"
    data = bytearray(record.read_bytes())
    data[-3] ^= 0x01  # flip one bit, size unchanged
    record.write_bytes(bytes(data))
    assert verify_manifest(root, entries, check_hashes=False).ok  # quick mode cannot see it
    report = verify_manifest(root, entries)
    assert report.checksum_mismatch == ["Telecom/record.csv"]
    assert not report.ok


def test_missing_truncated_and_extra_files(root: Path) -> None:
    entries = build_manifest(root, ["Telecom"])
    (root / "Telecom" / "query.csv").unlink()
    (root / "Telecom" / "record.csv").write_text("short", encoding="utf-8")
    (root / "Telecom" / "extra.csv").write_text("x", encoding="utf-8")
    report = verify_manifest(root, entries)
    assert report.missing == ["Telecom/query.csv"]
    assert report.size_mismatch == ["Telecom/record.csv"]
    assert report.unlisted == ["Telecom/extra.csv"]
    assert not report.ok


def test_empty_manifest_is_an_error(root: Path) -> None:
    with pytest.raises(ValueError, match="no entries"):
        verify_manifest(root, [])


def test_missing_system_directory_is_an_error(root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_manifest(root, ["Nope"])


def test_extra_root_level_file_is_listed_and_verified(root: Path) -> None:
    (root / "deploy.xlsx").write_bytes(b"sheet")
    entries = build_manifest(root, ["Telecom"], extra_files=[Path("deploy.xlsx")])
    assert any(e.path == "deploy.xlsx" for e in entries)
    assert verify_manifest(root, entries).ok
    (root / "deploy.xlsx").write_bytes(b"sheeT")
    assert verify_manifest(root, entries).checksum_mismatch == ["deploy.xlsx"]
    with pytest.raises(FileNotFoundError):
        build_manifest(root, ["Telecom"], extra_files=[Path("nope.xlsx")])
