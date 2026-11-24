"""E9 (ADR 0024): reproduction plan, table comparison and the cache bundle."""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

import pytest

from tcc_rca.bundle import compare_evidence, make_bundle, restore_bundle
from tcc_rca.config import Config, load_config
from tcc_rca.evaluation.reproduce_check import check, diff_values
from tcc_rca.reproduce import LEVELS, SYSTEMS, plan


def test_every_planned_config_exists(repo_root: Path) -> None:
    for level in LEVELS:
        for step in plan(level):
            for cfg in step.configs:
                assert (repo_root / cfg).is_file(), cfg


def test_principal_runs_the_thesis_chain_in_order() -> None:
    steps = plan("principal")
    labels = [s.command for s in steps]
    assert labels[0] == "verify"
    assert labels[-1] == "rq1-curves"
    for system in SYSTEMS:
        mine = [s.command for s in steps if s.system == system]
        assert mine == [
            "convert", "series", "tasks", "baselines", "lstm", "compare-e5", "xai",
            "evidence-select",
        ]  # fmt: skip
    completo = [s.command for s in plan("completo")]
    assert completo.count("alt-models") == 8  # 4 configurations x 2 systems


def test_agent_replays_merge_the_zero_budget_last() -> None:
    steps = plan("agente-cache", ["Telecom"])
    agents = [s for s in steps if s.command == "agent"]
    assert len(agents) == 9  # 3 models x 3 variants
    assert all(s.configs[-1].endswith("_replay.yaml") for s in agents)
    assert [s.command for s in steps[-3:]] == ["evaluate-agent", "compare-models", "compare-models"]


def test_argv_puts_user_files_last() -> None:
    step = plan("principal", ["Bank"])[1]  # convert, Bank
    argv = step.argv(["configs/base.yaml"], ["configs/local.yaml"])
    assert argv == [
        "convert", "--config", "configs/base.yaml", "configs/system/bank.yaml", "configs/local.yaml"
    ]  # fmt: skip
    with pytest.raises(ValueError, match="unknown level"):
        plan("tudo")


def test_diff_ignores_run_dirs_and_wall_clock_but_not_values() -> None:
    ref = {"run_dir": "20261119T170412-lstm", "elapsed_s": 3.0, "auc": 0.5, "n": 3, "s": "a"}
    same = {"run_dir": "20261201T000000-lstm", "elapsed_s": 9.0, "auc": 0.5, "n": 3, "s": "a"}
    assert list(diff_values(ref, same, 0.0, 0.0)) == []
    off = {**same, "auc": 0.5000001}
    assert list(diff_values(ref, off, 0.0, 0.0)) == ["/auc: 0.5 != 0.5000001"]
    assert list(diff_values(ref, off, 1e-6, 0.0)) == []  # within the tolerance
    assert list(diff_values(ref, {**same, "n": 4}, 1.0, 0.0)) != []  # 3 vs 4 is 33%: outside
    assert list(diff_values([["x", "1.0"]], [["x", "1.00"]], 0.0, 0.0)) == []  # CSV numbers
    assert list(diff_values({"a": 1}, {"b": 1}, 0.0, 0.0))  # missing key


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_check_compares_only_tables_written_after_the_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tables = tmp_path / "reports" / "tables"
    tables.mkdir(parents=True)
    for name in ("a.json", "b.json", "c.json"):
        (tables / name).write_text(json.dumps({"v": 1.0}), encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "ref")
    monkeypatch.chdir(tmp_path)
    old = time.time() - 100
    os.utime(tables / "c.json", (old, old))  # not regenerated
    start = time.time() - 50
    (tables / "b.json").write_text(json.dumps({"v": 2.0}), encoding="utf-8")
    (tables / "d.json").write_text("{}", encoding="utf-8")  # new, untracked
    out = check(Path("reports/tables"), start)
    status = {k: v["status"] for k, v in out["files"].items()}
    assert status == {"a.json": "igual", "b.json": "diferente", "c.json": "nao_regenerado",
                      "d.json": "novo"}  # fmt: skip
    assert not out["ok"]


def _bundle_config(tmp_path: Path, repo_root: Path, root: str) -> Config:
    base = tmp_path / root
    return load_config(
        repo_root / "configs/base.yaml",
        overrides={
            "paths": {
                "experiments_dir": str(base / "experiments"),
                "processed_dir": str(base / "processed"),
            },
            "llm": {"cache_dir": str(base / "experiments" / "llm_cache")},
        },
    )


def test_bundle_round_trip_checksums_and_evidence(tmp_path: Path, repo_root: Path) -> None:
    src = _bundle_config(tmp_path, repo_root, "origem")
    (src.llm.cache_dir / "ab").mkdir(parents=True)
    (src.llm.cache_dir / "ab" / "key.json").write_text('{"text": "ok"}', encoding="utf-8")
    run = src.paths.experiments_dir / "20261119T000000-agent_a1"
    (run / "artifacts").mkdir(parents=True)
    (run / "artifacts" / "predictions_s0.csv").write_text("row,prediction\n", encoding="utf-8")
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "agent_Telecom_agent_a1_claude-opus-5-5.json").write_text(
        json.dumps({"run_dir": run.name}), encoding="utf-8"
    )
    for system in SYSTEMS:
        ev = src.paths.processed_dir / system / "evidence" / src.agent.evidence_name
        ev.mkdir(parents=True)
        (ev / "query_001.json").write_text('{"k": 1}', encoding="utf-8")
    out = make_bundle(src, tables, tmp_path / "pacote.tar.gz")
    assert (out["files"], out["runs"]) == (4, 1)
    with tarfile.open(tmp_path / "pacote.tar.gz") as tar:
        assert {"LEIA-ME.txt", "MANIFEST.json"} <= set(tar.getnames())

    dst = _bundle_config(tmp_path, repo_root, "destino")
    ev_dst = dst.paths.processed_dir / "Telecom" / "evidence" / dst.agent.evidence_name
    ev_dst.mkdir(parents=True)
    (ev_dst / "query_001.json").write_text('{"k": 2}', encoding="utf-8")  # regenerated, differs
    restored = restore_bundle(dst, tmp_path / "pacote.tar.gz", use_reference_evidence=True)
    assert restored["copied"] == 2  # cache entry + run file
    assert (dst.paths.experiments_dir / run.name / "artifacts" / "predictions_s0.csv").exists()
    assert not restored["evidence"]["Telecom"]["equal"]
    assert (ev_dst / "query_001.json").read_text(encoding="utf-8") == '{"k": 1}'
    assert (ev_dst.with_name(ev_dst.name + "_regenerado") / "query_001.json").exists()
    reference = dst.paths.experiments_dir / "reproduce" / "pacote" / "referencia" / "evidence"
    assert compare_evidence(dst, reference)["Telecom"]["equal"]
    # a second restore finds identical files and copies nothing
    assert restore_bundle(dst, tmp_path / "pacote.tar.gz", False)["already_present"] == 2


def test_bundle_refuses_api_keys(tmp_path: Path, repo_root: Path) -> None:
    src = _bundle_config(tmp_path, repo_root, "origem")
    src.llm.cache_dir.mkdir(parents=True)
    (src.llm.cache_dir / "k.json").write_text("sk-ant-" + "a" * 40, encoding="utf-8")
    for system in SYSTEMS:
        (src.paths.processed_dir / system / "evidence" / src.agent.evidence_name).mkdir(
            parents=True
        )
    with pytest.raises(ValueError, match="API key"):
        make_bundle(src, tmp_path, tmp_path / "x.tar.gz")


def test_repo_audit_rules() -> None:
    from tcc_rca.repo_audit import FORBIDDEN_PATH, KEY

    forbidden = [
        "data/raw/openrca/Telecom/2020_04_11/metric/metric_app.csv",
        "data/processed/Telecom/series.parquet",
        "Telecom/record.csv",
        ".env",
        ".env.local",
        "experiments/llm_cache/ab/key.json",
        "model.pt",
    ]
    allowed = ["data/raw/.gitkeep", ".env.example", "experiments/.gitkeep", "src/tcc_rca/cli.py"]
    assert all(FORBIDDEN_PATH.search(p) for p in forbidden)
    assert not any(FORBIDDEN_PATH.search(p) for p in allowed)
    assert KEY.search(b"ANTHROPIC_API_KEY=" + b"x" * 20)
    assert KEY.search(b"sk-ant-api03-" + b"a" * 30)
    assert not KEY.search(b"# ANTHROPIC_API_KEY=\n")  # the empty template line


def test_preflight_names_missing_data_and_submodule(tmp_path: Path, repo_root: Path) -> None:
    from tcc_rca.reproduce import preflight

    cfg = load_config(
        repo_root / "configs/base.yaml",
        overrides={
            "paths": {
                "data_root": str(tmp_path / "openrca"),
                "openrca_repo": str(tmp_path / "OpenRCA"),
            }
        },
    )
    problems = preflight(cfg, SYSTEMS)
    assert len(problems) == 4  # Telecom, Bank, deployment list, empty submodule
    assert any("git submodule update --init" in p for p in problems)
    for system in SYSTEMS:
        (tmp_path / "openrca" / system).mkdir(parents=True)
        (tmp_path / "openrca" / system / "query.csv").write_text("", encoding="utf-8")
    assert cfg.series.deployment_list is not None
    (tmp_path / "openrca" / cfg.series.deployment_list).write_bytes(b"")
    (tmp_path / "OpenRCA" / "main").mkdir(parents=True)
    (tmp_path / "OpenRCA" / "main" / "evaluate.py").write_text("", encoding="utf-8")
    assert preflight(cfg, SYSTEMS) == []


def test_preflight_requires_the_restored_cache_for_agent_replays(
    tmp_path: Path, repo_root: Path
) -> None:
    from tcc_rca.reproduce import preflight

    tables = tmp_path / "tables"
    tables.mkdir()
    run = "experiments/20261119T194655-agent_a1"
    (tables / "agent_Telecom_agent_a1_claude-opus-5-5.json").write_text(
        json.dumps({"run_dir": run}), encoding="utf-8"
    )
    cfg = load_config(
        repo_root / "configs/base.yaml",
        overrides={
            "paths": {"experiments_dir": str(tmp_path / "exp"), "openrca_repo": str(repo_root)},
            "llm": {"cache_dir": str(tmp_path / "exp" / "llm_cache")},
        },
    )

    def cache_problems() -> list[str]:
        return [p for p in preflight(cfg, [], "agente-cache", tables) if "cache-restore" in p]

    assert not [p for p in preflight(cfg, [], "principal", tables) if "cache-restore" in p]
    assert len(cache_problems()) == 1
    (tmp_path / "exp" / "llm_cache").mkdir(parents=True)
    (tmp_path / "exp" / "llm_cache" / "x.json").write_text("{}", encoding="utf-8")
    assert "faltam 1" in cache_problems()[0]
    (tmp_path / "exp" / Path(run).name / "artifacts").mkdir(parents=True)
    assert cache_problems() == []


def test_restore_refuses_a_missing_bundle_without_touching_experiments(
    tmp_path: Path, repo_root: Path
) -> None:
    from tcc_rca.bundle import restore_bundle

    cfg = load_config(
        repo_root / "configs/base.yaml",
        overrides={"paths": {"experiments_dir": str(tmp_path / "exp")}},
    )
    with pytest.raises(FileNotFoundError, match="bundle not found"):
        restore_bundle(cfg, tmp_path / "nope.tar.gz", use_reference_evidence=False)
    assert not (tmp_path / "exp").exists()


def test_latest_runs_takes_one_dir_per_level(tmp_path: Path) -> None:
    from tcc_rca.reproduce import latest_runs

    for stamp, level in [
        ("20261123T1-principal", "principal"),
        ("20261123T2-principal", "principal"),
        ("20261124T1-agente-cache", "agente-cache"),
    ]:
        (tmp_path / stamp).mkdir()
        (tmp_path / stamp / "manifest.json").write_text(
            json.dumps({"level": level}), encoding="utf-8"
        )
    (tmp_path / "pacote").mkdir()
    assert [p.name for p in latest_runs(tmp_path)] == [
        "20261123T2-principal",
        "20261124T1-agente-cache",
    ]
