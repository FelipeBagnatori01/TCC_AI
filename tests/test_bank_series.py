"""Bank E2 (ADR 0012): rule catalog, container KPIs, logs, day clipping and adaptive filling."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from synthetic import BANK_DAY, BANK_START_S, make_bank_system
from tcc_rca.ingestion.telemetry import convert_day
from tcc_rca.series.bank import bank_metric_long, bank_trace_edges, normalize_kpi
from tcc_rca.series.catalog import KpiRule, build_feature_catalog, load_kpi_catalog
from tcc_rca.series.logs import LogTemplateModel, is_count_kpi, log_long
from tcc_rca.series.matrix import build_matrix, native_steps

BANK2 = "2021_03_05"
DAY_S = 86_400


@pytest.fixture
def catalog(repo_root: Path):  # type: ignore[no-untyped-def]
    return load_kpi_catalog(repo_root / "configs/data/kpi_catalog_bank.yaml")


def _write(system: Path, day: str, sub: str, name: str, text: str) -> None:
    path = system / "telemetry" / day / sub / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _copy_day(system: Path, day: str, shift_s: int) -> None:
    """Copy the synthetic Bank day to ``day`` shifting every epoch (s, or ms for traces)."""
    src = system / "telemetry" / BANK_DAY
    for path in src.rglob("*.csv"):
        text = path.read_text(encoding="utf-8")
        shift = shift_s * 1000 if "trace" in path.parts else shift_s
        base = BANK_START_S * 1000 if "trace" in path.parts else BANK_START_S
        out = _shift_epochs(text, base, shift)
        target = system / "telemetry" / day / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(out, encoding="utf-8")


def _shift_epochs(text: str, base: int, shift: int) -> str:
    digits = len(str(base))
    return re.sub(
        rf"\b{str(base)[:4]}\d{{{digits - 4}}}\b", lambda m: str(int(m.group(0)) + shift), text
    )


# ------------------------------------------------------------------------------- KPI catalog
def test_rule_catalog_first_match_named_groups_and_missing(catalog) -> None:  # type: ignore[no-untyped-def]
    info = catalog.lookup(
        "metric_container", "OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKPercentBusy"
    )
    assert (info.category, info.unit, info.description) == ("disco", "%", "Ocupação do disco sdb")
    generic = catalog.lookup("metric_container", "OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKRead")
    assert generic.description == "Disco sdb: Read"  # the more specific rule above did not match
    assert catalog.lookup("metric_container", "Container_CpuPercent").item_agg == "sum"
    assert catalog.lookup("log", "log_catalina_level_SEVERE").category == "erro"
    assert catalog.lookup("metric_container", "Something-Unknown") is None
    with pytest.raises(KeyError, match="without catalog entry"):
        build_feature_catalog({"X::Something-Unknown": "metric_container"}, catalog)


def test_rule_requires_full_match() -> None:
    rule = KpiRule(pattern=r"a_(?P<x>\d)", category="CPU", unit="%", description="n {x}")
    assert rule.match("a_1").description == "n 1"  # type: ignore[union-attr]
    assert rule.match("a_12") is None
    assert rule.match("xa_1") is None


# ----------------------------------------------------------------------------- container KPIs
def test_container_kpi_drops_restart_hash() -> None:
    names = pl.DataFrame(
        {
            "k": [
                "Container-DOCKER_CONTAINER_23bdcf67c7e3--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent",
                "Container-DOCKER_CONTAINER_ffff0000aaaa--bcou-role-st-uat-statefulset-0--bcou--UATWKR18_CpuPercent",
                "OSLinux-CPU_CPU_CPUCpuUtil",
            ]
        }
    )
    out = names.select(normalize_kpi(pl.col("k")).alias("k"))["k"].to_list()
    assert out == ["Container_CpuPercent", "Container_CpuPercent", "OSLinux-CPU_CPU_CPUCpuUtil"]


def test_overlapping_containers_combine_by_catalog_rule(tmp_path: Path, catalog) -> None:  # type: ignore[no-untyped-def]
    """During a restart old and new containers report together: CPU sums, MemPercent takes max."""
    system = make_bank_system(tmp_path)
    t = BANK_START_S
    rows = [
        (t, "aaaa", "CpuPercent", 10.0),
        (t, "bbbb", "CpuPercent", 5.0),
        (t, "aaaa", "MemPercent", 40.0),
        (t, "bbbb", "MemPercent", 70.0),
    ]
    body = "".join(
        f"{ts},dockerA1,Container-DOCKER_CONTAINER_{h}--pod-0--x_{m},{v}\n" for ts, h, m, v in rows
    )
    _write(
        system, BANK_DAY, "metric", "metric_container", "timestamp,cmdb_id,kpi_name,value\n" + body
    )
    convert_day(system, "Bank", BANK_DAY, tmp_path / "interim")
    long = bank_metric_long(tmp_path / "interim", "Bank", BANK_DAY, catalog, "60s")
    got = dict(zip(long["feature"], long["value"], strict=True))
    assert got == {"dockerA1::Container_CpuPercent": 15.0, "dockerA1::Container_MemPercent": 70.0}


# -------------------------------------------------------------------------------------- logs
LOG_HEADER = "log_id,timestamp,cmdb_id,log_name,value\n"


def _logs(system: Path, day: str, start: int, lines: list[tuple[int, str, str, str]]) -> None:
    body = "".join(
        f'l{i},{start + off},{comp},{name},"{val}"\n'
        for i, (off, comp, name, val) in enumerate(lines)
    )
    _write(system, day, "log", "log_service", LOG_HEADER + body)


def test_structured_logs_hand_computed(tmp_path: Path) -> None:
    system = make_bank_system(tmp_path)
    t = BANK_START_S
    _logs(
        system,
        BANK_DAY,
        t,
        [
            (
                1,
                "Tomcat01",
                "localhost_access_log",
                "IG01 POST /a.json HTTP/1.1 200 93 - k6 (api_url 260 0.260",
            ),
            (
                2,
                "Tomcat01",
                "localhost_access_log",
                "IG01 POST /a.json HTTP/1.1 503 10 - k6 (api_url 140 0.140",
            ),
            (
                3,
                "apache01",
                "apache_access_log",
                'IPAddress ""POST /b.json HTTP/1.1"" 404 391 ""-"" ""k6 (api_url 697220 0',
            ),
            (
                4,
                "Tomcat01",
                "gc",
                "1.5: [Full GC (Ergonomics) [Times: user=0.59 sys=0.00, real=0.20 secs]",
            ),
            (
                5,
                "Tomcat01",
                "gc",
                "2.0: [GC (Allocation Failure) [Times: user=0.1 sys=0.00, real=0.05 secs]",
            ),
            (6, "Tomcat01", "gc", "2.1: [CMS-concurrent-mark-start]"),
            (
                7,
                "Mysql01",
                "slow",
                "# Query_time: 0.5  Lock_time: 0.0 Rows_sent: 0  Rows_examined: 10",
            ),
            (
                8,
                "Mysql01",
                "slow",
                "# Query_time: 1.5  Lock_time: 0.0 Rows_sent: 0  Rows_examined: 30",
            ),
            (9, "Mysql01", "error", "0 [Note] InnoDB: page_cleaner took 7883ms"),
        ],
    )
    convert_day(system, "Bank", BANK_DAY, tmp_path / "interim")
    long = log_long(tmp_path / "interim", "Bank", BANK_DAY, "60s", LogTemplateModel(5))
    got = dict(zip(long["feature"], long["value"], strict=True))
    assert got["Tomcat01::access_requests"] == 2
    assert got["Tomcat01::access_errors"] == 1
    assert got["Tomcat01::access_rt_mean_ms"] == pytest.approx(200.0)  # (260 + 140) / 2 ms
    assert got["apache01::access_errors"] == 1
    assert got["apache01::access_rt_mean_ms"] == pytest.approx(697.22)  # microseconds -> ms
    assert (got["Tomcat01::gc_events"], got["Tomcat01::gc_full"]) == (3, 1)
    assert got["Tomcat01::gc_pause_s"] == pytest.approx(0.25)
    assert got["Mysql01::slow_queries"] == 2
    assert got["Mysql01::slow_query_time_mean_s"] == pytest.approx(1.0)
    assert got["Mysql01::slow_rows_examined_mean"] == pytest.approx(20.0)
    assert got["Mysql01::mysql_error_count"] == 1


def test_templates_are_fitted_on_fit_days_only(tmp_path: Path) -> None:
    model = LogTemplateModel(top_k=1).fit(
        {
            "catalina": [
                "INFO [main] Starting service 1",
                "INFO [main] Starting service 2",
                "SEVERE [main] Context failed 7",
            ]
        }
    )
    assert model.label("catalina", "INFO [main] Starting service 99") == "tpl_1"
    assert model.label("catalina", "SEVERE [main] Context failed 8") == "tpl_other"  # not top-1
    assert model.label("catalina", "WARNING totally new message") == "tpl_other"  # unseen
    assert model.label("localhost", "anything") == "tpl_other"  # log type absent in fit days
    assert [t["fit_count"] for t in model.templates] == [2]


def test_count_kpis_are_zero_filled() -> None:
    for kpi in (
        "trace_calls",
        "access_requests",
        "gc_pause_s",
        "log_catalina_tpl_3",
        "log_localhost_level_INFO",
    ):
        assert is_count_kpi(kpi), kpi
    for kpi in ("access_rt_mean_ms", "trace_latency_p95", "OSLinux-CPU_CPU_CPUCpuUtil"):
        assert not is_count_kpi(kpi), kpi


# ------------------------------------------------------------------------- traces and matrix
def test_trace_edges_follow_parent_span(tmp_path: Path) -> None:
    system = make_bank_system(tmp_path)
    ms = BANK_START_S * 1000
    _write(
        system,
        BANK_DAY,
        "trace",
        "trace_span",
        "timestamp,cmdb_id,parent_id,span_id,trace_id,duration\n"
        f"{ms},IG01,root,s1,t1,5\n{ms + 1},Tomcat01,s1,s2,t1,3\n{ms + 2},MG01,s2,s3,t1,1\n"
        f"{ms + 3},Tomcat01,s2,s4,t2,1\n",  # parent s2 lives in another trace: no edge
    )
    convert_day(system, "Bank", BANK_DAY, tmp_path / "interim")
    edges = bank_trace_edges(tmp_path / "interim", "Bank", BANK_DAY).sort("src")
    assert edges.rows() == [("IG01", "Tomcat01", 1), ("Tomcat01", "MG01", 1)]


def _two_day_system(tmp_path: Path) -> Path:
    system = make_bank_system(tmp_path)
    _copy_day(system, BANK2, DAY_S)
    interim = tmp_path / "interim"
    for d in (BANK_DAY, BANK2):
        convert_day(system, "Bank", d, interim)
    return interim


def test_bank_matrix_clips_spill_and_builds_every_source(tmp_path: Path, catalog) -> None:  # type: ignore[no-untyped-def]
    system = make_bank_system(tmp_path)
    spill = (BANK_START_S - 3600) * 1000  # one span stamped the previous evening
    path = system / "telemetry" / BANK_DAY / "trace" / "trace_span.csv"
    path.write_text(path.read_text(encoding="utf-8") + f"{spill},Tomcat01,p,s,t,1\n", "utf-8")
    convert_day(system, "Bank", BANK_DAY, tmp_path / "interim")
    model = LogTemplateModel(5).fit({"catalina": []})
    m = build_matrix(tmp_path / "interim", "Bank", [BANK_DAY], catalog, 60, 5, model)
    assert m.values["time_utc"].min() == datetime(2021, 3, 3, 16, 0, tzinfo=UTC)  # 00:00 UTC+8
    sources = set(m.sources.values())
    assert sources == {"metric_container", "metric_app", "trace", "log"}
    assert m.values["Tomcat01::trace_calls"].null_count() == 0  # counts are zero-filled
    build_feature_catalog(m.sources, catalog)  # every feature is catalogued


def test_adaptive_fill_uses_initial_block_only(tmp_path: Path, catalog) -> None:  # type: ignore[no-untyped-def]
    interim = _two_day_system(tmp_path)
    model = LogTemplateModel(5).fit({"catalina": []})
    m = build_matrix(interim, "Bank", [BANK_DAY, BANK2], catalog, 60, 2, model, [BANK_DAY], 10)
    assert m.fill_limits["Tomcat01::OSLinux-CPU_CPU_CPUCpuUtil"] == 2  # sampled every minute
    # Leakage check: thinning the second day's samples must not change limits fitted on day 1.
    obs = m.observed.with_columns(
        pl.when(pl.col("day") == BANK2)
        .then(False)
        .otherwise(pl.col("Tomcat01::OSLinux-CPU_CPU_CPUCpuUtil"))
        .alias("Tomcat01::OSLinux-CPU_CPU_CPUCpuUtil")
    )
    assert native_steps(obs, ["Tomcat01::OSLinux-CPU_CPU_CPUCpuUtil"], [BANK_DAY]) == native_steps(
        m.observed, ["Tomcat01::OSLinux-CPU_CPU_CPUCpuUtil"], [BANK_DAY]
    )


def test_native_steps_and_bounded_limits() -> None:
    observed = pl.DataFrame(
        {
            "day": ["d"] * 12,
            "a": [i % 4 == 0 for i in range(12)],  # every 4 steps
            "b": [True] * 12,
            "c": [i == 0 for i in range(12)],  # single sample: unknown step
        }
    )
    assert native_steps(observed, ["a", "b", "c"], ["d"]) == {"a": 4, "b": 1}
