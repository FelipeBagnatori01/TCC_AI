"""Tiny synthetic OpenRCA-shaped system used by ingestion tests (never real data)."""

from __future__ import annotations

from pathlib import Path

DAY = "2020_04_11"
START_MS = 1586534400000  # 2020-04-11 00:00:00 UTC+8 == 2020-04-10 16:00:00 UTC
MIN_MS = 60_000

METRIC_HEADER = "itemid,name,bomc_id,timestamp,value,cmdb_id"
APP_HEADER = "serviceName,startTime,avg_time,num,succee_num,succee_rate"
TRACE_HEADER = "callType,startTime,elapsedTime,success,traceId,id,pid,cmdb_id,dsName,serviceName"

# record.csv: failure on docker_001 at 00:05:00 UTC+8 (epoch 1586534700)
RECORD = (
    "level,reason,component,timestamp,datetime\n"
    "pod,CPU fault,docker_001,1586534700,2020-04-11 00:05:00\n"
)
INSTRUCTION = (
    "During the specified time range of April 11, 2020, from 00:00 to 00:30, there was one "
    "failure reported. The exact reason for this failure is currently unknown."
)
QUERY = (
    "task_index,instruction,scoring_points\n"
    f'task_2,"{INSTRUCTION}","The only predicted root cause reason is CPU fault\n"\n'
)


def _metric_rows(component: str, kpi: str, values: list[float]) -> str:
    return "".join(
        f"1,{kpi},B-1,{START_MS + i * MIN_MS},{v},{component}\n" for i, v in enumerate(values)
    )


def make_system(root: Path, system: str = "Sys") -> Path:
    """Create ``root/system`` with one day of telemetry and return the system directory."""
    system_dir = root / system
    metric_dir = system_dir / "telemetry" / DAY / "metric"
    trace_dir = system_dir / "telemetry" / DAY / "trace"
    metric_dir.mkdir(parents=True)
    trace_dir.mkdir(parents=True)
    (system_dir / "record.csv").write_text(RECORD, encoding="utf-8")
    (system_dir / "query.csv").write_text(QUERY, encoding="utf-8")

    n = 40
    cpu = [1.0] * 6 + [98.0] * 4 + [2.0] * (n - 10)
    (metric_dir / "metric_container.csv").write_text(
        METRIC_HEADER
        + "\n"
        + _metric_rows("docker_001", "container_cpu_used", cpu)
        + _metric_rows("docker_001", "container_fgc", [0.0] * n),  # constant KPI
        encoding="utf-8",
    )
    for name, comp, kpi in (
        ("metric_middleware", "redis_001", "connected_clients"),
        ("metric_node", "os_001", "CPU_util_pct"),
        ("metric_service", "db_001", "Sess_Active"),
    ):
        (metric_dir / f"{name}.csv").write_text(
            METRIC_HEADER + "\n" + _metric_rows(comp, kpi, [float(i % 5) for i in range(n)]),
            encoding="utf-8",
        )
    (metric_dir / "metric_app.csv").write_text(
        APP_HEADER + f"\nosb_001,{START_MS},0.5,10,10,1.0\n", encoding="utf-8"
    )
    spans = "".join(
        f"JDBC,{START_MS + i * MIN_MS},2.0,True,t{i},s{i},p{i},docker_001,db_001,\n"
        for i in range(n)
    )
    (trace_dir / "trace_span.csv").write_text(TRACE_HEADER + "\n" + spans, encoding="utf-8")
    return system_dir
