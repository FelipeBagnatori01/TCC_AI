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


def make_system(root: Path, system: str = "Telecom") -> Path:
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


DAY2 = "2020_04_12"
DAY_MS = 86_400_000


def add_shifted_day(system_dir: Path, day: str = DAY2, shift_ms: int = DAY_MS) -> None:
    """Copy the synthetic day to ``day`` with every epoch-ms timestamp shifted by ``shift_ms``."""
    src = system_dir / "telemetry" / DAY
    for path in src.rglob("*.csv"):
        lines = path.read_text(encoding="utf-8").splitlines()
        header = lines[0].split(",")
        time_col = header.index("timestamp" if "timestamp" in header else "startTime")
        out_lines = [lines[0]]
        for line in lines[1:]:
            cells = line.split(",")
            cells[time_col] = str(int(cells[time_col]) + shift_ms)
            out_lines.append(",".join(cells))
        target = system_dir / "telemetry" / day / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def write_metric(system_dir: Path, name: str, rows: list[tuple[int, str, int, float, str]]) -> None:
    """Overwrite a metric file of ``DAY`` with rows ``(itemid, kpi, timestamp_ms, value, cmdb)``."""
    body = "".join(f"{i},{k},B-1,{t},{v},{c}\n" for i, k, t, v, c in rows)
    path = system_dir / "telemetry" / DAY / "metric" / f"{name}.csv"
    path.write_text(METRIC_HEADER + "\n" + body, encoding="utf-8")


# ------------------------------------------------------------------ Bank-shaped synthetic system
BANK_DAY = "2021_03_04"
BANK_START_S = 1614787200  # 2021-03-04 00:00:00 UTC+8
BANK_RECORD = (
    "level,component,timestamp,datetime,reason\n"
    "pod,Tomcat01,1614787500.0,2021-03-04 00:05:00,high CPU usage\n"
)
BANK_QUERY = (
    "task_index,instruction,scoring_points\n"
    '"task_3","On March 4, 2021, within the time range of 00:00 to 00:30, a single failure was '
    'detected.","The only predicted root cause component is Tomcat01\n"\n'
)


def make_bank_system(root: Path) -> Path:
    """One Bank-shaped day: second-resolution metrics/logs, millisecond traces, multi-line logs."""
    system_dir = root / "Bank"
    day = system_dir / "telemetry" / BANK_DAY
    for sub in ("metric", "trace", "log"):
        (day / sub).mkdir(parents=True)
    (system_dir / "record.csv").write_text(BANK_RECORD, encoding="utf-8")
    (system_dir / "query.csv").write_text(BANK_QUERY, encoding="utf-8")
    n = 20

    def cpu(i: int) -> float:
        return 90.0 if 5 <= i < 8 else 10.0

    rows = "".join(
        f"{BANK_START_S + 60 * i},Tomcat01,OSLinux-CPU_CPU_CPUCpuUtil,{cpu(i)}\n"
        f"{BANK_START_S + 60 * i},Mysql01,Mysql-MySQL_3306_Uptime,1.0\n"
        for i in range(n)
    )
    (day / "metric" / "metric_container.csv").write_text(
        "timestamp,cmdb_id,kpi_name,value\n" + rows, encoding="utf-8"
    )
    (day / "metric" / "metric_app.csv").write_text(
        "timestamp,rr,sr,cnt,mrt,tc\n"
        + "".join(f"{BANK_START_S + 60 * i},100.0,100.0,5,50.0,ServiceTest1\n" for i in range(n)),
        encoding="utf-8",
    )
    (day / "trace" / "trace_span.csv").write_text(
        "timestamp,cmdb_id,parent_id,span_id,trace_id,duration\n"
        + "".join(
            f"{(BANK_START_S + 60 * i) * 1000 + 5},Tomcat01,p{i},s{i},t{i},{12 + i}\n"
            for i in range(n)
        ),
        encoding="utf-8",
    )
    (day / "log" / "log_service.csv").write_text(
        "log_id,timestamp,cmdb_id,log_name,value\n"
        f'a1,{BANK_START_S + 1},Tomcat01,gc,"[GC (Allocation Failure)\nline two, with comma]"\n'
        f"a2,{BANK_START_S + 400},Tomcat01,gc,plain line\n",
        encoding="utf-8",
    )
    return system_dir
