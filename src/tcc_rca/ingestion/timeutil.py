"""Time-zone handling. OpenRCA labels and queries are in UTC+8; the pipeline uses UTC (ADR 0002).

All conversions between the two live here so a single test can guard the 8-hour shift.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

OPENRCA_TZ = timezone(timedelta(hours=8), name="UTC+8")
LOCAL_FORMAT = "%Y-%m-%d %H:%M:%S"


def local_to_utc(naive_local: datetime) -> datetime:
    """Interpret a naive datetime as UTC+8 and return it as an aware UTC datetime."""
    if naive_local.tzinfo is not None:
        raise ValueError("expected a naive datetime in OpenRCA local time (UTC+8)")
    return naive_local.replace(tzinfo=OPENRCA_TZ).astimezone(UTC)


def parse_local(text: str) -> datetime:
    """Parse ``YYYY-MM-DD HH:MM:SS`` written in UTC+8 into aware UTC."""
    return local_to_utc(datetime.strptime(text.strip(), LOCAL_FORMAT))


def utc_to_local_str(moment: datetime) -> str:
    """Format an aware datetime as UTC+8 text, the format the official evaluator expects."""
    if moment.tzinfo is None:
        raise ValueError("expected an aware datetime")
    return moment.astimezone(OPENRCA_TZ).strftime(LOCAL_FORMAT)


def epoch_s_to_utc(seconds: int) -> datetime:
    return datetime.fromtimestamp(seconds, tz=UTC)


def epoch_ms_to_utc(millis: int) -> datetime:
    return datetime.fromtimestamp(millis / 1000, tz=UTC)
