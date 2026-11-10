from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tcc_rca.ingestion.timeutil import (
    epoch_ms_to_utc,
    epoch_s_to_utc,
    local_to_utc,
    parse_local,
    utc_to_local_str,
)


def test_known_utc_plus_8_timestamp_converts_to_utc() -> None:
    # Taken from the real Telecom record.csv: epoch 1586534700 <-> "2020-04-11 00:05:00" (UTC+8)
    assert parse_local("2020-04-11 00:05:00") == datetime(2020, 4, 10, 16, 5, tzinfo=UTC)
    assert parse_local("2020-04-11 00:05:00") == epoch_s_to_utc(1586534700)


def test_shift_is_exactly_eight_hours() -> None:
    naive = datetime(2020, 5, 1, 12, 0, 0)
    assert (naive.replace(tzinfo=UTC) - local_to_utc(naive)).total_seconds() == 8 * 3600


def test_round_trip_utc_to_local_text() -> None:
    text = "2020-05-31 23:59:00"
    assert utc_to_local_str(parse_local(text)) == text


def test_epoch_ms_matches_epoch_s() -> None:
    assert epoch_ms_to_utc(1586534700000) == epoch_s_to_utc(1586534700)


def test_naive_and_aware_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="naive"):
        local_to_utc(datetime(2020, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="aware"):
        utc_to_local_str(datetime(2020, 1, 1))
