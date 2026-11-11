"""Sliding windows over the minute matrix, their labels and the LSTM normal-period mask (ADR 0007).

A window of ``W`` minutes *ending at* ``t`` covers the matrix rows ``[t - (W - 1), t]`` of a
single day; windows never cross day boundaries. Only failure **onsets** are labelled in OpenRCA
(no end), so every label is a function of the distance between ``t`` and the onsets:

* positive  -- an onset falls in ``[t - D, t + H]`` (H = horizon, D = detection tolerance);
* negative  -- the window is *normal*: it does not touch ``[s - before, s + after]`` for any onset
  ``s``, where ``before = max(margin_before, H)`` so a positive can never be a normal window;
* ignored   -- everything else (null label): too close to a failure to be normal, but not in the
  positive band. Such windows are neither trained on as normal nor scored as negatives.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt
import polars as pl

_MINUTE = np.timedelta64(1, "m")


@dataclass(frozen=True)
class WindowSpec:
    window_minutes: int  # W
    margin_before_minutes: int
    margin_after_minutes: int
    horizon_minutes: int = 0  # H
    step_seconds: int = 60

    @property
    def effective_before(self) -> int:
        return max(self.margin_before_minutes, self.horizon_minutes)


def _as_minutes(times: pl.Series) -> npt.NDArray[np.datetime64]:
    return times.dt.replace_time_zone(None).to_numpy().astype("datetime64[m]")


def _onsets_array(onsets: Sequence[datetime]) -> npt.NDArray[np.datetime64]:
    naive = [o.replace(tzinfo=None) for o in onsets]
    return np.sort(np.array(naive, dtype="datetime64[m]"))


def build_window_index(
    index: pl.DataFrame, onsets: Sequence[datetime], spec: WindowSpec
) -> pl.DataFrame:
    """One row per complete window. ``index`` holds the matrix ``day`` and UTC ``time_utc`` columns
    in matrix row order; ``row`` in the output is the matrix row where the window ends."""
    if spec.step_seconds != 60:
        raise ValueError("windows are defined on the 1-minute grid (ADR 0005)")
    frame = index.select("day", "time_utc").with_row_index("row")
    times = _as_minutes(frame["time_utc"])
    days = frame["day"].to_numpy()
    w = spec.window_minutes

    ends: list[int] = []
    for day in dict.fromkeys(days.tolist()):  # matrix order, each day once
        rows = np.flatnonzero(days == day)
        if rows.size and np.any(np.diff(rows) != 1):
            raise ValueError(f"rows of day {day} are not contiguous in the matrix")
        if rows.size > 1 and np.any(np.diff(times[rows]) != _MINUTE):
            raise ValueError(f"day {day} is not a regular 1-minute grid")
        ends.extend(rows[w - 1 :].tolist())
    end_rows = np.array(ends, dtype=np.int64)
    end_t = times[end_rows]
    start_t = end_t - (w - 1) * _MINUTE

    s = _onsets_array(onsets)
    to_next = np.full(end_rows.size, np.nan)
    since_last = np.full(end_rows.size, np.nan)
    normal = np.ones(end_rows.size, dtype=np.bool_)
    if s.size:
        nxt = np.searchsorted(s, end_t, side="left")  # first onset >= t
        has_next = nxt < s.size
        to_next[has_next] = (s[nxt[has_next]] - end_t[has_next]) / _MINUTE
        prv = np.searchsorted(s, end_t, side="right") - 1  # last onset <= t
        has_prev = prv >= 0
        since_last[has_prev] = (end_t[has_prev] - s[prv[has_prev]]) / _MINUTE
        before, after = spec.effective_before * _MINUTE, spec.margin_after_minutes * _MINUTE
        for onset in s:  # few onsets: a loop is clearer than a broadcast
            normal &= ~((start_t <= onset + after) & (end_t >= onset - before))

    return pl.DataFrame(
        {
            "row": end_rows,
            "day": days[end_rows],
            "start_utc": start_t.astype("datetime64[us]"),
            "end_utc": end_t.astype("datetime64[us]"),
            "minutes_to_next_onset": to_next,
            "minutes_since_last_onset": since_last,
            "normal": normal,
        }
    ).with_columns(
        pl.col("start_utc", "end_utc").dt.replace_time_zone("UTC"),
        pl.col("minutes_to_next_onset", "minutes_since_last_onset").fill_nan(None),
    )


def label_windows(windows: pl.DataFrame, horizon_minutes: int, tolerance_minutes: int) -> pl.Series:
    """1 = onset in ``[t - D, t + H]``; 0 = normal window; null = ignored (see module docstring)."""
    positive = (pl.col("minutes_to_next_onset") <= horizon_minutes) | (
        pl.col("minutes_since_last_onset") <= tolerance_minutes
    )
    return windows.select(
        pl.when(positive.fill_null(False))
        .then(1)
        .when(pl.col("normal"))
        .then(0)
        .otherwise(None)
        .cast(pl.Int8)
        .alias("label")
    )["label"]
