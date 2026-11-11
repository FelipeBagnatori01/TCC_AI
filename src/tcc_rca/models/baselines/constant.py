"""E4 constant diagnosis baseline (A0): reads no telemetry.

Component and reason are the most frequent among the fold's training/validation failures; the
occurrence time is the start of the query window.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from tcc_rca.evaluation.openrca import Prediction
from tcc_rca.ingestion.cases import Failure, Query


def most_frequent(items: Iterable[str]) -> str:
    """Mode; ties broken alphabetically so the choice is reproducible."""
    counts = Counter(items)
    if not counts:
        raise ValueError("no items to take the mode of")
    top = max(counts.values())
    return min(item for item, n in counts.items() if n == top)


def constant_predictions(
    training_failures: Sequence[Failure], queries: Sequence[Query]
) -> list[Prediction]:
    component = most_frequent(f.component for f in training_failures)
    reason = most_frequent(f.reason for f in training_failures)
    return [Prediction(q.row, q.window_start_utc, component, reason) for q in queries]
