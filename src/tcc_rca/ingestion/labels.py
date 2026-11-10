"""Ground-truth access (``scoring_points``). FORBIDDEN for agent code and agent tools (R3).

Only evaluation modules may import this file; a test asserts that nothing under
``tcc_rca.agent`` does.
"""

from __future__ import annotations

import csv
from pathlib import Path


def read_scoring_points(query_csv: Path) -> list[str]:
    """Raw ``scoring_points`` strings, in the same row order as ``query.csv``."""
    with query_csv.open(encoding="utf-8", newline="") as handle:
        return [row["scoring_points"] for row in csv.DictReader(handle)]
