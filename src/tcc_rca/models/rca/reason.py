"""Root-cause reason given the chosen component (ADR 0020).

* R0: most frequent reason among past failures (the A0 constant);
* R1: most frequent reason among past failures of the same component, else R0;
* R2: fixed rule: category of the component's most anomalous KPI (C4 score) -> the candidate
  reasons whose ``explain.reason_categories`` contain it (fixed 2026-11-12), tie by R1 within
  them, else R1;
* R3: multinomial logistic regression on log(1 + max C4 score) per KPI category of the component
  plus the component type, trained on past single-failure queries; R0 with fewer than 2 classes.

"Past" always means failures on days before the scored query's day (labels the deployed system
would already have).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def most_frequent(items: Sequence[str], allowed: Sequence[str] | None = None) -> str | None:
    """Mode (ties alphabetical) among ``items``, restricted to ``allowed`` when given."""
    counts = Counter(i for i in items if allowed is None or i in allowed)
    if not counts:
        return None
    top = max(counts.values())
    return min(i for i, n in counts.items() if n == top)


def prior_reason(past_reasons: Sequence[str], candidates: Sequence[str]) -> str:
    """R0; with no past failure, the first candidate reason alphabetically."""
    return most_frequent(past_reasons) or sorted(candidates)[0]


def component_prior_reason(
    component: str, past: Sequence[tuple[str, str]], candidates: Sequence[str]
) -> str:
    """R1. ``past`` = (component, reason) of past failures."""
    own = most_frequent([r for c, r in past if c == component])
    return own or prior_reason([r for _, r in past], candidates)


def rule_reason(
    top_category: str | None,
    component: str,
    past: Sequence[tuple[str, str]],
    reason_categories: Mapping[str, Sequence[str]],
    candidates: Sequence[str],
) -> str:
    """R2. Reasons mapped to the category; ties by the component's then the global past
    frequency among them, then alphabetical."""
    mapped = sorted(r for r, cats in reason_categories.items() if top_category in cats)
    if not mapped:
        return component_prior_reason(component, past, candidates)
    return (
        most_frequent([r for c, r in past if c == component], mapped)
        or most_frequent([r for _, r in past], mapped)
        or mapped[0]
    )


def component_type(component: str) -> str:
    """Name without digits and separators: os_001 -> os, Tomcat01 -> Tomcat."""
    return re.sub(r"[\d_]+", "", component)


def reason_features(
    per_feature: FloatArray,
    columns: IntArray,
    categories: Sequence[str],
    category_names: Sequence[str],
) -> FloatArray:
    """log(1 + max C4 score) per KPI category over the component's features (0 if none)."""
    out = np.zeros(len(category_names), dtype=np.float64)
    pos = {c: i for i, c in enumerate(category_names)}
    for j in columns:
        i = pos[categories[j]]
        out[i] = max(out[i], float(np.log1p(max(per_feature[j], 0.0))))
    return out


class ReasonClassifier:
    """R3: scikit-learn ``LogisticRegression`` (lbfgs, multinomial) on category features plus a
    one-hot of the component type. Deterministic."""

    def __init__(self, types: Sequence[str], c: float, max_iter: int = 2000) -> None:
        self.types = sorted(set(types))
        self.c, self.max_iter = c, max_iter
        self.model: object | None = None
        self.constant: str | None = None

    def _x(self, feats: FloatArray, ctype: str) -> FloatArray:
        onehot = np.array([float(ctype == t) for t in self.types])
        return np.concatenate([feats, onehot])

    def fit(self, rows: Sequence[tuple[FloatArray, str, str]], fallback: str) -> ReasonClassifier:
        """``rows`` = (category features, component type, reason)."""
        classes = {r for _, _, r in rows}
        if len(classes) < 2:
            self.constant = next(iter(classes)) if classes else fallback
            return self
        from sklearn.linear_model import LogisticRegression

        x = np.stack([self._x(f, t) for f, t, _ in rows])
        y = np.array([r for _, _, r in rows])
        self.model = LogisticRegression(C=self.c, max_iter=self.max_iter).fit(x, y)
        return self

    def predict(self, feats: FloatArray, ctype: str) -> str:
        if self.model is None:
            assert self.constant is not None
            return self.constant
        pred = self.model.predict(self._x(feats, ctype)[None])  # type: ignore[attr-defined]
        return str(pred[0])
