"""ADR 0020: the agent's evidence chosen by prequential selection (onset, component, reason).

For every query of the initial block and of the tested days, a closed list of methods (fixed
before measuring) produces candidate onsets (T0-T4), component rankings (C0-C8) and reasons
(R0-R3) from agent-visible data plus models trained on earlier days. Those candidates are
computed WITHOUT labels (``_candidates``); labels are attached afterwards (``attach_gold``).

The selector then chooses, for each fold, the method of each stage that did best on the queries
of the days BEFORE the fold's test day (``select``), in the fixed order T -> C -> R, and applies
it to the test day. The reported result is the selector's; per-method test scores are
descriptive. Scored with the official OpenRCA script (and checked against the field hits
computed here). Evaluation code: reads ground truth, never imported by the agent.
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from tcc_rca.agent.tasks import failure_count, load_candidates, required_fields
from tcc_rca.config import Config
from tcc_rca.dataset.precursor import minute_scores
from tcc_rca.dataset.split import day_of
from tcc_rca.evaluation.baselines import _fit_fold_transforms, row_of
from tcc_rca.evaluation.bootstrap import cluster_bootstrap
from tcc_rca.evaluation.compare import verdict
from tcc_rca.evaluation.lstm import Experiment, _masks
from tcc_rca.evaluation.metrics import pessimistic_rank
from tcc_rca.evaluation.openrca import MultiPrediction, Prediction, score_with_official_script
from tcc_rca.evaluation.xai import _e5_artifacts, _FoldContext
from tcc_rca.explain.aggregate import component_scores, feature_importance
from tcc_rca.ingestion.cases import Failure, Query, build_cases, read_queries
from tcc_rca.ingestion.timeutil import parse_local
from tcc_rca.models.baselines.constant import constant_predictions
from tcc_rca.models.baselines.detection import choose_threshold
from tcc_rca.models.baselines.features import last_steps, window_rows
from tcc_rca.models.baselines.localization import zscore_component_scores
from tcc_rca.models.lstm.data import window_tensor
from tcc_rca.models.rca.localize import (
    component_max,
    nsigma_feature_scores,
    personalized_pagerank,
    ranking,
    reciprocal_rank_fusion,
    robust_feature_scores,
    seasonal_feature_scores,
    training_statistics,
)
from tcc_rca.models.rca.onset import bocpd_onsets, pick_onsets, threshold_onsets
from tcc_rca.models.rca.reason import (
    ReasonClassifier,
    component_prior_reason,
    component_type,
    prior_reason,
    reason_features,
    rule_reason,
)
from tcc_rca.tracking import Run

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

ONSET_METHODS = ("T0", "T1", "T2", "T3", "T4")
COMPONENT_METHODS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")
REASON_METHODS = ("R0", "R1", "R2", "R3")
LABELS = {
    "T0": "início da janela da consulta",
    "T1": "previsora LSTM: 1º minuto acima do limiar",
    "T2": "z-score local: 1º minuto acima do limiar",
    "T3": "BOCPD sobre o z-score local",
    "T4": "BOCPD sobre a previsora LSTM",
    "C0": "frequência no passado",
    "C1": "previsora LSTM + SHAP",
    "C2": "ranqueamento nativo da previsora",
    "C3": "z-score global",
    "C4": "pontuação robusta (BARO)",
    "C5": "N-Sigma",
    "C6": "desvio sazonal (dia anterior)",
    "C7": "PageRank no grafo de dependências",
    "C8": "fusão por posto (C1, C3, C4)",
    "R0": "mais frequente no passado",
    "R1": "mais frequente no passado para o componente",
    "R2": "regra categoria do KPI → motivo",
    "R3": "regressão logística",
}
FIELDS = ("occurrence_datetime", "component", "reason")


# ------------------------------------------------------------------------------ records
@dataclass
class FoldInfo:
    """What the reason stage needs from the fold that scored a query."""

    fold: int
    columns: dict[str, IntArray]  # component -> columns of the fold's kept features
    categories: list[str]  # KPI category of each kept feature


@dataclass
class Cause:
    """Evidence for one failure of a query under one onset method."""

    onset_row: int
    onset_utc: datetime
    scores: dict[str, dict[str, float]]  # C1..C8 -> candidate component -> score
    c4_features: FloatArray  # C4 per-feature score (reason stage)


@dataclass
class ScoredQuery:
    query: Query
    day: str
    info: FoldInfo
    n_causes: int
    fields: tuple[str, ...]
    prior_components: dict[str, float]  # C0: past root-cause counts per candidate
    onsets: dict[str, list[Cause]]  # T0..T4 -> one Cause per reported failure
    gold: tuple[Failure, ...] = ()  # attached by the evaluation step only


@dataclass(frozen=True)
class Past:
    """Labelled failures known before a day (component, reason) and the reason settings."""

    pairs_before: Callable[[str], list[tuple[str, str]]]
    reasons: tuple[str, ...]  # candidate reasons of the system
    reason_map: Mapping[str, Sequence[str]]
    category_names: tuple[str, ...]
    logistic_c: float


# ------------------------------------------------------------------- prediction per stage
def component_scores_of(sq: ScoredQuery, onset: str, method: str, cause: int) -> dict[str, float]:
    if method == "C0":
        return sq.prior_components
    return sq.onsets[onset][cause].scores[method]


def predict_components(sq: ScoredQuery, onset: str, method: str) -> list[str]:
    """Each cause takes its best-ranked component not used by an earlier cause."""
    chosen: list[str] = []
    for i in range(sq.n_causes):
        order = ranking(component_scores_of(sq, onset, method, i))
        pick = next((c for c in order if c not in chosen), order[0])
        chosen.append(pick)
    return chosen


def _reason_classifier(
    pool: Sequence[ScoredQuery], day: str, onset: str, comp: str, past: Past
) -> ReasonClassifier:
    """R3 trained on the pool's single-failure queries of days before ``day``."""
    rows: list[tuple[FloatArray, str, str]] = []
    for q in pool:
        if q.day >= day or len(q.gold) != 1:
            continue
        c = predict_components(q, onset, comp)[0]
        feats = reason_features(
            q.onsets[onset][0].c4_features,
            q.info.columns.get(c, np.zeros(0, dtype=np.int64)),
            q.info.categories,
            past.category_names,
        )
        rows.append((feats, component_type(c), q.gold[0].reason))
    types = [component_type(c) for q in pool for c in q.prior_components]
    fallback = prior_reason([r for _, r in past.pairs_before(day)], past.reasons)
    return ReasonClassifier(types, past.logistic_c).fit(rows, fallback)


def predict_reasons(
    sq: ScoredQuery,
    onset: str,
    comp: str,
    method: str,
    past: Past,
    pool: Sequence[ScoredQuery],
) -> list[str]:
    pairs = past.pairs_before(sq.day)
    comps = predict_components(sq, onset, comp)
    out: list[str] = []
    model = _reason_classifier(pool, sq.day, onset, comp, past) if method == "R3" else None
    for i, c in enumerate(comps):
        cols = sq.info.columns.get(c, np.zeros(0, dtype=np.int64))
        c4 = sq.onsets[onset][i].c4_features
        if method == "R0":
            out.append(prior_reason([r for _, r in pairs], past.reasons))
        elif method == "R1":
            out.append(component_prior_reason(c, pairs, past.reasons))
        elif method == "R2":
            top = (
                sq.info.categories[int(cols[int(np.argmax(c4[cols]))])]
                if cols.size and float(c4[cols].max()) > 0
                else None
            )
            out.append(rule_reason(top, c, pairs, past.reason_map, past.reasons))
        else:
            assert model is not None
            feats = reason_features(c4, cols, sq.info.categories, past.category_names)
            out.append(model.predict(feats, component_type(c)))
    return out


# ----------------------------------------------------------------------------- scoring
def _value_hit(f: str, pred: Any, gold: Failure) -> bool:
    if f == "occurrence_datetime":
        return bool(abs((pred - gold.start_utc).total_seconds()) <= 60)
    return bool(pred == (gold.component if f == "component" else gold.reason))


def best_hits(
    preds: Sequence[Mapping[str, Any]], gold: Sequence[Failure], fields: Sequence[str]
) -> int:
    """Hits of the best one-to-one assignment of predicted causes to gold failures, counting
    ``fields`` (the official scorer's rule; equal lengths required)."""
    if len(preds) != len(gold):
        raise ValueError("one predicted cause per gold failure is required")
    best = 0
    for perm in itertools.permutations(preds):
        hits = sum(_value_hit(f, p[f], g) for p, g in zip(perm, gold, strict=True) for f in fields)
        best = max(best, hits)
    return best


def _onset_times(sq: ScoredQuery, onset: str) -> list[datetime]:
    return [c.onset_utc for c in sq.onsets[onset]]


def _mrr(sq: ScoredQuery, onset: str, method: str) -> float:
    """Mean over gold failures of 1 / (best pessimistic rank across the query's causes)."""
    vals = []
    for g in sq.gold:
        rank = min(
            pessimistic_rank(g.component, component_scores_of(sq, onset, method, i))
            for i in range(sq.n_causes)
        )
        vals.append(1.0 / rank)
    return float(np.mean(vals))


@dataclass
class Selection:
    onset: str
    component: str
    reason: str
    reason_beats_prior: bool
    accuracy: dict[str, float] = field(default_factory=dict)  # method -> past accuracy
    n_past_queries: int = 0
    n_past_failures: int = 0


def _rate(hits: Sequence[int], totals: Sequence[int]) -> float:
    return float(sum(hits)) / float(sum(totals)) if sum(totals) else 0.0


def select(past_queries: Sequence[ScoredQuery], past: Past) -> Selection:
    """Choose T, then C given T, then R given T and C, by accuracy on ``past_queries`` (ties:
    MRR for C, then the fixed list order, simplest first). Uses only those queries' labels."""
    if not past_queries:
        raise ValueError("no past query to select on")
    n = [len(q.gold) for q in past_queries]
    acc: dict[str, float] = {}
    for t in ONSET_METHODS:
        hits = [
            best_hits(
                [{"occurrence_datetime": x} for x in _onset_times(q, t)],
                q.gold,
                ["occurrence_datetime"],
            )
            for q in past_queries
        ]
        acc[t] = _rate(hits, n)
    t_best = max(ONSET_METHODS, key=lambda m: (acc[m], -ONSET_METHODS.index(m)))
    mrr: dict[str, float] = {}
    for c in COMPONENT_METHODS:
        hits = [
            best_hits(
                [{"component": x} for x in predict_components(q, t_best, c)], q.gold, ["component"]
            )
            for q in past_queries
        ]
        acc[c] = _rate(hits, n)
        mrr[c] = float(np.mean([_mrr(q, t_best, c) for q in past_queries]))
    c_best = max(COMPONENT_METHODS, key=lambda m: (acc[m], mrr[m], -COMPONENT_METHODS.index(m)))
    for r in REASON_METHODS:
        hits = [
            best_hits(
                [{"reason": x} for x in predict_reasons(q, t_best, c_best, r, past, past_queries)],
                q.gold,
                ["reason"],
            )
            for q in past_queries
        ]
        acc[r] = _rate(hits, n)
    r_best = max(REASON_METHODS, key=lambda m: (acc[m], -REASON_METHODS.index(m)))
    return Selection(
        onset=t_best,
        component=c_best,
        reason=r_best,
        reason_beats_prior=acc[r_best] > acc["R0"],
        accuracy={k: round(v, 4) for k, v in acc.items()},
        n_past_queries=len(past_queries),
        n_past_failures=sum(n),
    )


def prequential_selections(
    scored: Sequence[ScoredQuery], test_days: Mapping[int, str], past: Past
) -> dict[int, Selection]:
    """One selection per fold, each from the queries of days BEFORE the fold's test day only."""
    return {
        k: select([s for s in scored if s.day < day], past) for k, day in sorted(test_days.items())
    }


def answer(
    sq: ScoredQuery,
    onset: str,
    comp: str,
    reason: str,
    past: Past,
    pool: Sequence[ScoredQuery],
) -> list[dict[str, Any]]:
    """One dict per cause with the three fields."""
    times = _onset_times(sq, onset)
    comps = predict_components(sq, onset, comp)
    reasons = predict_reasons(sq, onset, comp, reason, past, pool)
    return [
        {"occurrence_datetime": t, "component": c, "reason": r}
        for t, c, r in zip(times, comps, reasons, strict=True)
    ]


# ------------------------------------------------------------------ candidate evidence
def estimated_end_row(onset_row: int, day_rows: IntArray, tolerance: int, window: int) -> int:
    """Window end D minutes after the onset, clipped to the day's last row; a window that would
    start before the day becomes the first complete window of the day (as ``oracle_end_row``)."""
    first, last = int(day_rows[0]), int(day_rows[-1])
    end = min(onset_row + tolerance, last)
    if end - (window - 1) < first:
        end = first + window - 1
        if end > last:
            raise ValueError("day shorter than one window")
    return end


class _FoldEvidence:
    """Per-fold signals and per-onset component evidence (no labels)."""

    def __init__(
        self,
        config: Config,
        exp: Experiment,
        ctx: _FoldContext,
        forecast_threshold: float,
        candidates: Sequence[str],
        edges: Sequence[tuple[str, str]],
        prev_rows: Mapping[int, int],
    ) -> None:
        self.config, self.exp, self.ctx = config, exp, ctx
        self.t, self.e = config.tasks, config.evidence_select
        self.candidates, self.edges, self.prev_rows = list(candidates), edges, prev_rows
        fd, fold = ctx.fd, ctx.fold
        kept = fd.kept
        self.raw = exp.values.select(kept).to_numpy().astype(np.float64)
        fit_rows = exp.normal_min & np.isin(exp.day_arr, list(fold.train_days))
        self.stats = training_statistics(self.raw[fit_rows])
        kept_e4, self.z, _ = _fit_fold_transforms(
            exp.values, exp.features, fit_rows, config, exp.categories
        )
        if kept_e4 != kept:
            raise ValueError(f"fold {fold.index}: E4 and E5 feature selections differ")
        local = exp.inputs(fd, "local").matrix.astype(np.float64)
        self.local_minute = minute_scores(local, self.t.precursor.top_k)
        m = _masks(fd)
        val = m["val_labelled"]
        val_scores = last_steps(
            self.local_minute[window_rows(fd.end_rows[val], self.t.window_minutes)],
            self.t.score_minutes,
        ).max(axis=1)
        self.local_threshold, _ = choose_threshold(
            fd.y[val], val_scores, config.baselines.val_fallback_quantile
        )
        self.forecast_threshold = forecast_threshold
        self.times: list[datetime] = exp.values["time_utc"].to_list()
        self._cache: dict[int, Cause] = {}

    def forecast_minutes(self, rows: IntArray) -> FloatArray:
        """Forecaster score of the LAST step of the windows ending at ``rows``."""
        w = window_tensor(self.ctx.inputs.matrix, rows, self.t.window_minutes)
        err = self.ctx.main.scorer.normalized_errors(w)[:, -1, :]
        k = min(self.config.lstm.score_top_k, err.shape[1])
        out: FloatArray = np.partition(err, err.shape[1] - k, axis=1)[:, -k:].mean(axis=1)
        return out.astype(np.float64)

    def cause(self, onset_row: int, day_rows: IntArray) -> Cause:
        if onset_row in self._cache:
            return self._cache[onset_row]
        ctx, t, e = self.ctx, self.t, self.e
        end = estimated_end_row(
            onset_row, day_rows, t.detection_tolerance_minutes, t.window_minutes
        )
        win = window_tensor(ctx.inputs.matrix, np.array([end], dtype=np.int64), t.window_minutes)[0]
        module = ctx.main.modules["lstm_forecast"]
        imp = feature_importance(ctx.xp.shap(module, win, ctx.seed))
        scores: dict[str, dict[str, float]] = {
            "C1": component_scores(imp, ctx.columns, self.candidates)["sum"]
        }
        errors = ctx.main.scorer.normalized_errors(win[None])[0].max(axis=0)
        scores["C2"] = {
            c: float(errors[ctx.columns[c]].sum()) if c in ctx.columns else 0.0
            for c in self.candidates
        }
        rows = window_rows(np.array([end], dtype=np.int64), t.window_minutes)[0]
        scores["C3"] = zscore_component_scores(self.z, rows, ctx.columns, self.candidates)
        first = int(day_rows[0])
        lo = max(first, onset_row - e.reference_minutes)
        reference = self.raw[lo:onset_row]
        if reference.shape[0] < e.min_reference_minutes:
            reference = reference[:0]
        post_rows = np.arange(onset_row, max(end, onset_row) + 1)
        post = self.raw[post_rows]
        f4 = robust_feature_scores(reference, post, self.stats["median"], self.stats["iqr"])
        f5 = nsigma_feature_scores(reference, post, self.stats["mean"], self.stats["std"])
        prev = np.full_like(post, np.nan)
        for i, r in enumerate(post_rows):
            pr = self.prev_rows.get(int(r))
            if pr is not None:
                prev[i] = self.raw[pr]
        f6 = seasonal_feature_scores(post, prev, self.stats["iqr"])
        scores["C4"] = component_max(f4, ctx.columns, self.candidates)
        scores["C5"] = component_max(f5, ctx.columns, self.candidates)
        scores["C6"] = component_max(f6, ctx.columns, self.candidates)
        everyone = sorted(set(ctx.columns) | set(self.candidates))
        pr_scores = personalized_pagerank(
            self.edges,
            component_max(f4, ctx.columns, everyone),
            e.pagerank_damping,
            e.pagerank_backward_weight,
        )
        scores["C7"] = {c: pr_scores.get(c, 0.0) for c in self.candidates}
        scores["C8"] = reciprocal_rank_fusion(
            [scores["C1"], scores["C3"], scores["C4"]], self.candidates, e.rrf_k
        )
        out = Cause(onset_row, self.times[onset_row], scores, f4)
        self._cache[onset_row] = out
        return out


def _candidates(
    fe: _FoldEvidence, q: Query, day: str, n_causes: int, prior: dict[str, float]
) -> ScoredQuery:
    """Every onset method's causes for one query, from agent-visible data only."""
    exp, e, t = fe.exp, fe.e, fe.t
    day_rows = np.flatnonzero(exp.day_arr == day).astype(np.int64)
    inside = [int(r) for r in day_rows if q.window_start_utc <= fe.times[int(r)] < q.window_end_utc]
    if not inside:
        raise ValueError(f"query {q.row} has no matrix row on {day}")
    span = np.arange(max(int(day_rows[0]), inside[0] - e.reference_minutes), inside[-1] + 1)
    query_mask = np.isin(span, inside)
    complete = span - (t.window_minutes - 1) >= int(day_rows[0])
    sig_f = np.full(span.size, np.nan)
    if complete.any():
        sig_f[complete] = fe.forecast_minutes(span[complete])
    sig_z = fe.local_minute[span]
    hazard = 1.0 / e.hazard_minutes
    cands = {
        "T1": (threshold_onsets(sig_f, query_mask, fe.forecast_threshold), sig_f),
        "T2": (threshold_onsets(sig_z, query_mask, fe.local_threshold), sig_z),
        "T3": (bocpd_onsets(sig_z, query_mask, hazard), sig_z),
        "T4": (bocpd_onsets(sig_f, query_mask, hazard), sig_f),
    }
    onsets: dict[str, list[Cause]] = {
        "T0": [fe.cause(inside[0], day_rows)] * n_causes,
    }
    for name, (cand, sig) in cands.items():
        pos = pick_onsets(cand, sig, query_mask, n_causes, e.min_onset_separation_minutes)
        onsets[name] = [fe.cause(int(span[p]), day_rows) for p in pos]
    info = FoldInfo(fe.ctx.fold.index, fe.ctx.columns, fe.ctx.categories)
    return ScoredQuery(q, day, info, n_causes, required_fields(q.task_index), prior, onsets)


def _previous_day_rows(exp: Experiment, excluded: Sequence[str]) -> dict[int, int]:
    """Row -> row of the same UTC minute-of-day on the previous usable day."""
    times = exp.values["time_utc"].to_list()
    days = exp.values["day"].to_list()
    by_day: dict[str, dict[int, int]] = {}
    for i, (d, ts) in enumerate(zip(days, times, strict=True)):
        by_day.setdefault(d, {})[ts.hour * 60 + ts.minute] = i
    usable = sorted(d for d in by_day if d not in set(excluded))
    out: dict[int, int] = {}
    for prev, cur in itertools.pairwise(usable):
        for mod, row in by_day[cur].items():
            if mod in by_day[prev]:
                out[row] = by_day[prev][mod]
    return out


# ------------------------------------------------------------------------- evaluation
def attach_gold(
    scored: Sequence[ScoredQuery], queries: Sequence[Query], failures: Sequence[Failure]
) -> None:
    """Gold failures of each scored query (evaluation only); the count must equal the number
    parsed from the instruction (``failure_count``), as checked for every query in ADR 0017."""
    cases, _ = build_cases(list(queries), list(failures))
    by_row = {c.query.row: c.failures for c in cases}
    for sq in scored:
        sq.gold = by_row[sq.query.row]
        if len(sq.gold) != sq.n_causes:
            raise ValueError(
                f"query {sq.query.row}: {sq.n_causes} failures parsed, {len(sq.gold)} labelled"
            )


def _ratio_estimate(hits: FloatArray, totals: FloatArray, n: int, seed: int) -> dict[str, Any]:
    clusters = [np.array([i], dtype=np.int64) for i in range(hits.size)]
    return cluster_bootstrap(
        lambda i: float(hits[i].sum() / totals[i].sum()), clusters, n, seed
    ).as_dict()


def _diff_estimate(diff: FloatArray, n: int, seed: int) -> dict[str, Any]:
    clusters = [np.array([i], dtype=np.int64) for i in range(diff.size)]
    e = cluster_bootstrap(lambda i: float(diff[i].mean()), clusters, n, seed)
    return {**e.as_dict(), "verdict": verdict(e)}


def _accuracy_difference(diff_hits: FloatArray, totals: FloatArray) -> Callable[[IntArray], float]:
    """Difference of two field accuracies (hits / failures) on resampled queries."""

    def stat(i: IntArray) -> float:
        return float(diff_hits[i].sum() / totals[i].sum())

    return stat


def _prediction(sq: ScoredQuery, causes: Sequence[Mapping[str, Any]]) -> MultiPrediction:
    """Official prediction with only the fields the task asks for (like the agent's answers)."""
    f = sq.fields
    return MultiPrediction(
        sq.query.row,
        tuple(
            Prediction(
                sq.query.row,
                c["occurrence_datetime"] if "occurrence_datetime" in f else None,
                c["component"] if "component" in f else None,
                c["reason"] if "reason" in f else None,
            )
            for c in causes
        ),
    )


def current_package_answer(
    package: Mapping[str, Any], sq: ScoredQuery, candidates: Sequence[str], past: Past
) -> list[dict[str, Any]]:
    """What today's package suggests: peak of K1, component C1 (C2 for a second failure),
    reason from the suggested category (ties by past frequency, else R0). Fixed in ADR 0020."""
    time_utc = parse_local(package["kpis"][0]["peak_time_local"]) if package["kpis"] else None
    comps = [c["component"] for c in package["components"] if c["component"] in candidates]
    pairs = past.pairs_before(sq.day)
    cat = package.get("suggested_category")
    mapped = sorted(r for r, cats in past.reason_map.items() if cat in cats)
    reason = (
        prior_reason([r for _, r in pairs if r in mapped], mapped) if mapped else None
    ) or prior_reason([r for _, r in pairs], past.reasons)
    out = []
    for i in range(sq.n_causes):
        comp = comps[i] if i < len(comps) else (comps[0] if comps else sorted(candidates)[0])
        out.append(
            {
                "occurrence_datetime": time_utc or sq.query.window_start_utc,
                "component": comp,
                "reason": reason,
            }
        )
    return out


def _field_hits(sq: ScoredQuery, causes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {f: best_hits(causes, sq.gold, [f]) for f in FIELDS}


def _official_block(official: pl.DataFrame, n: int, seed: int) -> dict[str, Any]:
    partial = official["score"].to_numpy().astype(np.float64)
    strict = official["strict"].to_numpy().astype(np.float64)
    ones = np.ones_like(partial)
    by_task = official.group_by("task_index").agg(
        pl.len().alias("n"),
        pl.col("strict").sum().alias("strict_correct"),
        pl.col("score").mean().round(4).alias("partial_mean"),
    )
    return {
        "n_queries": int(partial.size),
        "partial_score": _ratio_estimate(partial, ones, n, seed + 1),
        "strict_accuracy": _ratio_estimate(strict, ones, n, seed + 2),
        "by_task": {
            r["task_index"]: {k: r[k] for k in ("n", "strict_correct", "partial_mean")}
            for r in by_task.sort("task_index").iter_rows(named=True)
        },
    }


def evaluate_test(
    config: Config,
    scored: Sequence[ScoredQuery],
    selections: Mapping[int, Selection],
    test_days: Mapping[int, str],
    past: Past,
    candidates: Sequence[str],
    packages_dir: Path,
    a0: Sequence[Prediction],
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Selector, current-package chain and A0 on the test queries; per-method descriptive
    accuracies; official scores (checked against the hits computed here)."""
    b, seed = config.baselines, config.experiment.seed
    query_csv = config.paths.data_root / config.target_system / "query.csv"
    test = [s for s in scored if s.day in set(test_days.values())]
    fold_of_day = {d: k for k, d in test_days.items()}
    rows: list[dict[str, Any]] = []
    sel_hits: dict[str, list[int]] = {f: [] for f in FIELDS}
    pkg_hits: dict[str, list[int]] = {f: [] for f in FIELDS}
    n_gold: list[int] = []
    desc: dict[str, list[float]] = {
        m: [] for m in ONSET_METHODS + COMPONENT_METHODS + REASON_METHODS
    }
    mrr: dict[str, list[float]] = {m: [] for m in COMPONENT_METHODS}
    sel_preds, pkg_preds = [], []
    for sq in test:
        sel = selections[fold_of_day[sq.day]]
        causes = answer(sq, sel.onset, sel.component, sel.reason, past, scored)
        package = json.loads(
            (packages_dir / f"query_{sq.query.row:03d}.json").read_text(encoding="utf-8")
        )
        pkg = current_package_answer(package, sq, candidates, past)
        n_gold.append(len(sq.gold))
        for f, h in _field_hits(sq, causes).items():
            sel_hits[f].append(h)
        for f, h in _field_hits(sq, pkg).items():
            pkg_hits[f].append(h)
        for t in ONSET_METHODS:
            desc[t].append(
                best_hits(
                    [{"occurrence_datetime": x} for x in _onset_times(sq, t)],
                    sq.gold,
                    ["occurrence_datetime"],
                )
            )
        for c in COMPONENT_METHODS:
            desc[c].append(
                best_hits(
                    [{"component": x} for x in predict_components(sq, sel.onset, c)],
                    sq.gold,
                    ["component"],
                )
            )
            mrr[c].append(_mrr(sq, sel.onset, c))
        for r in REASON_METHODS:
            desc[r].append(
                best_hits(
                    [
                        {"reason": x}
                        for x in predict_reasons(sq, sel.onset, sel.component, r, past, scored)
                    ],
                    sq.gold,
                    ["reason"],
                )
            )
        sel_preds.append(_prediction(sq, causes))
        pkg_preds.append(_prediction(sq, pkg))
        rows.append(
            {
                "query_row": sq.query.row,
                "fold": fold_of_day[sq.day],
                "task_index": sq.query.task_index,
                "n_failures": len(sq.gold),
                "selection": f"{sel.onset}/{sel.component}/{sel.reason}",
                "expected_selector": round(
                    best_hits(causes, sq.gold, sq.fields) / (len(sq.gold) * len(sq.fields)), 2
                ),
                "expected_package": round(
                    best_hits(pkg, sq.gold, sq.fields) / (len(sq.gold) * len(sq.fields)), 2
                ),
                "selector": json.dumps(
                    [{**c, "occurrence_datetime": str(c["occurrence_datetime"])} for c in causes],
                    ensure_ascii=False,
                ),
            }
        )
    official = {
        "selector": score_with_official_script(
            sel_preds, query_csv, config.paths.openrca_repo, out_dir / "selector"
        ),
        "current_package": score_with_official_script(
            pkg_preds, query_csv, config.paths.openrca_repo, out_dir / "current_package"
        ),
        "constant_a0": score_with_official_script(
            a0, query_csv, config.paths.openrca_repo, out_dir / "constant_a0"
        ),
    }
    by_row = {r["query_row"]: r for r in rows}
    for name, key in (("selector", "expected_selector"), ("current_package", "expected_package")):
        for rec in official[name].iter_rows(named=True):
            expected = by_row[rec["query_row"]][key]
            if abs(rec["score"] - expected) > 1e-9:
                raise RuntimeError(
                    f"{name}: official score {rec['score']} != computed {expected} for query "
                    f"{rec['query_row']}"
                )
            by_row[rec["query_row"]][f"official_{name}"] = rec["score"]
    totals = np.array(n_gold, dtype=np.float64)

    def fields_block(hits: Mapping[str, list[int]], offset: int) -> dict[str, Any]:
        return {
            f: _ratio_estimate(
                np.array(hits[f], dtype=np.float64), totals, b.bootstrap, seed + offset + i
            )
            for i, f in enumerate(FIELDS)
        }

    metrics: dict[str, Any] = {
        "n_test_queries": len(test),
        "n_test_failures": int(totals.sum()),
        "selector": {
            "fields": fields_block(sel_hits, 10),
            "official": _official_block(official["selector"], b.bootstrap, seed + 20),
        },
        "current_package": {
            "fields": fields_block(pkg_hits, 30),
            "official": _official_block(official["current_package"], b.bootstrap, seed + 40),
        },
        "constant_a0": {
            "official": _official_block(official["constant_a0"], b.bootstrap, seed + 50)
        },
        "descriptive": {m: round(float(np.sum(v) / totals.sum()), 4) for m, v in desc.items()},
        "descriptive_mrr": {m: round(float(np.mean(v)), 4) for m, v in mrr.items()},
    }
    sel_score = official["selector"].sort("query_row")
    paired: list[dict[str, Any]] = []
    for rival in ("current_package", "constant_a0"):
        other = official[rival].sort("query_row")
        for metric, col in (("nota parcial", "score"), ("acerto estrito", "strict")):
            d = sel_score[col].cast(pl.Float64).to_numpy() - other[col].cast(pl.Float64).to_numpy()
            paired.append(
                {
                    "rival": rival,
                    "metric": metric,
                    **_diff_estimate(d.astype(np.float64), b.bootstrap, seed + 60),
                }
            )
    for i, f in enumerate(FIELDS):
        # difference of field accuracies (hits / failures), resampling queries
        dh = (np.array(sel_hits[f]) - np.array(pkg_hits[f])).astype(np.float64)
        est = cluster_bootstrap(
            _accuracy_difference(dh, totals),
            [np.array([j], dtype=np.int64) for j in range(dh.size)],
            b.bootstrap,
            seed + 70 + i,
        )
        paired.append(
            {
                "rival": "current_package",
                "metric": f"campo {f}",
                **est.as_dict(),
                "verdict": verdict(est),
            }
        )
    metrics["paired"] = paired
    return metrics, list(by_row.values())


def oracle_component_ranks(
    fe: _FoldEvidence, failures: Sequence[Failure], prior: Callable[[str], dict[str, float]]
) -> list[dict[str, Any]]:
    """Descriptive: component methods on windows built from the LABELLED onset (comparable to
    E4-E6 oracle localization). Evaluation only."""
    exp = fe.exp
    out = []
    for f in failures:
        onset_row = row_of(exp.minute_to_row, f.start_utc.timestamp() / 60.0)
        day = exp.day_arr[onset_row]
        day_rows = np.flatnonzero(exp.day_arr == day).astype(np.int64)
        cause = fe.cause(onset_row, day_rows)
        for c in COMPONENT_METHODS:
            scores = prior(str(day)) if c == "C0" else cause.scores[c]
            out.append(
                {
                    "method": c,
                    "fold": fe.ctx.fold.index,
                    "component": f.component,
                    "reason": f.reason,
                    "rank": pessimistic_rank(f.component, scores),
                }
            )
    return out


def run_evidence_select(config: Config, tables_dir: Path) -> dict[str, Any]:
    system, name, e, t = (
        config.target_system,
        config.experiment.name,
        config.evidence_select,
        config.tasks,
    )
    lstm_table, models_dir, thresholds, catalog = _e5_artifacts(config, tables_dir)
    with Run(config) as run:
        started = time.perf_counter()
        exp = Experiment(config, tables_dir)
        index = exp.values.select("day", "time_utc")
        queries = read_queries(config.paths.data_root / system / "query.csv")
        q_days = [day_of(q.window_start_utc, index) for q in queries]
        cand = load_candidates(config.agent.prompts_dir, system)
        candidates, reasons = list(cand.components), tuple(cand.reasons)
        edges_df = pl.read_csv(tables_dir / f"dependency_edges_{system}.csv")
        edges = [
            (str(r["src"]), str(r["dst"]))
            for r in edges_df.filter(
                pl.col("relation").is_in(list(e.pagerank_relations))
            ).iter_rows(named=True)
        ]
        prev_rows = _previous_day_rows(exp, t.excluded_days)
        excluded = set(t.excluded_days)
        labelled = [
            (d, f.component, f.reason)
            for f, d in zip(exp.failures, exp.f_days, strict=True)
            if d not in excluded
        ]

        def pairs_before(day: str) -> list[tuple[str, str]]:
            return [(c, r) for d, c, r in labelled if d < day]

        def prior_components(day: str) -> dict[str, float]:
            counts = dict.fromkeys(candidates, 0.0)
            for c, _ in pairs_before(day):
                if c in counts:
                    counts[c] += 1.0
            return counts

        past = Past(
            pairs_before,
            reasons,
            config.explain.reason_categories.get(system, {}),
            tuple(sorted({v["category"] for v in catalog.values()})),
            e.reason_logistic_c,
        )
        folds = exp.folds[: e.max_folds] if e.max_folds else exp.folds
        initial = sorted(
            {str(d) for d in exp.day_arr if d < t.first_test_day and d not in excluded}
        )
        scored: list[ScoredQuery] = []
        oracle_rows: list[dict[str, Any]] = []
        for fold in folds:
            ctx = _FoldContext.build(config, exp, fold, models_dir, lstm_table, catalog)
            fe = _FoldEvidence(
                config,
                exp,
                ctx,
                thresholds[(fold.index, config.explain.explained_seed, "lstm_forecast")],
                candidates,
                edges,
                prev_rows,
            )
            days_here = {fold.test_day} | (set(initial) if fold.index == 0 else set())
            for q, d in zip(queries, q_days, strict=True):
                if d in days_here:
                    scored.append(
                        _candidates(fe, q, d, failure_count(q.instruction), prior_components(d))
                    )
            oracle_rows += oracle_component_ranks(fe, exp.tested(fold), prior_components)
            run.logger.info(
                "fold %d: candidates done (%.0fs)", fold.index, time.perf_counter() - started
            )

        attach_gold(scored, queries, exp.failures)
        test_days = {f.index: f.test_day for f in folds}
        selections = prequential_selections(scored, test_days, past)

        a0: list[Prediction] = []
        for fold in folds:
            known = [
                f
                for f, d in zip(exp.failures, exp.f_days, strict=True)
                if d < fold.test_day and d not in excluded
            ]
            fold_q = [q for q, d in zip(queries, q_days, strict=True) if d == fold.test_day]
            a0.extend(constant_predictions(known, fold_q))
        packages_dir = config.paths.processed_dir / system / "evidence" / e.package_name
        metrics, per_query = evaluate_test(
            config,
            scored,
            selections,
            test_days,
            past,
            candidates,
            packages_dir,
            a0,
            run.artifacts_dir / "official",
        )
        oracle = pl.DataFrame(oracle_rows)
        metrics["descriptive_oracle_onset"] = {
            c: {
                "hr@1": round(float(np.mean(sub["rank"].to_numpy() == 1)), 4),
                "mrr": round(float(np.mean(1.0 / sub["rank"].to_numpy())), 4),
            }
            for c in COMPONENT_METHODS
            for sub in [oracle.filter(pl.col("method") == c)]
        }
        metrics.update(
            {
                "system": system,
                "folds": len(folds),
                "explained_lstm_run": lstm_table["run_dir"],
                "current_packages": e.package_name,
                "n_scored_queries": len(scored),
                "selections": {
                    str(k): {
                        "test_day": test_days[k],
                        "onset": s.onset,
                        "component": s.component,
                        "reason": s.reason,
                        "reason_beats_prior": s.reason_beats_prior,
                        "n_past_queries": s.n_past_queries,
                        "n_past_failures": s.n_past_failures,
                        "past_accuracy": s.accuracy,
                    }
                    for k, s in selections.items()
                },
            }
        )
        base_name = config.explain.lstm_name.replace("lstm", "baselines", 1)
        base_path = tables_dir / f"baselines_{system}_{base_name}.json"
        if e.max_folds is None and base_path.exists():
            e4 = json.loads(base_path.read_text(encoding="utf-8"))["diagnosis_constant"]
            mine = metrics["constant_a0"]["official"]["partial_score"]["value"]
            metrics["constant_a0"]["matches_e4"] = abs(mine - e4["partial_score"]["value"]) < 1e-9
        run.log_metrics(metrics)
        pl.DataFrame(per_query).write_csv(run.artifacts_dir / "test_predictions.csv")
        oracle.write_csv(run.artifacts_dir / "oracle_component_ranks.csv")
        table = {
            **metrics,
            "run_dir": run.dir.name,
            "labels": LABELS,
            "seconds": round(time.perf_counter() - started, 1),
        }
        tables_dir.mkdir(parents=True, exist_ok=True)
        (tables_dir / f"evidence_select_{system}_{name}.json").write_text(
            json.dumps(table, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
        )
        write_markdown(tables_dir / f"evidence_select_{system}_{name}.md", table)
        return {
            "run_dir": str(run.dir),
            "seconds": table["seconds"],
            "selector_partial": metrics["selector"]["official"]["partial_score"]["value"],
        }


def _est(e: Mapping[str, Any]) -> str:
    return f"{e['value']:.2f} [{e['ci_low']:.2f}; {e['ci_high']:.2f}]"


def write_markdown(path: Path, t: Mapping[str, Any]) -> None:
    lab = t["labels"]
    lines = [
        f"# Evidência por seleção prequencial — {t['system']} (ADR 0020)",
        "",
        f"{t['n_test_queries']} consultas de teste ({t['n_test_failures']} falhas), "
        f"{t['folds']} dobras. Em cada dobra, cada etapa escolhe o método com maior acerto nas "
        "consultas dos dias anteriores ao teste (ordem: instante, componente dado o instante, "
        "motivo dados os dois). IC de 95% por bootstrap sobre consultas.",
        "",
        "## Escolhas por dobra",
        "",
        "| Dobra | Dia de teste | Consultas passadas | Instante | Componente | Motivo "
        "| Motivo supera R0 no passado |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, s in t["selections"].items():
        lines.append(
            f"| {k} | {s['test_day']} | {s['n_past_queries']} | {s['onset']} | {s['component']} "
            f"| {s['reason']} | {'sim' if s['reason_beats_prior'] else 'não'} |"
        )
    names = {
        "occurrence_datetime": "instante (≤ 60 s)",
        "component": "componente",
        "reason": "motivo",
    }
    lines += [
        "",
        "## Acerto por campo (todas as falhas de teste)",
        "",
        "| Cadeia | " + " | ".join(names.values()) + " |",
        "|---|---|---|---|",
    ]
    for key, label in (("selector", "seletor prequencial"), ("current_package", "pacote atual")):
        f = t[key]["fields"]
        lines.append(f"| {label} | " + " | ".join(_est(f[x]) for x in names) + " |")
    lines += [
        "",
        "## Nota oficial do OpenRCA (só os campos pedidos por consulta)",
        "",
        "| Cadeia | Nota parcial | Acerto estrito |",
        "|---|---|---|",
    ]
    for key, label in (
        ("selector", "seletor prequencial"),
        ("current_package", "pacote atual"),
        ("constant_a0", "A0 constante (E4)"),
    ):
        o = t[key]["official"]
        lines.append(f"| {label} | {_est(o['partial_score'])} | {_est(o['strict_accuracy'])} |")
    lines += [
        "",
        "## Comparações pareadas (seletor menos rival, mesmas consultas)",
        "",
        "| Rival | Métrica | Diferença | Veredito |",
        "|---|---|---|---|",
    ]
    for p in t["paired"]:
        lines.append(
            f"| {p['rival']} | {p['metric']} | {p['value']:+.3f} "
            f"[{p['ci_low']:+.3f}; {p['ci_high']:+.3f}] | {p['verdict']} |"
        )
    d, dm, do = t["descriptive"], t["descriptive_mrr"], t["descriptive_oracle_onset"]
    lines += [
        "",
        "## Descritivo: cada método no teste (não usado para escolher)",
        "",
        "Instante: acerto a até 60 s. Componente: acerto com o instante escolhido pelo seletor "
        "na dobra, e com o instante rotulado (janela-oráculo, comparável à E4 a E6). Motivo: "
        "acerto com o instante e o componente escolhidos pelo seletor.",
        "",
        "| Método | Acerto | MRR | HR@1 com instante rotulado | MRR com instante rotulado |",
        "|---|---|---|---|---|",
    ]
    for m in ONSET_METHODS:
        lines.append(f"| {m}: {lab[m]} | {d[m]:.2f} | — | — | — |")
    for m in COMPONENT_METHODS:
        lines.append(
            f"| {m}: {lab[m]} | {d[m]:.2f} | {dm[m]:.2f} | {do[m]['hr@1']:.2f} "
            f"| {do[m]['mrr']:.2f} |"
        )
    for m in REASON_METHODS:
        lines.append(f"| {m}: {lab[m]} | {d[m]:.2f} | — | — | — |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
