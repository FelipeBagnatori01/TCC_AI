"""E8: evaluation of the E7 agent runs (RQ3 diagnosis, RQ4 fidelity and cost; ADR 0022).

Every score comes from the official OpenRCA evaluator (``openrca.score_with_official_script``):
this module never re-implements it. On top of the per-query official scores it computes

* bootstrap CIs over queries and paired differences between chains (same queries);
* the exact McNemar test on strict hits, with Holm's correction among the agent variants;
* a per-field reading of the official report (approximate for two-failure queries, because the
  official evaluator picks the best cause assignment and here each criterion counts alone);
* automatic fidelity checks of each explanation against the tool outputs of its own transcript
  (issued evidence ids, component/KPI names, numbers);
* the blinded manual-audit sheet (generated once, never overwritten) and its summary once filled;
* the cache-replay check of the E7 acceptance criterion (zero paid calls, same predictions).

The A0 constant and the no-LLM package chain come from the official reports already written by
the ``evidence-select`` run (ADR 0020/0021) on the same queries.
"""

from __future__ import annotations

import csv
import json
import re
from ast import literal_eval
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from tcc_rca.config import Config
from tcc_rca.evaluation.bootstrap import Estimate, cluster_bootstrap
from tcc_rca.evaluation.compare import verdict
from tcc_rca.evaluation.openrca import score_with_official_script
from tcc_rca.tracking import Run

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

FIELDS = ("instante", "componente", "motivo")
REFERENCES = ("A0", "pacote")
AUDIT_LABELS = ("sustentada", "parcial", "nao_sustentada", "nao_verificavel")


# --------------------------------------------------------------------------- official scores
@dataclass(frozen=True)
class AgentAnswer:
    """One agent answer already in the official JSON format (``predictions_s0.csv``)."""

    query_row: int
    payload: str

    def as_official_json(self) -> str:
        return self.payload


@dataclass(frozen=True)
class Criterion:
    field: str  # instante | componente | motivo
    expected: str
    hit: bool


def parse_criteria(groundtruth: str, passed: Sequence[str]) -> list[Criterion]:
    """Criteria of one official report row: one line per field and failure in ``groundtruth``;
    a criterion is hit when its expected value is among the official ``passed`` values."""
    out = []
    for line in groundtruth.splitlines():
        if "occurrence time" in line:
            field, value = "instante", line.rsplit(" of ", 1)[1].strip()
        elif "component" in line:
            field, value = "componente", line.rsplit(" is ", 1)[1].strip()
        elif "reason" in line:
            field, value = "motivo", line.rsplit(" is ", 1)[1].strip()
        else:
            continue
        out.append(Criterion(field, value, value in passed))
    return out


def report_criteria(report_csv: Path) -> list[list[Criterion]]:
    """Criteria per row of an official report CSV, in file order."""
    report = pl.read_csv(report_csv).fill_null("")
    return [
        parse_criteria(r["groundtruth"], literal_eval(r["passed"]) if r["passed"] else [])
        for r in report.iter_rows(named=True)
    ]


@dataclass
class Chain:
    """Per-query official results of one chain (A0, package or agent variant), by query row."""

    name: str
    rows: list[int]
    task: dict[int, str]
    score: dict[int, float]
    criteria: dict[int, list[Criterion]]
    official_dir: str

    def array(self, rows: Sequence[int], kind: str = "score") -> FloatArray:
        if kind == "score":
            return np.array([self.score[r] for r in rows], dtype=np.float64)
        return np.array([float(self.score[r] == 1.0) for r in rows], dtype=np.float64)


def _chain(name: str, rows: Sequence[int], official_dir: Path) -> Chain:
    report = pl.read_csv(official_dir / "official_report.csv")
    if report.height != len(rows):
        raise RuntimeError(f"{official_dir}: {report.height} report rows, {len(rows)} predictions")
    ordered = sorted(rows)
    crit = report_criteria(official_dir / "official_report.csv")
    return Chain(
        name=name,
        rows=ordered,
        task=dict(zip(ordered, report["task_index"].to_list(), strict=True)),
        score=dict(zip(ordered, report["score"].cast(pl.Float64).to_list(), strict=True)),
        criteria=dict(zip(ordered, crit, strict=True)),
        official_dir=str(official_dir),
    )


# --------------------------------------------------------------------------- statistics
def mean_estimate(x: FloatArray, n_resamples: int, seed: int) -> Estimate:
    """Mean with a percentile CI from resampling queries (each query its own cluster)."""
    clusters = [np.array([i], dtype=np.int64) for i in range(x.size)]
    return cluster_bootstrap(lambda idx: float(x[idx].mean()), clusters, n_resamples, seed)


def ratio_estimate(num: FloatArray, den: FloatArray, n_resamples: int, seed: int) -> Estimate:
    """sum(num) / sum(den) over the units with den > 0, CI from resampling those units."""
    keep = np.flatnonzero(den > 0)
    if keep.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    clusters = [np.array([i], dtype=np.int64) for i in keep]
    return cluster_bootstrap(
        lambda idx: float(num[idx].sum() / den[idx].sum()), clusters, n_resamples, seed
    )


def mcnemar_exact(
    first: Sequence[bool] | npt.NDArray[np.bool_], second: Sequence[bool] | npt.NDArray[np.bool_]
) -> tuple[int, int, float]:
    """Exact (binomial) two-sided McNemar test on paired hits.

    Returns ``(b, c, p)``: b = only ``first`` hits, c = only ``second`` hits, p = two-sided
    binomial p-value of b out of b + c at 1/2 (1.0 without discordant pairs)."""
    from scipy.stats import binomtest

    a, s = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    b, c = int((a & ~s).sum()), int((~a & s).sum())
    p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    return b, c, p


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1), in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def paired(
    first: Chain, second: Chain, rows: Sequence[int], n_resamples: int, seed: int
) -> dict[str, Any]:
    """Paired partial-score difference (CI over queries) and McNemar on strict hits."""
    d = first.array(rows) - second.array(rows)
    e = mean_estimate(d, n_resamples, seed)
    b, c, p = mcnemar_exact(first.array(rows, "strict") > 0, second.array(rows, "strict") > 0)
    return {
        "comparison": f"{first.name} - {second.name}",
        "partial_difference": e.as_dict(),
        "verdict": verdict(e),
        "queries_better": int((d > 0).sum()),
        "queries_worse": int((d < 0).sum()),
        "mcnemar_only_first": b,
        "mcnemar_only_second": c,
        "mcnemar_p": round(p, 6),
    }


def field_estimate(
    chain: Chain, rows: Sequence[int], field: str, n_resamples: int, seed: int
) -> dict[str, Any]:
    """Hit rate of one field over its criteria (approximate for two-failure queries)."""
    hits = np.array([sum(c.hit for c in chain.criteria[r] if c.field == field) for r in rows])
    asked = np.array([sum(c.field == field for c in chain.criteria[r]) for r in rows])
    e = ratio_estimate(hits.astype(np.float64), asked.astype(np.float64), n_resamples, seed)
    return {**e.as_dict(), "criteria": int(asked.sum())}


def single_component_hits(chain: Chain, rows: Sequence[int]) -> dict[int, bool]:
    """Component hit of the single-failure queries that ask for the component."""
    out = {}
    for r in rows:
        comp = [c for c in chain.criteria[r] if c.field == "componente"]
        if len(comp) == 1:
            out[r] = comp[0].hit
    return out


def component_type(component: str) -> str:
    """os_001 -> os; Tomcat01 -> Tomcat (same rule as ``models.rca.reason``)."""
    return re.sub(r"[\d_]+", "", component)


def error_breakdown(chains: Sequence[Chain], rows: Sequence[int]) -> dict[str, Any]:
    """Hit rate per expected reason (reason criteria) and per expected component type
    (component criteria), for every chain."""
    out: dict[str, Any] = {"by_reason": {}, "by_component_type": {}}
    for key, field, label in (
        ("by_reason", "motivo", lambda v: v),
        ("by_component_type", "componente", component_type),
    ):
        groups: dict[str, dict[str, list[int]]] = {}
        for chain in chains:
            for r in rows:
                for c in chain.criteria[r]:
                    if c.field == field:
                        g = groups.setdefault(label(c.expected), {}).setdefault(chain.name, [0, 0])
                        g[0] += int(c.hit)
                        g[1] += 1
        out[key] = {
            g: {"n": next(iter(v.values()))[1], **{k: round(h / n, 4) for k, (h, n) in v.items()}}
            for g, v in sorted(groups.items())
        }
    return out


# --------------------------------------------------------------------------- fidelity
_NUMBER = re.compile(r"(?<![\w:./+-])(\d[\d.,]*\d|\d)(?![\d]*[:\-/])(?![.,]\d)")
_THOUSANDS = re.compile(r"[1-9]\d{0,2}(?:\.\d{3})+(?:,\d+)?")
_ID = re.compile(r"\b([TCKN]\d+)\b")
_IDENT = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")


_APPROX = re.compile(
    r"(cerca de|aproximadamente|aprox\.?|por volta de|em torno de|quase|~|≈)\s*$", re.IGNORECASE
)
CONTROL_FACTOR = 1.37  # chance level of the number check: stated values scaled by this factor


@dataclass(frozen=True)
class Number:
    value: float
    decimals: int
    text: str
    approx: bool = False  # preceded by 'cerca de', '~', ...


def parse_number(text: str) -> Number | None:
    """Portuguese-formatted number: '.' groups thousands, ',' is the decimal mark
    ('18.628' = 18628; '1.574,9' = 1574.9; '0,0094'); a lone '.' not grouping thousands is a
    decimal point ('0.5')."""
    t = text.strip(".,")
    if not t:
        return None
    if _THOUSANDS.fullmatch(t):
        t = t.replace(".", "")
    if "," in t:
        if t.count(",") > 1 or "." in t:
            return None
        whole, frac = t.split(",")
        return Number(float(f"{whole}.{frac}"), len(frac), text)
    if t.count(".") > 1:
        return None
    if "." in t:
        whole, frac = t.split(".")
        return Number(float(t), len(frac), text)
    return Number(float(t), 0, text)


def numbers_in_text(text: str, min_integer: float) -> list[Number]:
    """Numbers an explanation states as values: times (hh:mm), dates, ids (T7, K1, p99) and
    names (db_003) are excluded, and so are integers below ``min_integer`` (counts, minutes,
    ordinals and percentages, which the check cannot tell from KPI values)."""
    out = []
    for m in _NUMBER.finditer(text):
        n = parse_number(m.group(1))
        if n is None:
            continue
        if n.decimals == 0 and abs(n.value) < min_integer:
            continue
        approx = bool(_APPROX.search(text[max(0, m.start() - 25) : m.start()]))
        out.append(Number(n.value, n.decimals, n.text, approx))
    return out


def json_numbers(value: Any) -> list[float]:
    """Every numeric leaf of a parsed JSON value."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, dict):
        return [x for v in value.values() for x in json_numbers(v)]
    if isinstance(value, list):
        return [x for v in value for x in json_numbers(v)]
    return []


def grounded(n: Number, values: FloatArray, rel_tol: float) -> bool:
    """A stated number is grounded when some tool value equals it up to one unit of its last
    decimal (rounded or truncated: 8474,66 written as 8474); an approximate one ('cerca de
    68 ms') also within ``rel_tol``.

    A relative tolerance for every number would be meaningless: against the thousands of values
    of a transcript, 5% matches two thirds of numbers scaled by ``CONTROL_FACTOR`` (measured on
    the E7 runs), while rounding alone matches about one in ten."""
    if values.size == 0:
        return False
    rounding = 10.0 ** (-n.decimals)
    tol = max(rounding, rel_tol * abs(n.value)) if n.approx else rounding
    return bool(np.any(np.abs(values - n.value) <= tol + 1e-9))


@dataclass(frozen=True)
class Transcript:
    tool_outputs: list[Any]  # parsed JSON of every tool message
    tool_text: str  # raw text of every tool message
    tool_names: tuple[str, ...]


def read_transcript(path: Path) -> Transcript:
    data = json.loads(path.read_text(encoding="utf-8"))
    outputs, texts = [], []
    for m in data["messages"]:
        if m.get("role") != "tool":
            continue
        texts.append(m["content"])
        try:
            outputs.append(json.loads(m["content"]))
        except json.JSONDecodeError:
            outputs.append(None)
    names = tuple(sorted({t["tool"] for t in data.get("tools", [])}))  # tools actually called
    return Transcript(outputs, "\n".join(texts), names)


def issued_ids(outputs: Iterable[Any]) -> set[str]:
    """Evidence ids handed to the agent: every ``"id"`` value (T#, C#, K#, N#) in tool outputs."""
    found: set[str] = set()

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            for k, x in v.items():
                if k == "id" and isinstance(x, str) and _ID.fullmatch(x):
                    found.add(x)
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    for o in outputs:
        walk(o)
    return found


def fidelity_checks(
    explanation: str,
    transcript: Transcript,
    known_components: set[str],
    known_kpis: Sequence[str],
    rel_tol: float,
    min_integer: float,
    tool_names: Iterable[str] = (),
) -> dict[str, Any]:
    """Automatic, gold-free checks of one explanation against its own transcript.

    * ids: ``T#/C#/K#/N#`` written in the text that no tool output issued;
    * components: names shaped like a component of the system (same type prefix) that are
      neither a system component nor present in the tool outputs;
    * identifiers: KPI-like tokens (``Proc_Used_Pct``, ``access_rt_mean_ms``) found neither
      in any catalog KPI name (abbreviations allowed: substring, case-insensitive), nor in the
      tool outputs, nor among the tool names;
    * numbers: stated values (see ``numbers_in_text``) without a literal match in the tool
      outputs. Derived numbers (differences, ratios) also count here, so this is an upper
      bound of invented values; the manual audit measures support.
    """
    ids_text = set(_ID.findall(explanation))
    ids_bad = sorted(ids_text - issued_ids(transcript.tool_outputs))

    types = sorted({component_type(c) for c in known_components}, key=len, reverse=True)
    comp_re = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in types) + r")_?\d+\b")
    tool_lower = transcript.tool_text.lower()
    comps = set(comp_re.findall(explanation))
    comps_bad = sorted(c for c in comps if c not in known_components and c not in tool_lower)

    kpis_lower = [k.lower() for k in known_kpis]
    names = {t.lower() for t in (*transcript.tool_names, *tool_names)}
    idents = {t for t in _IDENT.findall(explanation) if not comp_re.fullmatch(t)}
    idents_bad = sorted(
        t
        for t in idents
        if t.lower() not in names
        and t.lower() not in tool_lower
        and not any(t.lower() in k for k in kpis_lower)
    )

    values = np.array(
        [x for o in transcript.tool_outputs for x in json_numbers(o)], dtype=np.float64
    )
    nums = numbers_in_text(explanation, min_integer)
    literal = set(re.findall(r"\d+(?:[.,]\d+)?", transcript.tool_text))

    def ok(n: Number) -> bool:  # numeric match, or the same digits in a name (port 7778)
        return grounded(n, values, rel_tol) or n.text in literal

    nums_bad = [n.text for n in nums if not ok(n)]
    control = [
        Number(round(n.value * CONTROL_FACTOR, n.decimals), n.decimals, n.text, n.approx)
        for n in nums
    ]
    return {
        "ids": len(ids_text),
        "ids_not_issued": ids_bad,
        "components": len(comps),
        "components_unknown": comps_bad,
        "identifiers": len(idents),
        "identifiers_unknown": idents_bad,
        "numbers": len(nums),
        "numbers_ungrounded": nums_bad,
        "numbers_control_grounded": sum(grounded(n, values, rel_tol) for n in control),
    }


# --------------------------------------------------------------------------- manual audit
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ(\"-])|\n+")


def factual_sentences(explanation: str, is_factual: Callable[[str], bool]) -> list[str]:
    """Sentences of an explanation that state something checkable (``is_factual``)."""
    parts = [s.strip(" -•\t") for s in _SENTENCE.split(explanation)]
    return [s for s in parts if len(s) > 15 and is_factual(s)]


def audit_sample(
    items: Sequence[tuple[str, str, int]], per_stratum: int, seed: int
) -> list[tuple[str, str, int]]:
    """``per_stratum`` (system, variant, query_row) items per (system, variant), drawn without
    replacement with a fixed seed, then shuffled together (the auditor sees no order)."""
    rng = np.random.default_rng(seed)
    strata: dict[tuple[str, str], list[tuple[str, str, int]]] = {}
    for it in sorted(items):
        strata.setdefault((it[0], it[1]), []).append(it)
    picked: list[tuple[str, str, int]] = []
    for key in sorted(strata):
        pool = strata[key]
        idx = rng.choice(len(pool), size=min(per_stratum, len(pool)), replace=False)
        picked.extend(pool[i] for i in sorted(idx))
    order = rng.permutation(len(picked))
    return [picked[i] for i in order]


def summarize_audit(sheet: Path, key: Path, n_resamples: int, seed: int) -> dict[str, Any]:
    """Share of audited claims per label and variant (CI over explanations) once filled."""
    rows = list(csv.DictReader(sheet.open(encoding="utf-8", newline="")))
    mapping = json.loads(key.read_text(encoding="utf-8"))
    filled = [r for r in rows if r["classificacao"].strip()]
    bad = sorted({r["classificacao"] for r in filled} - set(AUDIT_LABELS))
    if bad:
        raise ValueError(f"unknown audit labels {bad}; use {AUDIT_LABELS}")
    out: dict[str, Any] = {"claims": len(rows), "classified": len(filled), "by_variant": {}}
    if not filled:
        return out
    groups: dict[str, dict[str, list[str]]] = {}
    for r in filled:
        item = mapping[r["explicacao"]]
        groups.setdefault(item["variant"], {}).setdefault(r["explicacao"], []).append(
            r["classificacao"].strip()
        )
    for variant, by_expl in sorted(groups.items()):
        expl = list(by_expl.values())
        res: dict[str, Any] = {"explanations": len(expl), "claims": sum(map(len, expl))}
        for label in AUDIT_LABELS:
            num = np.array([sum(x == label for x in e) for e in expl], dtype=np.float64)
            den = np.array([len(e) for e in expl], dtype=np.float64)
            res[label] = ratio_estimate(num, den, n_resamples, seed).as_dict()
        out["by_variant"][variant] = res
    return out


# --------------------------------------------------------------------------- run
def agent_table(tables_dir: Path, system: str, run_name: str, model_slug: str) -> dict[str, Any]:
    path = tables_dir / f"agent_{system}_{run_name}_{model_slug}.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def run_tool_names(run_dir: Path) -> list[str]:
    """Tools offered to the agent in a run (first request of the LLM call log)."""
    with (run_dir / "artifacts" / "llm_calls.jsonl").open(encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    return [t["name"] for t in first["request"]["tools"]]


def _cases(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "artifacts" / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in lines]


def replay_check(original: Path, replay: Path, replay_table: Mapping[str, Any]) -> dict[str, Any]:
    """E7 criterion: the replay from the LLM cache made no paid call and reproduced the run."""
    same_pred = (original / "artifacts" / "predictions_s0.csv").read_bytes() == (
        replay / "artifacts" / "predictions_s0.csv"
    ).read_bytes()
    same_metrics = (original / "metrics.json").read_bytes() == (
        replay / "metrics.json"
    ).read_bytes()
    paid = int(replay_table["usage"]["paid_calls"])
    return {
        "replay_run": replay.name,
        "paid_calls": paid,
        "cache_hits": int(replay_table["usage"]["cache_hits"]),
        "same_predictions": same_pred,
        "same_metrics": same_metrics,
        "ok": paid == 0 and same_pred and same_metrics,
    }


def reference_chains(config: Config, tables_dir: Path) -> tuple[dict[str, Chain], dict[str, Any]]:
    """A0 and the no-LLM package chain from the ``evidence-select`` official reports; their
    partial scores must equal the versioned table."""
    system, name_es = config.target_system, config.agent_eval.evidence_select_name
    es_table = json.loads(
        (tables_dir / f"evidence_select_{system}_{name_es}.json").read_text(encoding="utf-8")
    )
    es_official = config.paths.experiments_dir / es_table["run_dir"] / "artifacts" / "official"
    chains: dict[str, Chain] = {}
    for name, folder in (("A0", "constant_a0"), ("pacote", "current_package")):
        pred = pl.read_csv(es_official / folder / "prediction.csv")["query_row"].to_list()
        chains[name] = _chain(name, pred, es_official / folder)
        ref = es_table[folder]["official"]["partial_score"]["value"]
        got = float(np.mean(list(chains[name].score.values())))
        if abs(got - ref) > 1e-3:
            raise RuntimeError(f"{name}: official partial {got:.4f} != table {ref:.4f}")
    return chains, es_table


def score_agent_run(
    name: str, run_dir: Path, query_csv: Path, openrca_repo: Path, out_dir: Path
) -> Chain:
    """Official scores of one agent run (``predictions_s0.csv``), written to ``out_dir``."""
    preds = pl.read_csv(run_dir / "artifacts" / "predictions_s0.csv")
    answers = [
        AgentAnswer(int(r), str(p))
        for r, p in zip(preds["row_id"], preds["prediction"], strict=True)
    ]
    score_with_official_script(answers, query_csv, openrca_repo, out_dir)
    return _chain(name, [a.query_row for a in answers], out_dir)


def _is_factual_factory(known_components: set[str]) -> Callable[[str], bool]:
    def is_factual(s: str) -> bool:
        return bool(
            re.search(r"\d", s)
            or "::" in s
            or _IDENT.search(s)
            or any(c in s for c in known_components)
        )

    return is_factual


def run_agent_eval(config: Config, tables_dir: Path) -> dict[str, Any]:
    """Official scores, paired comparisons, fidelity, cost and replay check of one system."""
    c, system = config.agent_eval, config.target_system
    nb, seed = c.bootstrap, config.experiment.seed
    from tcc_rca.agent.tasks import load_candidates

    candidates = load_candidates(config.agent.prompts_dir, system)
    catalog = pl.read_csv(tables_dir / f"feature_catalog_{system}.csv")
    known_components = set(catalog["component"].to_list()) | set(candidates.components)
    known_kpis = sorted(set(catalog["kpi"].to_list()))
    query_csv = config.paths.data_root / system / "query.csv"

    with Run(config) as run:
        chains, es_table = reference_chains(config, tables_dir)
        tables: dict[str, dict[str, Any]] = {}
        runs: dict[str, Path] = {}
        for v, run_name in zip(c.variants, c.run_names, strict=True):
            t = agent_table(tables_dir, system, run_name, c.model_slug)
            if t["case_set"] != "test" or t["budget_stopped"]:
                raise ValueError(f"{run_name}: not a complete test run")
            tables[v] = t
            runs[v] = config.paths.experiments_dir / t["run_dir"]
            chains[v] = score_agent_run(
                v, runs[v], query_csv, config.paths.openrca_repo, run.artifacts_dir / "official" / v
            )
        rows = chains["A0"].rows
        for name, chain in chains.items():
            if chain.rows != rows:
                raise RuntimeError(f"{name} answers other queries than A0")

        summary = {}
        for name, chain in chains.items():
            summary[name] = {
                "partial_score": mean_estimate(chain.array(rows), nb, seed).as_dict(),
                "strict_accuracy": mean_estimate(chain.array(rows, "strict"), nb, seed).as_dict(),
                "fields": {f: field_estimate(chain, rows, f, nb, seed) for f in FIELDS},
            }
        by_task: dict[str, Any] = {}
        for task in sorted(set(chains["A0"].task.values())):
            t_rows = [r for r in rows if chains["A0"].task[r] == task]
            by_task[task] = {"n": len(t_rows)} | {
                name: round(float(ch.array(t_rows).mean()), 4) for name, ch in chains.items()
            }

        primary = tuple(c.primary_comparison)
        between = [(a, b) for a, b in c.variant_comparisons]
        comparisons = [paired(chains[a], chains[b], rows, nb, seed) for a, b in between]
        for row, p_adj in zip(
            comparisons, holm([r["mcnemar_p"] for r in comparisons]), strict=True
        ):
            row["mcnemar_p_holm"] = round(p_adj, 6)
            row["role"] = (
                "primary" if row["comparison"] == f"{primary[0]} - {primary[1]}" else ("secondary")
            )
        references = [
            paired(chains[v], chains[ref], rows, nb, seed) | {"role": "reference"}
            for v in c.variants
            for ref in REFERENCES
        ]

        # localization (RQ2): component of single-failure queries, agent vs. package C1
        pkg_hits = single_component_hits(chains["pacote"], rows)
        loc_rows = sorted(pkg_hits)
        localization: dict[str, Any] = {"queries": len(loc_rows)}
        pkg_arr = np.array([float(pkg_hits[r]) for r in loc_rows])
        localization["pacote"] = mean_estimate(pkg_arr, nb, seed).as_dict()
        for v in c.variants:
            hits = single_component_hits(chains[v], rows)
            arr = np.array([float(hits[r]) for r in loc_rows])
            e = mean_estimate(arr - pkg_arr, nb, seed)
            b, cc, p = mcnemar_exact(arr > 0, pkg_arr > 0)
            localization[v] = {
                **mean_estimate(arr, nb, seed).as_dict(),
                "minus_pacote": e.as_dict(),
                "verdict": verdict(e),
                "mcnemar": [b, cc, round(p, 4)],
            }

        # fidelity, cost and evidence use per variant
        fidelity: dict[str, Any] = {}
        cost: dict[str, Any] = {}
        audit_pool: list[tuple[str, str, int]] = []
        explanations: dict[tuple[str, int], str] = {}
        details = []
        for v in c.variants:
            cases = _cases(runs[v])
            run_tools = run_tool_names(runs[v])
            per_case = []
            for case in cases:
                tr = read_transcript(
                    runs[v] / "artifacts" / "transcripts" / f"query_{case['query_row']:03d}_s0.json"
                )
                expl = case["answer"]["explicacao"]
                chk = fidelity_checks(
                    expl,
                    tr,
                    known_components,
                    known_kpis,
                    c.value_rel_tolerance,
                    c.min_integer,
                    run_tools,
                )
                per_case.append(chk)
                details.append(
                    {"variant": v, "query_row": case["query_row"]}
                    | {k: " | ".join(x) if isinstance(x, list) else x for k, x in chk.items()}
                )
                audit_pool.append((system, v, int(case["query_row"])))
                explanations[(v, int(case["query_row"]))] = expl
            fidelity[v] = _fidelity_summary(per_case, nb, seed)
            cols = ("cost_usd", "latency_s", "output_tokens", "tool_calls", "turns")
            frame = {col: np.array([float(x[col]) for x in cases]) for col in cols}
            cost[v] = {
                "cost_usd": round(float(tables[v]["cost_usd"]), 4),
                "paid_calls": int(tables[v]["usage"]["paid_calls"]),
                "valid_rate": tables[v]["valid_rate"],
                "valid_after_retry": tables[v]["valid_after_retry"],
                **{
                    f"{col}_mean": round(float(frame[col].mean()), 4)
                    for col in ("cost_usd", "latency_s", "output_tokens", "tool_calls", "turns")
                },
                **{
                    f"{col}_sd": round(float(frame[col].std(ddof=1)), 4)
                    for col in ("cost_usd", "latency_s", "output_tokens", "tool_calls")
                },
                "evidence_tool_use": int(sum("get_evidence" in x["tools_used"] for x in cases)),
                "scan_tool_use": int(sum("find_anomalous_kpis" in x["tools_used"] for x in cases)),
                "cites": {
                    k: int(
                        sum(
                            any(i.startswith(k) for i in x["answer"]["evidencias_citadas"])
                            for x in cases
                        )
                    )
                    for k in ("T", "N", "C", "K")
                },
            }
        pl.DataFrame(details).write_csv(run.artifacts_dir / "fidelity_details.csv")

        replay: dict[str, Any] = {}
        for v, run_name in zip(c.variants, c.run_names, strict=True):
            path = tables_dir / f"agent_{system}_{run_name}{c.replay_suffix}_{c.model_slug}.json"
            if path.exists():
                rt = json.loads(path.read_text(encoding="utf-8"))
                replay[v] = replay_check(runs[v], config.paths.experiments_dir / rt["run_dir"], rt)

        audit = _audit(config, system, audit_pool, explanations, known_components, runs, nb, seed)

        table = {
            "system": system,
            "queries": len(rows),
            "runs": {v: tables[v]["run_dir"] for v in c.variants},
            "evidence_select_run": es_table["run_dir"],
            "summary": summary,
            "by_task": by_task,
            "comparisons": comparisons,
            "references": references,
            "localization": localization,
            "errors": error_breakdown(list(chains.values()), rows),
            "fidelity": fidelity,
            "cost": cost,
            "replay": replay,
            "audit": audit,
        }
        run.log_metrics(table)
    table["run_dir"] = run.dir.name
    (tables_dir / f"agent_eval_{system}.json").write_text(
        json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_markdown(tables_dir / f"agent_eval_{system}.md", table)
    return {
        "run_dir": str(run.dir),
        "primary": next(r for r in comparisons if r["role"] == "primary"),
        "replay_ok": all(r["ok"] for r in replay.values()) if replay else None,
        "audit": {k: audit[k] for k in ("claims", "classified") if k in audit},
    }


def _fidelity_summary(per_case: Sequence[Mapping[str, Any]], nb: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {"explanations": len(per_case)}
    for total, bad, name in (
        ("ids", "ids_not_issued", "ids"),
        ("components", "components_unknown", "components"),
        ("identifiers", "identifiers_unknown", "identifiers"),
        ("numbers", "numbers_ungrounded", "numbers"),
    ):
        num = np.array([len(x[bad]) for x in per_case], dtype=np.float64)
        den = np.array([x[total] for x in per_case], dtype=np.float64)
        out[name] = {
            "checked": int(den.sum()),
            "flagged": int(num.sum()),
            "explanations_flagged": int((num > 0).sum()),
            "flagged_rate": ratio_estimate(num, den, nb, seed).as_dict(),
        }
    checked = sum(x["numbers"] for x in per_case)
    control = sum(x["numbers_control_grounded"] for x in per_case)
    out["numbers"]["control_grounded_rate"] = round(control / checked, 4) if checked else None
    return out


def _audit(
    config: Config,
    system: str,
    pool: Sequence[tuple[str, str, int]],
    explanations: Mapping[tuple[str, int], str],
    known_components: set[str],
    runs: Mapping[str, Path],
    nb: int,
    seed: int,
) -> dict[str, Any]:
    """Writes this system's part of the blinded audit sheet the first time; afterwards only
    reads it. The sheet holds both systems, so the second system appends its items once."""
    c = config.agent_eval
    sheet, key = c.audit_dir / "auditoria_fidelidade.csv", c.audit_dir / "auditoria_chave.json"
    evidence_dir = config.paths.experiments_dir / "auditoria_fidelidade"
    mapping: dict[str, Any] = json.loads(key.read_text(encoding="utf-8")) if key.exists() else {}
    if not any(m["system"] == system for m in mapping.values()):
        c.audit_dir.mkdir(parents=True, exist_ok=True)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(c.audit_seed + (0 if system == "Telecom" else 1))
        picked = audit_sample(pool, c.audit_per_stratum, c.audit_seed)
        is_factual = _is_factual_factory(known_components)
        start = len(mapping)
        new_rows = []
        for i, (_, v, row) in enumerate(picked, start + 1):
            code = f"E{i:02d}"
            expl = explanations[(v, row)]
            sentences = factual_sentences(expl, is_factual)
            take = sorted(
                rng.choice(len(sentences), size=min(c.audit_claims, len(sentences)), replace=False)
            )
            claims = [sentences[j] for j in take]
            mapping[code] = {"system": system, "variant": v, "query_row": row}
            transcript = runs[v] / "artifacts" / "transcripts" / f"query_{row:03d}_s0.json"
            _write_audit_evidence(evidence_dir / f"{code}.md", code, expl, claims, transcript)
            for j, claim in enumerate(claims, 1):
                new_rows.append(
                    {
                        "explicacao": code,
                        "afirmacao": j,
                        "texto": claim,
                        "evidencia": str(evidence_dir / f"{code}.md"),
                        "classificacao": "",
                        "observacao": "",
                    }
                )
        exists = sheet.exists()
        with sheet.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(new_rows[0]))
            if not exists:
                writer.writeheader()
            writer.writerows(new_rows)
        key.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = summarize_audit(sheet, key, nb, seed)
    codes = {k for k, m in mapping.items() if m["system"] == system}
    rows = list(csv.DictReader(sheet.open(encoding="utf-8", newline="")))
    mine = [r for r in rows if r["explicacao"] in codes]
    summary["system_claims"] = len(mine)
    summary["system_classified"] = sum(bool(r["classificacao"].strip()) for r in mine)
    return summary


def _write_audit_evidence(
    path: Path, code: str, explanation: str, claims: Sequence[str], transcript: Path
) -> None:
    """Everything the auditor needs for one explanation: the text, the sampled claims and every
    tool output the agent saw (no variant name, no ground truth)."""
    data = json.loads(transcript.read_text(encoding="utf-8"))
    lines = [f"# {code}", "", "## Explicação completa", "", explanation, "", "## Afirmações", ""]
    lines += [f"{j}. {c}" for j, c in enumerate(claims, 1)]
    lines += ["", "## Saídas das ferramentas (o que o agente viu)", ""]
    calls = {}
    for m in data["messages"]:
        if m.get("role") == "assistant":
            for call in m.get("tool_calls") or []:
                calls[call["id"]] = call
        if m.get("role") == "tool":
            call = calls.get(m.get("tool_call_id"), {})
            try:
                body = json.dumps(json.loads(m["content"]), indent=1, ensure_ascii=False)
            except json.JSONDecodeError:
                body = m["content"]
            args = json.dumps(call.get("arguments", {}), ensure_ascii=False)
            lines += [f"### {call.get('name', '?')} {args}", "", "```json", body, "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- markdown
def _p(x: float | str) -> str:
    if isinstance(x, str):
        return x
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def _ci(d: Mapping[str, Any], digits: int = 2) -> str:
    return f"{d['value']:.{digits}f} [{d['ci_low']:.{digits}f}; {d['ci_high']:.{digits}f}]"


def write_markdown(path: Path, t: Mapping[str, Any]) -> None:
    s = t["summary"]
    lines = [
        f"# Avaliação do agente (E8): {t['system']}",
        "",
        f"{t['queries']} consultas de teste; nota pelo script oficial do OpenRCA; IC de 95% por "
        "bootstrap sobre consultas.",
        "",
        "| Cadeia | Nota parcial | Acerto estrito | Instante | Componente | Motivo |",
        "|---|---|---|---|---|---|",
    ]
    for name, x in s.items():
        f = x["fields"]
        lines.append(
            f"| {name} | {_ci(x['partial_score'])} | {_ci(x['strict_accuracy'])} | "
            + " | ".join(_ci(f[k]) for k in FIELDS)
            + " |"
        )
    lines += [
        "",
        "Campos: acerto por critério do gabarito (aproximado em consultas com duas falhas).",
        "",
        "## Comparações pareadas",
        "",
        "| Comparação | Papel | Diferença parcial | Veredito | Melhores / piores | "
        "McNemar b / c | p | p (Holm) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in [*t["comparisons"], *t["references"]]:
        lines.append(
            f"| {r['comparison']} | {r['role']} | {_ci(r['partial_difference'], 3)} | "
            f"{r['verdict']} | {r['queries_better']} / {r['queries_worse']} | "
            f"{r['mcnemar_only_first']} / {r['mcnemar_only_second']} | {_p(r['mcnemar_p'])} | "
            f"{_p(r.get('mcnemar_p_holm', '—'))} |"
        )
    loc = t["localization"]
    lines += [
        "",
        f"## Localização (consultas de uma falha que pedem componente: {loc['queries']})",
        "",
        "| Cadeia | HR@1 | Menos o pacote | Veredito | McNemar b / c / p |",
        "|---|---|---|---|---|",
        f"| pacote (C1) | {_ci(loc['pacote'])} | — | — | — |",
    ]
    for v in t["runs"]:
        x = loc[v]
        b, c, p = x["mcnemar"]
        lines.append(
            f"| {v} | {_ci(x)} | {_ci(x['minus_pacote'])} | {x['verdict']} | {b} / {c} / {_p(p)} |"
        )
    lines += [
        "",
        "## Fidelidade automática (itens sinalizados / verificados)",
        "",
        "| Variante | IDs não emitidos | Componentes desconhecidos | Identificadores desconhecidos "
        "| Números sem lastro literal |",
        "|---|---|---|---|---|",
    ]
    for v, f in t["fidelity"].items():
        cells = [
            f"{f[k]['flagged']} / {f[k]['checked']} ({f[k]['explanations_flagged']} expl.)"
            for k in ("ids", "components", "identifiers", "numbers")
        ]
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Custo",
        "",
        "| Variante | US$ total | US$/consulta | Latência/consulta (s) | Ferramentas/consulta | "
        "Usa get_evidence | Usa find_anomalous_kpis |",
        "|---|---|---|---|---|---|---|",
    ]
    for v, x in t["cost"].items():
        lines.append(
            f"| {v} | {x['cost_usd']:.2f} | {x['cost_usd_mean']:.3f} ± {x['cost_usd_sd']:.3f} | "
            f"{x['latency_s_mean']:.1f} ± {x['latency_s_sd']:.1f} | "
            f"{x['tool_calls_mean']:.1f} ± {x['tool_calls_sd']:.1f} | {x['evidence_tool_use']} | "
            f"{x['scan_tool_use']} |"
        )
    if t["replay"]:
        lines += ["", "## Reprodução a partir do cache", ""]
        for v, r in t["replay"].items():
            lines.append(
                f"- {v}: {r['replay_run']}, {r['paid_calls']} chamadas pagas, "
                f"{r['cache_hits']} do cache, previsões iguais: {r['same_predictions']}, "
                f"metrics.json igual: {r['same_metrics']}"
            )
    a = t["audit"]
    lines += [
        "",
        "## Auditoria manual",
        "",
        f"{a['system_classified']} de {a['system_claims']} afirmações deste sistema classificadas.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
