"""The agent's fixed, read-only tools (ADR 0017).

Every tool returns a compact JSON summary, never whole raw series, tagged with an id ``T<n>``
the agent can cite. Time arguments are UTC+8 strings (``YYYY-MM-DD HH:MM:SS``) that must fall on
the query's day. Deviations are measured against a causal local reference: the 60 minutes before
the requested start on the same day, or, when the day has too little data before it (Telecom
days start at 00:00), the 60 minutes after the requested end; the summary says which one.

Variants (same tools for telemetry in all of them):

* A1: telemetry tools only;
* A2: + ``get_evidence`` with the LSTM detector score/alarm and the native ranking (``N#``);
* A3: A2 + the SHAP/LIME components (``C#``) and KPIs (``K#``). The suggested category is
  removed in every variant: E6 showed it does not beat the training prior (threat 13).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tcc_rca.agent.store import TelemetryStore
from tcc_rca.agent.tasks import TaskSpec
from tcc_rca.ingestion.timeutil import parse_local, utc_to_local_str
from tcc_rca.llm.types import ToolSpec

FloatArray = npt.NDArray[np.float64]
Variant = Literal["A1", "A2", "A3"]
REFERENCE_MINUTES = 60
MIN_REFERENCE_POINTS = 20
DEVIATION_THRESHOLD = 3.0
LIST_KPIS_MAX = 60  # above this, list_kpis without a category returns counts per category
DEVIATION_CAP = 50.0  # same cap as the causal local z of E3/E5 (tasks.precursor.local_cap)


class ToolError(ValueError):
    """A recoverable tool-usage error, returned to the model as an error result."""


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _NoArgs(_Args):
    pass


class _ComponentArgs(_Args):
    component: str = Field(description="nome do componente, por exemplo os_021 ou Tomcat01")


class _ListKpisArgs(_ComponentArgs):
    category: str | None = Field(
        default=None, description="restringe a uma categoria de KPI (opcional; null = todas)"
    )


class _RangeArgs(_Args):
    start: str = Field(description="início, YYYY-MM-DD HH:MM:SS em UTC+8")
    end: str = Field(description="fim, YYYY-MM-DD HH:MM:SS em UTC+8")


class _KpiArgs(_RangeArgs):
    component: str = Field(description="nome do componente")
    kpi: str = Field(description="nome do KPI, como listado por list_kpis")


class _ComponentRangeArgs(_RangeArgs):
    component: str = Field(description="nome do componente")


class _AnomalyArgs(_RangeArgs):
    component: str | None = Field(
        default=None, description="restringe a um componente (opcional; null = todos)"
    )


def _num(x: float) -> float | None:
    return None if x is None or not np.isfinite(x) else float(f"{x:.6g}")


def _local(t: datetime) -> str:
    return utc_to_local_str(t)


def _schema(model: type[_Args]) -> dict[str, Any]:
    """Strict-mode JSON Schema (both providers): every property required, no extras, no
    defaults; optional arguments are nullable instead."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    props = schema.setdefault("properties", {})
    for prop in props.values():
        prop.pop("title", None)
        prop.pop("default", None)
    schema["required"] = list(props)
    schema["additionalProperties"] = False
    return schema


def evidence_view(package: dict[str, Any], variant: Variant) -> dict[str, Any]:
    """What ``get_evidence`` shows in each variant (A1 has no such tool)."""
    if variant == "A1":
        raise ValueError("variant A1 has no evidence tool")
    keep = [
        "schema_version",
        "system",
        "query_window_local",
        "detector",
        "window_local",
        "detector_score",
        "detector_threshold",
        "alarm",
        "native_ranking",
    ]
    if variant == "A3":
        keep += ["explained_model", "category_weights", "components", "kpis"]
    return {k: package[k] for k in keep if k in package}


def evidence_ids(view: dict[str, Any]) -> set[str]:
    return {
        item["id"] for key in ("components", "kpis", "native_ranking") for item in view.get(key, [])
    }


class Toolbox:
    """Tool specs + dispatcher for ONE case: knows the query day, issues ``T#`` ids and counts
    calls against the per-case budget."""

    def __init__(
        self,
        store: TelemetryStore,
        task: TaskSpec,
        variant: Variant,
        max_tool_calls: int,
        anomaly_top_k: int,
    ) -> None:
        self.store, self.task, self.variant = store, task, variant
        self.max_tool_calls, self.anomaly_top_k = max_tool_calls, anomaly_top_k
        self.calls = 0
        self.known_ids: set[str] = set()
        self.log: list[dict[str, Any]] = []
        self._tools: dict[str, tuple[type[_Args], Callable[[Any], dict[str, Any]], str]] = {
            "list_components": (
                _NoArgs,
                self._list_components,
                "Lista os componentes com telemetria e quantos KPIs cada um tem.",
            ),
            "list_kpis": (
                _ListKpisArgs,
                self._list_kpis,
                "Lista os KPIs de um componente, com categoria, unidade, descrição e origem "
                f"(métrica, trace ou log). Componentes com mais de {LIST_KPIS_MAX} KPIs: sem "
                "categoria, retorna só a contagem por categoria; peça uma categoria para ver os "
                "nomes.",
            ),
            "query_kpi": (
                _KpiArgs,
                self._query_kpi,
                "Resumo estatístico de um KPI de um componente num intervalo do dia da "
                "consulta: n, mínimo, máximo, média, mediana, primeiro e último valor, medianas "
                "a cada 5 minutos, referência local (60 min antes do início, ou depois do fim "
                "quando não há dados antes), maior desvio em relação a ela com o instante do pico "
                "e o primeiro instante em que o desvio passa de 3 (desvio = (valor - mediana) / "
                "escala robusta da referência, limitado a 50).",
            ),
            "find_anomalous_kpis": (
                _AnomalyArgs,
                self._find_anomalous,
                "Ordena os KPIs (de todos os componentes, ou de um) pelo maior desvio no "
                "intervalo em relação à referência local, como em query_kpi (desvio limitado a "
                "50; empates no limite vêm pelo primeiro instante de desvio). Retorna os "
                f"{anomaly_top_k} primeiros com instante do pico e primeiro instante de desvio.",
            ),
            "get_trace_summary": (
                _ComponentRangeArgs,
                self._trace_summary,
                "Resumo dos KPIs de trace de um componente (chamadas, erros, latência) no "
                "intervalo, com o desvio de cada um em relação à referência local.",
            ),
            "get_dependencies": (
                _ComponentArgs,
                self._dependencies,
                "Arestas de dependência do componente: quem ele chama, quem o chama (traces) e "
                "onde está implantado.",
            ),
        }
        if variant != "A1":
            what = (
                "pontuação e alarme do detector LSTM e o ranqueamento nativo de componentes (N#)"
                if variant == "A2"
                else "pontuação e alarme do detector LSTM, ranqueamento nativo (N#), componentes "
                "(C#) e KPIs (K#) mais atribuídos por SHAP e LIME, com valor bruto no pico"
            )
            self._tools["get_evidence"] = (
                _NoArgs,
                self._get_evidence,
                f"Pacote de evidências do modelo para esta consulta: {what}.",
            )

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(
            ToolSpec(name=name, description=desc, input_schema=_schema(model))
            for name, (model, _, desc) in self._tools.items()
        )

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Run one tool; returns (JSON content, is_error). Never raises for model mistakes."""
        self.calls += 1
        entry: dict[str, Any] = {"tool": name, "arguments": arguments}
        result: dict[str, Any]
        error: bool
        if self.calls > self.max_tool_calls:
            result, error = (
                {"erro": "limite de chamadas de ferramenta atingido; responda agora"},
                True,
            )
        elif name not in self._tools:
            result, error = {"erro": f"ferramenta desconhecida: {name}"}, True
        else:
            model, func, _ = self._tools[name]
            try:
                result, error = func(model.model_validate(arguments)), False
            except ValidationError as exc:
                msgs = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
                result, error = {"erro": "argumentos inválidos", "detalhes": msgs}, True
            except ToolError as exc:
                result, error = {"erro": str(exc)}, True
        tool_id = f"T{self.calls}"
        if not error:
            self.known_ids.add(tool_id)
        payload = {"id": tool_id, **result}
        entry.update({"id": tool_id, "is_error": error})
        self.log.append(entry)
        return json.dumps(payload, ensure_ascii=False, sort_keys=False), error

    # ------------------------------------------------------------------ argument helpers
    def _component(self, component: str) -> str:
        if component not in self.store.components:
            raise ToolError(f"componente desconhecido: {component}")
        return component

    def _range(self, start: str, end: str) -> tuple[datetime, datetime]:
        try:
            t0, t1 = parse_local(start), parse_local(end)
        except ValueError:
            raise ToolError("use o formato YYYY-MM-DD HH:MM:SS (UTC+8)") from None
        day = self.task.window_local[0][:10]
        if _local(t0)[:10] != day or _local(t1)[:10] != day:
            raise ToolError(f"o intervalo deve ficar no dia da consulta ({day}, UTC+8)")
        if t1 < t0:
            raise ToolError("o fim vem antes do início")
        return t0, t1

    def _reference(self, t0: datetime, t1: datetime) -> tuple[pl.DataFrame, str]:
        frame = self.store.day(self.task.day)
        before = frame.filter(
            (pl.col("time_utc") >= t0 - timedelta(minutes=REFERENCE_MINUTES))
            & (pl.col("time_utc") < t0)
        )
        if before.height >= MIN_REFERENCE_POINTS:
            return before, f"{REFERENCE_MINUTES} min antes do início"
        after = frame.filter(
            (pl.col("time_utc") > t1)
            & (pl.col("time_utc") <= t1 + timedelta(minutes=REFERENCE_MINUTES))
        )
        return after, f"{REFERENCE_MINUTES} min depois do fim (não há dados suficientes antes)"

    # ------------------------------------------------------------------------ the tools
    def _list_components(self, _: _NoArgs) -> dict[str, Any]:
        counts = self.store.catalog.group_by("component").len().sort("component")
        return {"componentes": [{"componente": c, "kpis": n} for c, n in counts.iter_rows()]}

    def _list_kpis(self, args: _ListKpisArgs) -> dict[str, Any]:
        comp = self._component(args.component)
        rows = self.store.catalog.filter(pl.col("component") == comp).sort("kpi")
        if args.category is not None:
            rows = rows.filter(pl.col("category") == args.category)
            if rows.height == 0:
                raise ToolError(f"{comp} não tem KPIs na categoria '{args.category}'")
        elif rows.height > LIST_KPIS_MAX:
            counts = rows.group_by("category").len().sort("category")
            return {
                "componente": comp,
                "total_kpis": rows.height,
                "kpis_por_categoria": dict(counts.iter_rows()),
                "aviso": "muitos KPIs: chame list_kpis de novo com uma categoria",
            }
        if rows.height > LIST_KPIS_MAX:  # long category list: names and units only
            return {
                "componente": comp,
                "aviso": "lista longa: descrições omitidas (query_kpi mostra categoria e unidade)",
                "kpis": [
                    {"kpi": r["kpi"], "unidade": r["unit"]} for r in rows.iter_rows(named=True)
                ],
            }
        return {
            "componente": comp,
            "kpis": [
                {
                    "kpi": r["kpi"],
                    "categoria": r["category"],
                    "unidade": r["unit"],
                    "descricao": r["description"],
                    "origem": r["source"],
                }
                for r in rows.iter_rows(named=True)
            ],
        }

    def _deviations(
        self, features: list[str], t0: datetime, t1: datetime
    ) -> tuple[list[dict[str, Any]], str]:
        frame = self.store.day(self.task.day)
        cols = [f for f in features if f in frame.columns]
        window = frame.filter((pl.col("time_utc") >= t0) & (pl.col("time_utc") <= t1))
        ref, ref_kind = self._reference(t0, t1)
        if window.height == 0 or not cols:
            return [], ref_kind
        x = window.select(cols).cast(pl.Float64).fill_null(np.nan).to_numpy()
        r = ref.select(cols).cast(pl.Float64).fill_null(np.nan).to_numpy()
        times = window["time_utc"].to_list()
        out = []
        with np.errstate(all="ignore"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(r, axis=0) if r.shape[0] else np.full(len(cols), np.nan)
                q75, q25 = (
                    np.nanpercentile(r, [75, 25], axis=0)
                    if r.shape[0]
                    else (np.full(len(cols), np.nan), np.full(len(cols), np.nan))
                )
            scale = np.maximum((q75 - q25) / 1.349, np.maximum(0.01 * np.abs(med), 1e-6))
            dev = np.clip((x - med) / scale, -DEVIATION_CAP, DEVIATION_CAP)
        for j, feature in enumerate(cols):
            d = dev[:, j]
            if not np.isfinite(d).any():
                peak = first = None
                max_dev = float("nan")
            else:
                k = int(np.nanargmax(np.abs(d)))
                max_dev, peak = float(d[k]), k
                over = np.flatnonzero(np.abs(np.nan_to_num(d)) > DEVIATION_THRESHOLD)
                first = int(over[0]) if over.size else None
            out.append(
                {
                    "feature": feature,
                    "desvio_maximo": _num(max_dev),
                    "instante_pico": _local(times[peak]) if peak is not None else None,
                    "valor_pico": _num(float(x[peak, j])) if peak is not None else None,
                    "mediana_referencia": _num(float(med[j])),
                    "primeiro_desvio_acima_de_3": _local(times[first])
                    if first is not None
                    else None,
                }
            )
        return out, ref_kind

    def _query_kpi(self, args: _KpiArgs) -> dict[str, Any]:
        comp = self._component(args.component)
        feature = f"{comp}::{args.kpi}"
        if feature not in self.store.features:
            raise ToolError(f"KPI desconhecido para {comp}: {args.kpi} (use list_kpis)")
        t0, t1 = self._range(args.start, args.end)
        times, values = self.store.series(self.task.day, feature, t0, t1)
        info = self.store.catalog.filter(pl.col("feature") == feature).row(0, named=True)
        dev, ref_kind = self._deviations([feature], t0, t1)
        finite = values[np.isfinite(values)]
        per5 = []
        for i in range(0, len(times), 5):
            chunk = values[i : i + 5]
            chunk = chunk[np.isfinite(chunk)]
            per5.append(
                {
                    "inicio": _local(times[i]),
                    "mediana": _num(float(np.median(chunk))) if chunk.size else None,
                }
            )
        summary: dict[str, Any] = {
            "feature": feature,
            "categoria": info["category"],
            "unidade": info["unit"],
            "intervalo": [args.start, args.end],
            "n": int(finite.size),
            "ausentes_pct": _num(100.0 * (1 - finite.size / max(1, values.size))),
        }
        if finite.size:
            summary.update(
                {
                    "minimo": _num(float(finite.min())),
                    "maximo": _num(float(finite.max())),
                    "media": _num(float(finite.mean())),
                    "mediana": _num(float(np.median(finite))),
                    "primeiro": _num(float(finite[0])),
                    "ultimo": _num(float(finite[-1])),
                }
            )
        summary["medianas_5min"] = per5
        summary["referencia"] = ref_kind
        if dev:
            summary.update({k: v for k, v in dev[0].items() if k != "feature"})
        return summary

    def _find_anomalous(self, args: _AnomalyArgs) -> dict[str, Any]:
        t0, t1 = self._range(args.start, args.end)
        cat = self.store.catalog
        if args.component is not None:
            cat = cat.filter(pl.col("component") == self._component(args.component))
        rows, ref_kind = self._deviations(cat["feature"].to_list(), t0, t1)
        categories = dict(zip(cat["feature"].to_list(), cat["category"].to_list(), strict=True))
        # saturated deviations tie at the cap: the one that deviated first comes first
        ranked = sorted(
            (r for r in rows if r["desvio_maximo"] is not None),
            key=lambda r: (
                -abs(r["desvio_maximo"]),
                r["primeiro_desvio_acima_de_3"] or "9999",
                r["feature"],
            ),
        )[: self.anomaly_top_k]
        return {
            "intervalo": [args.start, args.end],
            "referencia": ref_kind,
            "kpis": [{**r, "categoria": categories[r["feature"]]} for r in ranked],
        }

    def _trace_summary(self, args: _ComponentRangeArgs) -> dict[str, Any]:
        comp = self._component(args.component)
        t0, t1 = self._range(args.start, args.end)
        feats = self.store.catalog.filter(
            (pl.col("component") == comp) & (pl.col("source") == "trace")
        )["feature"].to_list()
        if not feats:
            return {"componente": comp, "aviso": "componente sem KPIs de trace", "kpis": []}
        rows, ref_kind = self._deviations(feats, t0, t1)
        return {
            "componente": comp,
            "intervalo": [args.start, args.end],
            "referencia": ref_kind,
            "kpis": rows,
        }

    def _dependencies(self, args: _ComponentArgs) -> dict[str, Any]:
        comp = self._component(args.component)
        e = self.store.edges

        def rows(frame: pl.DataFrame, other: str) -> list[dict[str, Any]]:
            return [
                {"componente": r[other], "relacao": r["relation"], "peso": r["weight"]}
                for r in frame.sort(other).iter_rows(named=True)
            ]

        return {
            "componente": comp,
            "saida": rows(e.filter(pl.col("src") == comp), "dst"),
            "entrada": rows(e.filter(pl.col("dst") == comp), "src"),
        }

    def _get_evidence(self, _: _NoArgs) -> dict[str, Any]:
        package = self.store.evidence(self.task.query_row)
        if package is None:
            raise ToolError("não há pacote de evidências para esta consulta")
        view = evidence_view(package, self.variant)
        self.known_ids |= evidence_ids(view)
        return {"pacote": view}
