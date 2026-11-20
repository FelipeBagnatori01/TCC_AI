"""ADR 0023 (exploratory): can a cheaper model with the LSTM/XAI evidence match the E8 model?

The candidate model runs the same variants, tools, prompts and queries as the E7 runs; only
``configs/llm/<model>.yaml`` changes. Everything is scored by the official evaluator. Fixed
before the candidate runs:

* within the candidate, the E8 family (A3 vs A1 primary, McNemar with Holm);
* non-inferiority of each candidate variant to EVERY reference variant: the lower bound of the
  95% CI of the paired partial-score difference above ``-margin``. "Matches the reference" is
  claimed only when that holds against all reference variants (intersection-union: no
  multiplicity correction needed), and only for ``claim_variants``;
* agreement with the package answer (field by field, single-failure queries) and the share of
  onsets at the window start, the pilot failure of the cheaper models; both are descriptive.

The reference model's official reports are the ones written by the E8 ``evaluate-agent`` run
recorded in ``reports/tables/agent_eval_<system>.json`` (their scores must match that table).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from tcc_rca.config import Config
from tcc_rca.evaluation.agent_eval import (
    FIELDS,
    REFERENCES,
    _cases,
    _chain,
    _fidelity_summary,
    agent_table,
    error_breakdown,
    fidelity_checks,
    holm,
    mean_estimate,
    paired,
    ratio_estimate,
    read_transcript,
    reference_chains,
    replay_check,
    run_tool_names,
    score_agent_run,
    single_component_hits,
)
from tcc_rca.evaluation.bootstrap import Estimate
from tcc_rca.ingestion.cases import read_queries
from tcc_rca.ingestion.timeutil import utc_to_local_str
from tcc_rca.tracking import Run

# official answer keys -> Portuguese field names used in the tables
ANSWER_KEYS = {
    "root cause occurrence datetime": "instante",
    "root cause component": "componente",
    "root cause reason": "motivo",
}
LOCAL_FORMAT = "%Y-%m-%d %H:%M:%S"


def noninferiority(e: Estimate, margin: float) -> str:
    """'nao_inferior' when the CI lower bound of (candidate - reference) is above -margin,
    'inferior' when its upper bound is below -margin, else 'inconclusivo'."""
    if e.ci_low > -margin:
        return "nao_inferior"
    if e.ci_high < -margin:
        return "inferior"
    return "inconclusivo"


def matches_reference(verdicts: Sequence[str]) -> bool:
    """Intersection-union claim: non-inferior to every reference variant."""
    return bool(verdicts) and all(v == "nao_inferior" for v in verdicts)


def parse_causes(prediction: str) -> list[dict[str, str]]:
    """Causes of an official answer, in order: numbered (``{"1": {...}}``, the agent and
    multi-cause format) or a single unnumbered object (the package chain)."""
    data = json.loads(prediction)
    if not isinstance(data, dict):
        raise ValueError(f"not an answer object: {prediction!r}")
    if data and all(str(k).isdigit() for k in data):
        items = [data[k] for k in sorted(data, key=lambda k: int(k))]
    else:
        items = [data]
    return [{ANSWER_KEYS[k]: str(v) for k, v in item.items() if k in ANSWER_KEYS} for item in items]


def same_value(field: str, a: str, b: str) -> bool:
    """Datetimes agree within one minute (the official tolerance); names and reasons exactly."""
    if field == "instante":
        delta = datetime.strptime(a, LOCAL_FORMAT) - datetime.strptime(b, LOCAL_FORMAT)
        return abs(delta.total_seconds()) <= 60
    return a == b


def package_agreement(
    answers: Mapping[int, str],
    package: Mapping[int, str],
    rows: Sequence[int],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Per field, share of single-cause answers equal to the package chain's answer (queries
    where both answered that field)."""
    out: dict[str, Any] = {}
    for field in FIELDS:
        hits = []
        for r in rows:
            a, p = parse_causes(answers[r]), parse_causes(package[r])
            if len(a) == 1 and len(p) == 1 and field in a[0] and field in p[0]:
                hits.append(float(same_value(field, a[0][field], p[0][field])))
        arr = np.array(hits, dtype=np.float64)
        out[field] = (
            {**mean_estimate(arr, n_resamples, seed).as_dict(), "n": int(arr.size)}
            if arr.size
            else {"n": 0}
        )
    return out


def window_start_share(
    answers: Mapping[int, str],
    window_start_local: Mapping[int, str],
    rows: Sequence[int],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Share of answered onsets within one minute of the query window start (CI over queries)."""
    num, den = [], []
    for r in rows:
        onsets = [c["instante"] for c in parse_causes(answers[r]) if "instante" in c]
        num.append(float(sum(same_value("instante", t, window_start_local[r]) for t in onsets)))
        den.append(float(len(onsets)))
    e = ratio_estimate(np.array(num), np.array(den), n_resamples, seed)
    return {**e.as_dict(), "onsets": int(sum(den))}


def _answers(official_dir: Path) -> dict[int, str]:
    df = pl.read_csv(official_dir / "prediction.csv")
    return dict(zip(df["query_row"].to_list(), df["prediction"].to_list(), strict=True))


def run_model_compare(config: Config, tables_dir: Path) -> dict[str, Any]:
    c, m, system = config.agent_eval, config.model_compare, config.target_system
    nb, seed = c.bootstrap, config.experiment.seed
    if not set(m.claim_variants) <= set(c.variants):
        raise ValueError("model_compare.claim_variants must be agent_eval variants")
    e8 = json.loads((tables_dir / f"agent_eval_{system}.json").read_text(encoding="utf-8"))
    e8_official = config.paths.experiments_dir / e8["run_dir"] / "artifacts" / "official"
    query_csv = config.paths.data_root / system / "query.csv"
    ref, cand = m.reference_label, m.candidate_label

    with Run(config) as run:
        chains, _ = reference_chains(config, tables_dir)
        answers: dict[str, dict[int, str]] = {
            "pacote": _answers(Path(chains["pacote"].official_dir))
        }
        for v in c.variants:
            name = f"{ref} {v}"
            pred = pl.read_csv(e8_official / v / "prediction.csv")["query_row"].to_list()
            chains[name] = _chain(name, pred, e8_official / v)
            want = e8["summary"][v]["partial_score"]["value"]
            got = float(np.mean(list(chains[name].score.values())))
            if abs(got - want) > 1e-3:
                raise RuntimeError(f"{name}: official partial {got:.4f} != E8 table {want:.4f}")
            answers[name] = _answers(e8_official / v)

        tables: dict[str, dict[str, Any]] = {}
        runs: dict[str, Path] = {}
        for v, run_name in zip(c.variants, c.run_names, strict=True):
            name = f"{cand} {v}"
            t = agent_table(tables_dir, system, run_name, m.candidate_slug)
            if t["case_set"] != "test" or t["budget_stopped"]:
                raise ValueError(f"{run_name} ({m.candidate_slug}): not a complete test run")
            tables[v], runs[v] = t, config.paths.experiments_dir / t["run_dir"]
            out_dir = run.artifacts_dir / "official" / name.replace(" ", "_")
            chains[name] = score_agent_run(
                name, runs[v], query_csv, config.paths.openrca_repo, out_dir
            )
            answers[name] = _answers(out_dir)
        rows = chains["A0"].rows
        for name, chain in chains.items():
            if chain.rows != rows:
                raise RuntimeError(f"{name} answers other queries than A0")

        summary = {
            name: {
                "partial_score": mean_estimate(ch.array(rows), nb, seed).as_dict(),
                "strict_accuracy": mean_estimate(ch.array(rows, "strict"), nb, seed).as_dict(),
            }
            for name, ch in chains.items()
        }

        # within the candidate: the E8 family (primary A3 vs A1), Holm on McNemar
        within = [
            paired(chains[f"{cand} {a}"], chains[f"{cand} {b}"], rows, nb, seed)
            for a, b in c.variant_comparisons
        ]
        primary = f"{cand} {c.primary_comparison[0]} - {cand} {c.primary_comparison[1]}"
        for row, p_adj in zip(within, holm([r["mcnemar_p"] for r in within]), strict=True):
            row["mcnemar_p_holm"] = round(p_adj, 6)
            row["role"] = "primary" if row["comparison"] == primary else "secondary"

        # candidate vs. reference model: non-inferiority with the fixed margin
        versus: list[dict[str, Any]] = []
        matches: dict[str, Any] = {}
        for v in c.variants:
            verdicts = []
            for w in c.variants:
                row = paired(chains[f"{cand} {v}"], chains[f"{ref} {w}"], rows, nb, seed)
                d = row["partial_difference"]
                e = Estimate(d["value"], d["ci_low"], d["ci_high"], d["n_valid_resamples"])
                row["noninferiority"] = noninferiority(e, m.margin)
                verdicts.append(row["noninferiority"])
                versus.append(row)
            matches[v] = {
                "claim": v in m.claim_variants,
                "matches_reference": matches_reference(verdicts),
            }
        references = [
            paired(chains[f"{cand} {v}"], chains[r], rows, nb, seed) | {"role": "reference"}
            for v in c.variants
            for r in REFERENCES
        ]

        # localization: single-failure component hit vs. the package C1
        pkg_hits = single_component_hits(chains["pacote"], rows)
        loc_rows = sorted(pkg_hits)
        pkg_arr = np.array([float(pkg_hits[r]) for r in loc_rows])
        localization: dict[str, Any] = {
            "queries": len(loc_rows),
            "pacote": mean_estimate(pkg_arr, nb, seed).as_dict(),
        }
        for model in (ref, cand):
            for v in c.variants:
                hits = single_component_hits(chains[f"{model} {v}"], rows)
                arr = np.array([float(hits[r]) for r in loc_rows])
                localization[f"{model} {v}"] = mean_estimate(arr, nb, seed).as_dict()

        # descriptive: agreement with the package answer and onsets at the window start
        starts = {
            q.row: utc_to_local_str(q.window_start_utc)
            for q in read_queries(query_csv)
            if q.row in set(rows)
        }
        agents = [f"{model} {v}" for model in (ref, cand) for v in c.variants]
        agreement = {
            name: package_agreement(answers[name], answers["pacote"], rows, nb, seed)
            for name in agents
        }
        window_start = {
            name: window_start_share(answers[name], starts, rows, nb, seed)
            for name in [*agents, "pacote"]
        }

        # candidate fidelity (gold-free) and cost; the reference ones are in the E8 table
        catalog = pl.read_csv(tables_dir / f"feature_catalog_{system}.csv")
        from tcc_rca.agent.tasks import load_candidates

        known_components = set(catalog["component"].to_list()) | set(
            load_candidates(config.agent.prompts_dir, system).components
        )
        known_kpis = sorted(set(catalog["kpi"].to_list()))
        fidelity: dict[str, Any] = {}
        cost: dict[str, Any] = {}
        for v in c.variants:
            cases = _cases(runs[v])
            tools = run_tool_names(runs[v])
            per_case = [
                fidelity_checks(
                    case["answer"]["explicacao"],
                    read_transcript(
                        runs[v]
                        / "artifacts"
                        / "transcripts"
                        / f"query_{case['query_row']:03d}_s0.json"
                    ),
                    known_components,
                    known_kpis,
                    c.value_rel_tolerance,
                    c.min_integer,
                    tools,
                )
                for case in cases
            ]
            fidelity[f"{cand} {v}"] = _fidelity_summary(per_case, nb, seed)
            per_query = np.array([float(x["cost_usd"]) for x in cases])
            cost[f"{cand} {v}"] = {
                "cost_usd": round(float(tables[v]["cost_usd"]), 4),
                "cost_usd_mean": round(float(per_query.mean()), 5),
                "cost_usd_sd": round(float(per_query.std(ddof=1)), 5),
                "paid_calls": int(tables[v]["usage"]["paid_calls"]),
                "valid_rate": tables[v]["valid_rate"],
                "valid_after_retry": tables[v]["valid_after_retry"],
                "tool_calls_mean": round(float(np.mean([x["tool_calls"] for x in cases])), 2),
                "evidence_tool_use": int(sum("get_evidence" in x["tools_used"] for x in cases)),
            }
            ref_cost = e8["cost"][v]
            cost[f"{ref} {v}"] = {
                k: ref_cost[k] for k in ("cost_usd", "cost_usd_mean", "cost_usd_sd", "valid_rate")
            }

        replay: dict[str, Any] = {}
        for v, run_name in zip(c.variants, c.run_names, strict=True):
            path = (
                tables_dir / f"agent_{system}_{run_name}{c.replay_suffix}_{m.candidate_slug}.json"
            )
            if path.exists():
                rt = json.loads(path.read_text(encoding="utf-8"))
                replay[v] = replay_check(runs[v], config.paths.experiments_dir / rt["run_dir"], rt)

        table: dict[str, Any] = {
            "system": system,
            "queries": len(rows),
            "reference": {"label": ref, "model": c.model_slug, "e8_run": e8["run_dir"]},
            "candidate": {
                "label": cand,
                "model": m.candidate_slug,
                "runs": {v: tables[v]["run_dir"] for v in c.variants},
            },
            "margin": m.margin,
            "summary": summary,
            "within_candidate": within,
            "versus_reference": versus,
            "matches_reference": matches,
            "references": references,
            "localization": localization,
            "package_agreement": agreement,
            "window_start": window_start,
            "errors": error_breakdown(list(chains.values()), rows),
            "fidelity": fidelity,
            "cost": cost,
            "replay": replay,
        }
        run.log_metrics(table)
    table["run_dir"] = run.dir.name
    stem = f"model_compare_{system}_{m.candidate_slug}"
    (tables_dir / f"{stem}.json").write_text(
        json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_markdown(tables_dir / f"{stem}.md", table)
    return {
        "run_dir": str(run.dir),
        "primary": next(r for r in within if r["role"] == "primary"),
        "matches_reference": matches,
    }


# --------------------------------------------------------------------------- markdown
def _ci(d: Mapping[str, Any]) -> str:
    return f"{d['value']:.2f} [{d['ci_low']:.2f}; {d['ci_high']:.2f}]"


def _p(x: float) -> str:
    return "< 0,001" if x < 0.001 else f"{x:.3f}".replace(".", ",")


def write_markdown(path: Path, t: Mapping[str, Any]) -> None:
    ref, cand = t["reference"]["label"], t["candidate"]["label"]
    lines = [
        f"# {cand} vs. {ref} no agente — {t['system']} ({t['queries']} consultas; ADR 0023)",
        "",
        f"Exploratório. Modelos: `{t['reference']['model']}` e `{t['candidate']['model']}`; "
        f"margem de não inferioridade {t['margin']:.2f} na nota parcial oficial.",
        "",
        "## Nota oficial (IC 95% por bootstrap sobre consultas)",
        "",
        "| Cadeia | Nota parcial | Acerto estrito |",
        "|---|---|---|",
    ]
    for name, s in t["summary"].items():
        lines.append(f"| {name} | {_ci(s['partial_score'])} | {_ci(s['strict_accuracy'])} |")
    lines += [
        "",
        f"## Dentro do {cand} (família da E8; Holm no McNemar)",
        "",
        "| Comparação | Diferença | Veredito | McNemar (só 1º / só 2º) | p Holm | Papel |",
        "|---|---|---|---|---|---|",
    ]
    for r in t["within_candidate"]:
        lines.append(
            f"| {r['comparison']} | {_ci(r['partial_difference'])} | {r['verdict']} | "
            f"{r['mcnemar_only_first']}/{r['mcnemar_only_second']} | "
            f"{_p(r['mcnemar_p_holm'])} | {r['role']} |"
        )
    lines += [
        "",
        f"## {cand} vs. {ref} (não inferioridade, margem {t['margin']:.2f})",
        "",
        "| Comparação | Diferença | Não inferioridade | Veredito usual | McNemar | p |",
        "|---|---|---|---|---|---|",
    ]
    for r in t["versus_reference"]:
        lines.append(
            f"| {r['comparison']} | {_ci(r['partial_difference'])} | {r['noninferiority']} | "
            f"{r['verdict']} | {r['mcnemar_only_first']}/{r['mcnemar_only_second']} | "
            f"{_p(r['mcnemar_p'])} |"
        )
    lines += ["", f"Iguala o {ref} (não inferior a todas as variantes dele):", ""]
    for v, x in t["matches_reference"].items():
        tag = "afirmação registrada" if x["claim"] else "só contexto"
        lines.append(f"- {cand} {v}: {'sim' if x['matches_reference'] else 'não'} ({tag})")
    lines += [
        "",
        "## Contra as referências sem LLM",
        "",
        "| Comparação | Diferença | Veredito |",
        "|---|---|---|",
    ]
    for r in t["references"]:
        lines.append(f"| {r['comparison']} | {_ci(r['partial_difference'])} | {r['verdict']} |")
    loc = t["localization"]
    lines += [
        "",
        f"## Localização (consultas de uma falha que pedem componente: {loc['queries']})",
        "",
        "| Cadeia | Acerto do componente |",
        "|---|---|",
    ]
    for name, d in loc.items():
        if name != "queries":
            lines.append(f"| {name} | {_ci(d)} |")
    lines += [
        "",
        "## Concordância com a resposta do pacote e instante no início da janela (descritivo)",
        "",
        "| Cadeia | Instante = pacote | Componente = pacote | Motivo = pacote | "
        "Instante no início da janela |",
        "|---|---|---|---|---|",
    ]
    for name, w in t["window_start"].items():
        a = t["package_agreement"].get(name)
        cells = (
            [(_ci(a[f]) + f" (n={a[f]['n']})") if a[f]["n"] else "—" for f in FIELDS]
            if a
            else ["—"] * 3
        )
        lines.append(f"| {name} | {' | '.join(cells)} | {_ci(w)} (n={w['onsets']}) |")
    lines += [
        "",
        "## Custo",
        "",
        "| Cadeia | Total (US$) | Por consulta (US$, média ± dp) | Respostas válidas |",
        "|---|---|---|---|",
    ]
    for name, x in t["cost"].items():
        lines.append(
            f"| {name} | {x['cost_usd']:.2f} | {x['cost_usd_mean']:.3f} ± {x['cost_usd_sd']:.3f} | "
            f"{x['valid_rate']:.2f} |"
        )
    lines += [
        "",
        f"## Fidelidade automática ({cand}; sem gabarito)",
        "",
        "| Cadeia | IDs não emitidos | Componentes desconhecidos | KPIs desconhecidos | "
        "Números sem correspondência | Controle x1,37 casado |",
        "|---|---|---|---|---|---|",
    ]
    for name, f in t["fidelity"].items():
        cells = [
            f"{f[k]['flagged']}/{f[k]['checked']}"
            for k in ("ids", "components", "identifiers", "numbers")
        ]
        ctrl = f["numbers"]["control_grounded_rate"]
        lines.append(f"| {name} | {' | '.join(cells)} | {'—' if ctrl is None else f'{ctrl:.2f}'} |")
    if t["replay"]:
        lines += ["", "## Reprodução a partir do cache", ""]
        for v, r in t["replay"].items():
            lines.append(f"- {cand} {v}: {'ok' if r['ok'] else 'FALHOU'} ({r['paid_calls']} pagas)")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
