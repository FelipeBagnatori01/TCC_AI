"""E7 run: one variant of the agent over a case set, tracked, cached and budget-capped.

Outputs (``experiments/<run>/``):

* ``metrics.json``: reproducible counts (status per case, turns, tool calls, retries);
* ``artifacts/llm_calls.jsonl``: every model call, request and response (R6);
* ``artifacts/cases.jsonl``: one summary per case and repetition, with tokens and cost;
* ``artifacts/transcripts/``: full conversation per case and repetition;
* ``artifacts/predictions_s<k>.csv``: official-format predictions of repetition k (``row_id``
  = row of ``query.csv``), scored by the official OpenRCA script in E8;
* ``reports/tables/agent_<system>_<name>_<model>.{json,md}``: validity, tokens and cost summary.

Cost and latency depend on the cache state, so they stay out of ``metrics.json``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tcc_rca.agent.fake import FakeAgentProvider
from tcc_rca.agent.prompts import Prompts
from tcc_rca.agent.runner import CaseResult, run_case
from tcc_rca.agent.store import TelemetryStore
from tcc_rca.agent.tasks import TaskSpec, load_candidates, task_spec
from tcc_rca.agent.tools import Toolbox
from tcc_rca.config import Config
from tcc_rca.explain.evidence import PILOT_EVIDENCE_SUFFIX
from tcc_rca.ingestion.cases import read_queries
from tcc_rca.llm.client import BudgetExceededError, LLMClient
from tcc_rca.llm.providers import register_provider
from tcc_rca.tracking import Run

register_provider("fake_agent", FakeAgentProvider)


def select_cases(config: Config, store: TelemetryStore) -> list[TaskSpec]:
    """Test set = queries with an E6 package; pilot = queries before the first test day (they
    have a package only in the in-sample pilot evidence dir, used to check A2/A3 mechanics)."""
    system = config.target_system
    tasks = [
        task_spec(system, q.row, q.task_index, q.instruction, q.window_start_utc, q.window_end_utc)
        for q in read_queries(config.paths.data_root / system / "query.csv")
    ]
    in_sample = config.agent.evidence_name.endswith(PILOT_EVIDENCE_SUFFIX)
    if config.agent.case_set == "test":
        if in_sample:
            raise ValueError(
                "the test set needs the E6 test packages, not the in-sample pilot ones"
            )
        chosen = [t for t in tasks if store.evidence(t.query_row) is not None]
    else:
        first = config.tasks.first_test_day
        excluded = set(config.tasks.excluded_days)
        chosen = [t for t in tasks if t.day < first and t.day not in excluded]
        missing = [t.query_row for t in chosen if store.evidence(t.query_row) is None]
        if config.agent.variant != "A1" and (not in_sample or missing):
            raise ValueError(
                "pilot A2/A3 needs the in-sample pilot packages for every pilot query "
                f"(tcc-rca pilot-evidence; evidence_name=<name>{PILOT_EVIDENCE_SUFFIX}); "
                f"missing rows: {missing}"
            )
    chosen.sort(key=lambda t: t.query_row)
    return chosen[: config.agent.limit] if config.agent.limit else chosen


def _summarize(results: list[CaseResult], budget_stopped: bool) -> dict[str, Any]:
    n = len(results)
    status: dict[str, int] = {}
    for r in results:
        status[r.status] = status.get(r.status, 0) + 1
    valid = status.get("valid", 0)
    return {
        "case_runs": n,
        "status": dict(sorted(status.items())),
        "valid_rate": round(valid / n, 4) if n else None,
        "valid_after_retry": sum(1 for r in results if r.status == "valid" and r.retries > 0),
        "mean_turns": round(sum(r.turns for r in results) / n, 3) if n else None,
        "mean_tool_calls": round(sum(r.tool_calls for r in results) / n, 3) if n else None,
        "budget_stopped": budget_stopped,
    }


def _cost(results: list[CaseResult], client: LLMClient) -> dict[str, Any]:
    n = len(results)
    total = sum(r.cost_usd for r in results)
    return {
        "provider": client.config.provider,
        "model": client.config.model,
        "effort": client.config.effort,
        "budget_usd": client.config.budget_usd,
        "cost_usd": round(total, 6),
        "cost_per_case_run_usd": round(total / n, 6) if n else None,
        "max_case_cost_usd": round(max((r.cost_usd for r in results), default=0.0), 6),
        "usage": client.usage.as_dict(),
    }


def run_agent(config: Config, tables_dir: Path) -> dict[str, Any]:
    system, c = config.target_system, config.agent
    store = TelemetryStore(config.paths.processed_dir, tables_dir, system, c.evidence_name)
    candidates = load_candidates(c.prompts_dir, system)
    prompts = Prompts(c.prompts_dir)
    cases = select_cases(config, store)
    if not cases:
        raise ValueError(f"no {c.case_set} cases for {system}")
    results: list[CaseResult] = []
    budget_stopped = False
    with Run(config) as run:
        client = LLMClient(config.llm, call_log=run.artifacts_dir / "llm_calls.jsonl")
        transcripts = run.artifacts_dir / "transcripts"
        transcripts.mkdir(exist_ok=True)
        (run.artifacts_dir / "prompts.json").write_text(
            json.dumps(prompts.fingerprint(), indent=2) + "\n", encoding="utf-8"
        )
        try:
            for sample in range(c.repetitions):  # repetition-major: a budget stop keeps whole reps
                for task in cases:
                    toolbox = Toolbox(store, task, c.variant, c.max_tool_calls, c.anomaly_top_k)
                    result = run_case(
                        client,
                        toolbox,
                        task,
                        candidates,
                        prompts,
                        sample=sample,
                        max_turns=c.max_turns,
                        max_retries=c.max_retries,
                    )
                    results.append(result)
                    name = f"query_{task.query_row:03d}_s{sample}.json"
                    (transcripts / name).write_text(
                        json.dumps(
                            {
                                "summary": result.summary(),
                                "tools": result.tool_log,
                                "messages": result.transcript,
                            },
                            ensure_ascii=False,
                            indent=1,
                        ),
                        encoding="utf-8",
                    )
                    run.logger.info(
                        "rep %d query %d: %s (%d turns, %d tools, US$ %.4f)",
                        sample,
                        task.query_row,
                        result.status,
                        result.turns,
                        result.tool_calls,
                        result.cost_usd,
                    )
        except BudgetExceededError as exc:
            budget_stopped = True
            run.logger.warning("stopped by the budget cap: %s", exc)

        with (run.artifacts_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
            for r in results:
                handle.write(json.dumps(r.summary(), ensure_ascii=False) + "\n")
        for sample in sorted({r.sample for r in results}):
            path = run.artifacts_dir / f"predictions_s{sample}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["row_id", "task_index", "prediction"])
                for r in sorted(
                    (r for r in results if r.sample == sample), key=lambda r: r.query_row
                ):
                    writer.writerow([r.query_row, r.task_index, r.prediction])
        summary = _summarize(results, budget_stopped)
        cost = _cost(results, client)
        run.log_metrics({"variant": c.variant, "case_set": c.case_set, **summary})
        (run.artifacts_dir / "cost.json").write_text(json.dumps(cost, indent=2) + "\n", "utf-8")
        table = {
            "system": system,
            "variant": c.variant,
            "case_set": c.case_set,
            "cases": len(cases),
            "repetitions": c.repetitions,
            **summary,
            **cost,
            "run_dir": run.dir.name,
        }
        model_slug = config.llm.model.replace("/", "-").replace(".", "-")
        name = f"{config.experiment.name}_{model_slug}"  # one table per model (pilot compares)
        (tables_dir / f"agent_{system}_{name}.json").write_text(
            json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        write_agent_markdown(tables_dir / f"agent_{system}_{name}.md", table)
        return {
            "run_dir": str(run.dir),
            "case_runs": summary["case_runs"],
            "valid_rate": summary["valid_rate"],
            "cost_usd": cost["cost_usd"],
            "paid_calls": cost["usage"]["paid_calls"],
            "budget_stopped": budget_stopped,
        }


def write_agent_markdown(path: Path, t: dict[str, Any]) -> None:
    u = t["usage"]
    lines = [
        f"# Agente (E7) — {t['system']}, variante {t['variant']} ({t['case_set']})",
        "",
        f"Modelo: `{t['provider']}` / `{t['model']}` (esforço: {t['effort']}). "
        f"{t['cases']} consultas x {t['repetitions']} repetição(ões). Execução: `{t['run_dir']}`.",
        "",
        "| Medida | Valor |",
        "|---|---|",
        f"| Execuções de caso | {t['case_runs']} |",
        f"| Situação | {t['status']} |",
        f"| Respostas válidas | {t['valid_rate']} |",
        f"| Válidas só após nova tentativa | {t['valid_after_retry']} |",
        f"| Turnos por caso (média) | {t['mean_turns']} |",
        f"| Chamadas de ferramenta por caso (média) | {t['mean_tool_calls']} |",
        f"| Chamadas ao modelo (pagas / cache) | {u['paid_calls']} / {u['cache_hits']} |",
        f"| Tokens de entrada (cheio / cache / escrita) | {u['input_tokens']} / "
        f"{u['cached_input_tokens']} / {u['cache_write_tokens']} |",
        f"| Tokens de saída | {u['output_tokens']} |",
        f"| Custo total (US$) | {t['cost_usd']:.4f} (teto {t['budget_usd']}) |",
        f"| Custo por execução de caso (US$) | {t['cost_per_case_run_usd']} |",
        f"| Maior custo de um caso (US$) | {t['max_case_cost_usd']} |",
        f"| Parado pelo teto de custo | {t['budget_stopped']} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
