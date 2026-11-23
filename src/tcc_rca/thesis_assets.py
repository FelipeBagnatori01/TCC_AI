"""Tables and figures for the monograph, generated from the result tables only (E9, ADR 0024).

Output (``reports/monografia/``):

* ``tabelas.docx`` and ``figuras.docx``: every item numbered with its title above and its source
  below (ABNT NBR 14724; tables with horizontal rules only, IBGE presentation norms);
* ``tabelas/Tabela_NN_<slug>.csv``: the same cells (``;``-separated, UTF-8 with BOM, decimal
  comma, so spreadsheet software in Portuguese opens them as they are);
* ``figuras/Figura_NN_<slug>.png`` at 300 dpi;
* ``MANIFEST.csv``: number, title, file, source tables, command and commit of each item;
* ``numeros.csv``: the numbers the text cites, each with the JSON file and key it comes from.

No number is typed here: every cell is read from ``reports/tables`` through :class:`Source`,
which records the file and key, so the manifest lists what each item depends on.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

MINUS = "\u2212"  # typographic minus in the tables (ABNT)
SIGMA = "\u03c3"
DASH = "\u2013"  # ABNT caption separator (Tabela 1, dash, title)
SYSTEMS = ("Telecom", "Bank")
VARIANTS = ("A1", "A2", "A3")
MODELS = {"Opus": "claude-opus-5-5", "Sonnet": "claude-sonnet-5-5", "Haiku": "claude-haiku-5-5"}
# ADR 0023 addendum: the Sonnet Bank A3 run was interrupted (no credit) after US$ 2.70 and resumed
# from the cache; its table records only the resumed part.
INTERRUPTED_USD = {("Bank", "Sonnet", "A3"): 2.70}
SOURCE_TEXT = "Fonte: Elaborado pelo autor (2026), a partir dos dados do OpenRCA (XU et al., 2025)."
COMMAND = "uv run tcc-rca thesis-assets"
COLORS = {"Opus": "#2a78d6", "Sonnet": "#eb6834", "Haiku": "#1baf7a", "ref": "#52514e"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


class Source:
    """Reads values from the result tables and remembers which files each item used."""

    def __init__(self, tables_dir: Path) -> None:
        self.dir = tables_dir
        self._cache: dict[str, Any] = {}
        self.used: set[str] = set()

    def table(self, name: str) -> Any:
        if name not in self._cache:
            self._cache[name] = json.loads((self.dir / name).read_text(encoding="utf-8"))
        self.used.add(name)
        return self._cache[name]

    def get(self, name: str, *keys: str | int) -> Any:
        value = self.table(name)
        for k in keys:
            value = value[k]
        return value

    def take_used(self) -> list[str]:
        used, self.used = sorted(self.used), set()
        return used


def num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",").replace("-", MINUS)


def ci(d: dict[str, Any], digits: int = 2) -> str:
    return f"{num(d['value'], digits)} [{num(d['ci_low'], digits)}; {num(d['ci_high'], digits)}]"


def money(x: float) -> str:
    return "US$ " + num(x, 3)


@dataclass
class TableAsset:
    slug: str
    title: str
    header: list[str]
    rows: list[list[str]]
    note: str = ""
    sources: list[str] = field(default_factory=list)


@dataclass
class FigureAsset:
    slug: str
    title: str
    draw: Callable[[Source], Figure] | None
    note: str = ""
    copy_from: Path | None = None
    sources: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- tables
def table_data(s: Source) -> TableAsset:
    rows = []
    fields: list[tuple[str, Callable[[str], Any]]] = [
        ("Dias com telemetria", lambda sys_: s.get(f"inventory_{sys_}.json", "n_days")),
        ("Falhas registradas", lambda sys_: s.get(f"tasks_{sys_}.json", "failures_total")),
        ("Falhas testadas", lambda sys_: s.get(f"tasks_{sys_}.json", "failures_tested")),
        ("Consultas testadas", lambda sys_: s.get(f"tasks_{sys_}.json", "queries_tested")),
        ("Dobras de origem móvel", lambda sys_: s.get(f"tasks_{sys_}.json", "folds")),
        ("Janelas (W = 20 min)", lambda sys_: s.get(f"tasks_{sys_}.json", "windows")),
        (
            "Componentes candidatos",
            lambda sys_: s.get(f"tasks_{sys_}.json", "candidate_components"),
        ),
    ]
    for label, fn in fields:
        rows.append([label, *[str(fn(sys_)) for sys_ in SYSTEMS]])
    return TableAsset("dados", "Características dos dados usados", ["Item", *SYSTEMS], rows)


DETECTORS = [  # label, (file template, keys) ; LSTM/alt: summary mean over seeds; E4: value
    ("LSTM previsora", "lstm_{s}_lstm.json", ("detection", "lstm_forecast", "summary"), True),
    (
        "LSTM classificadora",
        "lstm_{s}_lstm.json",
        ("detection", "lstm_classifier", "summary"),
        True,
    ),
    ("Logística L1", "alt_models_{s}_alt_tabular.json", ("detection", "logistic", "summary"), True),
    ("TCN", "alt_models_{s}_alt_tcn.json", ("detection", "tcn", "summary"), True),
    ("LightGBM", "alt_models_{s}_alt_tabular.json", ("detection", "lightgbm", "summary"), True),
    (
        "Isolation Forest",
        "alt_models_{s}_alt_tabular.json",
        ("detection", "isolation_forest", "summary"),
        True,
    ),
    ("z-score local", "baselines_{s}_baselines.json", ("detection", "zscore_local"), False),
    ("z-score 3" + SIGMA, "baselines_{s}_baselines.json", ("detection", "zscore_3sigma"), False),
    ("Referência só de horário", "baselines_{s}_baselines.json", ("detection", "prior_horario"),
     False),
]  # fmt: skip


def table_detection(s: Source) -> TableAsset:
    rows = []
    for label, tmpl, keys, seeded in DETECTORS:
        cells = [label]
        for sys_ in SYSTEMS:
            d = s.get(tmpl.format(s=sys_), *keys)
            pooled = d["auc_pr"]["mean"] if seeded else d["auc_pr"]["value"]
            fold = d["auc_pr_fold_mean"]["mean"] if seeded else d["auc_pr_fold_mean"]
            if isinstance(fold, dict):
                fold = fold["value"]
            cells.append(f"{num(pooled)} ({num(fold)})")
        rows.append(cells)
    return TableAsset(
        "deteccao",
        "Detecção de falhas (RQ1): AUC-PR agrupada e, entre parênteses, a média por dobra",
        ["Detector", *SYSTEMS],
        rows,
        note="Nota: modelos com sementes (LSTM, logística, TCN, LightGBM, Isolation Forest) trazem "
        "a média das sementes finais; os baselines estatísticos, o valor único.",
    )


LOCALIZERS = [
    ("LSTM previsora + SHAP", "xai_{s}_xai.json", ("localization", "lstm_forecast/shap/sum")),
    ("LSTM previsora + LIME", "xai_{s}_xai.json", ("localization", "lstm_forecast/lime/sum")),
    (
        "LSTM classificadora + SHAP",
        "xai_{s}_xai.json",
        ("localization", "lstm_classifier/shap/sum"),
    ),
    ("Ranqueamento nativo da LSTM", "lstm_{s}_lstm.json", ("localization", "lstm_sum", "seeds", 0)),
    ("TCN + SHAP", "alt_models_{s}_alt_tcn.json", ("localization", "tcn/shap/sum")),
    (
        "Logística L1 + SHAP",
        "alt_models_{s}_alt_tabular.json",
        ("localization", "logistic/shap/sum"),
    ),
    ("z-score global", "baselines_{s}_baselines.json", ("localization",)),
    ("z-score local", "baselines_{s}_baselines.json", ("localization_local",)),
    ("Frequência no treino", "baselines_{s}_baselines.json", ("localization_prior",)),
]


def table_localization(s: Source) -> TableAsset:
    rows = []
    for label, tmpl, keys in LOCALIZERS:
        cells = [label]
        for sys_ in SYSTEMS:
            d = s.get(tmpl.format(s=sys_), *keys)
            cells += [ci(d["hr@1"]), num(d["mrr"]["value"])]
        rows.append(cells)
    return TableAsset(
        "localizacao",
        "Localização do componente (RQ2) nas janelas-oráculo: HR@1 [IC 95%] e MRR",
        ["Localizador", "Telecom HR@1", "Telecom MRR", "Bank HR@1", "Bank MRR"],
        rows,
        note="Nota: atribuições somadas por componente; semente 0.",
    )


CHAINS = [("A0", "A0 constante"), ("pacote", "Pacote sem LLM"), ("A1", "A1 telemetria"),
          ("A2", "A2 + LSTM"), ("A3", "A3 + SHAP/LIME")]  # fmt: skip


def table_agent(s: Source) -> TableAsset:
    rows = []
    for key, label in CHAINS:
        cells = [label]
        for sys_ in SYSTEMS:
            d = s.get(f"agent_eval_{sys_}.json", "summary", key)
            cells += [ci(d["partial_score"]), num(d["strict_accuracy"]["value"])]
        rows.append(cells)
    return TableAsset(
        "agente_notas",
        "Diagnóstico da causa raiz (RQ3): nota oficial do OpenRCA por cadeia (Claude Opus 5.5)",
        ["Cadeia", "Telecom parcial", "Telecom estrito", "Bank parcial", "Bank estrito"],
        rows,
        note="Nota: nota parcial = média da nota oficial por consulta [IC 95% por bootstrap sobre "
        "consultas]; acerto estrito = fração de consultas com nota 1.",
    )


def table_agent_comparisons(s: Source) -> TableAsset:
    rows = []
    for sys_ in SYSTEMS:
        e = s.table(f"agent_eval_{sys_}.json")
        for r in [*e["comparisons"], *e["references"]]:
            holm = r.get("mcnemar_p_holm")
            rows.append(
                [
                    sys_,
                    r["comparison"].replace(" - ", f" {MINUS} "),
                    ci(r["partial_difference"]),
                    r["verdict"],
                    f"{r['mcnemar_only_first']}/{r['mcnemar_only_second']}",
                    "—" if holm is None else ("< 0,001" if holm < 0.001 else num(holm, 3)),
                ]
            )
    return TableAsset(
        "agente_comparacoes",
        "Comparações pareadas do agente (RQ3): diferença da nota parcial e McNemar exato",
        ["Sistema", "Comparação", "Diferença [IC 95%]", "Veredito", "b/c", "p (Holm)"],
        rows,
        note="Nota: b/c = consultas em que só a primeira/só a segunda cadeia acerta (acerto "
        "estrito); Holm só entre as três comparações de variantes. A3 "
        + MINUS
        + " A1 é a comparação principal.",
    )


def table_fidelity_cost(s: Source) -> TableAsset:
    rows = []
    for sys_ in SYSTEMS:
        e = s.table(f"agent_eval_{sys_}.json")
        for v in VARIANTS:
            f, c = e["fidelity"][v], e["cost"][v]
            audit = e["audit"]["by_variant"][v]
            rows.append(
                [
                    sys_,
                    v,
                    f"{f['ids']['flagged']}/{f['ids']['checked']}",
                    f"{f['components']['flagged']}/{f['components']['checked']}",
                    f"{f['numbers']['flagged']}/{f['numbers']['checked']}",
                    num(audit["sustentada"]["value"]),
                    num(audit["parcial"]["value"]),
                    money(c["cost_usd_mean"]),
                    num(c["latency_s_mean"], 0) + " s",
                ]
            )
    return TableAsset(
        "agente_fidelidade_custo",
        "Fidelidade e custo das explicações do agente (RQ4)",
        [
            "Sistema",
            "Variante",
            "IDs não emitidos",
            "Componentes inexistentes",
            "Números sem correspondência",
            "Sustentadas",
            "Parciais",
            "Custo por consulta",
            "Latência média",
        ],
        rows,
        note="Nota: as três primeiras checagens são automáticas, sobre todas as explicações. "
        "Sustentadas e parciais vêm da auditoria de 20 afirmações por célula feita por LLM "
        "(Claude Opus 5.5, o mesmo modelo do agente), sem validação humana.",
    )


def _model_cost(s: Source, sys_: str, model: str, v: str) -> float:
    if model == "Opus":
        cost = s.get(f"agent_eval_{sys_}.json", "cost", v, "cost_usd")
        n = s.get(f"agent_eval_{sys_}.json", "queries")
    else:
        t = s.table(f"model_compare_{sys_}_{MODELS[model]}.json")
        cost, n = t["cost"][f"{model} {v}"]["cost_usd"], t["queries"]
    return (float(cost) + INTERRUPTED_USD.get((sys_, model, v), 0.0)) / int(n)


def table_models(s: Source) -> TableAsset:
    rows = []
    for model, slug in MODELS.items():
        for v in VARIANTS:
            cells = [model, v]
            for sys_ in SYSTEMS:
                if model == "Opus":
                    d = s.get(f"agent_eval_{sys_}.json", "summary", v, "partial_score")
                else:
                    d = s.get(f"model_compare_{sys_}_{slug}.json", "summary", f"{model} {v}",
                              "partial_score")  # fmt: skip
                cells += [ci(d), money(_model_cost(s, sys_, model, v))]
            rows.append(cells)
    for model, slug in MODELS.items():
        cells = [model, f"A3 {MINUS} A1"]
        for sys_ in SYSTEMS:
            if model == "Opus":
                r = s.get(f"agent_eval_{sys_}.json", "comparisons", 0)
            else:
                r = s.get(f"model_compare_{sys_}_{slug}.json", "within_candidate", 0)
            cells += [f"{ci(r['partial_difference'])} {r['verdict']}", "—"]
        rows.append(cells)
    return TableAsset(
        "modelos_llm",
        "Agente com modelos de LLM diferentes (exploratório): nota parcial e custo por consulta",
        ["Modelo", "Variante", "Telecom nota", "Telecom custo", "Bank nota", "Bank custo"],
        rows,
        note="Nota: mesmas ferramentas, prompts e consultas; análise decidida depois de ver os "
        "resultados do Opus (ADR 0023). Nenhum modelo mais barato é não inferior ao Opus "
        "(margem de 0,10).",
    )


# --------------------------------------------------------------------------- figures
COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))  # decimal comma on the axes


def _style(ax: Any) -> None:
    ax.xaxis.set_major_formatter(COMMA)
    ax.yaxis.set_major_formatter(COMMA)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def _dot(ax: Any, d: dict[str, Any], y: float, color: str, marker: str = "o") -> None:
    ax.errorbar(d["value"], y, xerr=[[d["value"] - d["ci_low"]], [d["ci_high"] - d["value"]]],
                fmt=marker, color=color, ms=6, elinewidth=1.8, capsize=0)  # fmt: skip
    ax.text(d["ci_high"] + 0.015, y, num(d["value"]), va="center", fontsize=7, color=INK)


def figure_localization(s: Source) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    for ax, sys_ in zip(axes, SYSTEMS, strict=True):
        for i, (label, tmpl, keys) in enumerate(LOCALIZERS):
            d = s.get(tmpl.format(s=sys_), *keys)["hr@1"]
            _dot(ax, d, i, COLORS["Opus"] if label.startswith("LSTM") else COLORS["ref"])
        ax.set_yticks(range(len(LOCALIZERS)), [x[0] for x in LOCALIZERS], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(sys_, fontsize=10)
        ax.set_xlabel("HR@1 (IC 95%)", fontsize=8)
        _style(ax)
    fig.tight_layout()
    return fig


def figure_agent(s: Source) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), sharex=True)
    for ax, sys_ in zip(axes, SYSTEMS, strict=True):
        for i, (key, _) in enumerate(CHAINS):
            d = s.get(f"agent_eval_{sys_}.json", "summary", key, "partial_score")
            _dot(ax, d, i, COLORS["Opus"] if key.startswith("A") and key != "A0" else COLORS["ref"])
        ax.set_yticks(range(len(CHAINS)), [c[1] for c in CHAINS], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(sys_, fontsize=10)
        ax.set_xlabel("nota oficial parcial (IC 95%)", fontsize=8)
        _style(ax)
    fig.tight_layout()
    return fig


def figure_models(s: Source) -> Figure:
    markers = {"A1": "o", "A2": "s", "A3": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, sys_ in zip(axes, SYSTEMS, strict=True):
        for model, slug in MODELS.items():
            for v in VARIANTS:
                if model == "Opus":
                    score = s.get(f"agent_eval_{sys_}.json", "summary", v, "partial_score", "value")
                else:
                    score = s.get(f"model_compare_{sys_}_{slug}.json", "summary", f"{model} {v}",
                                  "partial_score", "value")  # fmt: skip
                ax.scatter(_model_cost(s, sys_, model, v), score, color=COLORS[model],
                           marker=markers[v], s=45, zorder=3)  # fmt: skip
        pkg = s.get(f"agent_eval_{sys_}.json", "summary", "pacote", "partial_score", "value")
        ax.axhline(pkg, color=COLORS["ref"], lw=0.8, ls=":")
        ax.text(0.0045, pkg + 0.012, "pacote sem LLM", fontsize=7, color=COLORS["ref"])
        ax.set_xscale("log")
        ax.set_xlim(0.004, 0.4)
        ax.set_ylim(0, 0.7)
        ax.set_title(sys_, fontsize=10)
        ax.set_xlabel("custo por consulta (US$, escala log)", fontsize=8)
        _style(ax)
    axes[0].set_ylabel("nota oficial parcial", fontsize=8)
    handles = [Line2D([], [], ls="", marker="o", color=COLORS[m], label=m) for m in MODELS] + [
        Line2D([], [], ls="", marker=markers[v], color=MUTED, label=v) for v in VARIANTS
    ]
    axes[1].legend(handles=handles, loc="lower right", ncol=2, fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- numbers in the text
NUMBERS: list[tuple[str, str, str, tuple[str | int, ...]]] = [
    ("auc_pr_previsora_telecom", "AUC-PR agrupada da LSTM previsora, Telecom (média das sementes)",
     "lstm_Telecom_lstm.json", ("detection", "lstm_forecast", "summary", "auc_pr", "mean")),
    ("auc_pr_previsora_bank", "AUC-PR agrupada da LSTM previsora, Bank (média das sementes)",
     "lstm_Bank_lstm.json", ("detection", "lstm_forecast", "summary", "auc_pr", "mean")),
    ("auc_pr_zscore_local_telecom", "AUC-PR do z-score local, Telecom",
     "baselines_Telecom_baselines.json", ("detection", "zscore_local", "auc_pr", "value")),
    ("auc_pr_zscore_local_bank", "AUC-PR do z-score local, Bank",
     "baselines_Bank_baselines.json", ("detection", "zscore_local", "auc_pr", "value")),
    ("hr1_shap_telecom", "HR@1 da LSTM previsora + SHAP, Telecom", "xai_Telecom_xai.json",
     ("localization", "lstm_forecast/shap/sum", "hr@1", "value")),
    ("hr1_shap_bank", "HR@1 da LSTM previsora + SHAP, Bank", "xai_Bank_xai.json",
     ("localization", "lstm_forecast/shap/sum", "hr@1", "value")),
]  # fmt: skip
for _sys in SYSTEMS:
    _t = f"agent_eval_{_sys}.json"
    for _k in ("A0", "pacote", "A1", "A2", "A3"):
        NUMBERS.append((f"nota_parcial_{_k.lower()}_{_sys.lower()}",
                        f"Nota parcial oficial, {_k}, {_sys}", _t,
                        ("summary", _k, "partial_score", "value")))  # fmt: skip
    NUMBERS.append((f"a3_menos_a1_{_sys.lower()}", f"Diferença pareada A3 {MINUS} A1, {_sys}", _t,
                    ("comparisons", 0, "partial_difference", "value")))  # fmt: skip
    NUMBERS.append((f"a3_menos_a1_ic_inf_{_sys.lower()}",
                    f"IC 95% inferior de A3 {MINUS} A1, {_sys}",
                    _t, ("comparisons", 0, "partial_difference", "ci_low")))  # fmt: skip
    NUMBERS.append((f"a3_menos_a1_ic_sup_{_sys.lower()}",
                    f"IC 95% superior de A3 {MINUS} A1, {_sys}",
                    _t, ("comparisons", 0, "partial_difference", "ci_high")))  # fmt: skip
    for _m, _slug in (("Haiku", MODELS["Haiku"]), ("Sonnet", MODELS["Sonnet"])):
        NUMBERS.append((f"nota_parcial_{_m.lower()}_a3_{_sys.lower()}",
                        f"Nota parcial oficial, {_m} A3, {_sys}",
                        f"model_compare_{_sys}_{_slug}.json",
                        ("summary", f"{_m} A3", "partial_score", "value")))  # fmt: skip


def write_numbers(s: Source, out: Path) -> list[str]:
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        w = csv.writer(handle, delimiter=";", lineterminator="\n")
        w.writerow(["id", "descricao", "valor", "arquivo", "chave"])
        for key, desc, name, path in NUMBERS:
            w.writerow([key, desc, num(float(s.get(name, *path)), 4), f"reports/tables/{name}",
                        "/".join(str(p) for p in path)])  # fmt: skip
    return s.take_used()


# --------------------------------------------------------------------------- docx
def _rules(table: Any) -> None:
    """IBGE presentation: top and bottom rules, one under the header, no vertical lines."""
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge, val in (("top", "single"), ("bottom", "single"), ("left", "nil"), ("right", "nil"),
                      ("insideH", "nil"), ("insideV", "nil")):  # fmt: skip
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), "6")
        borders.append(el)
    tbl_pr.append(borders)
    for cell in table.rows[0].cells:
        tc_pr = cell._tc.get_or_add_tcPr()
        tc_borders = OxmlElement("w:tcBorders")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "6")
        tc_borders.append(bottom)
        tc_pr.append(tc_borders)


def _caption(doc: Any, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.size = Pt(10)


def _source(doc: Any, note: str) -> None:
    for text in ([note] if note else []) + [SOURCE_TEXT]:
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.size = Pt(9)
    doc.add_paragraph()


def _new_doc() -> Any:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)
    return doc


def write_tables(tables: Sequence[TableAsset], out_dir: Path) -> None:
    doc = _new_doc()
    (out_dir / "tabelas").mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(tables, 1):
        _caption(doc, f"Tabela {i} {DASH} {t.title}")
        table = doc.add_table(rows=1 + len(t.rows), cols=len(t.header))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for r, row in enumerate([t.header, *t.rows]):
            for c, text in enumerate(row):
                cell = table.cell(r, c)
                cell.text = text
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.size = Pt(8)
                        run.bold = r == 0
        _rules(table)
        _source(doc, t.note)
        with (out_dir / "tabelas" / f"Tabela_{i:02d}_{t.slug}.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            w = csv.writer(handle, delimiter=";", lineterminator="\n")
            w.writerow(t.header)
            w.writerows(t.rows)
    doc.save(str(out_dir / "tabelas.docx"))


def write_figures(figures: Sequence[FigureAsset], s: Source, out_dir: Path) -> None:
    doc = _new_doc()
    (out_dir / "figuras").mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(figures, 1):
        path = out_dir / "figuras" / f"Figura_{i:02d}_{f.slug}.png"
        if f.copy_from is not None:
            shutil.copyfile(f.copy_from, path)
        else:
            assert f.draw is not None
            fig = f.draw(s)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            f.sources = s.take_used()
        _caption(doc, f"Figura {i} {DASH} {f.title}")
        doc.add_picture(str(path), width=Cm(16))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        _source(doc, f.note)
    doc.save(str(out_dir / "figuras.docx"))


def _commit() -> str:
    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src", "reports/tables"])
        return head + ("-dirty" if dirty.returncode else "")
    except (OSError, subprocess.CalledProcessError):
        return "desconhecido"


def build(tables_dir: Path, figures_dir: Path, out_dir: Path) -> dict[str, Any]:
    s = Source(tables_dir)
    tables = []
    for builder in (table_data, table_detection, table_localization, table_agent,
                    table_agent_comparisons, table_fidelity_cost, table_models):  # fmt: skip
        t = builder(s)
        t.sources = s.take_used()
        tables.append(t)
    figures = [
        FigureAsset(
            "rq1_pr_roc",
            "Curvas PR e ROC dos detectores no teste (RQ1)",
            None,
            copy_from=figures_dir / "rq1_curvas_pr_roc.png",
            sources=["reports/figures/rq1_curvas_pr_roc.png (tcc-rca rq1-curves)"],
        ),
        FigureAsset("localizacao", "Localização do componente (RQ2): HR@1 com IC 95%",
                    figure_localization),
        FigureAsset("agente_notas", "Nota oficial do agente por cadeia (RQ3), com IC 95%",
                    figure_agent),
        FigureAsset("modelos_llm", "Custo por consulta e nota parcial por modelo de LLM",
                    figure_models,
                    note="Nota: cor = modelo; marcador = variante; escala logarítmica no custo."),
    ]  # fmt: skip
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    write_tables(tables, out_dir)
    write_figures(figures, s, out_dir)
    number_sources = write_numbers(s, out_dir / "numeros.csv")
    commit = _commit()
    with (out_dir / "MANIFEST.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        w = csv.writer(handle, delimiter=";", lineterminator="\n")
        w.writerow(["tipo", "numero", "titulo", "arquivo", "origem", "comando", "commit"])
        for i, t in enumerate(tables, 1):
            src = " ".join(f"reports/tables/{x}" for x in t.sources)
            w.writerow(["Tabela", i, t.title, f"tabelas/Tabela_{i:02d}_{t.slug}.csv", src,
                        COMMAND, commit])  # fmt: skip
        for i, f in enumerate(figures, 1):
            src = " ".join(x if "/" in x else f"reports/tables/{x}" for x in f.sources)
            w.writerow(["Figura", i, f.title, f"figuras/Figura_{i:02d}_{f.slug}.png", src,
                        COMMAND, commit])  # fmt: skip
    return {
        "tables": len(tables),
        "figures": len(figures),
        "numbers": len(NUMBERS),
        "number_sources": len(number_sources),
        "out": str(out_dir),
    }
