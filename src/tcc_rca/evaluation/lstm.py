"""E5: LSTM forecaster (main) and classifier (secondary) over the rolling-origin folds.

Protocol (ADR 0015): hyperparameters are searched on VALIDATION only (mean validation AUC-PR
over the folds whose validation day has failures, one seed); the selected configuration is then
trained with every seed of ``lstm.seeds`` on every fold and evaluated on test with the E4 metrics.
Reads ground truth (labels, record.csv): evaluation code, never agent code.
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import torch

from tcc_rca.config import Config
from tcc_rca.dataset.build import failure_days, feature_categories, normal_minute_mask
from tcc_rca.dataset.split import Fold, rolling_origin_folds, with_roles
from tcc_rca.evaluation.baselines import (
    detection_metrics,
    epoch_minutes,
    evaluate_localization,
    load_windows,
    row_of,
)
from tcc_rca.evaluation.metrics import auc_pr, pessimistic_rank
from tcc_rca.ingestion.cases import Failure, read_failures
from tcc_rca.models.baselines.detection import choose_threshold
from tcc_rca.models.baselines.localization import component_columns
from tcc_rca.models.lstm.data import FoldInputs, Normalization, normalize, window_tensor
from tcc_rca.models.lstm.model import n_parameters
from tcc_rca.models.lstm.train import (
    ForecastScorer,
    TrainSettings,
    classifier_scores,
    train_classifier,
    train_forecaster,
)
from tcc_rca.series.build import exempt_features
from tcc_rca.series.catalog import split_feature
from tcc_rca.series.transforms import FeatureSelector
from tcc_rca.tracking import Run

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]
MODELS = ("lstm_forecast", "lstm_classifier")
RANKINGS = ("lstm_sum", "lstm_top3")


@dataclass(frozen=True)
class Choice:
    hidden: int
    layers: int
    normalization: Normalization


@dataclass
class FoldData:
    fold: Fold
    kept: list[str]
    roled: pl.DataFrame  # windows with role (purged), non-null roles only
    end_rows: IntArray
    role: npt.NDArray[np.str_]
    y: npt.NDArray[np.int64]  # 0 for ignored windows too; use ``labelled``
    labelled: npt.NDArray[np.bool_]


class Experiment:
    """Everything shared by the search and the final runs of one system."""

    def __init__(self, config: Config, tables_dir: Path) -> None:
        self.config, self.t = config, config.tasks
        system = config.target_system
        self.values = pl.read_parquet(config.paths.processed_dir / system / "series.parquet")
        self.windows = load_windows(config, tables_dir)
        self.failures = read_failures(config.paths.data_root / system / "record.csv")
        index = self.values.select("day", "time_utc")
        self.f_days = failure_days(self.failures, index)
        days = index["day"].unique().sort().to_list()
        self.folds = rolling_origin_folds(
            days, self.t.excluded_days, self.t.first_test_day, self.t.val_days
        )
        self.categories = feature_categories(
            self.values, tables_dir / f"feature_catalog_{system}.csv"
        )
        self.features = [c for c in self.values.columns if "::" in c]
        self.normal_min = normal_minute_mask(index, self.failures, config)
        self.day_arr = self.values["day"].to_numpy()
        self.minute_to_row = {
            m: i
            for i, m in enumerate(
                self.values.select(epoch_minutes("time_utc"))["time_utc"].to_list()
            )
        }
        self.candidates = sorted({split_feature(f)[0] for f in self.features})
        self.guard = self.t.window_minutes + self.t.horizon_minutes

    def fold_data(self, fold: Fold) -> FoldData:
        fit_rows = self.normal_min & np.isin(self.day_arr, list(fold.train_days))
        sel = self.config.series.selection
        kept = (
            FeatureSelector(
                sel.max_missing_frac,
                sel.near_constant_mode_share,
                exempt_features(self.categories, sel.constant_exempt_categories),
            )
            .fit(self.values, self.features, fit_rows)
            .kept
        )
        roled = with_roles(self.windows, fold, self.guard).filter(pl.col("role").is_not_null())
        label = roled["label"].to_numpy().astype(np.float64)
        labelled = ~np.isnan(label)
        return FoldData(
            fold,
            kept,
            roled,
            roled["row"].to_numpy().astype(np.int64),
            roled["role"].to_numpy(),
            np.where(labelled, label, 0).astype(np.int64),
            labelled,
        )

    def inputs(self, fd: FoldData, normalization: Normalization) -> FoldInputs:
        fit_rows = self.normal_min & np.isin(self.day_arr, list(fd.fold.train_days))
        p, lc = self.t.precursor, self.config.lstm
        return normalize(
            self.values,
            fd.kept,
            fit_rows,
            normalization,
            lc.global_clip,
            p.local_window_minutes,
            p.local_cap,
        )

    def tensors(
        self, fd: FoldData, inputs: FoldInputs, mask: npt.NDArray[np.bool_]
    ) -> Float32Array:
        return window_tensor(inputs.matrix, fd.end_rows[mask], self.t.window_minutes)

    def settings(self, hidden: int, layers: int, seed: int) -> TrainSettings:
        lc = self.config.lstm
        return TrainSettings(
            hidden,
            layers,
            lc.dropout,
            lc.learning_rate,
            lc.batch_size,
            lc.max_epochs,
            lc.patience,
            seed,
        )

    def tested(self, fold: Fold) -> list[Failure]:
        return [f for f, d in zip(self.failures, self.f_days, strict=True) if d == fold.test_day]


def _masks(fd: FoldData) -> dict[str, npt.NDArray[np.bool_]]:
    return {
        "train_normal": (fd.role == "train") & fd.labelled & (fd.y == 0),
        "train_labelled": (fd.role == "train") & fd.labelled,
        "val_normal": (fd.role == "val") & fd.labelled & (fd.y == 0),
        "val_labelled": (fd.role == "val") & fd.labelled,
        "test": fd.role == "test",
    }


def _val_auc_pr(y: npt.NDArray[np.int64], scores: FloatArray) -> float | None:
    return auc_pr(y, scores) if y.sum() > 0 else None


def search(exp: Experiment, log: Callable[[str], None]) -> dict[str, Any]:
    """Grid search on validation (one seed). Returns choices and the full search table."""
    lc, seed = exp.config.lstm, exp.config.experiment.seed
    g = lc.forecaster_grid
    f_grid = [Choice(h, n, z) for z in g.normalization for h in g.hidden for n in g.layers]
    f_scores: dict[Choice, list[float]] = {c: [] for c in f_grid}
    c_scores: dict[tuple[Normalization, int, int], list[float]] = {}
    rows: list[dict[str, Any]] = []
    for fold in exp.folds:
        fd = exp.fold_data(fold)
        m = _masks(fd)
        if fd.y[m["val_labelled"]].sum() == 0:
            log(f"search: fold {fold.index} skipped (no validation failure)")
            continue
        for norm in g.normalization:
            inputs = exp.inputs(fd, norm)
            tr, vn = (
                exp.tensors(fd, inputs, m["train_normal"]),
                exp.tensors(fd, inputs, m["val_normal"]),
            )
            vl = exp.tensors(fd, inputs, m["val_labelled"])
            yv = fd.y[m["val_labelled"]]
            for choice in (c for c in f_grid if c.normalization == norm):
                model, hist = train_forecaster(
                    tr, vn, exp.settings(choice.hidden, choice.layers, seed)
                )
                score = _val_auc_pr(
                    yv, ForecastScorer.fit(model, tr, lc.score_top_k).score_windows(vl)
                )
                f_scores[choice].append(score or 0.0)
                rows.append(
                    {
                        "model": "lstm_forecast",
                        "fold": fold.index,
                        **asdict(choice),
                        "val_auc_pr": score,
                        "epochs": len(hist.train_loss),
                        "seconds": round(hist.seconds, 2),
                    }
                )
            # classifier: every normalization is tried here too, but only its size is chosen
            # later, inside the forecaster's selected normalization
            trl, ytr = exp.tensors(fd, inputs, m["train_labelled"]), fd.y[m["train_labelled"]]
            for h, n in itertools.product(lc.classifier_hidden, lc.classifier_layers):
                model_c, hist_c = train_classifier(trl, ytr, vl, yv, exp.settings(h, n, seed))
                score = _val_auc_pr(yv, classifier_scores(model_c, vl))
                c_scores.setdefault((norm, h, n), []).append(score or 0.0)
                rows.append(
                    {
                        "model": "lstm_classifier",
                        "fold": fold.index,
                        "hidden": h,
                        "layers": n,
                        "normalization": norm,
                        "val_auc_pr": score,
                        "epochs": len(hist_c.train_loss),
                        "seconds": round(hist_c.seconds, 2),
                    }
                )
            log(f"search: fold {fold.index} normalization {norm} done")

    # best mean validation AUC-PR; ties -> the smaller model
    def key_f(c: Choice) -> tuple[float, int, int]:
        return (float(np.mean(f_scores[c])), -c.hidden, -c.layers)

    best_f = max(f_grid, key=key_f)
    same_norm = [k for k in c_scores if k[0] == best_f.normalization]
    best_c = max(same_norm, key=lambda k: (float(np.mean(c_scores[k])), -k[1], -k[2]))
    return {
        "forecaster": asdict(best_f),
        "classifier": {"normalization": best_c[0], "hidden": best_c[1], "layers": best_c[2]},
        "forecaster_val_auc_pr": {
            f"{c.normalization}/h{c.hidden}/l{c.layers}": round(float(np.mean(v)), 4)
            for c, v in f_scores.items()
        },
        "classifier_val_auc_pr": {
            f"{k[0]}/h{k[1]}/l{k[2]}": round(float(np.mean(v)), 4) for k, v in c_scores.items()
        },
        "trainings": len(rows),
        "table": rows,
    }


def _component_scores(
    errors: Float32Array, columns: dict[str, IntArray], candidates: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Native ranking from normalized forecast errors of one window: per feature the max over
    steps, then per component the SUM (as the plan specifies) or the mean of the 3 largest."""
    per_feature = errors.max(axis=0)
    out: dict[str, dict[str, float]] = {"lstm_sum": {}, "lstm_top3": {}}
    for c in candidates:
        cols = columns.get(c)
        vals = per_feature[cols] if cols is not None else np.zeros(1, dtype=np.float32)
        out["lstm_sum"][c] = float(vals.sum())
        out["lstm_top3"][c] = float(np.sort(vals)[::-1][:3].mean())
    return out


def final_runs(
    exp: Experiment, selection: dict[str, Any], models_dir: Path, log: Callable[[str], None]
) -> tuple[pl.DataFrame, pl.DataFrame, list[dict[str, Any]]]:
    lc, base_seed = exp.config.lstm, exp.config.experiment.seed
    fc, cc = selection["forecaster"], selection["classifier"]
    norm: Normalization = fc["normalization"]
    score_parts: list[pl.DataFrame] = []
    rank_rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for fold in exp.folds:
        fd = exp.fold_data(fold)
        m = _masks(fd)
        inputs = exp.inputs(fd, norm)
        tr, vn = (
            exp.tensors(fd, inputs, m["train_normal"]),
            exp.tensors(fd, inputs, m["val_normal"]),
        )
        vl, te = exp.tensors(fd, inputs, m["val_labelled"]), exp.tensors(fd, inputs, m["test"])
        trl, ytr, yv = (
            exp.tensors(fd, inputs, m["train_labelled"]),
            fd.y[m["train_labelled"]],
            fd.y[m["val_labelled"]],
        )
        columns = component_columns(fd.kept)
        failures = exp.tested(fold)
        fail_rows = np.array(
            [
                row_of(
                    exp.minute_to_row,
                    f.start_utc.timestamp() / 60.0 + exp.t.detection_tolerance_minutes,
                )
                for f in failures
            ],
            dtype=np.int64,
        )
        fail_windows = window_tensor(inputs.matrix, fail_rows, exp.t.window_minutes)
        for offset in lc.seeds:
            seed = base_seed + offset
            model, hist = train_forecaster(tr, vn, exp.settings(fc["hidden"], fc["layers"], seed))
            scorer = ForecastScorer.fit(model, tr, lc.score_top_k)
            t0 = time.perf_counter()
            test_scores = scorer.score_windows(te)
            per_window_ms = (time.perf_counter() - t0) * 1000 / max(te.shape[0], 1)
            model_c, hist_c = train_classifier(
                trl, ytr, vl, yv, exp.settings(cc["hidden"], cc["layers"], seed)
            )
            outputs = {
                "lstm_forecast": (scorer.score_windows(vl), test_scores),
                "lstm_classifier": (classifier_scores(model_c, vl), classifier_scores(model_c, te)),
            }
            for name, (val_s, test_s) in outputs.items():
                thr, rule = choose_threshold(yv, val_s, exp.config.baselines.val_fallback_quantile)
                score_parts.append(
                    fd.roled.filter(pl.Series(m["test"])).select(
                        pl.lit(fold.index).alias("fold"),
                        pl.lit(offset).alias("seed"),
                        pl.lit(name).alias("method"),
                        "row",
                        "day",
                        "end_utc",
                        "label",
                        pl.Series("score", test_s),
                        pl.Series("alarm", test_s >= thr),
                    )
                )
                runs.append(
                    {
                        "fold": fold.index,
                        "seed": offset,
                        "model": name,
                        "threshold": thr,
                        "rule": rule,
                    }
                )
            errors = scorer.normalized_errors(fail_windows) if len(failures) else np.zeros((0,))
            for i, f in enumerate(failures):
                for method, comp in _component_scores(errors[i], columns, exp.candidates).items():
                    rank_rows.append(
                        {
                            "method": method,
                            "seed": offset,
                            "fold": fold.index,
                            "onset_utc": f.start_utc,
                            "component": f.component,
                            "reason": f.reason,
                            "rank": pessimistic_rank(f.component, comp),
                        }
                    )
            runs[-2].update(
                {
                    "params": n_parameters(model),
                    "train_seconds": round(hist.seconds, 2),
                    "epochs": len(hist.train_loss),
                    "best_epoch": hist.best_epoch,
                    "train_loss": hist.train_loss,
                    "val_loss": hist.val_loss,
                    "inference_ms_per_window": round(per_window_ms, 4),
                }
            )
            runs[-1].update(
                {
                    "params": n_parameters(model_c),
                    "train_seconds": round(hist_c.seconds, 2),
                    "epochs": len(hist_c.train_loss),
                    "best_epoch": hist_c.best_epoch,
                    "train_loss": hist_c.train_loss,
                    "val_loss": hist_c.val_loss,
                }
            )
            target = models_dir / f"fold{fold.index}" / f"seed{offset}"
            target.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), target / "forecaster.pt")
            torch.save(model_c.state_dict(), target / "classifier.pt")
            np.save(target / "forecast_error_scale.npy", scorer.scale)
            (target / "features.json").write_text(
                json.dumps({"features": fd.kept, "normalization": norm, **fc}), encoding="utf-8"
            )
            secs = f"{hist.seconds:.1f}s + {hist_c.seconds:.1f}s"
            log(f"final: fold {fold.index} seed {offset} done ({secs})")
    return pl.concat(score_parts), pl.DataFrame(rank_rows), runs


def _summary(per_seed: list[dict[str, Any]], key: str) -> dict[str, float]:
    vals = np.array([s[key]["value"] for s in per_seed], dtype=np.float64)
    return {
        "mean": round(float(vals.mean()), 4),
        "sd": round(float(vals.std(ddof=1)) if vals.size > 1 else 0.0, 4),
        "min": round(float(vals.min()), 4),
        "max": round(float(vals.max()), 4),
    }


def evaluate(exp: Experiment, scores: pl.DataFrame, ranks: pl.DataFrame) -> dict[str, Any]:
    tested = [
        f
        for f, d in zip(exp.failures, exp.f_days, strict=True)
        if d in {fo.test_day for fo in exp.folds}
    ]
    onsets = np.array([f.start_utc.timestamp() / 60.0 for f in tested])
    reasons = [f.reason for f in tested]
    out: dict[str, Any] = {"detection": {}, "localization": {}}
    for name in MODELS:
        per_seed = []
        for offset in exp.config.lstm.seeds:
            m = scores.filter((pl.col("method") == name) & (pl.col("seed") == offset))
            per_seed.append(detection_metrics(m, onsets, reasons, exp.config))
        keys = (
            "auc_pr",
            "auc_roc",
            "precision",
            "recall",
            "f1",
            "false_alarms_per_hour",
            "event_recall",
        )
        out["detection"][name] = {
            "seeds": per_seed,
            "summary": {k: _summary(per_seed, k) for k in keys},
        }
    for method in RANKINGS:
        per_seed = [
            evaluate_localization(
                ranks.filter((pl.col("method") == method) & (pl.col("seed") == o)), exp.config
            )
            for o in exp.config.lstm.seeds
        ]
        out["localization"][method] = {
            "seeds": per_seed,
            "summary": {k: _summary(per_seed, k) for k in ("hr@1", "hr@3", "hr@5", "mrr")},
        }
    return out


def run_lstm(config: Config, tables_dir: Path) -> dict[str, Any]:
    system, name = config.target_system, config.experiment.name
    with Run(config) as run:
        exp = Experiment(config, tables_dir)
        if config.lstm.reuse_selection_from is not None:
            source = Path(str(config.lstm.reuse_selection_from).format(system=system))
            reused = json.loads(source.read_text(encoding="utf-8"))
            selection = {
                "forecaster": reused["selection"]["forecaster"],
                "classifier": reused["selection"]["classifier"],
                "reused_from": str(source),
            }
            run.logger.info("reusing selection %s", selection)
        else:
            t0 = time.perf_counter()
            selection = search(exp, run.logger.info)
            selection["search_seconds"] = round(time.perf_counter() - t0, 1)
            run.write_artifact("search.json", selection)
        scores, ranks, runs = final_runs(
            exp, selection, run.artifacts_dir / "models", run.logger.info
        )
        metrics = evaluate(exp, scores, ranks)
        metrics.update(
            {
                "system": system,
                "folds": len(exp.folds),
                "seeds": list(config.lstm.seeds),
                "selection": {k: selection[k] for k in ("forecaster", "classifier")},
            }
        )
        run.log_metrics(metrics)
        run.write_artifact("runs.json", {"runs": runs})
        scores.write_parquet(run.artifacts_dir / "test_window_scores.parquet")
        ranks.write_csv(run.artifacts_dir / "localization_ranks.csv")
        cost = _cost(runs)
        table = {
            **metrics,
            "cost": cost,
            "run_dir": run.dir.name,
            "search": {k: v for k, v in selection.items() if k != "table"},
        }
        tables_dir.mkdir(parents=True, exist_ok=True)
        (tables_dir / f"lstm_{system}_{name}.json").write_text(
            json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        write_lstm_markdown(
            tables_dir / f"lstm_{system}_{name}.md",
            table,
            _baseline_table(tables_dir, system, name),
        )
        plot_curves(Path("reports/figures") / f"lstm_curves_{system}_{name}.png", runs, system)
        return {"run_dir": str(run.dir), "selection": metrics["selection"]}


def _cost(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in MODELS:
        r = [x for x in runs if x["model"] == name]
        out[name] = {
            "params": r[0]["params"],
            "train_seconds_mean": round(float(np.mean([x["train_seconds"] for x in r])), 2),
            "train_seconds_total": round(float(np.sum([x["train_seconds"] for x in r])), 1),
            "epochs_mean": round(float(np.mean([x["epochs"] for x in r])), 1),
        }
        if name == "lstm_forecast":
            out[name]["inference_ms_per_window"] = round(
                float(np.mean([x["inference_ms_per_window"] for x in r])), 4
            )
    return out


def _baseline_table(tables_dir: Path, system: str, name: str) -> dict[str, Any] | None:
    """The E4 table run under the same selection config (``lstm`` <-> ``baselines``)."""
    path = tables_dir / f"baselines_{system}_{name.replace('lstm', 'baselines', 1)}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _ms(s: dict[str, float], pct: bool = False) -> str:
    k = 100.0 if pct else 1.0
    return f"{s['mean'] * k:.2f} ± {s['sd'] * k:.2f}"


def _ci(per_seed: list[dict[str, Any]], key: str) -> str:
    lo = float(np.mean([p[key]["ci_low"] for p in per_seed]))
    hi = float(np.mean([p[key]["ci_high"] for p in per_seed]))
    return f"[{lo:.2f}, {hi:.2f}]"


def write_lstm_markdown(
    path: Path, table: dict[str, Any], baselines: dict[str, Any] | None
) -> None:
    sel, n_seeds = table["selection"], len(table["seeds"])
    lines = [
        f"# LSTM — {table['system']} (E5, ADR 0015)",
        "",
        f"Configuração escolhida na validação: previsora {sel['forecaster']}; classificadora "
        f"{sel['classifier']}. Valores: média ± desvio-padrão entre {n_seeds} seeds; "
        '"IC médio" = média, entre seeds, dos limites do IC de 95% por bootstrap. '
        "Sem point-adjust.",
        "",
        "## Detecção (Tarefa A, por janela)",
        "",
        "| Método | AUC-PR | IC médio | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, d in table["detection"].items():
        s = d["summary"]
        lines.append(
            f"| {name} | {_ms(s['auc_pr'])} | {_ci(d['seeds'], 'auc_pr')} | {_ms(s['auc_roc'])} "
            f"| {_ms(s['precision'])} | {_ms(s['recall'])} | {_ms(s['f1'])} "
            f"| {_ms(s['false_alarms_per_hour'])} |"
        )
    if baselines is not None:
        for name, d in baselines["detection"].items():
            e = d["auc_pr"]
            lines.append(
                f"| E4: {name} | {e['value']:.2f} | [{e['ci_low']:.2f}, {e['ci_high']:.2f}] "
                f"| {d['auc_roc']['value']:.2f} | {d['precision']['value']:.2f} "
                f"| {d['recall']['value']:.2f} | {d['f1']['value']:.2f} "
                f"| {d['false_alarms_per_hour']['value']:.2f} |"
            )
        pos = next(iter(baselines["detection"].values()))["positive_rate"]
        lines += ["", f"AUC-PR ao acaso (taxa de positivos): {pos:.3f}."]
    lines += [
        "",
        "## Localização (Tarefa B): ranqueamento nativo da LSTM",
        "",
        "| Ranqueamento | HR@1 | HR@3 | HR@5 | MRR |",
        "|---|---|---|---|---|",
    ]
    labels = {"lstm_sum": "LSTM: soma dos erros", "lstm_top3": "LSTM: média dos 3 maiores erros"}
    for method, d in table["localization"].items():
        s = d["summary"]
        lines.append(
            f"| {labels[method]} | {_ms(s['hr@1'])} | {_ms(s['hr@3'])} | {_ms(s['hr@5'])} "
            f"| {_ms(s['mrr'])} |"
        )
    if baselines is not None:
        for key, label in (
            ("localization_prior", "E4: referência (frequência no treino)"),
            ("localization", "E4: z-score global"),
            ("localization_local", "E4: z-score local"),
        ):
            loc = baselines[key]
            lines.append(
                f"| {label} | {loc['hr@1']['value']:.2f} | {loc['hr@3']['value']:.2f} "
                f"| {loc['hr@5']['value']:.2f} | {loc['mrr']['value']:.2f} |"
            )
    cost = table["cost"]
    lines += [
        "",
        "## Custo (CPU, Apple M4 Pro)",
        "",
        "| Modelo | Parâmetros | Treino por dobra (s) | Épocas | Inferência (ms/janela) |",
        "|---|---|---|---|---|",
    ]
    for name, c in cost.items():
        inf = c.get("inference_ms_per_window", "—")
        lines.append(
            f"| {name} | {c['params']:,} | {c['train_seconds_mean']} | {c['epochs_mean']} | {inf} |"
        )
    search = table["search"]
    if "forecaster_val_auc_pr" in search:
        lines += [
            "",
            f"## Busca de hiperparâmetros (validação, 1 seed; {search['trainings']} treinos, "
            f"{search['search_seconds']} s)",
            "",
            "| Modelo | Configuração | AUC-PR média na validação |",
            "|---|---|---|",
        ]
        for model, key in (
            ("previsora", "forecaster_val_auc_pr"),
            ("classificadora", "classifier_val_auc_pr"),
        ):
            lines += [f"| {model} | {cfg} | {v:.3f} |" for cfg, v in search[key].items()]
    else:
        lines += ["", f"Seleção reaproveitada de `{search.get('reused_from')}` (sem nova busca)."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_curves(path: Path, runs: list[dict[str, Any]], system: str) -> None:
    """Training/validation loss of the final forecaster (first seed), one panel per fold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first_seed = min(r["seed"] for r in runs)
    sel = [r for r in runs if r["model"] == "lstm_forecast" and r["seed"] == first_seed]
    cols = 5
    rows = -(-len(sel) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.0 * rows), squeeze=False)
    for ax, r in zip(axes.flat, sel, strict=False):
        epochs = np.arange(1, len(r["train_loss"]) + 1)
        ax.plot(epochs, r["train_loss"], color="#2a78d6", linewidth=1.6, label="treino")
        ax.plot(epochs, r["val_loss"], color="#eb6834", linewidth=1.6, label="validação")
        ax.axvline(r["best_epoch"] + 1, color="#898781", linewidth=0.8)
        ax.set_title(f"dobra {r['fold']}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(sel) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6, frameon=False)
    fig.suptitle(
        f"LSTM previsora — {system}: perda (MSE) por época; linha cinza = época escolhida",
        fontsize=9,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
