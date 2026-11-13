# Modelos alternativos explicáveis — Telecom (família `tcn`, ADR 0018)

Mesmas janelas, dobras (10) e protocolo da E5/E6. Seleção na validação: tcn {"normalization": "local", "params": {"blocks": 2, "channels": 32}}. Detecção: média ± desvio-padrão entre seeds (a logística é determinística: 1 seed).

## Detecção (por janela)

| Modelo | AUC-PR | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|
| tcn | 0.37 ± 0.01 | 0.74 ± 0.02 | 0.23 ± 0.02 | 0.46 ± 0.05 | 0.31 ± 0.02 | 14.91 ± 2.76 |
| lstm_forecast | 0.31 ± 0.02 | 0.71 ± 0.01 | 0.19 ± 0.04 | 0.48 ± 0.05 | 0.28 ± 0.04 | 19.94 ± 3.49 |
| lstm_classifier | 0.49 ± 0.02 | 0.84 ± 0.01 | 0.39 ± 0.04 | 0.55 ± 0.04 | 0.45 ± 0.03 | 8.63 ± 1.58 |

## Localização via XAI (janelas-oráculo, seed explicada)

| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| tcn/shap/sum | 0.22 [0.11, 0.36] | 0.40 [0.27, 0.53] | 0.53 [0.40, 0.67] | 0.36 [0.26, 0.47] |
| tcn/shap/top3 | 0.18 [0.07, 0.31] | 0.42 [0.27, 0.56] | 0.58 [0.42, 0.73] | 0.35 [0.25, 0.45] |
| tcn/lime/sum | 0.24 [0.13, 0.38] | 0.40 [0.27, 0.53] | 0.51 [0.36, 0.67] | 0.37 [0.26, 0.49] |
| tcn/lime/top3 | 0.18 [0.07, 0.31] | 0.44 [0.29, 0.58] | 0.58 [0.42, 0.73] | 0.36 [0.26, 0.47] |

## Comparações pareadas (diferença = alternativo menos referência)

Detecção: mesmas janelas de teste e mesma seed; localização: mesmas falhas. IC de 95% por bootstrap pareado; médias entre seeds e contagem de vereditos.

| Tarefa | Modelo | Referência | Métrica | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | tcn | lstm_forecast | auc_pr | +0.060 | [-0.108, +0.262] | 0 / 5 / 0 |
| detecção | tcn | lstm_classifier | auc_pr | -0.119 | [-0.242, +0.057] | 0 / 5 / 0 |
| localização | tcn/shap/sum | lstm_forecast/shap | hr@1 | +0.067 | [-0.089, +0.222] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_forecast/shap | mrr | +0.004 | [-0.128, +0.136] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_classifier/shap | hr@1 | -0.022 | [-0.156, +0.111] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_classifier/shap | mrr | -0.025 | [-0.145, +0.093] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_nativo_soma | hr@1 | +0.022 | [-0.044, +0.111] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_nativo_soma | mrr | -0.050 | [-0.122, +0.019] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_global | hr@1 | +0.089 | [-0.067, +0.244] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_global | mrr | +0.062 | [-0.083, +0.203] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_local | hr@1 | +0.111 | [+0.000, +0.222] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_local | mrr | +0.029 | [-0.059, +0.120] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_forecast/lime | hr@1 | +0.089 | [-0.044, +0.244] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_forecast/lime | mrr | +0.037 | [-0.078, +0.148] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_classifier/lime | hr@1 | +0.111 | [+0.000, +0.244] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_classifier/lime | mrr | +0.082 | [-0.018, +0.181] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_nativo_soma | hr@1 | +0.044 | [-0.044, +0.133] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_nativo_soma | mrr | -0.042 | [-0.112, +0.025] | 0 / 1 / 0 |
| localização | tcn/lime/sum | zscore_global | hr@1 | +0.111 | [-0.044, +0.289] | 0 / 1 / 0 |
| localização | tcn/lime/sum | zscore_global | mrr | +0.071 | [-0.070, +0.211] | 0 / 1 / 0 |
| localização | tcn/lime/sum | zscore_local | hr@1 | +0.133 | [+0.022, +0.244] | 1 / 0 / 0 |
| localização | tcn/lime/sum | zscore_local | mrr | +0.037 | [-0.049, +0.130] | 0 / 1 / 0 |

## Qualidade das explicações (médias por janela)

Deleção: menor = mais fiel; inserção (relativa à janela completa): maior = mais fiel; `random` = ordens aleatórias. Jaccard/Kendall nos top-10 atributos.

| Medida | tcn |
|---|---|
| deletion_shap | 0.615 |
| deletion_lime | 0.618 |
| deletion_random | 1.002 |
| insertion_shap | 2.066 |
| insertion_lime | 2.090 |
| insertion_random | 1.491 |
| shap_lime_jaccard | 0.612 |
| shap_lime_kendall | 0.499 |
| neighbour_jaccard | 0.793 |
| top_component_agree | 0.844 |
| seed_jaccard | 0.301 |
| seed_kendall | -0.142 |

## Custo (CPU)

| Modelo | Medida | Valor |
|---|---|---|
| tcn | train_seconds_mean | 8.81 |
| tcn | inference_ms_per_window | 0.558 |
| tcn | xai_seconds_per_window | 3.4 |
| tcn | params_mean | 200154.6 |
| tcn | epochs_mean | 8.3 |

Seleção reaproveitada de `reports/tables/alt_models_Telecom_alt_tcn.json` (sem nova busca).
