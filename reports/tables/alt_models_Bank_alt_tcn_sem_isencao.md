# Modelos alternativos explicáveis — Bank (família `tcn`, ADR 0018)

Mesmas janelas, dobras (8) e protocolo da E5/E6. Seleção na validação: tcn {"normalization": "local", "params": {"blocks": 3, "channels": 32}}. Detecção: média ± desvio-padrão entre seeds (a logística é determinística: 1 seed).

## Detecção (por janela)

| Modelo | AUC-PR | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|
| tcn | 0.17 ± 0.01 | 0.53 ± 0.03 | 0.15 ± 0.01 | 0.42 ± 0.08 | 0.22 ± 0.01 | 17.10 ± 3.71 |
| lstm_forecast | 0.20 ± 0.00 | 0.70 ± 0.00 | 0.22 ± 0.01 | 0.39 ± 0.03 | 0.28 ± 0.01 | 9.98 ± 0.50 |
| lstm_classifier | 0.21 ± 0.01 | 0.65 ± 0.01 | 0.16 ± 0.01 | 0.39 ± 0.05 | 0.23 ± 0.01 | 14.12 ± 2.17 |

## Localização via XAI (janelas-oráculo, seed explicada)

| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| tcn/shap/sum | 0.33 [0.24, 0.41] | 0.57 [0.48, 0.66] | 0.69 [0.61, 0.77] | 0.49 [0.43, 0.56] |
| tcn/shap/top3 | 0.42 [0.33, 0.51] | 0.60 [0.51, 0.69] | 0.67 [0.58, 0.75] | 0.55 [0.48, 0.62] |
| tcn/lime/sum | 0.24 [0.17, 0.32] | 0.50 [0.41, 0.59] | 0.68 [0.59, 0.76] | 0.42 [0.36, 0.48] |
| tcn/lime/top3 | 0.38 [0.29, 0.47] | 0.59 [0.51, 0.68] | 0.69 [0.61, 0.77] | 0.53 [0.47, 0.60] |

## Comparações pareadas (diferença = alternativo menos referência)

Detecção: mesmas janelas de teste e mesma seed; localização: mesmas falhas. IC de 95% por bootstrap pareado; médias entre seeds e contagem de vereditos.

| Tarefa | Modelo | Referência | Métrica | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | tcn | lstm_forecast | auc_pr | -0.032 | [-0.097, +0.036] | 0 / 5 / 0 |
| detecção | tcn | lstm_classifier | auc_pr | -0.038 | [-0.087, +0.027] | 0 / 5 / 0 |
| localização | tcn/shap/sum | lstm_forecast/shap | hr@1 | -0.033 | [-0.133, +0.083] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_forecast/shap | mrr | -0.011 | [-0.090, +0.074] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_classifier/shap | hr@1 | -0.108 | [-0.200, -0.017] | 0 / 0 / 1 |
| localização | tcn/shap/sum | lstm_classifier/shap | mrr | -0.113 | [-0.174, -0.054] | 0 / 0 / 1 |
| localização | tcn/shap/sum | lstm_nativo_soma | hr@1 | +0.000 | [-0.100, +0.108] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_nativo_soma | mrr | -0.016 | [-0.091, +0.061] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_global | hr@1 | +0.008 | [-0.092, +0.108] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_global | mrr | +0.021 | [-0.055, +0.098] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_local | hr@1 | +0.217 | [+0.117, +0.308] | 1 / 0 / 0 |
| localização | tcn/shap/sum | zscore_local | mrr | +0.155 | [+0.088, +0.219] | 1 / 0 / 0 |
| localização | tcn/lime/sum | lstm_forecast/lime | hr@1 | -0.083 | [-0.183, +0.025] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_forecast/lime | mrr | -0.051 | [-0.129, +0.034] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_classifier/lime | hr@1 | +0.033 | [-0.042, +0.108] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_classifier/lime | mrr | -0.005 | [-0.061, +0.051] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_nativo_soma | hr@1 | -0.083 | [-0.183, +0.017] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_nativo_soma | mrr | -0.088 | [-0.163, -0.012] | 0 / 0 / 1 |
| localização | tcn/lime/sum | zscore_global | hr@1 | -0.075 | [-0.175, +0.033] | 0 / 1 / 0 |
| localização | tcn/lime/sum | zscore_global | mrr | -0.050 | [-0.125, +0.028] | 0 / 1 / 0 |
| localização | tcn/lime/sum | zscore_local | hr@1 | +0.133 | [+0.042, +0.225] | 1 / 0 / 0 |
| localização | tcn/lime/sum | zscore_local | mrr | +0.083 | [+0.021, +0.146] | 1 / 0 / 0 |

## Qualidade das explicações (médias por janela)

Deleção: menor = mais fiel; inserção (relativa à janela completa): maior = mais fiel; `random` = ordens aleatórias. Jaccard/Kendall nos top-10 atributos.

| Medida | tcn |
|---|---|
| deletion_shap | 0.552 |
| deletion_lime | 0.564 |
| deletion_random | 1.019 |
| insertion_shap | 147.402 |
| insertion_lime | 138.968 |
| insertion_random | 118.694 |
| shap_lime_jaccard | 0.586 |
| shap_lime_kendall | 0.454 |
| neighbour_jaccard | 0.795 |
| top_component_agree | 0.642 |
| seed_jaccard | 0.249 |
| seed_kendall | -0.209 |

## Custo (CPU)

| Modelo | Medida | Valor |
|---|---|---|
| tcn | train_seconds_mean | 25.08 |
| tcn | inference_ms_per_window | 0.5399 |
| tcn | xai_seconds_per_window | 3.23 |
| tcn | params_mean | 176689.0 |
| tcn | epochs_mean | 6.4 |

Seleção reaproveitada de `reports/tables/alt_models_Bank_alt_tcn.json` (sem nova busca).
