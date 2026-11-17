# Modelos alternativos explicáveis — Bank (família `tcn`, ADR 0018)

Mesmas janelas, dobras (8) e protocolo da E5/E6. Seleção na validação: tcn {"normalization": "local", "params": {"blocks": 3, "channels": 32}}. Detecção: média ± desvio-padrão entre seeds (a logística é determinística: 1 seed); as linhas `E4:` são execuções únicas, com IC de 95% por bootstrap. Todos os candidatos (alternativos, LSTMs da E5 e baselines da E4) são avaliados nas mesmas janelas e falhas.

## Detecção (por janela)

| Modelo | AUC-PR | AUC-PR por dobra | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|---|
| tcn | 0.17 ± 0.01 | 0.20 ± 0.01 | 0.53 ± 0.03 | 0.15 ± 0.01 | 0.42 ± 0.08 | 0.22 ± 0.01 | 17.10 ± 3.71 |
| E5: lstm_forecast | 0.34 ± 0.00 | 0.38 ± 0.00 | 0.78 ± 0.00 | 0.31 ± 0.02 | 0.36 ± 0.02 | 0.33 ± 0.01 | 5.76 ± 0.83 |
| E5: lstm_classifier | 0.21 ± 0.01 | 0.25 ± 0.01 | 0.65 ± 0.01 | 0.16 ± 0.01 | 0.39 ± 0.05 | 0.23 ± 0.01 | 14.12 ± 2.17 |
| E4: zscore_3sigma | 0.10 [0.08, 0.13] | 0.31 [0.26, 0.37] | 0.48 [0.42, 0.54] | 0.11 [0.09, 0.13] | 1.00 [1.00, 1.00] | 0.19 [0.16, 0.23] | 60.00 [60.00, 60.00] |
| E4: zscore_local | 0.33 [0.26, 0.41] | 0.35 [0.30, 0.45] | 0.79 [0.75, 0.83] | 0.37 [0.29, 0.45] | 0.40 [0.32, 0.48] | 0.39 [0.31, 0.46] | 4.77 [3.85, 5.79] |
| E4: isolation_forest | 0.14 [0.11, 0.19] | 0.20 [0.17, 0.30] | 0.60 [0.54, 0.66] | 0.14 [0.11, 0.17] | 0.56 [0.48, 0.64] | 0.22 [0.18, 0.27] | 24.66 [21.03, 28.18] |
| E4: lightgbm | 0.18 [0.12, 0.26] | 0.25 [0.19, 0.33] | 0.51 [0.44, 0.58] | 0.16 [0.12, 0.21] | 0.42 [0.33, 0.50] | 0.24 [0.18, 0.29] | 14.92 [11.98, 18.17] |
| E4: prior_horario | 0.12 [0.10, 0.16] | 0.18 [0.15, 0.27] | 0.55 [0.49, 0.61] | 0.15 [0.11, 0.20] | 0.44 [0.35, 0.53] | 0.22 [0.17, 0.28] | 17.63 [14.42, 21.23] |

## Localização via XAI (janelas-oráculo, seed explicada)

| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| tcn/shap/sum | 0.33 [0.24, 0.41] | 0.57 [0.48, 0.66] | 0.70 [0.62, 0.78] | 0.49 [0.43, 0.56] |
| tcn/shap/top3 | 0.42 [0.33, 0.51] | 0.60 [0.51, 0.69] | 0.67 [0.58, 0.75] | 0.55 [0.48, 0.62] |
| tcn/lime/sum | 0.24 [0.17, 0.32] | 0.50 [0.41, 0.59] | 0.68 [0.60, 0.77] | 0.42 [0.36, 0.48] |
| tcn/lime/top3 | 0.38 [0.29, 0.47] | 0.59 [0.51, 0.68] | 0.69 [0.61, 0.77] | 0.53 [0.47, 0.60] |
| E6: lstm_forecast/shap/sum | 0.50 [0.42, 0.59] | 0.68 [0.59, 0.76] | 0.78 [0.71, 0.85] | 0.62 [0.55, 0.69] |
| E6: lstm_forecast/lime/sum | 0.44 [0.35, 0.53] | 0.65 [0.57, 0.73] | 0.79 [0.72, 0.86] | 0.58 [0.52, 0.65] |
| E6: lstm_classifier/shap/sum | 0.43 [0.34, 0.52] | 0.72 [0.63, 0.80] | 0.86 [0.79, 0.92] | 0.61 [0.54, 0.67] |
| E6: lstm_classifier/lime/sum | 0.21 [0.14, 0.28] | 0.56 [0.47, 0.64] | 0.77 [0.69, 0.84] | 0.43 [0.37, 0.49] |
| E4: prior | 0.11 [0.06, 0.17] | 0.27 [0.18, 0.35] | 0.38 [0.30, 0.47] | 0.26 [0.21, 0.31] |
| E4: zscore_global | 0.32 [0.23, 0.40] | 0.48 [0.39, 0.57] | 0.68 [0.59, 0.76] | 0.47 [0.41, 0.54] |
| E4: zscore_local | 0.11 [0.06, 0.17] | 0.47 [0.38, 0.56] | 0.66 [0.57, 0.74] | 0.34 [0.29, 0.39] |

O ranqueamento nativo da LSTM (E5, sem XAI) está nas comparações pareadas abaixo (`lstm_nativo_soma`) e em `lstm_<sistema>_<nome>.md`.

## Comparações pareadas (diferença = alternativo menos referência)

Detecção: mesmas janelas de teste e mesma seed; localização: mesmas falhas. IC de 95% por bootstrap pareado; médias entre seeds e contagem de vereditos.

| Tarefa | Modelo | Referência | Métrica | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | tcn | lstm_forecast | auc_pr | -0.168 | [-0.237, -0.092] | 0 / 0 / 5 |
| detecção | tcn | lstm_classifier | auc_pr | -0.038 | [-0.087, +0.027] | 0 / 5 / 0 |
| detecção | tcn | zscore_3sigma | auc_pr | +0.073 | [+0.021, +0.157] | 5 / 0 / 0 |
| detecção | tcn | zscore_local | auc_pr | -0.154 | [-0.226, -0.067] | 0 / 0 / 5 |
| detecção | tcn | isolation_forest | auc_pr | +0.034 | [-0.030, +0.120] | 0 / 5 / 0 |
| detecção | tcn | lightgbm | auc_pr | -0.003 | [-0.081, +0.084] | 0 / 5 / 0 |
| detecção | tcn | prior_horario | auc_pr | +0.048 | [-0.007, +0.134] | 1 / 4 / 0 |
| localização | tcn/shap/sum | lstm_forecast/shap | hr@1 | -0.175 | [-0.275, -0.067] | 0 / 0 / 1 |
| localização | tcn/shap/sum | lstm_forecast/shap | mrr | -0.129 | [-0.204, -0.054] | 0 / 0 / 1 |
| localização | tcn/shap/sum | lstm_classifier/shap | hr@1 | -0.108 | [-0.200, -0.017] | 0 / 0 / 1 |
| localização | tcn/shap/sum | lstm_classifier/shap | mrr | -0.112 | [-0.173, -0.052] | 0 / 0 / 1 |
| localização | tcn/shap/sum | lstm_nativo_soma | hr@1 | -0.008 | [-0.117, +0.100] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_nativo_soma | mrr | -0.018 | [-0.092, +0.060] | 0 / 1 / 0 |
| localização | tcn/shap/sum | prior | hr@1 | +0.217 | [+0.117, +0.317] | 1 / 0 / 0 |
| localização | tcn/shap/sum | prior | mrr | +0.236 | [+0.160, +0.312] | 1 / 0 / 0 |
| localização | tcn/shap/sum | zscore_global | hr@1 | +0.008 | [-0.092, +0.108] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_global | mrr | +0.023 | [-0.053, +0.100] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_local | hr@1 | +0.217 | [+0.117, +0.308] | 1 / 0 / 0 |
| localização | tcn/shap/sum | zscore_local | mrr | +0.153 | [+0.087, +0.217] | 1 / 0 / 0 |
| localização | tcn/lime/sum | lstm_forecast/lime | hr@1 | -0.200 | [-0.300, -0.092] | 0 / 0 / 1 |
| localização | tcn/lime/sum | lstm_forecast/lime | mrr | -0.162 | [-0.236, -0.084] | 0 / 0 / 1 |
| localização | tcn/lime/sum | lstm_classifier/lime | hr@1 | +0.033 | [-0.042, +0.108] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_classifier/lime | mrr | -0.006 | [-0.062, +0.050] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_nativo_soma | hr@1 | -0.092 | [-0.192, +0.008] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_nativo_soma | mrr | -0.090 | [-0.165, -0.014] | 0 / 0 / 1 |
| localização | tcn/lime/sum | prior | hr@1 | +0.133 | [+0.050, +0.233] | 1 / 0 / 0 |
| localização | tcn/lime/sum | prior | mrr | +0.164 | [+0.092, +0.239] | 1 / 0 / 0 |
| localização | tcn/lime/sum | zscore_global | hr@1 | -0.075 | [-0.175, +0.033] | 0 / 1 / 0 |
| localização | tcn/lime/sum | zscore_global | mrr | -0.049 | [-0.124, +0.029] | 0 / 1 / 0 |
| localização | tcn/lime/sum | zscore_local | hr@1 | +0.133 | [+0.042, +0.225] | 1 / 0 / 0 |
| localização | tcn/lime/sum | zscore_local | mrr | +0.081 | [+0.017, +0.145] | 1 / 0 / 0 |

## Qualidade das explicações (médias por janela)

Deleção: menor = mais fiel; inserção (relativa à janela completa): maior = mais fiel; `random` = ordens aleatórias. Jaccard/Kendall nos top-10 atributos.

| Medida | tcn |
|---|---|
| deletion_shap | 0.555 |
| deletion_lime | 0.568 |
| deletion_random | 1.019 |
| insertion_shap | 147.562 |
| insertion_lime | 139.137 |
| insertion_random | 118.829 |
| shap_lime_jaccard | 0.583 |
| shap_lime_kendall | 0.454 |
| neighbour_jaccard | 0.790 |
| top_component_agree | 0.633 |
| seed_jaccard | 0.249 |
| seed_kendall | -0.209 |

## Custo (CPU)

| Modelo | Medida | Valor |
|---|---|---|
| tcn | train_seconds_mean | 23.39 |
| tcn | inference_ms_per_window | 0.4748 |
| tcn | xai_seconds_per_window | 2.91 |
| tcn | params_mean | 176689.0 |
| tcn | epochs_mean | 6.4 |

Seleção reaproveitada de `reports/tables/alt_models_Bank_alt_tcn.json` (sem nova busca).
