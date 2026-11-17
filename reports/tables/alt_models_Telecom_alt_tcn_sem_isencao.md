# Modelos alternativos explicáveis — Telecom (família `tcn`, ADR 0018)

Mesmas janelas, dobras (10) e protocolo da E5/E6. Seleção na validação: tcn {"normalization": "local", "params": {"blocks": 2, "channels": 32}}. Detecção: média ± desvio-padrão entre seeds (a logística é determinística: 1 seed); as linhas `E4:` são execuções únicas, com IC de 95% por bootstrap. Todos os candidatos (alternativos, LSTMs da E5 e baselines da E4) são avaliados nas mesmas janelas e falhas.

## Detecção (por janela)

| Modelo | AUC-PR | AUC-PR por dobra | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|---|
| tcn | 0.37 ± 0.01 | 0.54 ± 0.04 | 0.74 ± 0.02 | 0.23 ± 0.02 | 0.46 ± 0.05 | 0.31 ± 0.02 | 14.91 ± 2.76 |
| E5: lstm_forecast | 0.43 ± 0.02 | 0.48 ± 0.02 | 0.80 ± 0.01 | 0.24 ± 0.02 | 0.43 ± 0.02 | 0.31 ± 0.01 | 13.24 ± 1.49 |
| E5: lstm_classifier | 0.49 ± 0.02 | 0.60 ± 0.01 | 0.84 ± 0.01 | 0.39 ± 0.04 | 0.55 ± 0.04 | 0.45 ± 0.03 | 8.63 ± 1.58 |
| E4: zscore_3sigma | 0.24 [0.16, 0.38] | 0.43 [0.38, 0.60] | 0.70 [0.61, 0.79] | 0.14 [0.10, 0.19] | 1.00 [1.00, 1.00] | 0.25 [0.18, 0.31] | 60.00 [60.00, 60.00] |
| E4: zscore_local | 0.38 [0.26, 0.53] | 0.48 [0.42, 0.68] | 0.75 [0.67, 0.82] | 0.28 [0.18, 0.40] | 0.39 [0.25, 0.53] | 0.33 [0.21, 0.44] | 9.77 [6.99, 12.73] |
| E4: isolation_forest | 0.18 [0.12, 0.32] | 0.42 [0.34, 0.58] | 0.63 [0.52, 0.73] | 0.20 [0.13, 0.28] | 0.69 [0.56, 0.82] | 0.31 [0.21, 0.41] | 27.89 [21.81, 34.66] |
| E4: lightgbm | 0.26 [0.15, 0.41] | 0.55 [0.50, 0.71] | 0.60 [0.49, 0.70] | 0.21 [0.13, 0.31] | 0.49 [0.35, 0.63] | 0.29 [0.20, 0.40] | 18.09 [12.63, 24.05] |
| E4: prior_horario | 0.15 [0.09, 0.27] | 0.28 [0.21, 0.47] | 0.44 [0.34, 0.55] | 0.15 [0.09, 0.22] | 0.48 [0.33, 0.63] | 0.22 [0.14, 0.31] | 27.15 [20.73, 33.57] |

## Localização via XAI (janelas-oráculo, seed explicada)

| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| tcn/shap/sum | 0.20 [0.09, 0.33] | 0.40 [0.27, 0.53] | 0.53 [0.40, 0.67] | 0.35 [0.25, 0.46] |
| tcn/shap/top3 | 0.18 [0.07, 0.31] | 0.42 [0.27, 0.56] | 0.58 [0.42, 0.73] | 0.35 [0.25, 0.45] |
| tcn/lime/sum | 0.22 [0.11, 0.36] | 0.40 [0.27, 0.53] | 0.51 [0.36, 0.67] | 0.36 [0.25, 0.48] |
| tcn/lime/top3 | 0.18 [0.07, 0.31] | 0.42 [0.27, 0.58] | 0.58 [0.42, 0.73] | 0.36 [0.26, 0.46] |
| E6: lstm_forecast/shap/sum | 0.24 [0.11, 0.38] | 0.62 [0.49, 0.76] | 0.76 [0.62, 0.87] | 0.46 [0.36, 0.56] |
| E6: lstm_forecast/lime/sum | 0.22 [0.11, 0.36] | 0.58 [0.42, 0.71] | 0.67 [0.51, 0.80] | 0.43 [0.33, 0.53] |
| E6: lstm_classifier/shap/sum | 0.22 [0.11, 0.36] | 0.42 [0.29, 0.56] | 0.53 [0.38, 0.67] | 0.37 [0.26, 0.48] |
| E6: lstm_classifier/lime/sum | 0.11 [0.02, 0.20] | 0.29 [0.16, 0.42] | 0.47 [0.31, 0.60] | 0.27 [0.19, 0.36] |
| E4: prior | 0.09 [0.02, 0.18] | 0.18 [0.09, 0.29] | 0.27 [0.13, 0.40] | 0.19 [0.12, 0.29] |
| E4: zscore_global | 0.13 [0.04, 0.24] | 0.38 [0.24, 0.51] | 0.56 [0.40, 0.71] | 0.31 [0.22, 0.41] |
| E4: zscore_local | 0.09 [0.02, 0.18] | 0.49 [0.33, 0.62] | 0.56 [0.42, 0.71] | 0.32 [0.25, 0.40] |

O ranqueamento nativo da LSTM (E5, sem XAI) está nas comparações pareadas abaixo (`lstm_nativo_soma`) e em `lstm_<sistema>_<nome>.md`.

## Comparações pareadas (diferença = alternativo menos referência)

Detecção: mesmas janelas de teste e mesma seed; localização: mesmas falhas. IC de 95% por bootstrap pareado; médias entre seeds e contagem de vereditos.

| Tarefa | Modelo | Referência | Métrica | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | tcn | lstm_forecast | auc_pr | -0.059 | [-0.211, +0.150] | 0 / 5 / 0 |
| detecção | tcn | lstm_classifier | auc_pr | -0.119 | [-0.242, +0.057] | 0 / 5 / 0 |
| detecção | tcn | zscore_3sigma | auc_pr | +0.133 | [-0.018, +0.329] | 1 / 4 / 0 |
| detecção | tcn | zscore_local | auc_pr | -0.014 | [-0.141, +0.174] | 0 / 5 / 0 |
| detecção | tcn | isolation_forest | auc_pr | +0.187 | [+0.037, +0.387] | 5 / 0 / 0 |
| detecção | tcn | lightgbm | auc_pr | +0.110 | [-0.022, +0.293] | 0 / 5 / 0 |
| detecção | tcn | prior_horario | auc_pr | +0.218 | [+0.073, +0.420] | 5 / 0 / 0 |
| localização | tcn/shap/sum | lstm_forecast/shap | hr@1 | -0.044 | [-0.200, +0.111] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_forecast/shap | mrr | -0.112 | [-0.244, +0.018] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_classifier/shap | hr@1 | -0.022 | [-0.178, +0.133] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_classifier/shap | mrr | -0.018 | [-0.147, +0.107] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_nativo_soma | hr@1 | +0.022 | [-0.067, +0.133] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_nativo_soma | mrr | -0.051 | [-0.135, +0.030] | 0 / 1 / 0 |
| localização | tcn/shap/sum | prior | hr@1 | +0.111 | [-0.022, +0.244] | 0 / 1 / 0 |
| localização | tcn/shap/sum | prior | mrr | +0.155 | [+0.038, +0.271] | 1 / 0 / 0 |
| localização | tcn/shap/sum | zscore_global | hr@1 | +0.067 | [-0.089, +0.222] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_global | mrr | +0.036 | [-0.104, +0.184] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_local | hr@1 | +0.111 | [+0.000, +0.222] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_local | mrr | +0.029 | [-0.060, +0.117] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_forecast/lime | hr@1 | +0.000 | [-0.156, +0.156] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_forecast/lime | mrr | -0.065 | [-0.184, +0.051] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_classifier/lime | hr@1 | +0.111 | [-0.022, +0.244] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_classifier/lime | mrr | +0.088 | [-0.015, +0.190] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_nativo_soma | hr@1 | +0.044 | [-0.044, +0.156] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_nativo_soma | mrr | -0.038 | [-0.116, +0.038] | 0 / 1 / 0 |
| localização | tcn/lime/sum | prior | hr@1 | +0.133 | [+0.022, +0.267] | 1 / 0 / 0 |
| localização | tcn/lime/sum | prior | mrr | +0.167 | [+0.064, +0.273] | 1 / 0 / 0 |
| localização | tcn/lime/sum | zscore_global | hr@1 | +0.089 | [-0.067, +0.267] | 0 / 1 / 0 |
| localização | tcn/lime/sum | zscore_global | mrr | +0.048 | [-0.089, +0.194] | 0 / 1 / 0 |
| localização | tcn/lime/sum | zscore_local | hr@1 | +0.133 | [+0.022, +0.267] | 1 / 0 / 0 |
| localização | tcn/lime/sum | zscore_local | mrr | +0.041 | [-0.043, +0.135] | 0 / 1 / 0 |

## Qualidade das explicações (médias por janela)

Deleção: menor = mais fiel; inserção (relativa à janela completa): maior = mais fiel; `random` = ordens aleatórias. Jaccard/Kendall nos top-10 atributos.

| Medida | tcn |
|---|---|
| deletion_shap | 0.612 |
| deletion_lime | 0.618 |
| deletion_random | 1.004 |
| insertion_shap | 2.336 |
| insertion_lime | 2.379 |
| insertion_random | 1.699 |
| shap_lime_jaccard | 0.608 |
| shap_lime_kendall | 0.490 |
| neighbour_jaccard | 0.791 |
| top_component_agree | 0.800 |
| seed_jaccard | 0.300 |
| seed_kendall | -0.142 |

## Custo (CPU)

| Modelo | Medida | Valor |
|---|---|---|
| tcn | train_seconds_mean | 7.45 |
| tcn | inference_ms_per_window | 0.4566 |
| tcn | xai_seconds_per_window | 2.76 |
| tcn | params_mean | 200154.6 |
| tcn | epochs_mean | 8.3 |

Seleção reaproveitada de `reports/tables/alt_models_Telecom_alt_tcn.json` (sem nova busca).
