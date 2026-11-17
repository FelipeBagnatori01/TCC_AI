# Modelos alternativos explicáveis — Bank (família `tcn`, ADR 0018)

Mesmas janelas, dobras (8) e protocolo da E5/E6. Seleção na validação: tcn {"normalization": "local", "params": {"blocks": 3, "channels": 32}}. Detecção: média ± desvio-padrão entre seeds (a logística é determinística: 1 seed); as linhas `E4:` são execuções únicas, com IC de 95% por bootstrap. Todos os candidatos (alternativos, LSTMs da E5 e baselines da E4) são avaliados nas mesmas janelas e falhas.

## Detecção (por janela)

| Modelo | AUC-PR | AUC-PR por dobra | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|---|
| tcn | 0.16 ± 0.01 | 0.20 ± 0.02 | 0.52 ± 0.02 | 0.14 ± 0.01 | 0.43 ± 0.04 | 0.21 ± 0.02 | 18.73 ± 2.00 |
| E5: lstm_forecast | 0.35 ± 0.00 | 0.38 ± 0.01 | 0.78 ± 0.00 | 0.31 ± 0.01 | 0.35 ± 0.02 | 0.33 ± 0.01 | 5.67 ± 0.36 |
| E5: lstm_classifier | 0.22 ± 0.01 | 0.26 ± 0.01 | 0.66 ± 0.01 | 0.17 ± 0.01 | 0.41 ± 0.03 | 0.24 ± 0.02 | 14.32 ± 0.56 |
| E4: zscore_3sigma | 0.10 [0.08, 0.13] | 0.31 [0.26, 0.37] | 0.48 [0.42, 0.54] | 0.11 [0.09, 0.13] | 1.00 [1.00, 1.00] | 0.19 [0.16, 0.23] | 60.00 [60.00, 60.00] |
| E4: zscore_local | 0.33 [0.26, 0.41] | 0.35 [0.30, 0.45] | 0.79 [0.75, 0.83] | 0.37 [0.29, 0.45] | 0.40 [0.32, 0.48] | 0.39 [0.31, 0.46] | 4.77 [3.85, 5.79] |
| E4: isolation_forest | 0.14 [0.11, 0.21] | 0.21 [0.17, 0.30] | 0.57 [0.51, 0.63] | 0.14 [0.11, 0.18] | 0.62 [0.54, 0.71] | 0.23 [0.18, 0.28] | 26.87 [23.28, 30.58] |
| E4: lightgbm | 0.19 [0.13, 0.27] | 0.27 [0.21, 0.35] | 0.51 [0.45, 0.58] | 0.15 [0.10, 0.20] | 0.33 [0.25, 0.41] | 0.20 [0.15, 0.26] | 13.34 [10.55, 16.50] |
| E4: prior_horario | 0.12 [0.10, 0.16] | 0.18 [0.15, 0.27] | 0.55 [0.49, 0.61] | 0.15 [0.11, 0.20] | 0.44 [0.35, 0.53] | 0.22 [0.17, 0.28] | 17.63 [14.42, 21.23] |

## Localização via XAI (janelas-oráculo, seed explicada)

| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| tcn/shap/sum | 0.25 [0.17, 0.33] | 0.49 [0.40, 0.57] | 0.66 [0.57, 0.74] | 0.42 [0.36, 0.48] |
| tcn/shap/top3 | 0.41 [0.33, 0.50] | 0.57 [0.47, 0.66] | 0.62 [0.53, 0.71] | 0.53 [0.46, 0.60] |
| tcn/lime/sum | 0.18 [0.12, 0.26] | 0.41 [0.33, 0.49] | 0.55 [0.46, 0.63] | 0.36 [0.30, 0.42] |
| tcn/lime/top3 | 0.38 [0.29, 0.46] | 0.57 [0.47, 0.66] | 0.63 [0.55, 0.72] | 0.51 [0.45, 0.59] |
| E6: lstm_forecast/shap/sum | 0.48 [0.40, 0.57] | 0.66 [0.57, 0.74] | 0.79 [0.72, 0.87] | 0.61 [0.54, 0.68] |
| E6: lstm_forecast/lime/sum | 0.45 [0.37, 0.54] | 0.64 [0.56, 0.73] | 0.76 [0.68, 0.83] | 0.59 [0.51, 0.65] |
| E6: lstm_classifier/shap/sum | 0.44 [0.36, 0.53] | 0.73 [0.65, 0.81] | 0.84 [0.78, 0.90] | 0.61 [0.55, 0.68] |
| E6: lstm_classifier/lime/sum | 0.22 [0.15, 0.29] | 0.52 [0.42, 0.60] | 0.75 [0.67, 0.82] | 0.42 [0.37, 0.48] |
| E4: prior | 0.11 [0.06, 0.17] | 0.27 [0.18, 0.35] | 0.38 [0.30, 0.47] | 0.26 [0.21, 0.31] |
| E4: zscore_global | 0.32 [0.23, 0.40] | 0.48 [0.39, 0.57] | 0.68 [0.59, 0.76] | 0.47 [0.41, 0.54] |
| E4: zscore_local | 0.11 [0.06, 0.17] | 0.47 [0.38, 0.56] | 0.66 [0.57, 0.74] | 0.34 [0.29, 0.39] |

O ranqueamento nativo da LSTM (E5, sem XAI) está nas comparações pareadas abaixo (`lstm_nativo_soma`) e em `lstm_<sistema>_<nome>.md`.

## Comparações pareadas (diferença = alternativo menos referência)

Detecção: mesmas janelas de teste e mesma seed; localização: mesmas falhas. IC de 95% por bootstrap pareado; médias entre seeds e contagem de vereditos.

| Tarefa | Modelo | Referência | Métrica | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | tcn | lstm_forecast | auc_pr | -0.186 | [-0.255, -0.111] | 0 / 0 / 5 |
| detecção | tcn | lstm_classifier | auc_pr | -0.059 | [-0.109, +0.005] | 0 / 4 / 1 |
| detecção | tcn | zscore_3sigma | auc_pr | +0.061 | [+0.014, +0.140] | 5 / 0 / 0 |
| detecção | tcn | zscore_local | auc_pr | -0.166 | [-0.236, -0.080] | 0 / 0 / 5 |
| detecção | tcn | isolation_forest | auc_pr | +0.020 | [-0.051, +0.103] | 0 / 5 / 0 |
| detecção | tcn | lightgbm | auc_pr | -0.029 | [-0.098, +0.057] | 0 / 5 / 0 |
| detecção | tcn | prior_horario | auc_pr | +0.036 | [-0.015, +0.118] | 0 / 5 / 0 |
| localização | tcn/shap/sum | lstm_forecast/shap | hr@1 | -0.233 | [-0.333, -0.125] | 0 / 0 / 1 |
| localização | tcn/shap/sum | lstm_forecast/shap | mrr | -0.195 | [-0.274, -0.116] | 0 / 0 / 1 |
| localização | tcn/shap/sum | lstm_classifier/shap | hr@1 | -0.192 | [-0.283, -0.092] | 0 / 0 / 1 |
| localização | tcn/shap/sum | lstm_classifier/shap | mrr | -0.195 | [-0.258, -0.126] | 0 / 0 / 1 |
| localização | tcn/shap/sum | lstm_nativo_soma | hr@1 | -0.092 | [-0.192, +0.008] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_nativo_soma | mrr | -0.100 | [-0.170, -0.027] | 0 / 0 / 1 |
| localização | tcn/shap/sum | prior | hr@1 | +0.142 | [+0.058, +0.225] | 1 / 0 / 0 |
| localização | tcn/shap/sum | prior | mrr | +0.160 | [+0.091, +0.232] | 1 / 0 / 0 |
| localização | tcn/shap/sum | zscore_global | hr@1 | -0.067 | [-0.167, +0.033] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_global | mrr | -0.053 | [-0.131, +0.026] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_local | hr@1 | +0.142 | [+0.058, +0.233] | 1 / 0 / 0 |
| localização | tcn/shap/sum | zscore_local | mrr | +0.077 | [+0.019, +0.142] | 1 / 0 / 0 |
| localização | tcn/lime/sum | lstm_forecast/lime | hr@1 | -0.267 | [-0.358, -0.167] | 0 / 0 / 1 |
| localização | tcn/lime/sum | lstm_forecast/lime | mrr | -0.226 | [-0.301, -0.149] | 0 / 0 / 1 |
| localização | tcn/lime/sum | lstm_classifier/lime | hr@1 | -0.033 | [-0.108, +0.042] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_classifier/lime | mrr | -0.064 | [-0.117, -0.011] | 0 / 0 / 1 |
| localização | tcn/lime/sum | lstm_nativo_soma | hr@1 | -0.158 | [-0.242, -0.075] | 0 / 0 / 1 |
| localização | tcn/lime/sum | lstm_nativo_soma | mrr | -0.160 | [-0.228, -0.090] | 0 / 0 / 1 |
| localização | tcn/lime/sum | prior | hr@1 | +0.075 | [-0.008, +0.158] | 0 / 1 / 0 |
| localização | tcn/lime/sum | prior | mrr | +0.100 | [+0.033, +0.171] | 1 / 0 / 0 |
| localização | tcn/lime/sum | zscore_global | hr@1 | -0.133 | [-0.233, -0.025] | 0 / 0 / 1 |
| localização | tcn/lime/sum | zscore_global | mrr | -0.113 | [-0.192, -0.030] | 0 / 0 / 1 |
| localização | tcn/lime/sum | zscore_local | hr@1 | +0.075 | [+0.000, +0.158] | 0 / 1 / 0 |
| localização | tcn/lime/sum | zscore_local | mrr | +0.017 | [-0.040, +0.079] | 0 / 1 / 0 |

## Qualidade das explicações (médias por janela)

Deleção: menor = mais fiel; inserção (relativa à janela completa): maior = mais fiel; `random` = ordens aleatórias. Jaccard/Kendall nos top-10 atributos.

| Medida | tcn |
|---|---|
| deletion_shap | 0.540 |
| deletion_lime | 0.557 |
| deletion_random | 1.038 |
| insertion_shap | 149958.045 |
| insertion_lime | 132147.265 |
| insertion_random | 122013.895 |
| shap_lime_jaccard | 0.541 |
| shap_lime_kendall | 0.355 |
| neighbour_jaccard | 0.779 |
| top_component_agree | 0.642 |
| seed_jaccard | 0.261 |
| seed_kendall | -0.192 |

## Custo (CPU)

| Modelo | Medida | Valor |
|---|---|---|
| tcn | train_seconds_mean | 24.18 |
| tcn | inference_ms_per_window | 0.4844 |
| tcn | xai_seconds_per_window | 2.98 |
| tcn | params_mean | 181041.0 |
| tcn | epochs_mean | 6.6 |

## Busca (validação, 1 seed; 56 treinos, 1618.0 s)

| Modelo | Configuração | AUC-PR média na validação |
|---|---|---|
| tcn | global/blocks=2/channels=32 | 0.169 |
| tcn | global/blocks=3/channels=32 | 0.190 |
| tcn | global/blocks=2/channels=64 | 0.168 |
| tcn | global/blocks=3/channels=64 | 0.137 |
| tcn | local/blocks=2/channels=32 | 0.217 |
| tcn | local/blocks=3/channels=32 | 0.249 |
| tcn | local/blocks=2/channels=64 | 0.245 |
| tcn | local/blocks=3/channels=64 | 0.243 |
