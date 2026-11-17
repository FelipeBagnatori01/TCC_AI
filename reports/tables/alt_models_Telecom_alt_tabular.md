# Modelos alternativos explicáveis — Telecom (família `tabular`, ADR 0018)

Mesmas janelas, dobras (10) e protocolo da E5/E6. Seleção na validação: lightgbm {"normalization": "global", "params": {"n_estimators": 600, "num_leaves": 31}}; isolation_forest {"normalization": "global", "params": {"max_samples": 1024, "n_estimators": 300}}; logistic {"normalization": "global", "params": {"c": 10.0}}. Detecção: média ± desvio-padrão entre seeds (a logística é determinística: 1 seed); as linhas `E4:` são execuções únicas, com IC de 95% por bootstrap. Todos os candidatos (alternativos, LSTMs da E5 e baselines da E4) são avaliados nas mesmas janelas e falhas.

## Detecção (por janela)

| Modelo | AUC-PR | AUC-PR por dobra | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|---|
| lightgbm | 0.20 ± 0.01 | 0.55 ± 0.03 | 0.56 ± 0.01 | 0.20 ± 0.01 | 0.53 ± 0.02 | 0.29 ± 0.01 | 20.85 ± 1.11 |
| isolation_forest | 0.17 ± 0.00 | 0.35 ± 0.02 | 0.60 ± 0.01 | 0.17 ± 0.01 | 0.65 ± 0.04 | 0.27 ± 0.01 | 30.98 ± 3.26 |
| logistic | 0.51 ± 0.00 | 0.64 ± 0.00 | 0.78 ± 0.00 | 0.38 ± 0.00 | 0.54 ± 0.00 | 0.44 ± 0.00 | 8.91 ± 0.00 |
| E5: lstm_forecast | 0.52 ± 0.02 | 0.56 ± 0.01 | 0.81 ± 0.01 | 0.31 ± 0.01 | 0.51 ± 0.02 | 0.38 ± 0.01 | 11.27 ± 0.97 |
| E5: lstm_classifier | 0.48 ± 0.02 | 0.60 ± 0.01 | 0.83 ± 0.01 | 0.37 ± 0.04 | 0.55 ± 0.02 | 0.45 ± 0.02 | 9.20 ± 1.29 |
| E4: zscore_3sigma | 0.24 [0.16, 0.38] | 0.43 [0.38, 0.60] | 0.70 [0.61, 0.79] | 0.14 [0.10, 0.19] | 1.00 [1.00, 1.00] | 0.25 [0.18, 0.31] | 60.00 [60.00, 60.00] |
| E4: zscore_local | 0.39 [0.27, 0.54] | 0.49 [0.42, 0.69] | 0.76 [0.68, 0.83] | 0.30 [0.19, 0.43] | 0.41 [0.27, 0.55] | 0.35 [0.23, 0.46] | 9.30 [6.56, 12.35] |
| E4: isolation_forest | 0.16 [0.11, 0.26] | 0.36 [0.29, 0.54] | 0.60 [0.50, 0.70] | 0.17 [0.11, 0.24] | 0.62 [0.48, 0.75] | 0.26 [0.18, 0.35] | 30.35 [23.75, 37.16] |
| E4: lightgbm | 0.25 [0.15, 0.41] | 0.56 [0.50, 0.70] | 0.59 [0.48, 0.68] | 0.22 [0.14, 0.33] | 0.50 [0.37, 0.64] | 0.31 [0.21, 0.41] | 17.50 [12.17, 23.17] |
| E4: prior_horario | 0.15 [0.09, 0.27] | 0.28 [0.21, 0.47] | 0.44 [0.34, 0.55] | 0.15 [0.09, 0.22] | 0.48 [0.33, 0.63] | 0.22 [0.14, 0.31] | 27.15 [20.73, 33.57] |

## Localização via XAI (janelas-oráculo, seed explicada)

| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| lightgbm/shap/sum | 0.04 [0.00, 0.11] | 0.24 [0.13, 0.38] | 0.27 [0.13, 0.40] | 0.16 [0.11, 0.24] |
| lightgbm/shap/top3 | 0.04 [0.00, 0.11] | 0.18 [0.07, 0.31] | 0.22 [0.11, 0.36] | 0.15 [0.10, 0.22] |
| lightgbm/lime/sum | 0.00 [0.00, 0.00] | 0.13 [0.04, 0.24] | 0.20 [0.09, 0.31] | 0.12 [0.08, 0.17] |
| lightgbm/lime/top3 | 0.00 [0.00, 0.00] | 0.18 [0.07, 0.31] | 0.20 [0.09, 0.31] | 0.13 [0.08, 0.18] |
| isolation_forest/shap/sum | 0.11 [0.02, 0.20] | 0.24 [0.13, 0.38] | 0.38 [0.24, 0.53] | 0.24 [0.16, 0.33] |
| isolation_forest/shap/top3 | 0.13 [0.04, 0.24] | 0.27 [0.13, 0.40] | 0.27 [0.13, 0.40] | 0.24 [0.15, 0.34] |
| isolation_forest/lime/sum | 0.09 [0.02, 0.18] | 0.20 [0.09, 0.33] | 0.29 [0.16, 0.42] | 0.20 [0.12, 0.29] |
| isolation_forest/lime/top3 | 0.11 [0.02, 0.22] | 0.27 [0.13, 0.40] | 0.36 [0.22, 0.49] | 0.24 [0.15, 0.33] |
| logistic/shap/sum | 0.13 [0.04, 0.24] | 0.16 [0.07, 0.27] | 0.24 [0.11, 0.38] | 0.21 [0.12, 0.32] |
| logistic/shap/top3 | 0.11 [0.02, 0.22] | 0.16 [0.07, 0.27] | 0.27 [0.13, 0.40] | 0.20 [0.12, 0.30] |
| logistic/lime/sum | 0.07 [0.00, 0.16] | 0.24 [0.13, 0.38] | 0.31 [0.18, 0.44] | 0.19 [0.13, 0.28] |
| logistic/lime/top3 | 0.09 [0.02, 0.18] | 0.20 [0.09, 0.33] | 0.27 [0.13, 0.40] | 0.19 [0.12, 0.28] |
| E6: lstm_forecast/shap/sum | 0.42 [0.29, 0.58] | 0.78 [0.64, 0.89] | 0.89 [0.80, 0.98] | 0.61 [0.50, 0.71] |
| E6: lstm_forecast/lime/sum | 0.44 [0.31, 0.60] | 0.76 [0.62, 0.87] | 0.82 [0.71, 0.93] | 0.60 [0.49, 0.71] |
| E6: lstm_classifier/shap/sum | 0.27 [0.13, 0.40] | 0.51 [0.36, 0.64] | 0.56 [0.40, 0.69] | 0.42 [0.32, 0.53] |
| E6: lstm_classifier/lime/sum | 0.13 [0.04, 0.22] | 0.33 [0.20, 0.47] | 0.49 [0.33, 0.62] | 0.29 [0.20, 0.39] |
| E4: prior | 0.09 [0.02, 0.18] | 0.18 [0.09, 0.29] | 0.27 [0.13, 0.40] | 0.19 [0.12, 0.29] |
| E4: zscore_global | 0.13 [0.04, 0.24] | 0.38 [0.24, 0.51] | 0.56 [0.40, 0.71] | 0.31 [0.22, 0.41] |
| E4: zscore_local | 0.09 [0.02, 0.18] | 0.49 [0.33, 0.62] | 0.60 [0.44, 0.73] | 0.33 [0.26, 0.41] |

O ranqueamento nativo da LSTM (E5, sem XAI) está nas comparações pareadas abaixo (`lstm_nativo_soma`) e em `lstm_<sistema>_<nome>.md`.

## Comparações pareadas (diferença = alternativo menos referência)

Detecção: mesmas janelas de teste e mesma seed; localização: mesmas falhas. IC de 95% por bootstrap pareado; médias entre seeds e contagem de vereditos.

| Tarefa | Modelo | Referência | Métrica | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | lightgbm | lstm_forecast | auc_pr | -0.313 | [-0.444, -0.149] | 0 / 0 / 5 |
| detecção | lightgbm | lstm_classifier | auc_pr | -0.279 | [-0.403, -0.149] | 0 / 0 / 5 |
| detecção | lightgbm | zscore_3sigma | auc_pr | -0.035 | [-0.164, +0.090] | 0 / 5 / 0 |
| detecção | lightgbm | zscore_local | auc_pr | -0.188 | [-0.322, -0.040] | 0 / 0 / 5 |
| detecção | lightgbm | isolation_forest | auc_pr | +0.040 | [-0.060, +0.168] | 0 / 5 / 0 |
| detecção | lightgbm | lightgbm | auc_pr | -0.044 | [-0.108, -0.003] | 0 / 1 / 4 |
| detecção | lightgbm | prior_horario | auc_pr | +0.050 | [-0.069, +0.174] | 0 / 5 / 0 |
| detecção | isolation_forest | lstm_forecast | auc_pr | -0.346 | [-0.470, -0.192] | 0 / 0 / 5 |
| detecção | isolation_forest | lstm_classifier | auc_pr | -0.312 | [-0.451, -0.170] | 0 / 0 / 5 |
| detecção | isolation_forest | zscore_3sigma | auc_pr | -0.068 | [-0.148, -0.010] | 0 / 0 / 5 |
| detecção | isolation_forest | zscore_local | auc_pr | -0.221 | [-0.350, -0.087] | 0 / 0 / 5 |
| detecção | isolation_forest | isolation_forest | auc_pr | +0.007 | [-0.012, +0.046] | 0 / 5 / 0 |
| detecção | isolation_forest | lightgbm | auc_pr | -0.077 | [-0.238, +0.053] | 0 / 5 / 0 |
| detecção | isolation_forest | prior_horario | auc_pr | +0.017 | [-0.069, +0.097] | 0 / 5 / 0 |
| detecção | logistic | lstm_forecast | auc_pr | +0.001 | [-0.154, +0.175] | 0 / 1 / 0 |
| detecção | logistic | lstm_classifier | auc_pr | +0.052 | [-0.102, +0.189] | 0 / 1 / 0 |
| detecção | logistic | zscore_3sigma | auc_pr | +0.269 | [+0.095, +0.419] | 1 / 0 / 0 |
| detecção | logistic | zscore_local | auc_pr | +0.116 | [-0.031, +0.274] | 0 / 1 / 0 |
| detecção | logistic | isolation_forest | auc_pr | +0.344 | [+0.179, +0.497] | 1 / 0 / 0 |
| detecção | logistic | lightgbm | auc_pr | +0.260 | [+0.122, +0.384] | 1 / 0 / 0 |
| detecção | logistic | prior_horario | auc_pr | +0.354 | [+0.177, +0.515] | 1 / 0 / 0 |
| localização | lightgbm/shap/sum | lstm_forecast/shap | hr@1 | -0.378 | [-0.511, -0.244] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_forecast/shap | mrr | -0.443 | [-0.543, -0.341] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_classifier/shap | hr@1 | -0.222 | [-0.356, -0.089] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_classifier/shap | mrr | -0.261 | [-0.366, -0.155] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_nativo_soma | hr@1 | -0.378 | [-0.533, -0.244] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_nativo_soma | mrr | -0.402 | [-0.504, -0.302] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | prior | hr@1 | -0.044 | [-0.156, +0.044] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | prior | mrr | -0.032 | [-0.124, +0.057] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | zscore_global | hr@1 | -0.089 | [-0.200, +0.023] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | zscore_global | mrr | -0.150 | [-0.266, -0.025] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | zscore_local | hr@1 | -0.044 | [-0.133, +0.044] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | zscore_local | mrr | -0.165 | [-0.233, -0.097] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_forecast/lime | hr@1 | -0.444 | [-0.600, -0.311] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_forecast/lime | mrr | -0.483 | [-0.588, -0.384] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_classifier/lime | hr@1 | -0.133 | [-0.244, -0.044] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_classifier/lime | mrr | -0.171 | [-0.261, -0.087] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_nativo_soma | hr@1 | -0.422 | [-0.578, -0.289] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_nativo_soma | mrr | -0.444 | [-0.559, -0.337] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | prior | hr@1 | -0.089 | [-0.178, -0.022] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | prior | mrr | -0.074 | [-0.162, +0.003] | 0 / 1 / 0 |
| localização | lightgbm/lime/sum | zscore_global | hr@1 | -0.133 | [-0.222, -0.044] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_global | mrr | -0.193 | [-0.292, -0.091] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_local | hr@1 | -0.089 | [-0.178, -0.022] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_local | mrr | -0.207 | [-0.285, -0.133] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_forecast/shap | hr@1 | -0.311 | [-0.489, -0.133] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_forecast/shap | mrr | -0.368 | [-0.497, -0.241] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_classifier/shap | hr@1 | -0.156 | [-0.311, +0.000] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | lstm_classifier/shap | mrr | -0.186 | [-0.304, -0.073] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_nativo_soma | hr@1 | -0.311 | [-0.467, -0.178] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_nativo_soma | mrr | -0.327 | [-0.441, -0.212] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | prior | hr@1 | +0.022 | [-0.044, +0.111] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | prior | mrr | +0.043 | [-0.036, +0.125] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_global | hr@1 | -0.022 | [-0.156, +0.133] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_global | mrr | -0.075 | [-0.199, +0.055] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_local | hr@1 | +0.022 | [-0.089, +0.133] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_local | mrr | -0.090 | [-0.190, +0.016] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | lstm_forecast/lime | hr@1 | -0.356 | [-0.511, -0.200] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_forecast/lime | mrr | -0.403 | [-0.522, -0.287] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_classifier/lime | hr@1 | -0.044 | [-0.156, +0.067] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | lstm_classifier/lime | mrr | -0.090 | [-0.182, -0.001] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_nativo_soma | hr@1 | -0.333 | [-0.489, -0.178] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_nativo_soma | mrr | -0.364 | [-0.486, -0.249] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | prior | hr@1 | +0.000 | [-0.111, +0.111] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | prior | mrr | +0.006 | [-0.095, +0.106] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | zscore_global | hr@1 | -0.044 | [-0.178, +0.111] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | zscore_global | mrr | -0.112 | [-0.229, +0.019] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | zscore_local | hr@1 | +0.000 | [-0.111, +0.111] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | zscore_local | mrr | -0.127 | [-0.234, -0.013] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_forecast/shap | hr@1 | -0.289 | [-0.444, -0.111] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_forecast/shap | mrr | -0.396 | [-0.521, -0.263] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_classifier/shap | hr@1 | -0.133 | [-0.289, +0.022] | 0 / 1 / 0 |
| localização | logistic/shap/sum | lstm_classifier/shap | mrr | -0.214 | [-0.339, -0.081] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_nativo_soma | hr@1 | -0.289 | [-0.444, -0.133] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_nativo_soma | mrr | -0.355 | [-0.482, -0.221] | 0 / 0 / 1 |
| localização | logistic/shap/sum | prior | hr@1 | +0.044 | [-0.067, +0.156] | 0 / 1 / 0 |
| localização | logistic/shap/sum | prior | mrr | +0.016 | [-0.079, +0.116] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_global | hr@1 | +0.000 | [-0.156, +0.156] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_global | mrr | -0.103 | [-0.235, +0.041] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_local | hr@1 | +0.044 | [-0.067, +0.156] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_local | mrr | -0.117 | [-0.212, -0.016] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_forecast/lime | hr@1 | -0.378 | [-0.533, -0.222] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_forecast/lime | mrr | -0.410 | [-0.521, -0.300] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_classifier/lime | hr@1 | -0.067 | [-0.178, +0.044] | 0 / 1 / 0 |
| localização | logistic/lime/sum | lstm_classifier/lime | mrr | -0.097 | [-0.195, -0.009] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_nativo_soma | hr@1 | -0.356 | [-0.511, -0.200] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_nativo_soma | mrr | -0.371 | [-0.486, -0.259] | 0 / 0 / 1 |
| localização | logistic/lime/sum | prior | hr@1 | -0.022 | [-0.111, +0.089] | 0 / 1 / 0 |
| localização | logistic/lime/sum | prior | mrr | -0.001 | [-0.077, +0.075] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_global | hr@1 | -0.067 | [-0.200, +0.067] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_global | mrr | -0.119 | [-0.231, +0.002] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_local | hr@1 | -0.022 | [-0.133, +0.089] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_local | mrr | -0.134 | [-0.216, -0.046] | 0 / 0 / 1 |

## Qualidade das explicações (médias por janela)

Deleção: menor = mais fiel; inserção (relativa à janela completa): maior = mais fiel; `random` = ordens aleatórias. Jaccard/Kendall nos top-10 atributos.

| Medida | lightgbm | isolation_forest | logistic |
|---|---|---|---|
| deletion_shap | 0.276 | 0.972 | 0.325 |
| deletion_lime | 0.324 | 0.973 | 0.311 |
| deletion_random | 0.987 | 1.000 | 1.036 |
| insertion_shap | 4.300 | 0.955 | 10.392 |
| insertion_lime | 2.976 | 0.959 | 10.797 |
| insertion_random | 0.269 | 0.933 | 5.748 |
| shap_lime_jaccard | 0.255 | 0.365 | 0.524 |
| shap_lime_kendall | -0.031 | 0.076 | 0.421 |
| neighbour_jaccard | 0.773 | 0.668 | 0.813 |
| top_component_agree | 0.378 | 0.533 | 0.667 |
| seed_jaccard | 0.615 | 0.034 | — |
| seed_kendall | 0.418 | -0.459 | — |

## Custo (CPU)

| Modelo | Medida | Valor |
|---|---|---|
| lightgbm | train_seconds_mean | 8.19 |
| lightgbm | inference_ms_per_window | 0.0226 |
| lightgbm | xai_seconds_per_window | 0.88 |
| isolation_forest | train_seconds_mean | 0.14 |
| isolation_forest | inference_ms_per_window | 0.0264 |
| isolation_forest | xai_seconds_per_window | 3.79 |
| logistic | train_seconds_mean | 0.2 |
| logistic | inference_ms_per_window | 0.0057 |
| logistic | xai_seconds_per_window | 0.44 |
| logistic | nonzero_coefficients_mean | 191.0 |
| logistic | not_converged | 0 |

## Busca (validação, 1 seed; 216 treinos, 585.4 s)

| Modelo | Configuração | AUC-PR média na validação |
|---|---|---|
| lightgbm | global/n_estimators=300/num_leaves=15 | 0.659 |
| lightgbm | global/n_estimators=600/num_leaves=15 | 0.657 |
| lightgbm | global/n_estimators=300/num_leaves=31 | 0.673 |
| lightgbm | global/n_estimators=600/num_leaves=31 | 0.675 |
| lightgbm | local/n_estimators=300/num_leaves=15 | 0.543 |
| lightgbm | local/n_estimators=600/num_leaves=15 | 0.553 |
| lightgbm | local/n_estimators=300/num_leaves=31 | 0.470 |
| lightgbm | local/n_estimators=600/num_leaves=31 | 0.492 |
| isolation_forest | global/max_samples=256/n_estimators=100 | 0.370 |
| isolation_forest | global/max_samples=1024/n_estimators=100 | 0.416 |
| isolation_forest | global/max_samples=256/n_estimators=300 | 0.439 |
| isolation_forest | global/max_samples=1024/n_estimators=300 | 0.447 |
| isolation_forest | local/max_samples=256/n_estimators=100 | 0.326 |
| isolation_forest | local/max_samples=1024/n_estimators=100 | 0.297 |
| isolation_forest | local/max_samples=256/n_estimators=300 | 0.339 |
| isolation_forest | local/max_samples=1024/n_estimators=300 | 0.320 |
| logistic | global/c=0.01 | 0.632 |
| logistic | global/c=0.1 | 0.641 |
| logistic | global/c=1 | 0.641 |
| logistic | global/c=10 | 0.678 |
| logistic | local/c=0.01 | 0.533 |
| logistic | local/c=0.1 | 0.558 |
| logistic | local/c=1 | 0.518 |
| logistic | local/c=10 | 0.431 |
