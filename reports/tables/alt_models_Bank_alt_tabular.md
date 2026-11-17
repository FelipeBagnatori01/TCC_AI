# Modelos alternativos explicáveis — Bank (família `tabular`, ADR 0018)

Mesmas janelas, dobras (8) e protocolo da E5/E6. Seleção na validação: lightgbm {"normalization": "local", "params": {"n_estimators": 600, "num_leaves": 15}}; isolation_forest {"normalization": "global", "params": {"max_samples": 256, "n_estimators": 300}}; logistic {"normalization": "global", "params": {"c": 0.1}}. Detecção: média ± desvio-padrão entre seeds (a logística é determinística: 1 seed); as linhas `E4:` são execuções únicas, com IC de 95% por bootstrap. Todos os candidatos (alternativos, LSTMs da E5 e baselines da E4) são avaliados nas mesmas janelas e falhas.

## Detecção (por janela)

| Modelo | AUC-PR | AUC-PR por dobra | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|---|
| lightgbm | 0.18 ± 0.01 | 0.22 ± 0.01 | 0.50 ± 0.01 | 0.14 ± 0.01 | 0.51 ± 0.02 | 0.22 ± 0.01 | 22.94 ± 2.26 |
| isolation_forest | 0.14 ± 0.00 | 0.22 ± 0.01 | 0.60 ± 0.01 | 0.14 ± 0.00 | 0.62 ± 0.10 | 0.23 ± 0.01 | 25.96 ± 4.36 |
| logistic | 0.10 ± 0.00 | 0.19 ± 0.00 | 0.50 ± 0.00 | 0.12 ± 0.00 | 0.36 ± 0.00 | 0.18 ± 0.00 | 18.65 ± 0.00 |
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
| lightgbm/shap/sum | 0.12 [0.07, 0.18] | 0.28 [0.20, 0.36] | 0.40 [0.32, 0.48] | 0.28 [0.23, 0.34] |
| lightgbm/shap/top3 | 0.08 [0.04, 0.13] | 0.26 [0.17, 0.34] | 0.45 [0.36, 0.54] | 0.25 [0.20, 0.30] |
| lightgbm/lime/sum | 0.02 [0.00, 0.04] | 0.13 [0.07, 0.20] | 0.38 [0.30, 0.47] | 0.18 [0.16, 0.21] |
| lightgbm/lime/top3 | 0.08 [0.04, 0.13] | 0.25 [0.17, 0.33] | 0.44 [0.36, 0.53] | 0.25 [0.20, 0.29] |
| isolation_forest/shap/sum | 0.12 [0.06, 0.17] | 0.30 [0.23, 0.38] | 0.53 [0.44, 0.62] | 0.29 [0.24, 0.34] |
| isolation_forest/shap/top3 | 0.15 [0.09, 0.22] | 0.29 [0.21, 0.38] | 0.51 [0.42, 0.60] | 0.31 [0.26, 0.37] |
| isolation_forest/lime/sum | 0.03 [0.01, 0.07] | 0.27 [0.19, 0.35] | 0.47 [0.38, 0.57] | 0.22 [0.19, 0.26] |
| isolation_forest/lime/top3 | 0.12 [0.07, 0.17] | 0.28 [0.20, 0.36] | 0.43 [0.34, 0.53] | 0.28 [0.23, 0.33] |
| logistic/shap/sum | 0.15 [0.09, 0.21] | 0.32 [0.23, 0.41] | 0.39 [0.30, 0.48] | 0.30 [0.24, 0.36] |
| logistic/shap/top3 | 0.15 [0.09, 0.22] | 0.34 [0.25, 0.43] | 0.42 [0.33, 0.51] | 0.31 [0.25, 0.37] |
| logistic/lime/sum | 0.08 [0.03, 0.13] | 0.25 [0.17, 0.33] | 0.42 [0.34, 0.52] | 0.25 [0.20, 0.29] |
| logistic/lime/top3 | 0.17 [0.11, 0.23] | 0.34 [0.26, 0.43] | 0.47 [0.38, 0.57] | 0.33 [0.27, 0.39] |
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
| detecção | lightgbm | lstm_forecast | auc_pr | -0.166 | [-0.262, -0.082] | 0 / 0 / 5 |
| detecção | lightgbm | lstm_classifier | auc_pr | -0.038 | [-0.107, +0.023] | 0 / 4 / 1 |
| detecção | lightgbm | zscore_3sigma | auc_pr | +0.082 | [+0.021, +0.144] | 5 / 0 / 0 |
| detecção | lightgbm | zscore_local | auc_pr | -0.145 | [-0.229, -0.067] | 0 / 0 / 5 |
| detecção | lightgbm | isolation_forest | auc_pr | +0.040 | [-0.037, +0.104] | 0 / 5 / 0 |
| detecção | lightgbm | lightgbm | auc_pr | -0.009 | [-0.057, +0.033] | 0 / 5 / 0 |
| detecção | lightgbm | prior_horario | auc_pr | +0.057 | [-0.003, +0.119] | 2 / 3 / 0 |
| detecção | isolation_forest | lstm_forecast | auc_pr | -0.209 | [-0.297, -0.128] | 0 / 0 / 5 |
| detecção | isolation_forest | lstm_classifier | auc_pr | -0.081 | [-0.158, -0.013] | 0 / 0 / 5 |
| detecção | isolation_forest | zscore_3sigma | auc_pr | +0.039 | [+0.008, +0.079] | 5 / 0 / 0 |
| detecção | isolation_forest | zscore_local | auc_pr | -0.189 | [-0.261, -0.116] | 0 / 0 / 5 |
| detecção | isolation_forest | isolation_forest | auc_pr | -0.003 | [-0.036, +0.015] | 0 / 5 / 0 |
| detecção | isolation_forest | lightgbm | auc_pr | -0.052 | [-0.126, +0.018] | 0 / 5 / 0 |
| detecção | isolation_forest | prior_horario | auc_pr | +0.014 | [-0.023, +0.059] | 0 / 5 / 0 |
| detecção | logistic | lstm_forecast | auc_pr | -0.252 | [-0.338, -0.175] | 0 / 0 / 1 |
| detecção | logistic | lstm_classifier | auc_pr | -0.123 | [-0.198, -0.064] | 0 / 0 / 1 |
| detecção | logistic | zscore_3sigma | auc_pr | +0.000 | [-0.024, +0.022] | 0 / 1 / 0 |
| detecção | logistic | zscore_local | auc_pr | -0.227 | [-0.302, -0.161] | 0 / 0 / 1 |
| detecção | logistic | isolation_forest | auc_pr | -0.042 | [-0.098, -0.007] | 0 / 0 / 1 |
| detecção | logistic | lightgbm | auc_pr | -0.090 | [-0.160, -0.028] | 0 / 0 / 1 |
| detecção | logistic | prior_horario | auc_pr | -0.025 | [-0.056, +0.002] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | lstm_forecast/shap | hr@1 | -0.367 | [-0.467, -0.267] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_forecast/shap | mrr | -0.333 | [-0.415, -0.252] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_classifier/shap | hr@1 | -0.325 | [-0.433, -0.225] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_classifier/shap | mrr | -0.333 | [-0.410, -0.256] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_nativo_soma | hr@1 | -0.225 | [-0.333, -0.125] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_nativo_soma | mrr | -0.238 | [-0.323, -0.155] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | prior | hr@1 | +0.008 | [-0.067, +0.075] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | prior | mrr | +0.022 | [-0.044, +0.084] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | zscore_global | hr@1 | -0.200 | [-0.308, -0.092] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | zscore_global | mrr | -0.191 | [-0.275, -0.103] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | zscore_local | hr@1 | +0.008 | [-0.067, +0.083] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | zscore_local | mrr | -0.061 | [-0.127, +0.005] | 0 / 1 / 0 |
| localização | lightgbm/lime/sum | lstm_forecast/lime | hr@1 | -0.433 | [-0.525, -0.342] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_forecast/lime | mrr | -0.403 | [-0.475, -0.328] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_classifier/lime | hr@1 | -0.200 | [-0.275, -0.125] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_classifier/lime | mrr | -0.240 | [-0.300, -0.181] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_nativo_soma | hr@1 | -0.325 | [-0.417, -0.242] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_nativo_soma | mrr | -0.336 | [-0.403, -0.270] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | prior | hr@1 | -0.092 | [-0.150, -0.033] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | prior | mrr | -0.076 | [-0.135, -0.020] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_global | hr@1 | -0.300 | [-0.392, -0.208] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_global | mrr | -0.289 | [-0.364, -0.214] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_local | hr@1 | -0.092 | [-0.150, -0.033] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_local | mrr | -0.159 | [-0.212, -0.104] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_forecast/shap | hr@1 | -0.367 | [-0.467, -0.258] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_forecast/shap | mrr | -0.324 | [-0.399, -0.239] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_classifier/shap | hr@1 | -0.325 | [-0.425, -0.217] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_classifier/shap | mrr | -0.324 | [-0.398, -0.244] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_nativo_soma | hr@1 | -0.225 | [-0.317, -0.133] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_nativo_soma | mrr | -0.229 | [-0.297, -0.158] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | prior | hr@1 | +0.008 | [-0.075, +0.092] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | prior | mrr | +0.031 | [-0.045, +0.111] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_global | hr@1 | -0.200 | [-0.308, -0.092] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | zscore_global | mrr | -0.181 | [-0.272, -0.096] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | zscore_local | hr@1 | +0.008 | [-0.067, +0.083] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_local | mrr | -0.051 | [-0.109, +0.014] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | lstm_forecast/lime | hr@1 | -0.417 | [-0.508, -0.325] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_forecast/lime | mrr | -0.361 | [-0.433, -0.288] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_classifier/lime | hr@1 | -0.183 | [-0.267, -0.100] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_classifier/lime | mrr | -0.199 | [-0.260, -0.137] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_nativo_soma | hr@1 | -0.308 | [-0.392, -0.225] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_nativo_soma | mrr | -0.295 | [-0.359, -0.231] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | prior | hr@1 | -0.075 | [-0.142, -0.008] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | prior | mrr | -0.035 | [-0.099, +0.026] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | zscore_global | hr@1 | -0.283 | [-0.375, -0.192] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | zscore_global | mrr | -0.248 | [-0.325, -0.172] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | zscore_local | hr@1 | -0.075 | [-0.133, -0.017] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | zscore_local | mrr | -0.118 | [-0.169, -0.066] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_forecast/shap | hr@1 | -0.333 | [-0.442, -0.217] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_forecast/shap | mrr | -0.316 | [-0.402, -0.224] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_classifier/shap | hr@1 | -0.292 | [-0.392, -0.183] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_classifier/shap | mrr | -0.316 | [-0.392, -0.234] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_nativo_soma | hr@1 | -0.192 | [-0.300, -0.083] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_nativo_soma | mrr | -0.221 | [-0.307, -0.136] | 0 / 0 / 1 |
| localização | logistic/shap/sum | prior | hr@1 | +0.042 | [-0.042, +0.125] | 0 / 1 / 0 |
| localização | logistic/shap/sum | prior | mrr | +0.039 | [-0.026, +0.109] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_global | hr@1 | -0.167 | [-0.258, -0.067] | 0 / 0 / 1 |
| localização | logistic/shap/sum | zscore_global | mrr | -0.174 | [-0.244, -0.098] | 0 / 0 / 1 |
| localização | logistic/shap/sum | zscore_local | hr@1 | +0.042 | [-0.042, +0.133] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_local | mrr | -0.044 | [-0.112, +0.028] | 0 / 1 / 0 |
| localização | logistic/lime/sum | lstm_forecast/lime | hr@1 | -0.367 | [-0.467, -0.258] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_forecast/lime | mrr | -0.338 | [-0.418, -0.256] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_classifier/lime | hr@1 | -0.133 | [-0.217, -0.058] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_classifier/lime | mrr | -0.175 | [-0.237, -0.114] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_nativo_soma | hr@1 | -0.258 | [-0.350, -0.167] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_nativo_soma | mrr | -0.271 | [-0.345, -0.196] | 0 / 0 / 1 |
| localização | logistic/lime/sum | prior | hr@1 | -0.025 | [-0.100, +0.050] | 0 / 1 / 0 |
| localização | logistic/lime/sum | prior | mrr | -0.011 | [-0.071, +0.051] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_global | hr@1 | -0.233 | [-0.325, -0.142] | 0 / 0 / 1 |
| localização | logistic/lime/sum | zscore_global | mrr | -0.224 | [-0.294, -0.150] | 0 / 0 / 1 |
| localização | logistic/lime/sum | zscore_local | hr@1 | -0.025 | [-0.100, +0.050] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_local | mrr | -0.094 | [-0.155, -0.031] | 0 / 0 / 1 |

## Qualidade das explicações (médias por janela)

Deleção: menor = mais fiel; inserção (relativa à janela completa): maior = mais fiel; `random` = ordens aleatórias. Jaccard/Kendall nos top-10 atributos.

| Medida | lightgbm | isolation_forest | logistic |
|---|---|---|---|
| deletion_shap | 0.369 | 0.969 | 0.196 |
| deletion_lime | 0.283 | 0.970 | 0.192 |
| deletion_random | 1.014 | 0.999 | 2.421 |
| insertion_shap | 14.432 | 0.803 | 379162.906 |
| insertion_lime | 16.247 | 0.802 | 384500.403 |
| insertion_random | 0.786 | 0.773 | 149368.570 |
| shap_lime_jaccard | 0.204 | 0.380 | 0.558 |
| shap_lime_kendall | -0.157 | 0.083 | 0.439 |
| neighbour_jaccard | 0.858 | 0.734 | 0.876 |
| top_component_agree | 0.108 | 0.625 | 0.317 |
| seed_jaccard | 0.595 | 0.029 | — |
| seed_kendall | 0.406 | -0.482 | — |

## Custo (CPU)

| Modelo | Medida | Valor |
|---|---|---|
| lightgbm | train_seconds_mean | 17.49 |
| lightgbm | inference_ms_per_window | 0.0221 |
| lightgbm | xai_seconds_per_window | 0.73 |
| isolation_forest | train_seconds_mean | 0.33 |
| isolation_forest | inference_ms_per_window | 0.0123 |
| isolation_forest | xai_seconds_per_window | 1.37 |
| logistic | train_seconds_mean | 1.11 |
| logistic | inference_ms_per_window | 0.0033 |
| logistic | xai_seconds_per_window | 0.36 |
| logistic | nonzero_coefficients_mean | 214.4 |
| logistic | not_converged | 0 |

## Busca (validação, 1 seed; 168 treinos, 1079.3 s)

| Modelo | Configuração | AUC-PR média na validação |
|---|---|---|
| lightgbm | global/n_estimators=300/num_leaves=15 | 0.236 |
| lightgbm | global/n_estimators=600/num_leaves=15 | 0.240 |
| lightgbm | global/n_estimators=300/num_leaves=31 | 0.216 |
| lightgbm | global/n_estimators=600/num_leaves=31 | 0.234 |
| lightgbm | local/n_estimators=300/num_leaves=15 | 0.236 |
| lightgbm | local/n_estimators=600/num_leaves=15 | 0.243 |
| lightgbm | local/n_estimators=300/num_leaves=31 | 0.204 |
| lightgbm | local/n_estimators=600/num_leaves=31 | 0.213 |
| isolation_forest | global/max_samples=256/n_estimators=100 | 0.213 |
| isolation_forest | global/max_samples=1024/n_estimators=100 | 0.192 |
| isolation_forest | global/max_samples=256/n_estimators=300 | 0.221 |
| isolation_forest | global/max_samples=1024/n_estimators=300 | 0.214 |
| isolation_forest | local/max_samples=256/n_estimators=100 | 0.193 |
| isolation_forest | local/max_samples=1024/n_estimators=100 | 0.190 |
| isolation_forest | local/max_samples=256/n_estimators=300 | 0.207 |
| isolation_forest | local/max_samples=1024/n_estimators=300 | 0.200 |
| logistic | global/c=0.01 | 0.237 |
| logistic | global/c=0.1 | 0.279 |
| logistic | global/c=1 | 0.265 |
| logistic | global/c=10 | 0.248 |
| logistic | local/c=0.01 | 0.218 |
| logistic | local/c=0.1 | 0.239 |
| logistic | local/c=1 | 0.241 |
| logistic | local/c=10 | 0.230 |
