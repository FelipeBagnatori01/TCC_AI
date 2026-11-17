# Modelos alternativos explicáveis — Bank (família `tabular`, ADR 0018)

Mesmas janelas, dobras (8) e protocolo da E5/E6. Seleção na validação: lightgbm {"normalization": "local", "params": {"n_estimators": 600, "num_leaves": 15}}; isolation_forest {"normalization": "global", "params": {"max_samples": 256, "n_estimators": 300}}; logistic {"normalization": "global", "params": {"c": 0.1}}. Detecção: média ± desvio-padrão entre seeds (a logística é determinística: 1 seed); as linhas `E4:` são execuções únicas, com IC de 95% por bootstrap. Todos os candidatos (alternativos, LSTMs da E5 e baselines da E4) são avaliados nas mesmas janelas e falhas.

## Detecção (por janela)

| Modelo | AUC-PR | AUC-PR por dobra | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|---|
| lightgbm | 0.18 ± 0.01 | 0.22 ± 0.01 | 0.50 ± 0.01 | 0.14 ± 0.01 | 0.51 ± 0.02 | 0.22 ± 0.01 | 22.94 ± 2.26 |
| isolation_forest | 0.14 ± 0.00 | 0.21 ± 0.01 | 0.60 ± 0.01 | 0.14 ± 0.01 | 0.67 ± 0.06 | 0.23 ± 0.01 | 28.72 ± 3.78 |
| logistic | 0.10 ± 0.00 | 0.19 ± 0.00 | 0.50 ± 0.00 | 0.12 ± 0.00 | 0.35 ± 0.00 | 0.18 ± 0.00 | 18.47 ± 0.00 |
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
| lightgbm/shap/sum | 0.12 [0.07, 0.18] | 0.28 [0.20, 0.36] | 0.40 [0.32, 0.48] | 0.28 [0.23, 0.34] |
| lightgbm/shap/top3 | 0.08 [0.04, 0.13] | 0.26 [0.17, 0.34] | 0.45 [0.36, 0.54] | 0.25 [0.20, 0.30] |
| lightgbm/lime/sum | 0.02 [0.00, 0.04] | 0.13 [0.07, 0.19] | 0.33 [0.25, 0.41] | 0.18 [0.16, 0.21] |
| lightgbm/lime/top3 | 0.10 [0.05, 0.16] | 0.29 [0.22, 0.38] | 0.43 [0.34, 0.53] | 0.27 [0.22, 0.32] |
| isolation_forest/shap/sum | 0.09 [0.04, 0.14] | 0.38 [0.29, 0.47] | 0.54 [0.45, 0.63] | 0.29 [0.24, 0.33] |
| isolation_forest/shap/top3 | 0.14 [0.08, 0.21] | 0.34 [0.27, 0.42] | 0.52 [0.43, 0.60] | 0.32 [0.27, 0.38] |
| isolation_forest/lime/sum | 0.04 [0.01, 0.07] | 0.26 [0.18, 0.34] | 0.49 [0.40, 0.58] | 0.24 [0.20, 0.27] |
| isolation_forest/lime/top3 | 0.13 [0.07, 0.20] | 0.29 [0.21, 0.38] | 0.48 [0.39, 0.57] | 0.30 [0.24, 0.35] |
| logistic/shap/sum | 0.15 [0.09, 0.21] | 0.32 [0.23, 0.41] | 0.39 [0.30, 0.48] | 0.30 [0.24, 0.36] |
| logistic/shap/top3 | 0.15 [0.09, 0.22] | 0.34 [0.25, 0.43] | 0.42 [0.33, 0.51] | 0.31 [0.25, 0.37] |
| logistic/lime/sum | 0.07 [0.03, 0.11] | 0.24 [0.17, 0.33] | 0.47 [0.38, 0.56] | 0.24 [0.20, 0.29] |
| logistic/lime/top3 | 0.14 [0.08, 0.20] | 0.34 [0.26, 0.43] | 0.45 [0.36, 0.54] | 0.31 [0.26, 0.37] |
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
| detecção | lightgbm | lstm_forecast | auc_pr | -0.160 | [-0.255, -0.077] | 0 / 0 / 5 |
| detecção | lightgbm | lstm_classifier | auc_pr | -0.029 | [-0.097, +0.033] | 0 / 5 / 0 |
| detecção | lightgbm | zscore_3sigma | auc_pr | +0.082 | [+0.021, +0.144] | 5 / 0 / 0 |
| detecção | lightgbm | zscore_local | auc_pr | -0.145 | [-0.229, -0.067] | 0 / 0 / 5 |
| detecção | lightgbm | isolation_forest | auc_pr | +0.043 | [-0.024, +0.107] | 0 / 5 / 0 |
| detecção | lightgbm | lightgbm | auc_pr | +0.006 | [-0.054, +0.051] | 0 / 5 / 0 |
| detecção | lightgbm | prior_horario | auc_pr | +0.057 | [-0.003, +0.119] | 2 / 3 / 0 |
| detecção | isolation_forest | lstm_forecast | auc_pr | -0.206 | [-0.293, -0.127] | 0 / 0 / 5 |
| detecção | isolation_forest | lstm_classifier | auc_pr | -0.075 | [-0.150, -0.009] | 0 / 1 / 4 |
| detecção | isolation_forest | zscore_3sigma | auc_pr | +0.036 | [+0.006, +0.075] | 5 / 0 / 0 |
| detecção | isolation_forest | zscore_local | auc_pr | -0.192 | [-0.264, -0.119] | 0 / 0 / 5 |
| detecção | isolation_forest | isolation_forest | auc_pr | -0.003 | [-0.027, +0.016] | 0 / 5 / 0 |
| detecção | isolation_forest | lightgbm | auc_pr | -0.041 | [-0.119, +0.025] | 0 / 5 / 0 |
| detecção | isolation_forest | prior_horario | auc_pr | +0.011 | [-0.026, +0.054] | 0 / 5 / 0 |
| detecção | logistic | lstm_forecast | auc_pr | -0.245 | [-0.332, -0.170] | 0 / 0 / 1 |
| detecção | logistic | lstm_classifier | auc_pr | -0.108 | [-0.180, -0.052] | 0 / 0 / 1 |
| detecção | logistic | zscore_3sigma | auc_pr | +0.001 | [-0.024, +0.023] | 0 / 1 / 0 |
| detecção | logistic | zscore_local | auc_pr | -0.226 | [-0.301, -0.159] | 0 / 0 / 1 |
| detecção | logistic | isolation_forest | auc_pr | -0.037 | [-0.081, -0.006] | 0 / 0 / 1 |
| detecção | logistic | lightgbm | auc_pr | -0.075 | [-0.152, -0.017] | 0 / 0 / 1 |
| detecção | logistic | prior_horario | auc_pr | -0.024 | [-0.056, +0.003] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | lstm_forecast/shap | hr@1 | -0.383 | [-0.483, -0.283] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_forecast/shap | mrr | -0.343 | [-0.425, -0.262] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_classifier/shap | hr@1 | -0.317 | [-0.425, -0.217] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_classifier/shap | mrr | -0.326 | [-0.404, -0.247] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_nativo_soma | hr@1 | -0.217 | [-0.325, -0.117] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_nativo_soma | mrr | -0.232 | [-0.316, -0.149] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | prior | hr@1 | +0.008 | [-0.067, +0.075] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | prior | mrr | +0.022 | [-0.044, +0.084] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | zscore_global | hr@1 | -0.200 | [-0.308, -0.092] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | zscore_global | mrr | -0.191 | [-0.275, -0.103] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | zscore_local | hr@1 | +0.008 | [-0.067, +0.083] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | zscore_local | mrr | -0.061 | [-0.127, +0.005] | 0 / 1 / 0 |
| localização | lightgbm/lime/sum | lstm_forecast/lime | hr@1 | -0.425 | [-0.508, -0.333] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_forecast/lime | mrr | -0.404 | [-0.474, -0.331] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_classifier/lime | hr@1 | -0.192 | [-0.275, -0.117] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_classifier/lime | mrr | -0.248 | [-0.311, -0.187] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_nativo_soma | hr@1 | -0.317 | [-0.408, -0.233] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_nativo_soma | mrr | -0.333 | [-0.402, -0.267] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | prior | hr@1 | -0.092 | [-0.158, -0.033] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | prior | mrr | -0.079 | [-0.136, -0.026] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_global | hr@1 | -0.300 | [-0.392, -0.208] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_global | mrr | -0.292 | [-0.367, -0.216] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_local | hr@1 | -0.092 | [-0.150, -0.033] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_local | mrr | -0.162 | [-0.213, -0.109] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_forecast/shap | hr@1 | -0.408 | [-0.508, -0.308] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_forecast/shap | mrr | -0.336 | [-0.408, -0.259] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_classifier/shap | hr@1 | -0.342 | [-0.442, -0.242] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_classifier/shap | mrr | -0.318 | [-0.388, -0.248] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_nativo_soma | hr@1 | -0.242 | [-0.325, -0.158] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_nativo_soma | mrr | -0.224 | [-0.289, -0.160] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | prior | hr@1 | -0.017 | [-0.092, +0.067] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | prior | mrr | +0.029 | [-0.040, +0.097] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_global | hr@1 | -0.225 | [-0.325, -0.125] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | zscore_global | mrr | -0.183 | [-0.261, -0.106] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | zscore_local | hr@1 | -0.017 | [-0.083, +0.058] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_local | mrr | -0.053 | [-0.109, +0.001] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | lstm_forecast/lime | hr@1 | -0.400 | [-0.483, -0.308] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_forecast/lime | mrr | -0.349 | [-0.415, -0.281] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_classifier/lime | hr@1 | -0.167 | [-0.242, -0.092] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_classifier/lime | mrr | -0.193 | [-0.254, -0.133] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_nativo_soma | hr@1 | -0.292 | [-0.375, -0.217] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_nativo_soma | mrr | -0.277 | [-0.341, -0.217] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | prior | hr@1 | -0.067 | [-0.133, -0.008] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | prior | mrr | -0.023 | [-0.082, +0.034] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | zscore_global | hr@1 | -0.275 | [-0.367, -0.192] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | zscore_global | mrr | -0.236 | [-0.308, -0.166] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | zscore_local | hr@1 | -0.067 | [-0.125, -0.008] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | zscore_local | mrr | -0.106 | [-0.157, -0.057] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_forecast/shap | hr@1 | -0.350 | [-0.458, -0.233] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_forecast/shap | mrr | -0.326 | [-0.410, -0.240] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_classifier/shap | hr@1 | -0.283 | [-0.375, -0.183] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_classifier/shap | mrr | -0.309 | [-0.383, -0.232] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_nativo_soma | hr@1 | -0.183 | [-0.292, -0.075] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_nativo_soma | mrr | -0.215 | [-0.300, -0.129] | 0 / 0 / 1 |
| localização | logistic/shap/sum | prior | hr@1 | +0.042 | [-0.042, +0.125] | 0 / 1 / 0 |
| localização | logistic/shap/sum | prior | mrr | +0.039 | [-0.026, +0.109] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_global | hr@1 | -0.167 | [-0.258, -0.067] | 0 / 0 / 1 |
| localização | logistic/shap/sum | zscore_global | mrr | -0.174 | [-0.244, -0.098] | 0 / 0 / 1 |
| localização | logistic/shap/sum | zscore_local | hr@1 | +0.042 | [-0.042, +0.133] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_local | mrr | -0.044 | [-0.112, +0.028] | 0 / 1 / 0 |
| localização | logistic/lime/sum | lstm_forecast/lime | hr@1 | -0.375 | [-0.475, -0.275] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_forecast/lime | mrr | -0.341 | [-0.420, -0.263] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_classifier/lime | hr@1 | -0.142 | [-0.217, -0.067] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_classifier/lime | mrr | -0.185 | [-0.247, -0.125] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_nativo_soma | hr@1 | -0.267 | [-0.367, -0.175] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_nativo_soma | mrr | -0.270 | [-0.342, -0.195] | 0 / 0 / 1 |
| localização | logistic/lime/sum | prior | hr@1 | -0.042 | [-0.108, +0.025] | 0 / 1 / 0 |
| localização | logistic/lime/sum | prior | mrr | -0.016 | [-0.072, +0.043] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_global | hr@1 | -0.250 | [-0.342, -0.158] | 0 / 0 / 1 |
| localização | logistic/lime/sum | zscore_global | mrr | -0.229 | [-0.298, -0.157] | 0 / 0 / 1 |
| localização | logistic/lime/sum | zscore_local | hr@1 | -0.042 | [-0.108, +0.025] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_local | mrr | -0.099 | [-0.157, -0.038] | 0 / 0 / 1 |

## Qualidade das explicações (médias por janela)

Deleção: menor = mais fiel; inserção (relativa à janela completa): maior = mais fiel; `random` = ordens aleatórias. Jaccard/Kendall nos top-10 atributos.

| Medida | lightgbm | isolation_forest | logistic |
|---|---|---|---|
| deletion_shap | 0.369 | 0.969 | 0.196 |
| deletion_lime | 0.292 | 0.970 | 0.191 |
| deletion_random | 1.054 | 0.999 | 1.257 |
| insertion_shap | 14.432 | 0.810 | 380987.107 |
| insertion_lime | 23.155 | 0.810 | 389621.246 |
| insertion_random | 0.771 | 0.780 | 186776.820 |
| shap_lime_jaccard | 0.203 | 0.426 | 0.571 |
| shap_lime_kendall | -0.159 | 0.141 | 0.459 |
| neighbour_jaccard | 0.858 | 0.690 | 0.876 |
| top_component_agree | 0.100 | 0.675 | 0.383 |
| seed_jaccard | 0.595 | 0.028 | — |
| seed_kendall | 0.406 | -0.486 | — |

## Custo (CPU)

| Modelo | Medida | Valor |
|---|---|---|
| lightgbm | train_seconds_mean | 17.69 |
| lightgbm | inference_ms_per_window | 0.0228 |
| lightgbm | xai_seconds_per_window | 0.73 |
| isolation_forest | train_seconds_mean | 0.33 |
| isolation_forest | inference_ms_per_window | 0.0122 |
| isolation_forest | xai_seconds_per_window | 1.37 |
| logistic | train_seconds_mean | 1.13 |
| logistic | inference_ms_per_window | 0.0041 |
| logistic | xai_seconds_per_window | 0.36 |
| logistic | nonzero_coefficients_mean | 214.0 |
| logistic | not_converged | 0 |

Seleção reaproveitada de `reports/tables/alt_models_Bank_alt_tabular.json` (sem nova busca).
