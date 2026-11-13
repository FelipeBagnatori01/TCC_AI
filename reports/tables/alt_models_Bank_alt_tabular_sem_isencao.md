# Modelos alternativos explicáveis — Bank (família `tabular`, ADR 0018)

Mesmas janelas, dobras (8) e protocolo da E5/E6. Seleção na validação: lightgbm {"normalization": "local", "params": {"n_estimators": 600, "num_leaves": 15}}; isolation_forest {"normalization": "global", "params": {"max_samples": 256, "n_estimators": 300}}; logistic {"normalization": "global", "params": {"c": 0.1}}. Detecção: média ± desvio-padrão entre seeds (a logística é determinística: 1 seed).

## Detecção (por janela)

| Modelo | AUC-PR | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|
| lightgbm | 0.18 ± 0.01 | 0.50 ± 0.01 | 0.14 ± 0.01 | 0.51 ± 0.02 | 0.22 ± 0.01 | 22.94 ± 2.26 |
| isolation_forest | 0.14 ± 0.00 | 0.60 ± 0.01 | 0.14 ± 0.01 | 0.67 ± 0.06 | 0.23 ± 0.01 | 28.72 ± 3.78 |
| logistic | 0.10 ± 0.00 | 0.50 ± 0.00 | 0.12 ± 0.00 | 0.35 ± 0.00 | 0.18 ± 0.00 | 18.47 ± 0.00 |
| lstm_forecast | 0.20 ± 0.00 | 0.70 ± 0.00 | 0.22 ± 0.01 | 0.39 ± 0.03 | 0.28 ± 0.01 | 9.98 ± 0.50 |
| lstm_classifier | 0.21 ± 0.01 | 0.65 ± 0.01 | 0.16 ± 0.01 | 0.39 ± 0.05 | 0.23 ± 0.01 | 14.12 ± 2.17 |

## Localização via XAI (janelas-oráculo, seed explicada)

| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| lightgbm/shap/sum | 0.12 [0.07, 0.18] | 0.28 [0.20, 0.36] | 0.40 [0.32, 0.48] | 0.28 [0.23, 0.34] |
| lightgbm/shap/top3 | 0.08 [0.04, 0.13] | 0.25 [0.17, 0.33] | 0.44 [0.35, 0.53] | 0.25 [0.20, 0.29] |
| lightgbm/lime/sum | 0.02 [0.00, 0.04] | 0.12 [0.07, 0.18] | 0.32 [0.24, 0.40] | 0.18 [0.16, 0.20] |
| lightgbm/lime/top3 | 0.09 [0.04, 0.14] | 0.28 [0.21, 0.37] | 0.42 [0.33, 0.52] | 0.26 [0.22, 0.31] |
| isolation_forest/shap/sum | 0.09 [0.04, 0.14] | 0.38 [0.29, 0.47] | 0.54 [0.45, 0.63] | 0.29 [0.24, 0.33] |
| isolation_forest/shap/top3 | 0.14 [0.08, 0.21] | 0.34 [0.27, 0.42] | 0.51 [0.42, 0.59] | 0.32 [0.26, 0.37] |
| isolation_forest/lime/sum | 0.04 [0.01, 0.07] | 0.25 [0.17, 0.33] | 0.48 [0.39, 0.57] | 0.23 [0.20, 0.27] |
| isolation_forest/lime/top3 | 0.13 [0.07, 0.20] | 0.29 [0.21, 0.38] | 0.48 [0.39, 0.57] | 0.30 [0.24, 0.35] |
| logistic/shap/sum | 0.15 [0.09, 0.21] | 0.32 [0.23, 0.41] | 0.39 [0.30, 0.48] | 0.30 [0.25, 0.36] |
| logistic/shap/top3 | 0.15 [0.09, 0.22] | 0.34 [0.25, 0.43] | 0.42 [0.33, 0.51] | 0.31 [0.25, 0.37] |
| logistic/lime/sum | 0.07 [0.03, 0.11] | 0.24 [0.17, 0.33] | 0.47 [0.38, 0.57] | 0.24 [0.21, 0.29] |
| logistic/lime/top3 | 0.14 [0.08, 0.20] | 0.34 [0.26, 0.43] | 0.45 [0.36, 0.54] | 0.31 [0.26, 0.37] |

## Comparações pareadas (diferença = alternativo menos referência)

Detecção: mesmas janelas de teste e mesma seed; localização: mesmas falhas. IC de 95% por bootstrap pareado; médias entre seeds e contagem de vereditos.

| Tarefa | Modelo | Referência | Métrica | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | lightgbm | lstm_forecast | auc_pr | -0.023 | [-0.109, +0.042] | 0 / 5 / 0 |
| detecção | lightgbm | lstm_classifier | auc_pr | -0.029 | [-0.097, +0.033] | 0 / 5 / 0 |
| detecção | isolation_forest | lstm_forecast | auc_pr | -0.069 | [-0.143, -0.019] | 0 / 0 / 5 |
| detecção | isolation_forest | lstm_classifier | auc_pr | -0.075 | [-0.150, -0.009] | 0 / 1 / 4 |
| detecção | logistic | lstm_forecast | auc_pr | -0.098 | [-0.176, -0.054] | 0 / 0 / 1 |
| detecção | logistic | lstm_classifier | auc_pr | -0.108 | [-0.180, -0.052] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_forecast/shap | hr@1 | -0.242 | [-0.350, -0.141] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_forecast/shap | mrr | -0.223 | [-0.309, -0.135] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_classifier/shap | hr@1 | -0.317 | [-0.425, -0.217] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_classifier/shap | mrr | -0.325 | [-0.404, -0.247] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_nativo_soma | hr@1 | -0.208 | [-0.317, -0.100] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_nativo_soma | mrr | -0.229 | [-0.313, -0.145] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | zscore_global | hr@1 | -0.200 | [-0.308, -0.092] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | zscore_global | mrr | -0.191 | [-0.275, -0.103] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | zscore_local | hr@1 | +0.008 | [-0.067, +0.083] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | zscore_local | mrr | -0.058 | [-0.124, +0.008] | 0 / 1 / 0 |
| localização | lightgbm/lime/sum | lstm_forecast/lime | hr@1 | -0.308 | [-0.392, -0.225] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_forecast/lime | mrr | -0.295 | [-0.366, -0.226] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_classifier/lime | hr@1 | -0.192 | [-0.275, -0.117] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_classifier/lime | mrr | -0.249 | [-0.311, -0.187] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_nativo_soma | hr@1 | -0.308 | [-0.392, -0.233] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_nativo_soma | mrr | -0.332 | [-0.395, -0.269] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_global | hr@1 | -0.300 | [-0.392, -0.208] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_global | mrr | -0.294 | [-0.370, -0.217] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_local | hr@1 | -0.092 | [-0.150, -0.033] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_local | mrr | -0.161 | [-0.212, -0.107] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_forecast/shap | hr@1 | -0.267 | [-0.350, -0.175] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_forecast/shap | mrr | -0.215 | [-0.285, -0.144] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_classifier/shap | hr@1 | -0.342 | [-0.442, -0.242] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_classifier/shap | mrr | -0.318 | [-0.387, -0.247] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_nativo_soma | hr@1 | -0.233 | [-0.325, -0.150] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_nativo_soma | mrr | -0.221 | [-0.286, -0.156] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | zscore_global | hr@1 | -0.225 | [-0.325, -0.125] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | zscore_global | mrr | -0.183 | [-0.261, -0.106] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | zscore_local | hr@1 | -0.017 | [-0.083, +0.058] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_local | mrr | -0.050 | [-0.106, +0.006] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | lstm_forecast/lime | hr@1 | -0.283 | [-0.367, -0.200] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_forecast/lime | mrr | -0.239 | [-0.304, -0.173] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_classifier/lime | hr@1 | -0.167 | [-0.242, -0.092] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_classifier/lime | mrr | -0.193 | [-0.254, -0.133] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_nativo_soma | hr@1 | -0.283 | [-0.367, -0.208] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_nativo_soma | mrr | -0.276 | [-0.338, -0.215] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | zscore_global | hr@1 | -0.275 | [-0.367, -0.192] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | zscore_global | mrr | -0.238 | [-0.310, -0.168] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | zscore_local | hr@1 | -0.067 | [-0.125, -0.008] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | zscore_local | mrr | -0.105 | [-0.156, -0.056] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_forecast/shap | hr@1 | -0.208 | [-0.317, -0.092] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_forecast/shap | mrr | -0.206 | [-0.290, -0.114] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_classifier/shap | hr@1 | -0.283 | [-0.375, -0.183] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_classifier/shap | mrr | -0.308 | [-0.381, -0.231] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_nativo_soma | hr@1 | -0.175 | [-0.283, -0.067] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_nativo_soma | mrr | -0.211 | [-0.295, -0.127] | 0 / 0 / 1 |
| localização | logistic/shap/sum | zscore_global | hr@1 | -0.167 | [-0.258, -0.067] | 0 / 0 / 1 |
| localização | logistic/shap/sum | zscore_global | mrr | -0.173 | [-0.243, -0.097] | 0 / 0 / 1 |
| localização | logistic/shap/sum | zscore_local | hr@1 | +0.042 | [-0.042, +0.133] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_local | mrr | -0.040 | [-0.107, +0.032] | 0 / 1 / 0 |
| localização | logistic/lime/sum | lstm_forecast/lime | hr@1 | -0.258 | [-0.350, -0.158] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_forecast/lime | mrr | -0.229 | [-0.306, -0.152] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_classifier/lime | hr@1 | -0.142 | [-0.217, -0.067] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_classifier/lime | mrr | -0.183 | [-0.244, -0.124] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_nativo_soma | hr@1 | -0.258 | [-0.350, -0.167] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_nativo_soma | mrr | -0.265 | [-0.334, -0.193] | 0 / 0 / 1 |
| localização | logistic/lime/sum | zscore_global | hr@1 | -0.250 | [-0.342, -0.158] | 0 / 0 / 1 |
| localização | logistic/lime/sum | zscore_global | mrr | -0.227 | [-0.297, -0.155] | 0 / 0 / 1 |
| localização | logistic/lime/sum | zscore_local | hr@1 | -0.042 | [-0.108, +0.025] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_local | mrr | -0.094 | [-0.152, -0.035] | 0 / 0 / 1 |

## Qualidade das explicações (médias por janela)

Deleção: menor = mais fiel; inserção (relativa à janela completa): maior = mais fiel; `random` = ordens aleatórias. Jaccard/Kendall nos top-10 atributos.

| Medida | lightgbm | isolation_forest | logistic |
|---|---|---|---|
| deletion_shap | 0.366 | 0.969 | 0.198 |
| deletion_lime | 0.292 | 0.970 | 0.193 |
| deletion_random | 1.058 | 0.999 | 1.255 |
| insertion_shap | 14.620 | 0.810 | 380958.756 |
| insertion_lime | 23.084 | 0.810 | 389592.898 |
| insertion_random | 0.771 | 0.780 | 186763.447 |
| shap_lime_jaccard | 0.204 | 0.424 | 0.574 |
| shap_lime_kendall | -0.158 | 0.140 | 0.461 |
| neighbour_jaccard | 0.861 | 0.692 | 0.878 |
| top_component_agree | 0.100 | 0.667 | 0.383 |
| seed_jaccard | 0.595 | 0.028 | — |
| seed_kendall | 0.404 | -0.486 | — |

## Custo (CPU)

| Modelo | Medida | Valor |
|---|---|---|
| lightgbm | train_seconds_mean | 21.36 |
| lightgbm | inference_ms_per_window | 0.0256 |
| lightgbm | xai_seconds_per_window | 0.81 |
| isolation_forest | train_seconds_mean | 0.35 |
| isolation_forest | inference_ms_per_window | 0.0147 |
| isolation_forest | xai_seconds_per_window | 1.51 |
| logistic | train_seconds_mean | 1.26 |
| logistic | inference_ms_per_window | 0.0045 |
| logistic | xai_seconds_per_window | 0.39 |
| logistic | nonzero_coefficients_mean | 214.0 |
| logistic | not_converged | 0 |

Seleção reaproveitada de `reports/tables/alt_models_Bank_alt_tabular.json` (sem nova busca).
