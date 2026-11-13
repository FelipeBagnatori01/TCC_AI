# Modelos alternativos explicáveis — Telecom (família `tabular`, ADR 0018)

Mesmas janelas, dobras (10) e protocolo da E5/E6. Seleção na validação: lightgbm {"normalization": "global", "params": {"n_estimators": 600, "num_leaves": 31}}; isolation_forest {"normalization": "global", "params": {"max_samples": 1024, "n_estimators": 300}}; logistic {"normalization": "global", "params": {"c": 10.0}}. Detecção: média ± desvio-padrão entre seeds (a logística é determinística: 1 seed).

## Detecção (por janela)

| Modelo | AUC-PR | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|
| lightgbm | 0.21 ± 0.01 | 0.57 ± 0.01 | 0.20 ± 0.01 | 0.54 ± 0.03 | 0.29 ± 0.01 | 20.94 ± 0.70 |
| isolation_forest | 0.17 ± 0.01 | 0.61 ± 0.01 | 0.17 ± 0.01 | 0.66 ± 0.06 | 0.27 ± 0.01 | 32.47 ± 2.49 |
| logistic | 0.50 ± 0.00 | 0.77 ± 0.00 | 0.37 ± 0.00 | 0.57 ± 0.00 | 0.45 ± 0.00 | 9.38 ± 0.00 |
| lstm_forecast | 0.31 ± 0.02 | 0.71 ± 0.01 | 0.19 ± 0.04 | 0.48 ± 0.05 | 0.28 ± 0.04 | 19.94 ± 3.49 |
| lstm_classifier | 0.49 ± 0.02 | 0.84 ± 0.01 | 0.39 ± 0.04 | 0.55 ± 0.04 | 0.45 ± 0.03 | 8.63 ± 1.58 |

## Localização via XAI (janelas-oráculo, seed explicada)

| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| lightgbm/shap/sum | 0.07 [0.00, 0.16] | 0.22 [0.11, 0.36] | 0.27 [0.16, 0.40] | 0.19 [0.12, 0.28] |
| lightgbm/shap/top3 | 0.07 [0.00, 0.16] | 0.20 [0.09, 0.33] | 0.27 [0.16, 0.40] | 0.18 [0.11, 0.26] |
| lightgbm/lime/sum | 0.02 [0.00, 0.07] | 0.13 [0.04, 0.24] | 0.22 [0.11, 0.36] | 0.14 [0.09, 0.20] |
| lightgbm/lime/top3 | 0.00 [0.00, 0.00] | 0.11 [0.02, 0.20] | 0.20 [0.09, 0.31] | 0.11 [0.08, 0.15] |
| isolation_forest/shap/sum | 0.18 [0.07, 0.29] | 0.22 [0.11, 0.36] | 0.33 [0.20, 0.47] | 0.26 [0.16, 0.37] |
| isolation_forest/shap/top3 | 0.07 [0.00, 0.16] | 0.22 [0.11, 0.36] | 0.29 [0.16, 0.42] | 0.20 [0.13, 0.27] |
| isolation_forest/lime/sum | 0.13 [0.04, 0.24] | 0.24 [0.13, 0.38] | 0.33 [0.20, 0.49] | 0.24 [0.16, 0.34] |
| isolation_forest/lime/top3 | 0.16 [0.07, 0.27] | 0.27 [0.16, 0.40] | 0.38 [0.24, 0.51] | 0.27 [0.17, 0.37] |
| logistic/shap/sum | 0.09 [0.02, 0.18] | 0.16 [0.07, 0.27] | 0.22 [0.11, 0.33] | 0.18 [0.11, 0.28] |
| logistic/shap/top3 | 0.04 [0.00, 0.11] | 0.16 [0.07, 0.27] | 0.27 [0.13, 0.40] | 0.16 [0.10, 0.23] |
| logistic/lime/sum | 0.13 [0.04, 0.24] | 0.18 [0.07, 0.29] | 0.31 [0.18, 0.44] | 0.22 [0.13, 0.32] |
| logistic/lime/top3 | 0.11 [0.02, 0.20] | 0.18 [0.07, 0.29] | 0.31 [0.18, 0.44] | 0.20 [0.12, 0.30] |

## Comparações pareadas (diferença = alternativo menos referência)

Detecção: mesmas janelas de teste e mesma seed; localização: mesmas falhas. IC de 95% por bootstrap pareado; médias entre seeds e contagem de vereditos.

| Tarefa | Modelo | Referência | Métrica | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | lightgbm | lstm_forecast | auc_pr | -0.102 | [-0.266, +0.052] | 0 / 5 / 0 |
| detecção | lightgbm | lstm_classifier | auc_pr | -0.281 | [-0.405, -0.150] | 0 / 0 / 5 |
| detecção | isolation_forest | lstm_forecast | auc_pr | -0.138 | [-0.290, -0.013] | 0 / 1 / 4 |
| detecção | isolation_forest | lstm_classifier | auc_pr | -0.318 | [-0.452, -0.181] | 0 / 0 / 5 |
| detecção | logistic | lstm_forecast | auc_pr | +0.187 | [-0.010, +0.368] | 0 / 1 / 0 |
| detecção | logistic | lstm_classifier | auc_pr | +0.040 | [-0.115, +0.193] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | lstm_forecast/shap | hr@1 | -0.089 | [-0.222, +0.044] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | lstm_forecast/shap | mrr | -0.168 | [-0.282, -0.048] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_classifier/shap | hr@1 | -0.178 | [-0.311, -0.044] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_classifier/shap | mrr | -0.197 | [-0.308, -0.081] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_nativo_soma | hr@1 | -0.133 | [-0.244, -0.044] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_nativo_soma | mrr | -0.223 | [-0.305, -0.143] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | zscore_global | hr@1 | -0.067 | [-0.200, +0.067] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | zscore_global | mrr | -0.110 | [-0.233, +0.025] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | zscore_local | hr@1 | -0.044 | [-0.156, +0.067] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | zscore_local | mrr | -0.143 | [-0.229, -0.054] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_forecast/lime | hr@1 | -0.133 | [-0.244, -0.022] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_forecast/lime | mrr | -0.196 | [-0.297, -0.099] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_classifier/lime | hr@1 | -0.111 | [-0.222, +0.000] | 0 / 1 / 0 |
| localização | lightgbm/lime/sum | lstm_classifier/lime | mrr | -0.150 | [-0.247, -0.062] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_nativo_soma | hr@1 | -0.178 | [-0.311, -0.067] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_nativo_soma | mrr | -0.275 | [-0.369, -0.187] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_global | hr@1 | -0.111 | [-0.222, +0.000] | 0 / 1 / 0 |
| localização | lightgbm/lime/sum | zscore_global | mrr | -0.162 | [-0.268, -0.056] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_local | hr@1 | -0.089 | [-0.200, +0.000] | 0 / 1 / 0 |
| localização | lightgbm/lime/sum | zscore_local | mrr | -0.195 | [-0.285, -0.110] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_forecast/shap | hr@1 | +0.022 | [-0.133, +0.200] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | lstm_forecast/shap | mrr | -0.100 | [-0.231, +0.044] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | lstm_classifier/shap | hr@1 | -0.067 | [-0.222, +0.089] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | lstm_classifier/shap | mrr | -0.128 | [-0.253, -0.006] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_nativo_soma | hr@1 | -0.022 | [-0.178, +0.111] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | lstm_nativo_soma | mrr | -0.154 | [-0.260, -0.052] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | zscore_global | hr@1 | +0.044 | [-0.111, +0.200] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_global | mrr | -0.041 | [-0.177, +0.094] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_local | hr@1 | +0.067 | [-0.067, +0.200] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_local | mrr | -0.074 | [-0.186, +0.045] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | lstm_forecast/lime | hr@1 | -0.022 | [-0.156, +0.111] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | lstm_forecast/lime | mrr | -0.091 | [-0.202, +0.026] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | lstm_classifier/lime | hr@1 | +0.000 | [-0.133, +0.111] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | lstm_classifier/lime | mrr | -0.045 | [-0.152, +0.057] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | lstm_nativo_soma | hr@1 | -0.067 | [-0.200, +0.067] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | lstm_nativo_soma | mrr | -0.169 | [-0.265, -0.078] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | zscore_global | hr@1 | +0.000 | [-0.133, +0.111] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | zscore_global | mrr | -0.057 | [-0.178, +0.060] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | zscore_local | hr@1 | +0.022 | [-0.111, +0.156] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | zscore_local | mrr | -0.090 | [-0.201, +0.025] | 0 / 1 / 0 |
| localização | logistic/shap/sum | lstm_forecast/shap | hr@1 | -0.067 | [-0.200, +0.067] | 0 / 1 / 0 |
| localização | logistic/shap/sum | lstm_forecast/shap | mrr | -0.176 | [-0.290, -0.048] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_classifier/shap | hr@1 | -0.156 | [-0.289, -0.022] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_classifier/shap | mrr | -0.204 | [-0.322, -0.086] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_nativo_soma | hr@1 | -0.111 | [-0.222, -0.022] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_nativo_soma | mrr | -0.230 | [-0.318, -0.144] | 0 / 0 / 1 |
| localização | logistic/shap/sum | zscore_global | hr@1 | -0.044 | [-0.178, +0.089] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_global | mrr | -0.117 | [-0.245, +0.025] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_local | hr@1 | -0.022 | [-0.111, +0.067] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_local | mrr | -0.150 | [-0.238, -0.058] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_forecast/lime | hr@1 | -0.022 | [-0.156, +0.111] | 0 / 1 / 0 |
| localização | logistic/lime/sum | lstm_forecast/lime | mrr | -0.114 | [-0.231, +0.009] | 0 / 1 / 0 |
| localização | logistic/lime/sum | lstm_classifier/lime | hr@1 | +0.000 | [-0.133, +0.133] | 0 / 1 / 0 |
| localização | logistic/lime/sum | lstm_classifier/lime | mrr | -0.069 | [-0.168, +0.030] | 0 / 1 / 0 |
| localização | logistic/lime/sum | lstm_nativo_soma | hr@1 | -0.067 | [-0.178, +0.044] | 0 / 1 / 0 |
| localização | logistic/lime/sum | lstm_nativo_soma | mrr | -0.193 | [-0.287, -0.098] | 0 / 0 / 1 |
| localização | logistic/lime/sum | zscore_global | hr@1 | +0.000 | [-0.156, +0.156] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_global | mrr | -0.080 | [-0.214, +0.060] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_local | hr@1 | +0.022 | [-0.111, +0.156] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_local | mrr | -0.114 | [-0.208, -0.007] | 0 / 0 / 1 |

## Qualidade das explicações (médias por janela)

Deleção: menor = mais fiel; inserção (relativa à janela completa): maior = mais fiel; `random` = ordens aleatórias. Jaccard/Kendall nos top-10 atributos.

| Medida | lightgbm | isolation_forest | logistic |
|---|---|---|---|
| deletion_shap | 0.299 | 0.976 | 0.318 |
| deletion_lime | 0.340 | 0.976 | 0.311 |
| deletion_random | 1.000 | 1.000 | 1.038 |
| insertion_shap | 2.651 | 0.965 | 10.778 |
| insertion_lime | 2.140 | 0.969 | 11.039 |
| insertion_random | 0.203 | 0.945 | 5.873 |
| shap_lime_jaccard | 0.233 | 0.331 | 0.509 |
| shap_lime_kendall | -0.067 | -0.006 | 0.403 |
| neighbour_jaccard | 0.750 | 0.603 | 0.810 |
| top_component_agree | 0.400 | 0.422 | 0.667 |
| seed_jaccard | 0.639 | 0.026 | — |
| seed_kendall | 0.485 | -0.483 | — |

## Custo (CPU)

| Modelo | Medida | Valor |
|---|---|---|
| lightgbm | train_seconds_mean | 10.12 |
| lightgbm | inference_ms_per_window | 0.0264 |
| lightgbm | xai_seconds_per_window | 0.97 |
| isolation_forest | train_seconds_mean | 0.16 |
| isolation_forest | inference_ms_per_window | 0.0316 |
| isolation_forest | xai_seconds_per_window | 4.23 |
| logistic | train_seconds_mean | 0.24 |
| logistic | inference_ms_per_window | 0.0031 |
| logistic | xai_seconds_per_window | 0.47 |
| logistic | nonzero_coefficients_mean | 201.3 |
| logistic | not_converged | 0 |

Seleção reaproveitada de `reports/tables/alt_models_Telecom_alt_tabular.json` (sem nova busca).
