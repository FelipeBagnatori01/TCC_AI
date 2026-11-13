# Modelos alternativos explicáveis — Telecom (família `tabular`, ADR 0018)

Mesmas janelas, dobras (10) e protocolo da E5/E6. Seleção na validação: lightgbm {"normalization": "global", "params": {"n_estimators": 600, "num_leaves": 31}}; isolation_forest {"normalization": "global", "params": {"max_samples": 1024, "n_estimators": 300}}; logistic {"normalization": "global", "params": {"c": 10.0}}. Detecção: média ± desvio-padrão entre seeds (a logística é determinística: 1 seed).

## Detecção (por janela)

| Modelo | AUC-PR | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|
| lightgbm | 0.20 ± 0.01 | 0.56 ± 0.01 | 0.20 ± 0.01 | 0.53 ± 0.02 | 0.29 ± 0.01 | 20.85 ± 1.11 |
| isolation_forest | 0.17 ± 0.00 | 0.60 ± 0.01 | 0.17 ± 0.01 | 0.65 ± 0.04 | 0.27 ± 0.01 | 30.98 ± 3.26 |
| logistic | 0.51 ± 0.00 | 0.78 ± 0.00 | 0.38 ± 0.00 | 0.54 ± 0.00 | 0.44 ± 0.00 | 8.91 ± 0.00 |
| lstm_forecast | 0.40 ± 0.02 | 0.74 ± 0.01 | 0.19 ± 0.03 | 0.43 ± 0.04 | 0.26 ± 0.02 | 18.03 ± 3.61 |
| lstm_classifier | 0.48 ± 0.02 | 0.83 ± 0.01 | 0.37 ± 0.04 | 0.55 ± 0.02 | 0.45 ± 0.02 | 9.20 ± 1.29 |

## Localização via XAI (janelas-oráculo, seed explicada)

| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| lightgbm/shap/sum | 0.07 [0.00, 0.16] | 0.24 [0.13, 0.38] | 0.27 [0.13, 0.40] | 0.18 [0.11, 0.26] |
| lightgbm/shap/top3 | 0.07 [0.00, 0.16] | 0.20 [0.09, 0.33] | 0.22 [0.11, 0.36] | 0.17 [0.10, 0.25] |
| lightgbm/lime/sum | 0.02 [0.00, 0.07] | 0.16 [0.07, 0.27] | 0.20 [0.09, 0.31] | 0.14 [0.09, 0.20] |
| lightgbm/lime/top3 | 0.00 [0.00, 0.00] | 0.18 [0.07, 0.31] | 0.20 [0.09, 0.31] | 0.13 [0.08, 0.18] |
| isolation_forest/shap/sum | 0.09 [0.02, 0.18] | 0.24 [0.13, 0.38] | 0.38 [0.24, 0.53] | 0.23 [0.15, 0.31] |
| isolation_forest/shap/top3 | 0.11 [0.02, 0.22] | 0.24 [0.11, 0.38] | 0.24 [0.13, 0.38] | 0.22 [0.14, 0.31] |
| isolation_forest/lime/sum | 0.09 [0.02, 0.18] | 0.20 [0.09, 0.33] | 0.29 [0.16, 0.42] | 0.20 [0.12, 0.29] |
| isolation_forest/lime/top3 | 0.09 [0.02, 0.18] | 0.27 [0.13, 0.40] | 0.38 [0.24, 0.51] | 0.23 [0.15, 0.31] |
| logistic/shap/sum | 0.13 [0.04, 0.24] | 0.16 [0.07, 0.27] | 0.22 [0.11, 0.33] | 0.21 [0.12, 0.31] |
| logistic/shap/top3 | 0.11 [0.02, 0.22] | 0.16 [0.07, 0.27] | 0.24 [0.13, 0.38] | 0.20 [0.12, 0.30] |
| logistic/lime/sum | 0.09 [0.02, 0.18] | 0.24 [0.13, 0.38] | 0.31 [0.18, 0.44] | 0.21 [0.14, 0.30] |
| logistic/lime/top3 | 0.11 [0.02, 0.20] | 0.20 [0.09, 0.33] | 0.31 [0.18, 0.44] | 0.21 [0.13, 0.30] |

## Comparações pareadas (diferença = alternativo menos referência)

Detecção: mesmas janelas de teste e mesma seed; localização: mesmas falhas. IC de 95% por bootstrap pareado; médias entre seeds e contagem de vereditos.

| Tarefa | Modelo | Referência | Métrica | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | lightgbm | lstm_forecast | auc_pr | -0.193 | [-0.345, -0.039] | 0 / 1 / 4 |
| detecção | lightgbm | lstm_classifier | auc_pr | -0.279 | [-0.403, -0.149] | 0 / 0 / 5 |
| detecção | isolation_forest | lstm_forecast | auc_pr | -0.226 | [-0.379, -0.078] | 0 / 0 / 5 |
| detecção | isolation_forest | lstm_classifier | auc_pr | -0.312 | [-0.451, -0.170] | 0 / 0 / 5 |
| detecção | logistic | lstm_forecast | auc_pr | +0.112 | [-0.054, +0.293] | 0 / 1 / 0 |
| detecção | logistic | lstm_classifier | auc_pr | +0.052 | [-0.102, +0.189] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | lstm_forecast/shap | hr@1 | -0.267 | [-0.400, -0.133] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_forecast/shap | mrr | -0.315 | [-0.421, -0.205] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_classifier/shap | hr@1 | -0.244 | [-0.378, -0.111] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_classifier/shap | mrr | -0.278 | [-0.388, -0.170] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_nativo_soma | hr@1 | -0.378 | [-0.533, -0.244] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_nativo_soma | mrr | -0.398 | [-0.503, -0.296] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | zscore_global | hr@1 | -0.067 | [-0.200, +0.067] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | zscore_global | mrr | -0.124 | [-0.242, +0.006] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | zscore_local | hr@1 | -0.044 | [-0.156, +0.067] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | zscore_local | mrr | -0.165 | [-0.245, -0.080] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_forecast/lime | hr@1 | -0.356 | [-0.489, -0.222] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_forecast/lime | mrr | -0.372 | [-0.494, -0.254] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_classifier/lime | hr@1 | -0.133 | [-0.222, -0.044] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_classifier/lime | mrr | -0.173 | [-0.264, -0.089] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_nativo_soma | hr@1 | -0.422 | [-0.578, -0.289] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_nativo_soma | mrr | -0.438 | [-0.552, -0.328] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_global | hr@1 | -0.111 | [-0.222, +0.000] | 0 / 1 / 0 |
| localização | lightgbm/lime/sum | zscore_global | mrr | -0.163 | [-0.272, -0.047] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_local | hr@1 | -0.089 | [-0.200, +0.022] | 0 / 1 / 0 |
| localização | lightgbm/lime/sum | zscore_local | mrr | -0.204 | [-0.299, -0.110] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_forecast/shap | hr@1 | -0.244 | [-0.400, -0.089] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_forecast/shap | mrr | -0.266 | [-0.390, -0.140] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_classifier/shap | hr@1 | -0.222 | [-0.378, -0.067] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_classifier/shap | mrr | -0.229 | [-0.346, -0.117] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_nativo_soma | hr@1 | -0.356 | [-0.511, -0.200] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_nativo_soma | mrr | -0.349 | [-0.465, -0.230] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | zscore_global | hr@1 | -0.044 | [-0.178, +0.089] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_global | mrr | -0.075 | [-0.194, +0.053] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_local | hr@1 | -0.022 | [-0.133, +0.089] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_local | mrr | -0.116 | [-0.220, -0.010] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_forecast/lime | hr@1 | -0.289 | [-0.444, -0.156] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_forecast/lime | mrr | -0.309 | [-0.438, -0.191] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_classifier/lime | hr@1 | -0.067 | [-0.178, +0.044] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | lstm_classifier/lime | mrr | -0.110 | [-0.211, -0.016] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_nativo_soma | hr@1 | -0.356 | [-0.511, -0.200] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_nativo_soma | mrr | -0.374 | [-0.499, -0.256] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | zscore_global | hr@1 | -0.044 | [-0.178, +0.111] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | zscore_global | mrr | -0.100 | [-0.217, +0.032] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | zscore_local | hr@1 | -0.022 | [-0.156, +0.111] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | zscore_local | mrr | -0.141 | [-0.260, -0.023] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_forecast/shap | hr@1 | -0.200 | [-0.356, -0.044] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_forecast/shap | mrr | -0.284 | [-0.411, -0.155] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_classifier/shap | hr@1 | -0.178 | [-0.333, -0.022] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_classifier/shap | mrr | -0.247 | [-0.374, -0.115] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_nativo_soma | hr@1 | -0.311 | [-0.467, -0.156] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_nativo_soma | mrr | -0.367 | [-0.497, -0.232] | 0 / 0 / 1 |
| localização | logistic/shap/sum | zscore_global | hr@1 | +0.000 | [-0.156, +0.156] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_global | mrr | -0.093 | [-0.228, +0.053] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_local | hr@1 | +0.022 | [-0.089, +0.133] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_local | mrr | -0.134 | [-0.232, -0.027] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_forecast/lime | hr@1 | -0.289 | [-0.467, -0.133] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_forecast/lime | mrr | -0.299 | [-0.419, -0.177] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_classifier/lime | hr@1 | -0.067 | [-0.178, +0.044] | 0 / 1 / 0 |
| localização | logistic/lime/sum | lstm_classifier/lime | mrr | -0.100 | [-0.189, -0.010] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_nativo_soma | hr@1 | -0.356 | [-0.511, -0.200] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_nativo_soma | mrr | -0.364 | [-0.477, -0.255] | 0 / 0 / 1 |
| localização | logistic/lime/sum | zscore_global | hr@1 | -0.044 | [-0.178, +0.089] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_global | mrr | -0.089 | [-0.209, +0.035] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_local | hr@1 | -0.022 | [-0.156, +0.111] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_local | mrr | -0.130 | [-0.222, -0.032] | 0 / 0 / 1 |

## Qualidade das explicações (médias por janela)

Deleção: menor = mais fiel; inserção (relativa à janela completa): maior = mais fiel; `random` = ordens aleatórias. Jaccard/Kendall nos top-10 atributos.

| Medida | lightgbm | isolation_forest | logistic |
|---|---|---|---|
| deletion_shap | 0.281 | 0.972 | 0.327 |
| deletion_lime | 0.333 | 0.972 | 0.318 |
| deletion_random | 0.982 | 1.000 | 1.032 |
| insertion_shap | 4.256 | 0.952 | 9.911 |
| insertion_lime | 2.985 | 0.956 | 10.257 |
| insertion_random | 0.269 | 0.930 | 5.473 |
| shap_lime_jaccard | 0.248 | 0.366 | 0.530 |
| shap_lime_kendall | -0.047 | 0.072 | 0.431 |
| neighbour_jaccard | 0.759 | 0.656 | 0.797 |
| top_component_agree | 0.422 | 0.489 | 0.644 |
| seed_jaccard | 0.624 | 0.033 | — |
| seed_kendall | 0.424 | -0.463 | — |

## Custo (CPU)

| Modelo | Medida | Valor |
|---|---|---|
| lightgbm | train_seconds_mean | 10.03 |
| lightgbm | inference_ms_per_window | 0.0263 |
| lightgbm | xai_seconds_per_window | 1.01 |
| isolation_forest | train_seconds_mean | 0.15 |
| isolation_forest | inference_ms_per_window | 0.0322 |
| isolation_forest | xai_seconds_per_window | 4.21 |
| logistic | train_seconds_mean | 0.24 |
| logistic | inference_ms_per_window | 0.0041 |
| logistic | xai_seconds_per_window | 0.49 |
| logistic | nonzero_coefficients_mean | 191.0 |
| logistic | not_converged | 0 |

## Busca (validação, 1 seed; 216 treinos, 703.0 s)

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
