# Modelos alternativos explicáveis — Bank (família `tabular`, ADR 0018)

Mesmas janelas, dobras (8) e protocolo da E5/E6. Seleção na validação: lightgbm {"normalization": "local", "params": {"n_estimators": 600, "num_leaves": 15}}; isolation_forest {"normalization": "global", "params": {"max_samples": 256, "n_estimators": 300}}; logistic {"normalization": "global", "params": {"c": 0.1}}. Detecção: média ± desvio-padrão entre seeds (a logística é determinística: 1 seed).

## Detecção (por janela)

| Modelo | AUC-PR | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|
| lightgbm | 0.18 ± 0.01 | 0.50 ± 0.01 | 0.14 ± 0.01 | 0.51 ± 0.02 | 0.22 ± 0.01 | 22.94 ± 2.26 |
| isolation_forest | 0.14 ± 0.00 | 0.60 ± 0.01 | 0.14 ± 0.00 | 0.62 ± 0.10 | 0.23 ± 0.01 | 25.96 ± 4.36 |
| logistic | 0.10 ± 0.00 | 0.50 ± 0.00 | 0.12 ± 0.00 | 0.36 ± 0.00 | 0.18 ± 0.00 | 18.65 ± 0.00 |
| lstm_forecast | 0.20 ± 0.00 | 0.70 ± 0.00 | 0.21 ± 0.02 | 0.39 ± 0.04 | 0.27 ± 0.01 | 10.54 ± 1.73 |
| lstm_classifier | 0.22 ± 0.01 | 0.66 ± 0.01 | 0.17 ± 0.01 | 0.41 ± 0.03 | 0.24 ± 0.02 | 14.32 ± 0.56 |

## Localização via XAI (janelas-oráculo, seed explicada)

| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| lightgbm/shap/sum | 0.12 [0.07, 0.18] | 0.28 [0.20, 0.36] | 0.40 [0.32, 0.48] | 0.28 [0.23, 0.34] |
| lightgbm/shap/top3 | 0.08 [0.04, 0.13] | 0.25 [0.17, 0.33] | 0.44 [0.35, 0.53] | 0.25 [0.20, 0.29] |
| lightgbm/lime/sum | 0.02 [0.00, 0.04] | 0.13 [0.07, 0.20] | 0.38 [0.29, 0.46] | 0.18 [0.16, 0.21] |
| lightgbm/lime/top3 | 0.08 [0.04, 0.13] | 0.24 [0.17, 0.32] | 0.43 [0.34, 0.53] | 0.24 [0.20, 0.29] |
| isolation_forest/shap/sum | 0.12 [0.06, 0.17] | 0.30 [0.23, 0.38] | 0.53 [0.44, 0.62] | 0.29 [0.24, 0.34] |
| isolation_forest/shap/top3 | 0.16 [0.10, 0.23] | 0.29 [0.21, 0.38] | 0.51 [0.42, 0.60] | 0.32 [0.27, 0.37] |
| isolation_forest/lime/sum | 0.03 [0.01, 0.07] | 0.26 [0.18, 0.34] | 0.47 [0.38, 0.57] | 0.22 [0.19, 0.26] |
| isolation_forest/lime/top3 | 0.12 [0.07, 0.17] | 0.28 [0.20, 0.36] | 0.43 [0.34, 0.53] | 0.28 [0.23, 0.33] |
| logistic/shap/sum | 0.15 [0.09, 0.21] | 0.32 [0.23, 0.41] | 0.39 [0.30, 0.48] | 0.30 [0.25, 0.36] |
| logistic/shap/top3 | 0.15 [0.09, 0.22] | 0.34 [0.25, 0.43] | 0.42 [0.33, 0.51] | 0.31 [0.25, 0.37] |
| logistic/lime/sum | 0.08 [0.03, 0.13] | 0.25 [0.17, 0.33] | 0.42 [0.34, 0.52] | 0.25 [0.20, 0.29] |
| logistic/lime/top3 | 0.17 [0.11, 0.23] | 0.34 [0.26, 0.43] | 0.47 [0.38, 0.56] | 0.33 [0.27, 0.39] |

## Comparações pareadas (diferença = alternativo menos referência)

Detecção: mesmas janelas de teste e mesma seed; localização: mesmas falhas. IC de 95% por bootstrap pareado; médias entre seeds e contagem de vereditos.

| Tarefa | Modelo | Referência | Métrica | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | lightgbm | lstm_forecast | auc_pr | -0.021 | [-0.105, +0.045] | 0 / 5 / 0 |
| detecção | lightgbm | lstm_classifier | auc_pr | -0.038 | [-0.107, +0.023] | 0 / 4 / 1 |
| detecção | isolation_forest | lstm_forecast | auc_pr | -0.064 | [-0.135, -0.015] | 0 / 0 / 5 |
| detecção | isolation_forest | lstm_classifier | auc_pr | -0.081 | [-0.158, -0.013] | 0 / 0 / 5 |
| detecção | logistic | lstm_forecast | auc_pr | -0.098 | [-0.164, -0.056] | 0 / 0 / 1 |
| detecção | logistic | lstm_classifier | auc_pr | -0.123 | [-0.198, -0.064] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_forecast/shap | hr@1 | -0.242 | [-0.350, -0.133] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_forecast/shap | mrr | -0.216 | [-0.305, -0.126] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_classifier/shap | hr@1 | -0.325 | [-0.433, -0.225] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_classifier/shap | mrr | -0.333 | [-0.410, -0.256] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_nativo_soma | hr@1 | -0.225 | [-0.333, -0.125] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_nativo_soma | mrr | -0.237 | [-0.323, -0.152] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | zscore_global | hr@1 | -0.200 | [-0.308, -0.092] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | zscore_global | mrr | -0.191 | [-0.275, -0.103] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | zscore_local | hr@1 | +0.008 | [-0.067, +0.083] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | zscore_local | mrr | -0.058 | [-0.124, +0.008] | 0 / 1 / 0 |
| localização | lightgbm/lime/sum | lstm_forecast/lime | hr@1 | -0.283 | [-0.367, -0.200] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_forecast/lime | mrr | -0.279 | [-0.346, -0.211] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_classifier/lime | hr@1 | -0.200 | [-0.275, -0.125] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_classifier/lime | mrr | -0.241 | [-0.300, -0.181] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_nativo_soma | hr@1 | -0.325 | [-0.417, -0.242] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_nativo_soma | mrr | -0.336 | [-0.401, -0.272] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_global | hr@1 | -0.300 | [-0.392, -0.208] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_global | mrr | -0.290 | [-0.364, -0.215] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_local | hr@1 | -0.092 | [-0.150, -0.033] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_local | mrr | -0.157 | [-0.210, -0.102] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_forecast/shap | hr@1 | -0.242 | [-0.333, -0.150] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_forecast/shap | mrr | -0.207 | [-0.284, -0.130] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_classifier/shap | hr@1 | -0.325 | [-0.425, -0.217] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_classifier/shap | mrr | -0.324 | [-0.399, -0.246] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_nativo_soma | hr@1 | -0.225 | [-0.317, -0.133] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_nativo_soma | mrr | -0.228 | [-0.295, -0.156] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | zscore_global | hr@1 | -0.200 | [-0.308, -0.092] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | zscore_global | mrr | -0.182 | [-0.272, -0.097] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | zscore_local | hr@1 | +0.008 | [-0.067, +0.083] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_local | mrr | -0.049 | [-0.108, +0.015] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | lstm_forecast/lime | hr@1 | -0.267 | [-0.350, -0.183] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_forecast/lime | mrr | -0.237 | [-0.300, -0.172] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_classifier/lime | hr@1 | -0.183 | [-0.267, -0.100] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_classifier/lime | mrr | -0.199 | [-0.260, -0.137] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_nativo_soma | hr@1 | -0.308 | [-0.392, -0.225] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_nativo_soma | mrr | -0.294 | [-0.359, -0.232] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | zscore_global | hr@1 | -0.283 | [-0.375, -0.192] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | zscore_global | mrr | -0.248 | [-0.325, -0.173] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | zscore_local | hr@1 | -0.075 | [-0.133, -0.017] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | zscore_local | mrr | -0.115 | [-0.167, -0.063] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_forecast/shap | hr@1 | -0.208 | [-0.317, -0.092] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_forecast/shap | mrr | -0.198 | [-0.283, -0.106] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_classifier/shap | hr@1 | -0.292 | [-0.392, -0.183] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_classifier/shap | mrr | -0.315 | [-0.391, -0.233] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_nativo_soma | hr@1 | -0.192 | [-0.300, -0.075] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_nativo_soma | mrr | -0.219 | [-0.306, -0.131] | 0 / 0 / 1 |
| localização | logistic/shap/sum | zscore_global | hr@1 | -0.167 | [-0.258, -0.067] | 0 / 0 / 1 |
| localização | logistic/shap/sum | zscore_global | mrr | -0.173 | [-0.243, -0.097] | 0 / 0 / 1 |
| localização | logistic/shap/sum | zscore_local | hr@1 | +0.042 | [-0.042, +0.133] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_local | mrr | -0.040 | [-0.107, +0.032] | 0 / 1 / 0 |
| localização | logistic/lime/sum | lstm_forecast/lime | hr@1 | -0.217 | [-0.308, -0.125] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_forecast/lime | mrr | -0.212 | [-0.288, -0.134] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_classifier/lime | hr@1 | -0.133 | [-0.217, -0.058] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_classifier/lime | mrr | -0.174 | [-0.236, -0.113] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_nativo_soma | hr@1 | -0.258 | [-0.350, -0.167] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_nativo_soma | mrr | -0.269 | [-0.344, -0.194] | 0 / 0 / 1 |
| localização | logistic/lime/sum | zscore_global | hr@1 | -0.233 | [-0.325, -0.142] | 0 / 0 / 1 |
| localização | logistic/lime/sum | zscore_global | mrr | -0.223 | [-0.293, -0.149] | 0 / 0 / 1 |
| localização | logistic/lime/sum | zscore_local | hr@1 | -0.025 | [-0.100, +0.050] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_local | mrr | -0.090 | [-0.150, -0.028] | 0 / 0 / 1 |

## Qualidade das explicações (médias por janela)

Deleção: menor = mais fiel; inserção (relativa à janela completa): maior = mais fiel; `random` = ordens aleatórias. Jaccard/Kendall nos top-10 atributos.

| Medida | lightgbm | isolation_forest | logistic |
|---|---|---|---|
| deletion_shap | 0.366 | 0.969 | 0.198 |
| deletion_lime | 0.282 | 0.970 | 0.194 |
| deletion_random | 1.017 | 0.999 | 2.405 |
| insertion_shap | 14.620 | 0.802 | 379134.543 |
| insertion_lime | 16.149 | 0.801 | 384472.040 |
| insertion_random | 0.786 | 0.772 | 149353.187 |
| shap_lime_jaccard | 0.207 | 0.382 | 0.561 |
| shap_lime_kendall | -0.155 | 0.085 | 0.441 |
| neighbour_jaccard | 0.861 | 0.736 | 0.878 |
| top_component_agree | 0.100 | 0.617 | 0.317 |
| seed_jaccard | 0.595 | 0.029 | — |
| seed_kendall | 0.404 | -0.483 | — |

## Custo (CPU)

| Modelo | Medida | Valor |
|---|---|---|
| lightgbm | train_seconds_mean | 21.43 |
| lightgbm | inference_ms_per_window | 0.0259 |
| lightgbm | xai_seconds_per_window | 0.85 |
| isolation_forest | train_seconds_mean | 0.35 |
| isolation_forest | inference_ms_per_window | 0.0149 |
| isolation_forest | xai_seconds_per_window | 1.56 |
| logistic | train_seconds_mean | 1.29 |
| logistic | inference_ms_per_window | 0.0043 |
| logistic | xai_seconds_per_window | 0.41 |
| logistic | nonzero_coefficients_mean | 214.4 |
| logistic | not_converged | 0 |

## Busca (validação, 1 seed; 168 treinos, 1296.5 s)

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
