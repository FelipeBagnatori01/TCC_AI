# Modelos alternativos explicáveis — Telecom (família `tabular`, ADR 0018)

Mesmas janelas, dobras (10) e protocolo da E5/E6. Seleção na validação: lightgbm {"normalization": "global", "params": {"n_estimators": 600, "num_leaves": 31}}; isolation_forest {"normalization": "global", "params": {"max_samples": 1024, "n_estimators": 300}}; logistic {"normalization": "global", "params": {"c": 10.0}}. Detecção: média ± desvio-padrão entre seeds (a logística é determinística: 1 seed); as linhas `E4:` são execuções únicas, com IC de 95% por bootstrap. Todos os candidatos (alternativos, LSTMs da E5 e baselines da E4) são avaliados nas mesmas janelas e falhas.

## Detecção (por janela)

| Modelo | AUC-PR | AUC-PR por dobra | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|---|
| lightgbm | 0.21 ± 0.01 | 0.57 ± 0.01 | 0.57 ± 0.01 | 0.20 ± 0.01 | 0.54 ± 0.03 | 0.29 ± 0.01 | 20.94 ± 0.70 |
| isolation_forest | 0.17 ± 0.01 | 0.37 ± 0.03 | 0.61 ± 0.01 | 0.17 ± 0.01 | 0.66 ± 0.06 | 0.27 ± 0.01 | 32.47 ± 2.49 |
| logistic | 0.50 ± 0.00 | 0.62 ± 0.00 | 0.77 ± 0.00 | 0.37 ± 0.00 | 0.57 ± 0.00 | 0.45 ± 0.00 | 9.38 ± 0.00 |
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
| lightgbm/shap/sum | 0.04 [0.00, 0.11] | 0.22 [0.11, 0.36] | 0.27 [0.16, 0.40] | 0.18 [0.12, 0.26] |
| lightgbm/shap/top3 | 0.04 [0.00, 0.11] | 0.18 [0.07, 0.29] | 0.27 [0.16, 0.40] | 0.16 [0.10, 0.24] |
| lightgbm/lime/sum | 0.02 [0.00, 0.07] | 0.13 [0.04, 0.24] | 0.22 [0.11, 0.36] | 0.14 [0.09, 0.20] |
| lightgbm/lime/top3 | 0.00 [0.00, 0.00] | 0.11 [0.02, 0.20] | 0.22 [0.11, 0.36] | 0.11 [0.08, 0.15] |
| isolation_forest/shap/sum | 0.18 [0.07, 0.29] | 0.22 [0.11, 0.36] | 0.33 [0.20, 0.47] | 0.26 [0.16, 0.37] |
| isolation_forest/shap/top3 | 0.07 [0.00, 0.16] | 0.22 [0.11, 0.36] | 0.29 [0.16, 0.42] | 0.20 [0.13, 0.27] |
| isolation_forest/lime/sum | 0.13 [0.04, 0.24] | 0.24 [0.13, 0.38] | 0.33 [0.20, 0.49] | 0.24 [0.16, 0.34] |
| isolation_forest/lime/top3 | 0.16 [0.07, 0.27] | 0.27 [0.16, 0.40] | 0.38 [0.24, 0.51] | 0.27 [0.17, 0.37] |
| logistic/shap/sum | 0.09 [0.02, 0.18] | 0.16 [0.07, 0.27] | 0.22 [0.11, 0.33] | 0.19 [0.11, 0.28] |
| logistic/shap/top3 | 0.04 [0.00, 0.11] | 0.16 [0.07, 0.27] | 0.27 [0.13, 0.40] | 0.16 [0.10, 0.23] |
| logistic/lime/sum | 0.11 [0.02, 0.22] | 0.16 [0.07, 0.27] | 0.31 [0.18, 0.44] | 0.20 [0.12, 0.30] |
| logistic/lime/top3 | 0.09 [0.02, 0.18] | 0.18 [0.07, 0.29] | 0.31 [0.18, 0.44] | 0.19 [0.11, 0.28] |
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
| detecção | lightgbm | lstm_forecast | auc_pr | -0.220 | [-0.367, -0.053] | 0 / 0 / 5 |
| detecção | lightgbm | lstm_classifier | auc_pr | -0.281 | [-0.405, -0.150] | 0 / 0 / 5 |
| detecção | lightgbm | zscore_3sigma | auc_pr | -0.029 | [-0.161, +0.103] | 0 / 5 / 0 |
| detecção | lightgbm | zscore_local | auc_pr | -0.175 | [-0.311, -0.024] | 0 / 0 / 5 |
| detecção | lightgbm | isolation_forest | auc_pr | +0.026 | [-0.105, +0.163] | 0 / 5 / 0 |
| detecção | lightgbm | lightgbm | auc_pr | -0.052 | [-0.100, -0.008] | 0 / 0 / 5 |
| detecção | lightgbm | prior_horario | auc_pr | +0.056 | [-0.064, +0.185] | 0 / 5 / 0 |
| detecção | isolation_forest | lstm_forecast | auc_pr | -0.257 | [-0.393, -0.113] | 0 / 0 / 5 |
| detecção | isolation_forest | lstm_classifier | auc_pr | -0.318 | [-0.452, -0.181] | 0 / 0 / 5 |
| detecção | isolation_forest | zscore_3sigma | auc_pr | -0.066 | [-0.143, -0.007] | 0 / 1 / 4 |
| detecção | isolation_forest | zscore_local | auc_pr | -0.212 | [-0.342, -0.078] | 0 / 0 / 5 |
| detecção | isolation_forest | isolation_forest | auc_pr | -0.011 | [-0.049, +0.013] | 0 / 5 / 0 |
| detecção | isolation_forest | lightgbm | auc_pr | -0.088 | [-0.239, +0.050] | 0 / 5 / 0 |
| detecção | isolation_forest | prior_horario | auc_pr | +0.020 | [-0.066, +0.103] | 0 / 5 / 0 |
| detecção | logistic | lstm_forecast | auc_pr | +0.053 | [-0.125, +0.238] | 0 / 1 / 0 |
| detecção | logistic | lstm_classifier | auc_pr | +0.040 | [-0.115, +0.193] | 0 / 1 / 0 |
| detecção | logistic | zscore_3sigma | auc_pr | +0.266 | [+0.090, +0.418] | 1 / 0 / 0 |
| detecção | logistic | zscore_local | auc_pr | +0.119 | [-0.027, +0.274] | 0 / 1 / 0 |
| detecção | logistic | isolation_forest | auc_pr | +0.320 | [+0.140, +0.476] | 1 / 0 / 0 |
| detecção | logistic | lightgbm | auc_pr | +0.243 | [+0.107, +0.373] | 1 / 0 / 0 |
| detecção | logistic | prior_horario | auc_pr | +0.351 | [+0.173, +0.513] | 1 / 0 / 0 |
| localização | lightgbm/shap/sum | lstm_forecast/shap | hr@1 | -0.200 | [-0.311, -0.089] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_forecast/shap | mrr | -0.281 | [-0.383, -0.180] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_classifier/shap | hr@1 | -0.178 | [-0.311, -0.067] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_classifier/shap | mrr | -0.187 | [-0.293, -0.079] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_nativo_soma | hr@1 | -0.133 | [-0.244, -0.044] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | lstm_nativo_soma | mrr | -0.219 | [-0.294, -0.144] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | prior | hr@1 | -0.044 | [-0.156, +0.044] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | prior | mrr | -0.013 | [-0.105, +0.077] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | zscore_global | hr@1 | -0.089 | [-0.200, +0.023] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | zscore_global | mrr | -0.132 | [-0.250, -0.003] | 0 / 0 / 1 |
| localização | lightgbm/shap/sum | zscore_local | hr@1 | -0.044 | [-0.133, +0.044] | 0 / 1 / 0 |
| localização | lightgbm/shap/sum | zscore_local | mrr | -0.139 | [-0.210, -0.064] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_forecast/lime | hr@1 | -0.200 | [-0.333, -0.067] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_forecast/lime | mrr | -0.288 | [-0.399, -0.185] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_classifier/lime | hr@1 | -0.089 | [-0.200, +0.000] | 0 / 1 / 0 |
| localização | lightgbm/lime/sum | lstm_classifier/lime | mrr | -0.135 | [-0.226, -0.050] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_nativo_soma | hr@1 | -0.156 | [-0.289, -0.044] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | lstm_nativo_soma | mrr | -0.262 | [-0.352, -0.179] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | prior | hr@1 | -0.067 | [-0.178, +0.022] | 0 / 1 / 0 |
| localização | lightgbm/lime/sum | prior | mrr | -0.057 | [-0.144, +0.024] | 0 / 1 / 0 |
| localização | lightgbm/lime/sum | zscore_global | hr@1 | -0.111 | [-0.222, +0.000] | 0 / 1 / 0 |
| localização | lightgbm/lime/sum | zscore_global | mrr | -0.175 | [-0.281, -0.071] | 0 / 0 / 1 |
| localização | lightgbm/lime/sum | zscore_local | hr@1 | -0.067 | [-0.178, +0.022] | 0 / 1 / 0 |
| localização | lightgbm/lime/sum | zscore_local | mrr | -0.182 | [-0.266, -0.102] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_forecast/shap | hr@1 | -0.067 | [-0.244, +0.111] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | lstm_forecast/shap | mrr | -0.202 | [-0.338, -0.073] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | lstm_classifier/shap | hr@1 | -0.044 | [-0.200, +0.111] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | lstm_classifier/shap | mrr | -0.108 | [-0.231, +0.006] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | lstm_nativo_soma | hr@1 | +0.000 | [-0.133, +0.133] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | lstm_nativo_soma | mrr | -0.140 | [-0.241, -0.041] | 0 / 0 / 1 |
| localização | isolation_forest/shap/sum | prior | hr@1 | +0.089 | [-0.022, +0.222] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | prior | mrr | +0.065 | [-0.049, +0.187] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_global | hr@1 | +0.044 | [-0.111, +0.200] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_global | mrr | -0.053 | [-0.189, +0.080] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_local | hr@1 | +0.089 | [-0.022, +0.200] | 0 / 1 / 0 |
| localização | isolation_forest/shap/sum | zscore_local | mrr | -0.060 | [-0.167, +0.051] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | lstm_forecast/lime | hr@1 | -0.089 | [-0.222, +0.044] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | lstm_forecast/lime | mrr | -0.181 | [-0.294, -0.071] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | lstm_classifier/lime | hr@1 | +0.022 | [-0.089, +0.133] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | lstm_classifier/lime | mrr | -0.027 | [-0.130, +0.066] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | lstm_nativo_soma | hr@1 | -0.044 | [-0.178, +0.067] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | lstm_nativo_soma | mrr | -0.154 | [-0.244, -0.068] | 0 / 0 / 1 |
| localização | isolation_forest/lime/sum | prior | hr@1 | +0.044 | [-0.089, +0.178] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | prior | mrr | +0.051 | [-0.070, +0.180] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | zscore_global | hr@1 | +0.000 | [-0.133, +0.111] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | zscore_global | mrr | -0.068 | [-0.189, +0.050] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | zscore_local | hr@1 | +0.044 | [-0.067, +0.156] | 0 / 1 / 0 |
| localização | isolation_forest/lime/sum | zscore_local | mrr | -0.075 | [-0.180, +0.033] | 0 / 1 / 0 |
| localização | logistic/shap/sum | lstm_forecast/shap | hr@1 | -0.156 | [-0.289, -0.022] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_forecast/shap | mrr | -0.276 | [-0.388, -0.160] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_classifier/shap | hr@1 | -0.133 | [-0.267, +0.000] | 0 / 1 / 0 |
| localização | logistic/shap/sum | lstm_classifier/shap | mrr | -0.182 | [-0.299, -0.067] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_nativo_soma | hr@1 | -0.089 | [-0.178, -0.022] | 0 / 0 / 1 |
| localização | logistic/shap/sum | lstm_nativo_soma | mrr | -0.214 | [-0.294, -0.135] | 0 / 0 / 1 |
| localização | logistic/shap/sum | prior | hr@1 | +0.000 | [-0.111, +0.111] | 0 / 1 / 0 |
| localização | logistic/shap/sum | prior | mrr | -0.009 | [-0.102, +0.090] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_global | hr@1 | -0.044 | [-0.178, +0.089] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_global | mrr | -0.127 | [-0.253, +0.009] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_local | hr@1 | +0.000 | [-0.089, +0.089] | 0 / 1 / 0 |
| localização | logistic/shap/sum | zscore_local | mrr | -0.135 | [-0.218, -0.045] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_forecast/lime | hr@1 | -0.111 | [-0.244, +0.022] | 0 / 1 / 0 |
| localização | logistic/lime/sum | lstm_forecast/lime | mrr | -0.222 | [-0.335, -0.111] | 0 / 0 / 1 |
| localização | logistic/lime/sum | lstm_classifier/lime | hr@1 | +0.000 | [-0.133, +0.133] | 0 / 1 / 0 |
| localização | logistic/lime/sum | lstm_classifier/lime | mrr | -0.069 | [-0.162, +0.025] | 0 / 1 / 0 |
| localização | logistic/lime/sum | lstm_nativo_soma | hr@1 | -0.067 | [-0.178, +0.044] | 0 / 1 / 0 |
| localização | logistic/lime/sum | lstm_nativo_soma | mrr | -0.195 | [-0.289, -0.101] | 0 / 0 / 1 |
| localização | logistic/lime/sum | prior | hr@1 | +0.022 | [-0.067, +0.133] | 0 / 1 / 0 |
| localização | logistic/lime/sum | prior | mrr | +0.010 | [-0.077, +0.098] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_global | hr@1 | -0.022 | [-0.156, +0.133] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_global | mrr | -0.109 | [-0.232, +0.026] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_local | hr@1 | +0.022 | [-0.089, +0.133] | 0 / 1 / 0 |
| localização | logistic/lime/sum | zscore_local | mrr | -0.116 | [-0.205, -0.021] | 0 / 0 / 1 |

## Qualidade das explicações (médias por janela)

Deleção: menor = mais fiel; inserção (relativa à janela completa): maior = mais fiel; `random` = ordens aleatórias. Jaccard/Kendall nos top-10 atributos.

| Medida | lightgbm | isolation_forest | logistic |
|---|---|---|---|
| deletion_shap | 0.297 | 0.976 | 0.316 |
| deletion_lime | 0.335 | 0.976 | 0.304 |
| deletion_random | 0.998 | 1.000 | 1.042 |
| insertion_shap | 2.647 | 0.967 | 11.030 |
| insertion_lime | 2.127 | 0.971 | 11.323 |
| insertion_random | 0.202 | 0.947 | 6.016 |
| shap_lime_jaccard | 0.235 | 0.327 | 0.509 |
| shap_lime_kendall | -0.054 | -0.004 | 0.392 |
| neighbour_jaccard | 0.764 | 0.611 | 0.823 |
| top_component_agree | 0.378 | 0.422 | 0.689 |
| seed_jaccard | 0.633 | 0.026 | — |
| seed_kendall | 0.479 | -0.481 | — |

## Custo (CPU)

| Modelo | Medida | Valor |
|---|---|---|
| lightgbm | train_seconds_mean | 8.32 |
| lightgbm | inference_ms_per_window | 0.0229 |
| lightgbm | xai_seconds_per_window | 0.87 |
| isolation_forest | train_seconds_mean | 0.14 |
| isolation_forest | inference_ms_per_window | 0.0265 |
| isolation_forest | xai_seconds_per_window | 3.78 |
| logistic | train_seconds_mean | 0.2 |
| logistic | inference_ms_per_window | 0.0031 |
| logistic | xai_seconds_per_window | 0.42 |
| logistic | nonzero_coefficients_mean | 201.3 |
| logistic | not_converged | 0 |

Seleção reaproveitada de `reports/tables/alt_models_Telecom_alt_tabular.json` (sem nova busca).
