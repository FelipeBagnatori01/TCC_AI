# Modelos alternativos explicáveis — Telecom (família `tcn`, ADR 0018)

Mesmas janelas, dobras (10) e protocolo da E5/E6. Seleção na validação: tcn {"normalization": "local", "params": {"blocks": 2, "channels": 32}}. Detecção: média ± desvio-padrão entre seeds (a logística é determinística: 1 seed); as linhas `E4:` são execuções únicas, com IC de 95% por bootstrap. Todos os candidatos (alternativos, LSTMs da E5 e baselines da E4) são avaliados nas mesmas janelas e falhas.

## Detecção (por janela)

| Modelo | AUC-PR | AUC-PR por dobra | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|---|
| tcn | 0.34 ± 0.02 | 0.53 ± 0.03 | 0.72 ± 0.01 | 0.20 ± 0.03 | 0.41 ± 0.04 | 0.27 ± 0.03 | 16.02 ± 1.68 |
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
| tcn/shap/sum | 0.27 [0.13, 0.40] | 0.44 [0.31, 0.58] | 0.60 [0.47, 0.73] | 0.42 [0.31, 0.54] |
| tcn/shap/top3 | 0.31 [0.18, 0.44] | 0.53 [0.40, 0.67] | 0.64 [0.51, 0.78] | 0.47 [0.36, 0.59] |
| tcn/lime/sum | 0.22 [0.11, 0.36] | 0.40 [0.27, 0.53] | 0.60 [0.44, 0.73] | 0.37 [0.27, 0.49] |
| tcn/lime/top3 | 0.29 [0.18, 0.42] | 0.58 [0.44, 0.71] | 0.69 [0.56, 0.82] | 0.46 [0.36, 0.58] |
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
| detecção | tcn | lstm_forecast | auc_pr | -0.172 | [-0.306, +0.038] | 0 / 5 / 0 |
| detecção | tcn | lstm_classifier | auc_pr | -0.138 | [-0.256, +0.037] | 0 / 4 / 1 |
| detecção | tcn | zscore_3sigma | auc_pr | +0.106 | [-0.035, +0.298] | 0 / 5 / 0 |
| detecção | tcn | zscore_local | auc_pr | -0.047 | [-0.170, +0.140] | 0 / 5 / 0 |
| detecção | tcn | isolation_forest | auc_pr | +0.181 | [+0.055, +0.382] | 5 / 0 / 0 |
| detecção | tcn | lightgbm | auc_pr | +0.097 | [-0.027, +0.271] | 0 / 5 / 0 |
| detecção | tcn | prior_horario | auc_pr | +0.191 | [+0.056, +0.390] | 5 / 0 / 0 |
| localização | tcn/shap/sum | lstm_forecast/shap | hr@1 | -0.156 | [-0.333, +0.022] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_forecast/shap | mrr | -0.187 | [-0.328, -0.052] | 0 / 0 / 1 |
| localização | tcn/shap/sum | lstm_classifier/shap | hr@1 | +0.000 | [-0.156, +0.156] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_classifier/shap | mrr | -0.005 | [-0.128, +0.113] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_nativo_soma | hr@1 | -0.156 | [-0.311, +0.000] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_nativo_soma | mrr | -0.146 | [-0.266, -0.034] | 0 / 0 / 1 |
| localização | tcn/shap/sum | prior | hr@1 | +0.178 | [+0.044, +0.311] | 1 / 0 / 0 |
| localização | tcn/shap/sum | prior | mrr | +0.224 | [+0.106, +0.337] | 1 / 0 / 0 |
| localização | tcn/shap/sum | zscore_global | hr@1 | +0.133 | [-0.044, +0.311] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_global | mrr | +0.106 | [-0.046, +0.260] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_local | hr@1 | +0.178 | [+0.067, +0.311] | 1 / 0 / 0 |
| localização | tcn/shap/sum | zscore_local | mrr | +0.091 | [-0.003, +0.190] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_forecast/lime | hr@1 | -0.222 | [-0.400, -0.044] | 0 / 0 / 1 |
| localização | tcn/lime/sum | lstm_forecast/lime | mrr | -0.230 | [-0.369, -0.098] | 0 / 0 / 1 |
| localização | tcn/lime/sum | lstm_classifier/lime | hr@1 | +0.089 | [-0.044, +0.222] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_classifier/lime | mrr | +0.083 | [-0.014, +0.178] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_nativo_soma | hr@1 | -0.200 | [-0.356, -0.044] | 0 / 0 / 1 |
| localização | tcn/lime/sum | lstm_nativo_soma | mrr | -0.190 | [-0.312, -0.074] | 0 / 0 / 1 |
| localização | tcn/lime/sum | prior | hr@1 | +0.133 | [+0.000, +0.267] | 0 / 1 / 0 |
| localização | tcn/lime/sum | prior | mrr | +0.180 | [+0.061, +0.294] | 1 / 0 / 0 |
| localização | tcn/lime/sum | zscore_global | hr@1 | +0.089 | [-0.089, +0.267] | 0 / 1 / 0 |
| localização | tcn/lime/sum | zscore_global | mrr | +0.061 | [-0.091, +0.215] | 0 / 1 / 0 |
| localização | tcn/lime/sum | zscore_local | hr@1 | +0.133 | [+0.000, +0.267] | 0 / 1 / 0 |
| localização | tcn/lime/sum | zscore_local | mrr | +0.047 | [-0.051, +0.146] | 0 / 1 / 0 |

## Qualidade das explicações (médias por janela)

Deleção: menor = mais fiel; inserção (relativa à janela completa): maior = mais fiel; `random` = ordens aleatórias. Jaccard/Kendall nos top-10 atributos.

| Medida | tcn |
|---|---|
| deletion_shap | 0.590 |
| deletion_lime | 0.592 |
| deletion_random | 1.002 |
| insertion_shap | 2.251 |
| insertion_lime | 2.308 |
| insertion_random | 1.439 |
| shap_lime_jaccard | 0.664 |
| shap_lime_kendall | 0.541 |
| neighbour_jaccard | 0.796 |
| top_component_agree | 0.778 |
| seed_jaccard | 0.331 |
| seed_kendall | -0.112 |

## Custo (CPU)

| Modelo | Medida | Valor |
|---|---|---|
| tcn | train_seconds_mean | 8.17 |
| tcn | inference_ms_per_window | 0.4766 |
| tcn | xai_seconds_per_window | 2.89 |
| tcn | params_mean | 210650.6 |
| tcn | epochs_mean | 8.7 |

## Busca (validação, 1 seed; 72 treinos, 791.0 s)

| Modelo | Configuração | AUC-PR média na validação |
|---|---|---|
| tcn | global/blocks=2/channels=32 | 0.568 |
| tcn | global/blocks=3/channels=32 | 0.542 |
| tcn | global/blocks=2/channels=64 | 0.530 |
| tcn | global/blocks=3/channels=64 | 0.544 |
| tcn | local/blocks=2/channels=32 | 0.582 |
| tcn | local/blocks=3/channels=32 | 0.551 |
| tcn | local/blocks=2/channels=64 | 0.486 |
| tcn | local/blocks=3/channels=64 | 0.533 |
