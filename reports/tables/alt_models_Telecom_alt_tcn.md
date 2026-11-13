# Modelos alternativos explicáveis — Telecom (família `tcn`, ADR 0018)

Mesmas janelas, dobras (10) e protocolo da E5/E6. Seleção na validação: tcn {"normalization": "local", "params": {"blocks": 2, "channels": 32}}. Detecção: média ± desvio-padrão entre seeds (a logística é determinística: 1 seed).

## Detecção (por janela)

| Modelo | AUC-PR | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|
| tcn | 0.34 ± 0.02 | 0.72 ± 0.01 | 0.20 ± 0.03 | 0.41 ± 0.04 | 0.27 ± 0.03 | 16.02 ± 1.68 |
| lstm_forecast | 0.40 ± 0.02 | 0.74 ± 0.01 | 0.19 ± 0.03 | 0.43 ± 0.04 | 0.26 ± 0.02 | 18.03 ± 3.61 |
| lstm_classifier | 0.48 ± 0.02 | 0.83 ± 0.01 | 0.37 ± 0.04 | 0.55 ± 0.02 | 0.45 ± 0.02 | 9.20 ± 1.29 |

## Localização via XAI (janelas-oráculo, seed explicada)

| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| tcn/shap/sum | 0.27 [0.13, 0.40] | 0.44 [0.31, 0.58] | 0.60 [0.47, 0.73] | 0.42 [0.31, 0.54] |
| tcn/shap/top3 | 0.31 [0.18, 0.44] | 0.53 [0.40, 0.67] | 0.64 [0.51, 0.78] | 0.47 [0.36, 0.59] |
| tcn/lime/sum | 0.22 [0.11, 0.36] | 0.40 [0.27, 0.53] | 0.60 [0.44, 0.73] | 0.37 [0.27, 0.49] |
| tcn/lime/top3 | 0.29 [0.18, 0.42] | 0.58 [0.44, 0.71] | 0.69 [0.56, 0.82] | 0.46 [0.36, 0.58] |

## Comparações pareadas (diferença = alternativo menos referência)

Detecção: mesmas janelas de teste e mesma seed; localização: mesmas falhas. IC de 95% por bootstrap pareado; médias entre seeds e contagem de vereditos.

| Tarefa | Modelo | Referência | Métrica | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | tcn | lstm_forecast | auc_pr | -0.052 | [-0.197, +0.146] | 0 / 5 / 0 |
| detecção | tcn | lstm_classifier | auc_pr | -0.138 | [-0.256, +0.037] | 0 / 4 / 1 |
| localização | tcn/shap/sum | lstm_forecast/shap | hr@1 | -0.067 | [-0.244, +0.111] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_forecast/shap | mrr | -0.074 | [-0.202, +0.053] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_classifier/shap | hr@1 | -0.044 | [-0.178, +0.089] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_classifier/shap | mrr | -0.037 | [-0.150, +0.067] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_nativo_soma | hr@1 | -0.178 | [-0.311, -0.044] | 0 / 0 / 1 |
| localização | tcn/shap/sum | lstm_nativo_soma | mrr | -0.157 | [-0.272, -0.048] | 0 / 0 / 1 |
| localização | tcn/shap/sum | zscore_global | hr@1 | +0.133 | [-0.044, +0.311] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_global | mrr | +0.117 | [-0.036, +0.269] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_local | hr@1 | +0.156 | [+0.044, +0.289] | 1 / 0 / 0 |
| localização | tcn/shap/sum | zscore_local | mrr | +0.076 | [-0.013, +0.173] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_forecast/lime | hr@1 | -0.156 | [-0.311, +0.000] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_forecast/lime | mrr | -0.136 | [-0.261, -0.013] | 0 / 0 / 1 |
| localização | tcn/lime/sum | lstm_classifier/lime | hr@1 | +0.067 | [-0.067, +0.200] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_classifier/lime | mrr | +0.063 | [-0.026, +0.151] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_nativo_soma | hr@1 | -0.222 | [-0.378, -0.067] | 0 / 0 / 1 |
| localização | tcn/lime/sum | lstm_nativo_soma | mrr | -0.201 | [-0.318, -0.088] | 0 / 0 / 1 |
| localização | tcn/lime/sum | zscore_global | hr@1 | +0.089 | [-0.089, +0.267] | 0 / 1 / 0 |
| localização | tcn/lime/sum | zscore_global | mrr | +0.073 | [-0.074, +0.227] | 0 / 1 / 0 |
| localização | tcn/lime/sum | zscore_local | hr@1 | +0.111 | [-0.022, +0.244] | 0 / 1 / 0 |
| localização | tcn/lime/sum | zscore_local | mrr | +0.032 | [-0.059, +0.129] | 0 / 1 / 0 |

## Qualidade das explicações (médias por janela)

Deleção: menor = mais fiel; inserção (relativa à janela completa): maior = mais fiel; `random` = ordens aleatórias. Jaccard/Kendall nos top-10 atributos.

| Medida | tcn |
|---|---|
| deletion_shap | 0.596 |
| deletion_lime | 0.599 |
| deletion_random | 1.003 |
| insertion_shap | 2.223 |
| insertion_lime | 2.280 |
| insertion_random | 1.421 |
| shap_lime_jaccard | 0.655 |
| shap_lime_kendall | 0.547 |
| neighbour_jaccard | 0.790 |
| top_component_agree | 0.800 |
| seed_jaccard | 0.334 |
| seed_kendall | -0.107 |

## Custo (CPU)

| Modelo | Medida | Valor |
|---|---|---|
| tcn | train_seconds_mean | 9.7 |
| tcn | inference_ms_per_window | 0.586 |
| tcn | xai_seconds_per_window | 3.56 |
| tcn | params_mean | 210650.6 |
| tcn | epochs_mean | 8.7 |

## Busca (validação, 1 seed; 72 treinos, 908.2 s)

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
