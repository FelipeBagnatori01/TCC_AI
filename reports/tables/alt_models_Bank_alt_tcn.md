# Modelos alternativos explicáveis — Bank (família `tcn`, ADR 0018)

Mesmas janelas, dobras (8) e protocolo da E5/E6. Seleção na validação: tcn {"normalization": "local", "params": {"blocks": 3, "channels": 32}}. Detecção: média ± desvio-padrão entre seeds (a logística é determinística: 1 seed).

## Detecção (por janela)

| Modelo | AUC-PR | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|
| tcn | 0.16 ± 0.01 | 0.52 ± 0.02 | 0.14 ± 0.01 | 0.43 ± 0.04 | 0.21 ± 0.02 | 18.73 ± 2.00 |
| lstm_forecast | 0.20 ± 0.00 | 0.70 ± 0.00 | 0.21 ± 0.02 | 0.39 ± 0.04 | 0.27 ± 0.01 | 10.54 ± 1.73 |
| lstm_classifier | 0.22 ± 0.01 | 0.66 ± 0.01 | 0.17 ± 0.01 | 0.41 ± 0.03 | 0.24 ± 0.02 | 14.32 ± 0.56 |

## Localização via XAI (janelas-oráculo, seed explicada)

| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| tcn/shap/sum | 0.25 [0.17, 0.33] | 0.49 [0.40, 0.57] | 0.65 [0.57, 0.73] | 0.42 [0.35, 0.48] |
| tcn/shap/top3 | 0.41 [0.33, 0.50] | 0.56 [0.47, 0.65] | 0.61 [0.52, 0.69] | 0.53 [0.45, 0.60] |
| tcn/lime/sum | 0.18 [0.12, 0.26] | 0.41 [0.33, 0.49] | 0.55 [0.46, 0.63] | 0.36 [0.30, 0.42] |
| tcn/lime/top3 | 0.38 [0.29, 0.46] | 0.57 [0.47, 0.66] | 0.63 [0.55, 0.72] | 0.51 [0.45, 0.59] |

## Comparações pareadas (diferença = alternativo menos referência)

Detecção: mesmas janelas de teste e mesma seed; localização: mesmas falhas. IC de 95% por bootstrap pareado; médias entre seeds e contagem de vereditos.

| Tarefa | Modelo | Referência | Métrica | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | tcn | lstm_forecast | auc_pr | -0.041 | [-0.102, +0.026] | 0 / 5 / 0 |
| detecção | tcn | lstm_classifier | auc_pr | -0.059 | [-0.109, +0.005] | 0 / 4 / 1 |
| localização | tcn/shap/sum | lstm_forecast/shap | hr@1 | -0.108 | [-0.208, +0.000] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_forecast/shap | mrr | -0.079 | [-0.155, +0.004] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_classifier/shap | hr@1 | -0.192 | [-0.283, -0.092] | 0 / 0 / 1 |
| localização | tcn/shap/sum | lstm_classifier/shap | mrr | -0.196 | [-0.259, -0.129] | 0 / 0 / 1 |
| localização | tcn/shap/sum | lstm_nativo_soma | hr@1 | -0.092 | [-0.192, +0.008] | 0 / 1 / 0 |
| localização | tcn/shap/sum | lstm_nativo_soma | mrr | -0.100 | [-0.175, -0.024] | 0 / 0 / 1 |
| localização | tcn/shap/sum | zscore_global | hr@1 | -0.067 | [-0.167, +0.033] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_global | mrr | -0.054 | [-0.131, +0.024] | 0 / 1 / 0 |
| localização | tcn/shap/sum | zscore_local | hr@1 | +0.142 | [+0.058, +0.233] | 1 / 0 / 0 |
| localização | tcn/shap/sum | zscore_local | mrr | +0.079 | [+0.020, +0.144] | 1 / 0 / 0 |
| localização | tcn/lime/sum | lstm_forecast/lime | hr@1 | -0.117 | [-0.208, -0.017] | 0 / 0 / 1 |
| localização | tcn/lime/sum | lstm_forecast/lime | mrr | -0.101 | [-0.176, -0.023] | 0 / 0 / 1 |
| localização | tcn/lime/sum | lstm_classifier/lime | hr@1 | -0.033 | [-0.108, +0.042] | 0 / 1 / 0 |
| localização | tcn/lime/sum | lstm_classifier/lime | mrr | -0.064 | [-0.116, -0.010] | 0 / 0 / 1 |
| localização | tcn/lime/sum | lstm_nativo_soma | hr@1 | -0.158 | [-0.242, -0.075] | 0 / 0 / 1 |
| localização | tcn/lime/sum | lstm_nativo_soma | mrr | -0.158 | [-0.228, -0.086] | 0 / 0 / 1 |
| localização | tcn/lime/sum | zscore_global | hr@1 | -0.133 | [-0.233, -0.025] | 0 / 0 / 1 |
| localização | tcn/lime/sum | zscore_global | mrr | -0.113 | [-0.192, -0.030] | 0 / 0 / 1 |
| localização | tcn/lime/sum | zscore_local | hr@1 | +0.075 | [+0.000, +0.158] | 0 / 1 / 0 |
| localização | tcn/lime/sum | zscore_local | mrr | +0.021 | [-0.036, +0.082] | 0 / 1 / 0 |

## Qualidade das explicações (médias por janela)

Deleção: menor = mais fiel; inserção (relativa à janela completa): maior = mais fiel; `random` = ordens aleatórias. Jaccard/Kendall nos top-10 atributos.

| Medida | tcn |
|---|---|
| deletion_shap | 0.537 |
| deletion_lime | 0.555 |
| deletion_random | 1.037 |
| insertion_shap | 149957.729 |
| insertion_lime | 132146.956 |
| insertion_random | 122013.712 |
| shap_lime_jaccard | 0.543 |
| shap_lime_kendall | 0.358 |
| neighbour_jaccard | 0.784 |
| top_component_agree | 0.650 |
| seed_jaccard | 0.259 |
| seed_kendall | -0.195 |

## Custo (CPU)

| Modelo | Medida | Valor |
|---|---|---|
| tcn | train_seconds_mean | 28.28 |
| tcn | inference_ms_per_window | 0.6008 |
| tcn | xai_seconds_per_window | 3.58 |
| tcn | params_mean | 181041.0 |
| tcn | epochs_mean | 6.6 |

## Busca (validação, 1 seed; 56 treinos, 1883.6 s)

| Modelo | Configuração | AUC-PR média na validação |
|---|---|---|
| tcn | global/blocks=2/channels=32 | 0.169 |
| tcn | global/blocks=3/channels=32 | 0.190 |
| tcn | global/blocks=2/channels=64 | 0.168 |
| tcn | global/blocks=3/channels=64 | 0.137 |
| tcn | local/blocks=2/channels=32 | 0.217 |
| tcn | local/blocks=3/channels=32 | 0.249 |
| tcn | local/blocks=2/channels=64 | 0.245 |
| tcn | local/blocks=3/channels=64 | 0.243 |
