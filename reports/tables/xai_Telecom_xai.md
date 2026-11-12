# Explicabilidade (E6) — Telecom

Janelas-oráculo (terminam 5 min após o início de cada falha testada, como na E4/E5); modelos da seed explicada. IC de 95% por bootstrap sobre falhas.

## Localização via XAI (RQ2)

| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| lstm_forecast/shap/sum | 0.33 [0.20, 0.47] | 0.62 [0.49, 0.76] | 0.76 [0.62, 0.87] | 0.49 [0.38, 0.60] |
| lstm_forecast/shap/top3 | 0.33 [0.20, 0.47] | 0.62 [0.49, 0.76] | 0.76 [0.62, 0.87] | 0.49 [0.38, 0.60] |
| lstm_forecast/lime/sum | 0.38 [0.24, 0.51] | 0.60 [0.44, 0.73] | 0.67 [0.51, 0.80] | 0.51 [0.39, 0.63] |
| lstm_forecast/lime/top3 | 0.33 [0.20, 0.47] | 0.73 [0.60, 0.84] | 0.78 [0.64, 0.89] | 0.51 [0.40, 0.62] |
| lstm_classifier/shap/sum | 0.31 [0.18, 0.44] | 0.53 [0.38, 0.67] | 0.58 [0.42, 0.71] | 0.46 [0.34, 0.57] |
| lstm_classifier/shap/top3 | 0.22 [0.11, 0.36] | 0.44 [0.31, 0.58] | 0.53 [0.38, 0.69] | 0.40 [0.29, 0.51] |
| lstm_classifier/lime/sum | 0.16 [0.07, 0.27] | 0.36 [0.22, 0.49] | 0.51 [0.38, 0.64] | 0.31 [0.22, 0.41] |
| lstm_classifier/lime/top3 | 0.20 [0.09, 0.33] | 0.33 [0.20, 0.47] | 0.51 [0.36, 0.64] | 0.35 [0.25, 0.46] |

Comparação pareada (agregação soma; diferença = XAI menos rival):

| XAI | Rival | Métrica | Diferença | Veredito |
|---|---|---|---|---|
| lstm_forecast/shap/sum | lstm_nativo_soma | hr@1 | -0.111 [-0.222, -0.044] | inferior |
| lstm_forecast/shap/sum | lstm_nativo_soma | mrr | -0.084 [-0.148, -0.029] | inferior |
| lstm_forecast/shap/sum | zscore_global | hr@1 | +0.200 [+0.044, +0.378] | superior |
| lstm_forecast/shap/sum | zscore_global | mrr | +0.191 [+0.042, +0.350] | superior |
| lstm_forecast/shap/sum | zscore_local | hr@1 | +0.222 [+0.067, +0.378] | superior |
| lstm_forecast/shap/sum | zscore_local | mrr | +0.150 [+0.015, +0.281] | superior |
| lstm_forecast/lime/sum | lstm_nativo_soma | hr@1 | -0.067 [-0.156, +0.000] | empate |
| lstm_forecast/lime/sum | lstm_nativo_soma | mrr | -0.065 [-0.123, -0.017] | inferior |
| lstm_forecast/lime/sum | zscore_global | hr@1 | +0.244 [+0.067, +0.422] | superior |
| lstm_forecast/lime/sum | zscore_global | mrr | +0.209 [+0.053, +0.373] | superior |
| lstm_forecast/lime/sum | zscore_local | hr@1 | +0.267 [+0.133, +0.400] | superior |
| lstm_forecast/lime/sum | zscore_local | mrr | +0.168 [+0.039, +0.295] | superior |
| lstm_classifier/shap/sum | lstm_nativo_soma | hr@1 | -0.133 [-0.289, +0.022] | empate |
| lstm_classifier/shap/sum | lstm_nativo_soma | mrr | -0.120 [-0.233, -0.006] | inferior |
| lstm_classifier/shap/sum | zscore_global | hr@1 | +0.178 [+0.000, +0.356] | empate |
| lstm_classifier/shap/sum | zscore_global | mrr | +0.154 [+0.002, +0.311] | superior |
| lstm_classifier/shap/sum | zscore_local | hr@1 | +0.200 [+0.067, +0.333] | superior |
| lstm_classifier/shap/sum | zscore_local | mrr | +0.113 [+0.002, +0.231] | superior |
| lstm_classifier/lime/sum | lstm_nativo_soma | hr@1 | -0.289 [-0.467, -0.133] | inferior |
| lstm_classifier/lime/sum | lstm_nativo_soma | mrr | -0.264 [-0.380, -0.152] | inferior |
| lstm_classifier/lime/sum | zscore_global | hr@1 | +0.022 [-0.133, +0.178] | empate |
| lstm_classifier/lime/sum | zscore_global | mrr | +0.010 [-0.128, +0.153] | empate |
| lstm_classifier/lime/sum | zscore_local | hr@1 | +0.044 [-0.067, +0.156] | empate |
| lstm_classifier/lime/sum | zscore_local | mrr | -0.031 [-0.120, +0.065] | empate |

## Categoria de motivo sugerida (acerto contra o mapa fixado antes)

Referências: `prior_treino` = categoria mais frequente nas falhas de treino/validação da dobra (sem olhar a janela); `uniforme` = categoria sorteada entre as presentes (taxa esperada).

| Modelo / explicador | Acerto |
|---|---|
| lstm_forecast/shap | 0.11 [0.02, 0.20] |
| lstm_forecast/lime | 0.04 [0.00, 0.11] |
| lstm_classifier/shap | 0.44 [0.31, 0.58] |
| lstm_classifier/lime | 0.22 [0.11, 0.33] |
| referência/prior_treino | 0.33 [0.20, 0.47] |
| referência/uniforme | 0.17 [0.15, 0.18] |

Comparação pareada com `prior_treino` (diferença = XAI menos prior):

| XAI | Diferença | Veredito |
|---|---|---|
| lstm_forecast/shap | -0.222 [-0.400, -0.044] | inferior |
| lstm_forecast/lime | -0.289 [-0.444, -0.133] | inferior |
| lstm_classifier/shap | +0.111 [-0.133, +0.333] | empate |
| lstm_classifier/lime | -0.111 [-0.311, +0.089] | empate |

## Qualidade das explicações (médias por janela)

Deleção: pontuação média ao retirar os atributos mais importantes, relativa à original (menor = mais fiel). Inserção: idem ao inseri-los na referência normal, relativa à pontuação da janela completa (maior = mais fiel). `random` = ordem aleatória (média de 5). Jaccard/Kendall nos top-10 atributos.

| Medida | lstm_forecast | lstm_classifier |
|---|---|---|
| deletion_shap | 0.594 | 0.754 |
| deletion_lime | 0.584 | 0.753 |
| deletion_random | 0.995 | 0.999 |
| insertion_shap | 0.699 | 0.790 |
| insertion_lime | 0.680 | 0.840 |
| insertion_random | 0.046 | 0.323 |
| shap_lime_jaccard | 0.663 | 0.455 |
| shap_lime_kendall | 0.609 | 0.172 |
| neighbour_jaccard | 0.886 | 0.444 |
| ig_shap_jaccard | 0.895 | 0.667 |
| ig_shap_kendall | 0.877 | 0.504 |
| top_component_agree | 0.911 | 0.622 |
| seed_jaccard | 0.810 | 0.200 |
| seed_kendall | 0.723 | -0.266 |
