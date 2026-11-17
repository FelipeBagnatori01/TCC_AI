# Explicabilidade (E6) — Telecom

Janelas-oráculo (terminam 5 min após o início de cada falha testada, como na E4/E5); modelos da seed explicada. IC de 95% por bootstrap sobre falhas.

## Localização via XAI (RQ2)

| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| lstm_forecast/shap/sum | 0.42 [0.29, 0.58] | 0.78 [0.64, 0.89] | 0.89 [0.80, 0.98] | 0.61 [0.50, 0.71] |
| lstm_forecast/shap/top3 | 0.44 [0.29, 0.60] | 0.78 [0.64, 0.89] | 0.89 [0.80, 0.98] | 0.62 [0.51, 0.72] |
| lstm_forecast/lime/sum | 0.44 [0.31, 0.60] | 0.76 [0.62, 0.87] | 0.82 [0.71, 0.93] | 0.60 [0.49, 0.71] |
| lstm_forecast/lime/top3 | 0.44 [0.29, 0.60] | 0.80 [0.67, 0.91] | 0.87 [0.76, 0.96] | 0.61 [0.50, 0.72] |
| lstm_classifier/shap/sum | 0.27 [0.13, 0.40] | 0.51 [0.36, 0.64] | 0.56 [0.40, 0.69] | 0.42 [0.32, 0.53] |
| lstm_classifier/shap/top3 | 0.20 [0.09, 0.31] | 0.40 [0.27, 0.53] | 0.51 [0.38, 0.67] | 0.38 [0.28, 0.48] |
| lstm_classifier/lime/sum | 0.13 [0.04, 0.22] | 0.33 [0.20, 0.47] | 0.49 [0.33, 0.62] | 0.29 [0.20, 0.39] |
| lstm_classifier/lime/top3 | 0.18 [0.07, 0.29] | 0.31 [0.18, 0.44] | 0.49 [0.36, 0.62] | 0.33 [0.24, 0.44] |

Comparação pareada (agregação soma; diferença = XAI menos rival):

| XAI | Rival | Métrica | Diferença | Veredito |
|---|---|---|---|---|
| lstm_forecast/shap/sum | lstm_nativo_soma | hr@1 | +0.000 [-0.111, +0.111] | empate |
| lstm_forecast/shap/sum | lstm_nativo_soma | mrr | +0.041 [-0.038, +0.118] | empate |
| lstm_forecast/shap/sum | zscore_global | hr@1 | +0.289 [+0.111, +0.467] | superior |
| lstm_forecast/shap/sum | zscore_global | mrr | +0.293 [+0.149, +0.435] | superior |
| lstm_forecast/shap/sum | zscore_local | hr@1 | +0.333 [+0.178, +0.489] | superior |
| lstm_forecast/shap/sum | zscore_local | mrr | +0.279 [+0.156, +0.399] | superior |
| lstm_forecast/lime/sum | lstm_nativo_soma | hr@1 | +0.022 [-0.067, +0.111] | empate |
| lstm_forecast/lime/sum | lstm_nativo_soma | mrr | +0.039 [-0.024, +0.106] | empate |
| lstm_forecast/lime/sum | zscore_global | hr@1 | +0.311 [+0.133, +0.489] | superior |
| lstm_forecast/lime/sum | zscore_global | mrr | +0.291 [+0.142, +0.440] | superior |
| lstm_forecast/lime/sum | zscore_local | hr@1 | +0.356 [+0.200, +0.511] | superior |
| lstm_forecast/lime/sum | zscore_local | mrr | +0.277 [+0.153, +0.401] | superior |
| lstm_classifier/shap/sum | lstm_nativo_soma | hr@1 | -0.156 [-0.311, +0.000] | empate |
| lstm_classifier/shap/sum | lstm_nativo_soma | mrr | -0.141 [-0.251, -0.026] | inferior |
| lstm_classifier/shap/sum | zscore_global | hr@1 | +0.133 [-0.022, +0.311] | empate |
| lstm_classifier/shap/sum | zscore_global | mrr | +0.111 [-0.033, +0.265] | empate |
| lstm_classifier/shap/sum | zscore_local | hr@1 | +0.178 [+0.067, +0.311] | superior |
| lstm_classifier/shap/sum | zscore_local | mrr | +0.096 [-0.012, +0.213] | empate |
| lstm_classifier/lime/sum | lstm_nativo_soma | hr@1 | -0.289 [-0.467, -0.133] | inferior |
| lstm_classifier/lime/sum | lstm_nativo_soma | mrr | -0.274 [-0.390, -0.158] | inferior |
| lstm_classifier/lime/sum | zscore_global | hr@1 | +0.000 [-0.156, +0.156] | empate |
| lstm_classifier/lime/sum | zscore_global | mrr | -0.022 [-0.156, +0.122] | empate |
| lstm_classifier/lime/sum | zscore_local | hr@1 | +0.044 [-0.067, +0.156] | empate |
| lstm_classifier/lime/sum | zscore_local | mrr | -0.036 [-0.126, +0.060] | empate |

## Categoria de motivo sugerida (acerto contra o mapa fixado antes)

Referências: `prior_treino` = categoria mais frequente nas falhas de treino/validação da dobra (sem olhar a janela); `uniforme` = categoria sorteada entre as presentes (taxa esperada).

| Modelo / explicador | Acerto |
|---|---|
| lstm_forecast/shap | 0.09 [0.02, 0.18] |
| lstm_forecast/lime | 0.07 [0.00, 0.16] |
| lstm_classifier/shap | 0.44 [0.31, 0.58] |
| lstm_classifier/lime | 0.22 [0.11, 0.33] |
| referência/prior_treino | 0.33 [0.20, 0.47] |
| referência/uniforme | 0.17 [0.15, 0.18] |

Comparação pareada com `prior_treino` (diferença = XAI menos prior):

| XAI | Diferença | Veredito |
|---|---|---|
| lstm_forecast/shap | -0.244 [-0.422, -0.067] | inferior |
| lstm_forecast/lime | -0.267 [-0.422, -0.111] | inferior |
| lstm_classifier/shap | +0.111 [-0.133, +0.333] | empate |
| lstm_classifier/lime | -0.111 [-0.311, +0.089] | empate |

## Qualidade das explicações (médias por janela)

Deleção: pontuação média ao retirar os atributos mais importantes, relativa à original (menor = mais fiel). Inserção: idem ao inseri-los na referência normal, relativa à pontuação da janela completa (maior = mais fiel). `random` = ordem aleatória (média de 5). Jaccard/Kendall nos top-10 atributos.

| Medida | lstm_forecast | lstm_classifier |
|---|---|---|
| deletion_shap | 0.537 | 0.757 |
| deletion_lime | 0.530 | 0.760 |
| deletion_random | 0.994 | 0.999 |
| insertion_shap | 0.687 | 0.807 |
| insertion_lime | 0.674 | 0.866 |
| insertion_random | 0.041 | 0.328 |
| shap_lime_jaccard | 0.732 | 0.452 |
| shap_lime_kendall | 0.735 | 0.166 |
| neighbour_jaccard | 0.748 | 0.449 |
| ig_shap_jaccard | 0.897 | 0.642 |
| ig_shap_kendall | 0.863 | 0.472 |
| top_component_agree | 0.978 | 0.578 |
| seed_jaccard | 0.849 | 0.201 |
| seed_kendall | 0.794 | -0.256 |
