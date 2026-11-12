# Explicabilidade (E6) — Telecom

Janelas-oráculo (terminam 5 min após o início de cada falha testada, como na E4/E5); modelos da seed explicada. IC de 95% por bootstrap sobre falhas.

## Localização via XAI (RQ2)

| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| lstm_forecast/shap/sum | 0.16 [0.07, 0.27] | 0.51 [0.38, 0.65] | 0.62 [0.47, 0.76] | 0.36 [0.27, 0.45] |
| lstm_forecast/shap/top3 | 0.16 [0.07, 0.27] | 0.51 [0.38, 0.65] | 0.62 [0.47, 0.76] | 0.36 [0.27, 0.45] |
| lstm_forecast/lime/sum | 0.16 [0.07, 0.27] | 0.42 [0.27, 0.56] | 0.56 [0.42, 0.69] | 0.34 [0.24, 0.43] |
| lstm_forecast/lime/top3 | 0.16 [0.07, 0.27] | 0.51 [0.38, 0.64] | 0.62 [0.49, 0.76] | 0.36 [0.26, 0.45] |
| lstm_classifier/shap/sum | 0.24 [0.13, 0.38] | 0.44 [0.29, 0.58] | 0.56 [0.40, 0.69] | 0.39 [0.28, 0.50] |
| lstm_classifier/shap/top3 | 0.24 [0.11, 0.38] | 0.42 [0.29, 0.56] | 0.53 [0.38, 0.67] | 0.39 [0.29, 0.50] |
| lstm_classifier/lime/sum | 0.13 [0.04, 0.24] | 0.31 [0.18, 0.44] | 0.49 [0.33, 0.64] | 0.29 [0.20, 0.39] |
| lstm_classifier/lime/top3 | 0.18 [0.07, 0.29] | 0.40 [0.24, 0.53] | 0.49 [0.33, 0.62] | 0.35 [0.25, 0.45] |

Comparação pareada (agregação soma; diferença = XAI menos rival):

| XAI | Rival | Métrica | Diferença | Veredito |
|---|---|---|---|---|
| lstm_forecast/shap/sum | lstm_nativo_soma | hr@1 | -0.044 [-0.178, +0.089] | empate |
| lstm_forecast/shap/sum | lstm_nativo_soma | mrr | -0.054 [-0.164, +0.048] | empate |
| lstm_forecast/shap/sum | zscore_global | hr@1 | +0.022 [-0.089, +0.133] | empate |
| lstm_forecast/shap/sum | zscore_global | mrr | +0.059 [-0.047, +0.168] | empate |
| lstm_forecast/shap/sum | zscore_local | hr@1 | +0.044 [-0.089, +0.178] | empate |
| lstm_forecast/shap/sum | zscore_local | mrr | +0.025 [-0.092, +0.135] | empate |
| lstm_forecast/lime/sum | lstm_nativo_soma | hr@1 | -0.044 [-0.156, +0.067] | empate |
| lstm_forecast/lime/sum | lstm_nativo_soma | mrr | -0.079 [-0.167, +0.008] | empate |
| lstm_forecast/lime/sum | zscore_global | hr@1 | +0.022 [-0.111, +0.156] | empate |
| lstm_forecast/lime/sum | zscore_global | mrr | +0.034 [-0.087, +0.157] | empate |
| lstm_forecast/lime/sum | zscore_local | hr@1 | +0.044 [-0.089, +0.178] | empate |
| lstm_forecast/lime/sum | zscore_local | mrr | +0.001 [-0.110, +0.113] | empate |
| lstm_classifier/shap/sum | lstm_nativo_soma | hr@1 | +0.044 [-0.089, +0.178] | empate |
| lstm_classifier/shap/sum | lstm_nativo_soma | mrr | -0.026 [-0.111, +0.064] | empate |
| lstm_classifier/shap/sum | zscore_global | hr@1 | +0.111 [-0.044, +0.289] | empate |
| lstm_classifier/shap/sum | zscore_global | mrr | +0.087 [-0.059, +0.248] | empate |
| lstm_classifier/shap/sum | zscore_local | hr@1 | +0.133 [+0.022, +0.267] | superior |
| lstm_classifier/shap/sum | zscore_local | mrr | +0.054 [-0.047, +0.164] | empate |
| lstm_classifier/lime/sum | lstm_nativo_soma | hr@1 | -0.067 [-0.200, +0.067] | empate |
| lstm_classifier/lime/sum | lstm_nativo_soma | mrr | -0.124 [-0.213, -0.031] | inferior |
| lstm_classifier/lime/sum | zscore_global | hr@1 | +0.000 [-0.133, +0.156] | empate |
| lstm_classifier/lime/sum | zscore_global | mrr | -0.012 [-0.150, +0.135] | empate |
| lstm_classifier/lime/sum | zscore_local | hr@1 | +0.022 [-0.067, +0.112] | empate |
| lstm_classifier/lime/sum | zscore_local | mrr | -0.045 [-0.127, +0.045] | empate |

## Categoria de motivo sugerida (acerto contra o mapa fixado antes)

Referências: `prior_treino` = categoria mais frequente nas falhas de treino/validação da dobra (sem olhar a janela); `uniforme` = categoria sorteada entre as presentes (taxa esperada).

| Modelo / explicador | Acerto |
|---|---|
| lstm_forecast/shap | 0.16 [0.07, 0.24] |
| lstm_forecast/lime | 0.09 [0.02, 0.18] |
| lstm_classifier/shap | 0.38 [0.24, 0.51] |
| lstm_classifier/lime | 0.24 [0.13, 0.38] |
| referência/prior_treino | 0.33 [0.20, 0.47] |
| referência/uniforme | 0.18 [0.16, 0.19] |

Comparação pareada com `prior_treino` (diferença = XAI menos prior):

| XAI | Diferença | Veredito |
|---|---|---|
| lstm_forecast/shap | -0.178 [-0.378, +0.000] | empate |
| lstm_forecast/lime | -0.244 [-0.422, -0.089] | inferior |
| lstm_classifier/shap | +0.044 [-0.200, +0.267] | empate |
| lstm_classifier/lime | -0.089 [-0.311, +0.111] | empate |

## Qualidade das explicações (médias por janela)

Deleção: pontuação média ao retirar os atributos mais importantes, relativa à original (menor = mais fiel). Inserção: idem ao inseri-los na referência normal, relativa à pontuação da janela completa (maior = mais fiel). `random` = ordem aleatória (média de 5). Jaccard/Kendall nos top-10 atributos.

| Medida | lstm_forecast | lstm_classifier |
|---|---|---|
| deletion_shap | 0.639 | 0.754 |
| deletion_lime | 0.636 | 0.756 |
| deletion_random | 0.995 | 0.998 |
| insertion_shap | 0.674 | 0.725 |
| insertion_lime | 0.661 | 0.760 |
| insertion_random | 0.056 | 0.327 |
| shap_lime_jaccard | 0.700 | 0.462 |
| shap_lime_kendall | 0.658 | 0.191 |
| neighbour_jaccard | 0.894 | 0.439 |
| ig_shap_jaccard | 0.910 | 0.681 |
| ig_shap_kendall | 0.881 | 0.559 |
| top_component_agree | 0.822 | 0.600 |
| seed_jaccard | 0.837 | 0.171 |
| seed_kendall | 0.729 | -0.305 |
