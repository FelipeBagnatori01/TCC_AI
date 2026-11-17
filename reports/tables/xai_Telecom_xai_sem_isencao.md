# Explicabilidade (E6) — Telecom

Janelas-oráculo (terminam 5 min após o início de cada falha testada, como na E4/E5); modelos da seed explicada. IC de 95% por bootstrap sobre falhas.

## Localização via XAI (RQ2)

| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| lstm_forecast/shap/sum | 0.24 [0.11, 0.38] | 0.62 [0.49, 0.76] | 0.76 [0.62, 0.87] | 0.46 [0.36, 0.56] |
| lstm_forecast/shap/top3 | 0.27 [0.13, 0.40] | 0.62 [0.49, 0.76] | 0.73 [0.60, 0.87] | 0.47 [0.37, 0.58] |
| lstm_forecast/lime/sum | 0.22 [0.11, 0.36] | 0.58 [0.42, 0.71] | 0.67 [0.51, 0.80] | 0.43 [0.33, 0.53] |
| lstm_forecast/lime/top3 | 0.24 [0.13, 0.38] | 0.60 [0.44, 0.73] | 0.71 [0.58, 0.84] | 0.44 [0.34, 0.55] |
| lstm_classifier/shap/sum | 0.22 [0.11, 0.36] | 0.42 [0.29, 0.56] | 0.53 [0.38, 0.67] | 0.37 [0.26, 0.48] |
| lstm_classifier/shap/top3 | 0.22 [0.11, 0.36] | 0.40 [0.24, 0.53] | 0.51 [0.36, 0.64] | 0.37 [0.27, 0.48] |
| lstm_classifier/lime/sum | 0.11 [0.02, 0.20] | 0.29 [0.16, 0.42] | 0.47 [0.31, 0.60] | 0.27 [0.19, 0.36] |
| lstm_classifier/lime/top3 | 0.16 [0.07, 0.27] | 0.38 [0.22, 0.51] | 0.47 [0.31, 0.60] | 0.33 [0.24, 0.42] |

Comparação pareada (agregação soma; diferença = XAI menos rival):

| XAI | Rival | Métrica | Diferença | Veredito |
|---|---|---|---|---|
| lstm_forecast/shap/sum | lstm_nativo_soma | hr@1 | +0.067 [-0.067, +0.200] | empate |
| lstm_forecast/shap/sum | lstm_nativo_soma | mrr | +0.062 [-0.035, +0.161] | empate |
| lstm_forecast/shap/sum | zscore_global | hr@1 | +0.111 [-0.022, +0.244] | empate |
| lstm_forecast/shap/sum | zscore_global | mrr | +0.149 [+0.031, +0.268] | superior |
| lstm_forecast/shap/sum | zscore_local | hr@1 | +0.156 [+0.022, +0.311] | superior |
| lstm_forecast/shap/sum | zscore_local | mrr | +0.142 [+0.034, +0.254] | superior |
| lstm_forecast/lime/sum | lstm_nativo_soma | hr@1 | +0.044 [-0.067, +0.178] | empate |
| lstm_forecast/lime/sum | lstm_nativo_soma | mrr | +0.027 [-0.058, +0.115] | empate |
| lstm_forecast/lime/sum | zscore_global | hr@1 | +0.089 [-0.067, +0.244] | empate |
| lstm_forecast/lime/sum | zscore_global | mrr | +0.113 [-0.016, +0.248] | empate |
| lstm_forecast/lime/sum | zscore_local | hr@1 | +0.133 [+0.000, +0.267] | empate |
| lstm_forecast/lime/sum | zscore_local | mrr | +0.106 [+0.004, +0.216] | superior |
| lstm_classifier/shap/sum | lstm_nativo_soma | hr@1 | +0.044 [-0.089, +0.178] | empate |
| lstm_classifier/shap/sum | lstm_nativo_soma | mrr | -0.032 [-0.119, +0.058] | empate |
| lstm_classifier/shap/sum | zscore_global | hr@1 | +0.089 [-0.067, +0.267] | empate |
| lstm_classifier/shap/sum | zscore_global | mrr | +0.055 [-0.088, +0.208] | empate |
| lstm_classifier/shap/sum | zscore_local | hr@1 | +0.133 [+0.022, +0.267] | superior |
| lstm_classifier/shap/sum | zscore_local | mrr | +0.048 [-0.056, +0.158] | empate |
| lstm_classifier/lime/sum | lstm_nativo_soma | hr@1 | -0.067 [-0.200, +0.067] | empate |
| lstm_classifier/lime/sum | lstm_nativo_soma | mrr | -0.127 [-0.214, -0.035] | inferior |
| lstm_classifier/lime/sum | zscore_global | hr@1 | -0.022 [-0.156, +0.111] | empate |
| lstm_classifier/lime/sum | zscore_global | mrr | -0.040 [-0.172, +0.102] | empate |
| lstm_classifier/lime/sum | zscore_local | hr@1 | +0.022 [-0.067, +0.112] | empate |
| lstm_classifier/lime/sum | zscore_local | mrr | -0.047 [-0.132, +0.048] | empate |

## Categoria de motivo sugerida (acerto contra o mapa fixado antes)

Referências: `prior_treino` = categoria mais frequente nas falhas de treino/validação da dobra (sem olhar a janela); `uniforme` = categoria sorteada entre as presentes (taxa esperada).

| Modelo / explicador | Acerto |
|---|---|
| lstm_forecast/shap | 0.16 [0.04, 0.27] |
| lstm_forecast/lime | 0.09 [0.02, 0.18] |
| lstm_classifier/shap | 0.40 [0.27, 0.53] |
| lstm_classifier/lime | 0.27 [0.13, 0.40] |
| referência/prior_treino | 0.33 [0.20, 0.47] |
| referência/uniforme | 0.18 [0.16, 0.19] |

Comparação pareada com `prior_treino` (diferença = XAI menos prior):

| XAI | Diferença | Veredito |
|---|---|---|
| lstm_forecast/shap | -0.178 [-0.378, +0.000] | empate |
| lstm_forecast/lime | -0.244 [-0.422, -0.089] | inferior |
| lstm_classifier/shap | +0.067 [-0.178, +0.289] | empate |
| lstm_classifier/lime | -0.067 [-0.289, +0.156] | empate |

## Qualidade das explicações (médias por janela)

Deleção: pontuação média ao retirar os atributos mais importantes, relativa à original (menor = mais fiel). Inserção: idem ao inseri-los na referência normal, relativa à pontuação da janela completa (maior = mais fiel). `random` = ordem aleatória (média de 5). Jaccard/Kendall nos top-10 atributos.

| Medida | lstm_forecast | lstm_classifier |
|---|---|---|
| deletion_shap | 0.577 | 0.750 |
| deletion_lime | 0.579 | 0.753 |
| deletion_random | 0.993 | 0.996 |
| insertion_shap | 0.666 | 0.755 |
| insertion_lime | 0.656 | 0.786 |
| insertion_random | 0.054 | 0.342 |
| shap_lime_jaccard | 0.798 | 0.458 |
| shap_lime_kendall | 0.738 | 0.188 |
| neighbour_jaccard | 0.736 | 0.435 |
| ig_shap_jaccard | 0.905 | 0.682 |
| ig_shap_kendall | 0.906 | 0.561 |
| top_component_agree | 0.889 | 0.600 |
| seed_jaccard | 0.871 | 0.171 |
| seed_kendall | 0.792 | -0.301 |
