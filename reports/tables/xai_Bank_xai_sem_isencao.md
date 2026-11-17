# Explicabilidade (E6) — Bank

Janelas-oráculo (terminam 5 min após o início de cada falha testada, como na E4/E5); modelos da seed explicada. IC de 95% por bootstrap sobre falhas.

## Localização via XAI (RQ2)

| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| lstm_forecast/shap/sum | 0.50 [0.42, 0.59] | 0.68 [0.59, 0.76] | 0.78 [0.71, 0.85] | 0.62 [0.55, 0.69] |
| lstm_forecast/shap/top3 | 0.51 [0.42, 0.60] | 0.67 [0.58, 0.75] | 0.78 [0.71, 0.86] | 0.63 [0.55, 0.69] |
| lstm_forecast/lime/sum | 0.44 [0.35, 0.53] | 0.65 [0.57, 0.73] | 0.79 [0.72, 0.86] | 0.58 [0.52, 0.65] |
| lstm_forecast/lime/top3 | 0.48 [0.40, 0.57] | 0.71 [0.62, 0.79] | 0.78 [0.70, 0.85] | 0.62 [0.54, 0.68] |
| lstm_classifier/shap/sum | 0.43 [0.34, 0.52] | 0.72 [0.63, 0.80] | 0.86 [0.79, 0.92] | 0.61 [0.54, 0.67] |
| lstm_classifier/shap/top3 | 0.48 [0.39, 0.57] | 0.78 [0.71, 0.86] | 0.89 [0.83, 0.94] | 0.65 [0.59, 0.71] |
| lstm_classifier/lime/sum | 0.21 [0.14, 0.28] | 0.56 [0.47, 0.64] | 0.77 [0.69, 0.84] | 0.43 [0.37, 0.49] |
| lstm_classifier/lime/top3 | 0.48 [0.40, 0.57] | 0.75 [0.67, 0.82] | 0.84 [0.78, 0.90] | 0.64 [0.57, 0.70] |

Comparação pareada (agregação soma; diferença = XAI menos rival):

| XAI | Rival | Métrica | Diferença | Veredito |
|---|---|---|---|---|
| lstm_forecast/shap/sum | lstm_nativo_soma | hr@1 | +0.167 [+0.083, +0.242] | superior |
| lstm_forecast/shap/sum | lstm_nativo_soma | mrr | +0.111 [+0.053, +0.171] | superior |
| lstm_forecast/shap/sum | zscore_global | hr@1 | +0.183 [+0.083, +0.292] | superior |
| lstm_forecast/shap/sum | zscore_global | mrr | +0.152 [+0.077, +0.231] | superior |
| lstm_forecast/shap/sum | zscore_local | hr@1 | +0.392 [+0.292, +0.483] | superior |
| lstm_forecast/shap/sum | zscore_local | mrr | +0.282 [+0.210, +0.351] | superior |
| lstm_forecast/lime/sum | lstm_nativo_soma | hr@1 | +0.108 [+0.033, +0.183] | superior |
| lstm_forecast/lime/sum | lstm_nativo_soma | mrr | +0.071 [+0.018, +0.122] | superior |
| lstm_forecast/lime/sum | zscore_global | hr@1 | +0.125 [+0.017, +0.225] | superior |
| lstm_forecast/lime/sum | zscore_global | mrr | +0.113 [+0.034, +0.190] | superior |
| lstm_forecast/lime/sum | zscore_local | hr@1 | +0.333 [+0.242, +0.433] | superior |
| lstm_forecast/lime/sum | zscore_local | mrr | +0.242 [+0.174, +0.310] | superior |
| lstm_classifier/shap/sum | lstm_nativo_soma | hr@1 | +0.100 [+0.008, +0.200] | superior |
| lstm_classifier/shap/sum | lstm_nativo_soma | mrr | +0.094 [+0.026, +0.161] | superior |
| lstm_classifier/shap/sum | zscore_global | hr@1 | +0.117 [+0.008, +0.225] | superior |
| lstm_classifier/shap/sum | zscore_global | mrr | +0.135 [+0.054, +0.215] | superior |
| lstm_classifier/shap/sum | zscore_local | hr@1 | +0.325 [+0.225, +0.425] | superior |
| lstm_classifier/shap/sum | zscore_local | mrr | +0.265 [+0.201, +0.328] | superior |
| lstm_classifier/lime/sum | lstm_nativo_soma | hr@1 | -0.125 [-0.225, -0.025] | inferior |
| lstm_classifier/lime/sum | lstm_nativo_soma | mrr | -0.084 [-0.157, -0.013] | inferior |
| lstm_classifier/lime/sum | zscore_global | hr@1 | -0.108 [-0.208, -0.008] | inferior |
| lstm_classifier/lime/sum | zscore_global | mrr | -0.043 [-0.120, +0.033] | empate |
| lstm_classifier/lime/sum | zscore_local | hr@1 | +0.100 [+0.017, +0.183] | superior |
| lstm_classifier/lime/sum | zscore_local | mrr | +0.087 [+0.025, +0.149] | superior |

## Categoria de motivo sugerida (acerto contra o mapa fixado antes)

Referências: `prior_treino` = categoria mais frequente nas falhas de treino/validação da dobra (sem olhar a janela); `uniforme` = categoria sorteada entre as presentes (taxa esperada).

| Modelo / explicador | Acerto |
|---|---|
| lstm_forecast/shap | 0.57 [0.47, 0.65] |
| lstm_forecast/lime | 0.26 [0.18, 0.34] |
| lstm_classifier/shap | 0.38 [0.29, 0.47] |
| lstm_classifier/lime | 0.19 [0.12, 0.27] |
| referência/prior_treino | 0.38 [0.28, 0.47] |
| referência/uniforme | 0.16 [0.15, 0.17] |

Comparação pareada com `prior_treino` (diferença = XAI menos prior):

| XAI | Diferença | Veredito |
|---|---|---|
| lstm_forecast/shap | +0.192 [+0.067, +0.317] | superior |
| lstm_forecast/lime | -0.117 [-0.250, +0.025] | empate |
| lstm_classifier/shap | +0.000 [-0.142, +0.150] | empate |
| lstm_classifier/lime | -0.183 [-0.317, -0.050] | inferior |

## Qualidade das explicações (médias por janela)

Deleção: pontuação média ao retirar os atributos mais importantes, relativa à original (menor = mais fiel). Inserção: idem ao inseri-los na referência normal, relativa à pontuação da janela completa (maior = mais fiel). `random` = ordem aleatória (média de 5). Jaccard/Kendall nos top-10 atributos.

| Medida | lstm_forecast | lstm_classifier |
|---|---|---|
| deletion_shap | 0.618 | 0.686 |
| deletion_lime | 0.624 | 0.693 |
| deletion_random | 0.995 | 1.000 |
| insertion_shap | 0.659 | 3.269 |
| insertion_lime | 0.647 | 3.361 |
| insertion_random | 0.061 | 0.930 |
| shap_lime_jaccard | 0.771 | 0.493 |
| shap_lime_kendall | 0.727 | 0.292 |
| neighbour_jaccard | 0.692 | 0.544 |
| ig_shap_jaccard | 0.818 | 0.527 |
| ig_shap_kendall | 0.795 | 0.293 |
| top_component_agree | 0.800 | 0.517 |
| seed_jaccard | 0.864 | 0.252 |
| seed_kendall | 0.793 | -0.150 |
