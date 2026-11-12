# Explicabilidade (E6) — Bank

Janelas-oráculo (terminam 5 min após o início de cada falha testada, como na E4/E5); modelos da seed explicada. IC de 95% por bootstrap sobre falhas.

## Localização via XAI (RQ2)

| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| lstm_forecast/shap/sum | 0.36 [0.28, 0.44] | 0.53 [0.43, 0.61] | 0.68 [0.60, 0.77] | 0.50 [0.44, 0.57] |
| lstm_forecast/shap/top3 | 0.35 [0.27, 0.43] | 0.53 [0.44, 0.62] | 0.66 [0.57, 0.74] | 0.50 [0.43, 0.57] |
| lstm_forecast/lime/sum | 0.33 [0.24, 0.41] | 0.52 [0.42, 0.60] | 0.65 [0.57, 0.73] | 0.47 [0.41, 0.54] |
| lstm_forecast/lime/top3 | 0.38 [0.29, 0.47] | 0.55 [0.46, 0.63] | 0.62 [0.53, 0.70] | 0.50 [0.44, 0.58] |
| lstm_classifier/shap/sum | 0.43 [0.34, 0.52] | 0.72 [0.64, 0.80] | 0.85 [0.78, 0.91] | 0.61 [0.54, 0.67] |
| lstm_classifier/shap/top3 | 0.48 [0.39, 0.57] | 0.78 [0.70, 0.85] | 0.88 [0.82, 0.93] | 0.65 [0.58, 0.71] |
| lstm_classifier/lime/sum | 0.21 [0.14, 0.28] | 0.56 [0.47, 0.64] | 0.76 [0.68, 0.83] | 0.43 [0.37, 0.48] |
| lstm_classifier/lime/top3 | 0.48 [0.40, 0.57] | 0.74 [0.66, 0.82] | 0.83 [0.77, 0.89] | 0.64 [0.57, 0.70] |

Comparação pareada (agregação soma; diferença = XAI menos rival):

| XAI | Rival | Métrica | Diferença | Veredito |
|---|---|---|---|---|
| lstm_forecast/shap/sum | lstm_nativo_soma | hr@1 | +0.033 [-0.042, +0.108] | empate |
| lstm_forecast/shap/sum | lstm_nativo_soma | mrr | -0.005 [-0.059, +0.046] | empate |
| lstm_forecast/shap/sum | zscore_global | hr@1 | +0.042 [-0.058, +0.133] | empate |
| lstm_forecast/shap/sum | zscore_global | mrr | +0.032 [-0.044, +0.106] | empate |
| lstm_forecast/shap/sum | zscore_local | hr@1 | +0.250 [+0.158, +0.342] | superior |
| lstm_forecast/shap/sum | zscore_local | mrr | +0.166 [+0.094, +0.235] | superior |
| lstm_forecast/lime/sum | lstm_nativo_soma | hr@1 | +0.000 [-0.067, +0.067] | empate |
| lstm_forecast/lime/sum | lstm_nativo_soma | mrr | -0.036 [-0.085, +0.012] | empate |
| lstm_forecast/lime/sum | zscore_global | hr@1 | +0.008 [-0.092, +0.108] | empate |
| lstm_forecast/lime/sum | zscore_global | mrr | +0.002 [-0.074, +0.079] | empate |
| lstm_forecast/lime/sum | zscore_local | hr@1 | +0.217 [+0.133, +0.300] | superior |
| lstm_forecast/lime/sum | zscore_local | mrr | +0.135 [+0.064, +0.205] | superior |
| lstm_classifier/shap/sum | lstm_nativo_soma | hr@1 | +0.108 [+0.008, +0.208] | superior |
| lstm_classifier/shap/sum | lstm_nativo_soma | mrr | +0.097 [+0.026, +0.164] | superior |
| lstm_classifier/shap/sum | zscore_global | hr@1 | +0.117 [+0.008, +0.225] | superior |
| lstm_classifier/shap/sum | zscore_global | mrr | +0.135 [+0.053, +0.215] | superior |
| lstm_classifier/shap/sum | zscore_local | hr@1 | +0.325 [+0.225, +0.425] | superior |
| lstm_classifier/shap/sum | zscore_local | mrr | +0.268 [+0.204, +0.330] | superior |
| lstm_classifier/lime/sum | lstm_nativo_soma | hr@1 | -0.117 [-0.217, -0.017] | inferior |
| lstm_classifier/lime/sum | lstm_nativo_soma | mrr | -0.082 [-0.155, -0.014] | inferior |
| lstm_classifier/lime/sum | zscore_global | hr@1 | -0.108 [-0.208, -0.008] | inferior |
| lstm_classifier/lime/sum | zscore_global | mrr | -0.045 [-0.121, +0.032] | empate |
| lstm_classifier/lime/sum | zscore_local | hr@1 | +0.100 [+0.017, +0.183] | superior |
| lstm_classifier/lime/sum | zscore_local | mrr | +0.088 [+0.026, +0.150] | superior |

## Categoria de motivo sugerida (acerto contra o mapa fixado antes)

Referências: `prior_treino` = categoria mais frequente nas falhas de treino/validação da dobra (sem olhar a janela); `uniforme` = categoria sorteada entre as presentes (taxa esperada).

| Modelo / explicador | Acerto |
|---|---|
| lstm_forecast/shap | 0.44 [0.35, 0.53] |
| lstm_forecast/lime | 0.20 [0.13, 0.27] |
| lstm_classifier/shap | 0.38 [0.29, 0.47] |
| lstm_classifier/lime | 0.19 [0.12, 0.27] |
| referência/prior_treino | 0.38 [0.28, 0.47] |
| referência/uniforme | 0.16 [0.15, 0.17] |

Comparação pareada com `prior_treino` (diferença = XAI menos prior):

| XAI | Diferença | Veredito |
|---|---|---|
| lstm_forecast/shap | +0.067 [-0.058, +0.183] | empate |
| lstm_forecast/lime | -0.175 [-0.300, -0.050] | inferior |
| lstm_classifier/shap | +0.000 [-0.142, +0.150] | empate |
| lstm_classifier/lime | -0.183 [-0.317, -0.050] | inferior |

## Qualidade das explicações (médias por janela)

Deleção: pontuação média ao retirar os atributos mais importantes, relativa à original (menor = mais fiel). Inserção: idem ao inseri-los na referência normal, relativa à pontuação da janela completa (maior = mais fiel). `random` = ordem aleatória (média de 5). Jaccard/Kendall nos top-10 atributos.

| Medida | lstm_forecast | lstm_classifier |
|---|---|---|
| deletion_shap | 0.653 | 0.686 |
| deletion_lime | 0.649 | 0.693 |
| deletion_random | 0.996 | 1.001 |
| insertion_shap | 0.668 | 3.265 |
| insertion_lime | 0.655 | 3.366 |
| insertion_random | 0.058 | 0.930 |
| shap_lime_jaccard | 0.696 | 0.489 |
| shap_lime_kendall | 0.650 | 0.280 |
| neighbour_jaccard | 0.874 | 0.547 |
| ig_shap_jaccard | 0.858 | 0.517 |
| ig_shap_kendall | 0.837 | 0.273 |
| top_component_agree | 0.817 | 0.517 |
| seed_jaccard | 0.817 | 0.253 |
| seed_kendall | 0.696 | -0.146 |
