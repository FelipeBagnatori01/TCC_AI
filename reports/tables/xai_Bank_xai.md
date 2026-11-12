# Explicabilidade (E6) — Bank

Janelas-oráculo (terminam 5 min após o início de cada falha testada, como na E4/E5); modelos da seed explicada. IC de 95% por bootstrap sobre falhas.

## Localização via XAI (RQ2)

| Modelo / explicador / agregação | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| lstm_forecast/shap/sum | 0.36 [0.28, 0.44] | 0.50 [0.42, 0.59] | 0.66 [0.57, 0.74] | 0.50 [0.43, 0.57] |
| lstm_forecast/shap/top3 | 0.35 [0.27, 0.43] | 0.49 [0.40, 0.57] | 0.64 [0.56, 0.72] | 0.49 [0.42, 0.56] |
| lstm_forecast/lime/sum | 0.30 [0.22, 0.38] | 0.52 [0.42, 0.60] | 0.62 [0.54, 0.71] | 0.46 [0.40, 0.52] |
| lstm_forecast/lime/top3 | 0.36 [0.28, 0.45] | 0.55 [0.46, 0.63] | 0.61 [0.53, 0.69] | 0.50 [0.43, 0.57] |
| lstm_classifier/shap/sum | 0.44 [0.36, 0.53] | 0.73 [0.65, 0.81] | 0.84 [0.78, 0.90] | 0.61 [0.55, 0.68] |
| lstm_classifier/shap/top3 | 0.47 [0.38, 0.56] | 0.76 [0.68, 0.83] | 0.85 [0.78, 0.91] | 0.64 [0.57, 0.70] |
| lstm_classifier/lime/sum | 0.22 [0.15, 0.29] | 0.52 [0.42, 0.60] | 0.75 [0.67, 0.82] | 0.42 [0.36, 0.48] |
| lstm_classifier/lime/top3 | 0.45 [0.36, 0.54] | 0.70 [0.62, 0.78] | 0.84 [0.77, 0.90] | 0.61 [0.54, 0.67] |

Comparação pareada (agregação soma; diferença = XAI menos rival):

| XAI | Rival | Métrica | Diferença | Veredito |
|---|---|---|---|---|
| lstm_forecast/shap/sum | lstm_nativo_soma | hr@1 | +0.017 [-0.058, +0.092] | empate |
| lstm_forecast/shap/sum | lstm_nativo_soma | mrr | -0.021 [-0.074, +0.031] | empate |
| lstm_forecast/shap/sum | zscore_global | hr@1 | +0.042 [-0.058, +0.142] | empate |
| lstm_forecast/shap/sum | zscore_global | mrr | +0.025 [-0.053, +0.101] | empate |
| lstm_forecast/shap/sum | zscore_local | hr@1 | +0.250 [+0.167, +0.342] | superior |
| lstm_forecast/shap/sum | zscore_local | mrr | +0.158 [+0.084, +0.230] | superior |
| lstm_forecast/lime/sum | lstm_nativo_soma | hr@1 | -0.042 [-0.100, +0.017] | empate |
| lstm_forecast/lime/sum | lstm_nativo_soma | mrr | -0.057 [-0.104, -0.011] | inferior |
| lstm_forecast/lime/sum | zscore_global | hr@1 | -0.017 [-0.125, +0.092] | empate |
| lstm_forecast/lime/sum | zscore_global | mrr | -0.011 [-0.092, +0.068] | empate |
| lstm_forecast/lime/sum | zscore_local | hr@1 | +0.192 [+0.108, +0.275] | superior |
| lstm_forecast/lime/sum | zscore_local | mrr | +0.122 [+0.052, +0.190] | superior |
| lstm_classifier/shap/sum | lstm_nativo_soma | hr@1 | +0.100 [-0.008, +0.200] | empate |
| lstm_classifier/shap/sum | lstm_nativo_soma | mrr | +0.096 [+0.022, +0.166] | superior |
| lstm_classifier/shap/sum | zscore_global | hr@1 | +0.125 [+0.008, +0.233] | superior |
| lstm_classifier/shap/sum | zscore_global | mrr | +0.142 [+0.057, +0.222] | superior |
| lstm_classifier/shap/sum | zscore_local | hr@1 | +0.333 [+0.233, +0.433] | superior |
| lstm_classifier/shap/sum | zscore_local | mrr | +0.275 [+0.208, +0.337] | superior |
| lstm_classifier/lime/sum | lstm_nativo_soma | hr@1 | -0.125 [-0.217, -0.033] | inferior |
| lstm_classifier/lime/sum | lstm_nativo_soma | mrr | -0.095 [-0.164, -0.028] | inferior |
| lstm_classifier/lime/sum | zscore_global | hr@1 | -0.100 [-0.208, +0.017] | empate |
| lstm_classifier/lime/sum | zscore_global | mrr | -0.049 [-0.131, +0.037] | empate |
| lstm_classifier/lime/sum | zscore_local | hr@1 | +0.108 [+0.025, +0.192] | superior |
| lstm_classifier/lime/sum | zscore_local | mrr | +0.084 [+0.018, +0.148] | superior |

## Categoria de motivo sugerida (acerto contra o mapa fixado antes)

Referências: `prior_treino` = categoria mais frequente nas falhas de treino/validação da dobra (sem olhar a janela); `uniforme` = categoria sorteada entre as presentes (taxa esperada).

| Modelo / explicador | Acerto |
|---|---|
| lstm_forecast/shap | 0.43 [0.34, 0.53] |
| lstm_forecast/lime | 0.20 [0.13, 0.27] |
| lstm_classifier/shap | 0.33 [0.24, 0.41] |
| lstm_classifier/lime | 0.19 [0.12, 0.27] |
| referência/prior_treino | 0.38 [0.28, 0.47] |
| referência/uniforme | 0.16 [0.15, 0.17] |

Comparação pareada com `prior_treino` (diferença = XAI menos prior):

| XAI | Diferença | Veredito |
|---|---|---|
| lstm_forecast/shap | +0.058 [-0.067, +0.175] | empate |
| lstm_forecast/lime | -0.175 [-0.300, -0.050] | inferior |
| lstm_classifier/shap | -0.050 [-0.192, +0.083] | empate |
| lstm_classifier/lime | -0.183 [-0.317, -0.050] | inferior |

## Qualidade das explicações (médias por janela)

Deleção: pontuação média ao retirar os atributos mais importantes, relativa à original (menor = mais fiel). Inserção: idem ao inseri-los na referência normal, relativa à pontuação da janela completa (maior = mais fiel). `random` = ordem aleatória (média de 5). Jaccard/Kendall nos top-10 atributos.

| Medida | lstm_forecast | lstm_classifier |
|---|---|---|
| deletion_shap | 0.653 | 0.695 |
| deletion_lime | 0.648 | 0.704 |
| deletion_random | 0.994 | 1.000 |
| insertion_shap | 0.669 | 3.469 |
| insertion_lime | 0.655 | 4.495 |
| insertion_random | 0.055 | 1.403 |
| shap_lime_jaccard | 0.714 | 0.476 |
| shap_lime_kendall | 0.645 | 0.248 |
| neighbour_jaccard | 0.857 | 0.544 |
| ig_shap_jaccard | 0.856 | 0.501 |
| ig_shap_kendall | 0.842 | 0.287 |
| top_component_agree | 0.792 | 0.550 |
| seed_jaccard | 0.822 | 0.262 |
| seed_kendall | 0.718 | -0.135 |
