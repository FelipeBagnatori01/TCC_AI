# LSTM — Telecom (E5, ADR 0015)

Configuração escolhida na validação: previsora {'hidden': 128, 'layers': 1, 'normalization': 'local'}; classificadora {'normalization': 'local', 'hidden': 128, 'layers': 1}. Valores: média ± desvio-padrão entre 5 seeds; "IC médio" = média, entre seeds, dos limites do IC de 95% por bootstrap. Sem point-adjust. A pontuação da previsora lê os últimos 6 minutos da janela (ADR 0019); "AUC-PR por dobra" = média das AUC-PR de cada dobra com falha.

## Detecção (Tarefa A, por janela)

| Método | AUC-PR | IC médio | AUC-PR por dobra | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|---|---|
| lstm_forecast | 0.43 ± 0.02 | [0.29, 0.59] | 0.48 ± 0.02 | 0.80 ± 0.01 | 0.24 ± 0.02 | 0.43 ± 0.02 | 0.31 ± 0.01 | 13.24 ± 1.49 |
| lstm_classifier | 0.49 ± 0.02 | [0.36, 0.64] | 0.60 ± 0.01 | 0.84 ± 0.01 | 0.39 ± 0.04 | 0.55 ± 0.04 | 0.45 ± 0.03 | 8.63 ± 1.58 |
| E4: zscore_3sigma | 0.24 | [0.16, 0.38] | 0.43 | 0.70 | 0.14 | 1.00 | 0.25 | 60.00 |
| E4: zscore_val | 0.24 | [0.16, 0.38] | 0.43 | 0.70 | 0.26 | 0.65 | 0.37 | 18.24 |
| E4: zscore_local | 0.38 | [0.26, 0.53] | 0.48 | 0.75 | 0.28 | 0.39 | 0.33 | 9.77 |
| E4: isolation_forest | 0.18 | [0.12, 0.32] | 0.42 | 0.63 | 0.20 | 0.69 | 0.31 | 27.89 |
| E4: lightgbm | 0.26 | [0.15, 0.41] | 0.55 | 0.60 | 0.21 | 0.49 | 0.29 | 18.09 |
| E4: prior_horario | 0.15 | [0.09, 0.27] | 0.28 | 0.44 | 0.15 | 0.48 | 0.22 | 27.15 |

AUC-PR ao acaso (taxa de positivos): 0.141.

## Localização (Tarefa B): ranqueamento nativo da LSTM

| Ranqueamento | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| LSTM: soma dos erros | 0.20 ± 0.02 | 0.57 ± 0.02 | 0.69 ± 0.00 | 0.42 ± 0.01 |
| LSTM: média dos 3 maiores erros | 0.14 ± 0.03 | 0.58 ± 0.01 | 0.75 ± 0.03 | 0.39 ± 0.01 |
| E4: referência (frequência no treino) | 0.09 | 0.18 | 0.27 | 0.19 |
| E4: z-score global | 0.13 | 0.38 | 0.56 | 0.31 |
| E4: z-score local | 0.09 | 0.49 | 0.56 | 0.32 |

## Custo (CPU, Apple M4 Pro)

| Modelo | Parâmetros | Treino por dobra (s) | Épocas | Inferência (ms/janela) |
|---|---|---|---|---|
| lstm_forecast | 937,038 | 2.29 | 16.0 | 0.2498 |
| lstm_classifier | 761,985 | 1.1 | 7.9 | — |

Seleção reaproveitada de `reports/tables/lstm_Telecom_lstm.json` (sem nova busca).
