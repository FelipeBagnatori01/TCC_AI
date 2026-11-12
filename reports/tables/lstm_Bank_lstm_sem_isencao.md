# LSTM — Bank (E5, ADR 0015)

Configuração escolhida na validação: previsora {'hidden': 128, 'layers': 2, 'normalization': 'local'}; classificadora {'normalization': 'local', 'hidden': 128, 'layers': 1}. Valores: média ± desvio-padrão entre 5 seeds; "IC médio" = média, entre seeds, dos limites do IC de 95% por bootstrap. Sem point-adjust.

## Detecção (Tarefa A, por janela)

| Método | AUC-PR | IC médio | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|---|
| lstm_forecast | 0.20 ± 0.00 | [0.16, 0.29] | 0.70 ± 0.00 | 0.22 ± 0.01 | 0.39 ± 0.03 | 0.28 ± 0.01 | 9.98 ± 0.50 |
| lstm_classifier | 0.21 ± 0.01 | [0.15, 0.29] | 0.65 ± 0.01 | 0.16 ± 0.01 | 0.39 ± 0.05 | 0.23 ± 0.01 | 14.12 ± 2.17 |
| E4: zscore_3sigma | 0.10 | [0.08, 0.13] | 0.48 | 0.11 | 1.00 | 0.19 | 60.00 |
| E4: zscore_val | 0.10 | [0.08, 0.13] | 0.48 | 0.12 | 0.68 | 0.20 | 35.80 |
| E4: zscore_local | 0.19 | [0.15, 0.24] | 0.69 | 0.19 | 0.43 | 0.26 | 13.07 |
| E4: isolation_forest | 0.14 | [0.11, 0.19] | 0.60 | 0.14 | 0.56 | 0.22 | 24.66 |
| E4: lightgbm | 0.18 | [0.12, 0.26] | 0.51 | 0.16 | 0.42 | 0.24 | 14.92 |

AUC-PR ao acaso (taxa de positivos): 0.105.

## Localização (Tarefa B): ranqueamento nativo da LSTM

| Ranqueamento | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| LSTM: soma dos erros | 0.32 ± 0.01 | 0.60 ± 0.01 | 0.78 ± 0.01 | 0.51 ± 0.00 |
| LSTM: média dos 3 maiores erros | 0.32 ± 0.02 | 0.57 ± 0.02 | 0.71 ± 0.01 | 0.49 ± 0.01 |
| E4: referência (frequência no treino) | 0.11 | 0.27 | 0.38 | 0.26 |
| E4: z-score global | 0.32 | 0.48 | 0.68 | 0.47 |
| E4: z-score local | 0.11 | 0.46 | 0.65 | 0.34 |

## Custo (CPU, Apple M4 Pro)

| Modelo | Parâmetros | Treino por dobra (s) | Épocas | Inferência (ms/janela) |
|---|---|---|---|---|
| lstm_forecast | 1,003,752 | 8.32 | 9.6 | 0.2489 |
| lstm_classifier | 709,761 | 3.29 | 6.7 | — |

Seleção reaproveitada de `reports/tables/lstm_Bank_lstm.json` (sem nova busca).
