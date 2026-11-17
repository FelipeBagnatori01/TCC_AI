# LSTM — Bank (E5, ADR 0015)

Configuração escolhida na validação: previsora {'hidden': 128, 'layers': 1, 'normalization': 'local'}; classificadora {'normalization': 'local', 'hidden': 128, 'layers': 1}. Valores: média ± desvio-padrão entre 5 seeds; "IC médio" = média, entre seeds, dos limites do IC de 95% por bootstrap. Sem point-adjust. A pontuação da previsora lê os últimos 6 minutos da janela (ADR 0019); "AUC-PR por dobra" = média das AUC-PR de cada dobra com falha.

## Detecção (Tarefa A, por janela)

| Método | AUC-PR | IC médio | AUC-PR por dobra | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|---|---|
| lstm_forecast | 0.34 ± 0.00 | [0.27, 0.43] | 0.38 ± 0.00 | 0.78 ± 0.00 | 0.31 ± 0.02 | 0.36 ± 0.02 | 0.33 ± 0.01 | 5.76 ± 0.83 |
| lstm_classifier | 0.21 ± 0.01 | [0.15, 0.29] | 0.25 ± 0.01 | 0.65 ± 0.01 | 0.16 ± 0.01 | 0.39 ± 0.05 | 0.23 ± 0.01 | 14.12 ± 2.17 |
| E4: zscore_3sigma | 0.10 | [0.08, 0.13] | 0.31 | 0.48 | 0.11 | 1.00 | 0.19 | 60.00 |
| E4: zscore_val | 0.10 | [0.08, 0.13] | 0.31 | 0.48 | 0.12 | 0.68 | 0.20 | 35.06 |
| E4: zscore_local | 0.33 | [0.26, 0.41] | 0.35 | 0.79 | 0.37 | 0.40 | 0.39 | 4.77 |
| E4: isolation_forest | 0.14 | [0.11, 0.19] | 0.20 | 0.60 | 0.14 | 0.56 | 0.22 | 24.66 |
| E4: lightgbm | 0.18 | [0.12, 0.26] | 0.25 | 0.51 | 0.16 | 0.42 | 0.24 | 14.92 |
| E4: prior_horario | 0.12 | [0.10, 0.16] | 0.18 | 0.55 | 0.15 | 0.44 | 0.22 | 17.63 |

AUC-PR ao acaso (taxa de positivos): 0.105.

## Localização (Tarefa B): ranqueamento nativo da LSTM

| Ranqueamento | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| LSTM: soma dos erros | 0.33 ± 0.00 | 0.58 ± 0.01 | 0.80 ± 0.00 | 0.51 ± 0.00 |
| LSTM: média dos 3 maiores erros | 0.34 ± 0.02 | 0.56 ± 0.00 | 0.74 ± 0.01 | 0.50 ± 0.01 |
| E4: referência (frequência no treino) | 0.11 | 0.27 | 0.38 | 0.26 |
| E4: z-score global | 0.32 | 0.48 | 0.68 | 0.47 |
| E4: z-score local | 0.11 | 0.47 | 0.66 | 0.34 |

## Custo (CPU, Apple M4 Pro)

| Modelo | Parâmetros | Treino por dobra (s) | Épocas | Inferência (ms/janela) |
|---|---|---|---|---|
| lstm_forecast | 871,656 | 5.5 | 9.6 | 0.218 |
| lstm_classifier | 709,761 | 3.24 | 6.7 | — |

Seleção reaproveitada de `reports/tables/lstm_Bank_lstm.json` (sem nova busca).
