# LSTM — Bank (E5, ADR 0015)

Configuração escolhida na validação: previsora {'hidden': 128, 'layers': 1, 'normalization': 'local'}; classificadora {'normalization': 'local', 'hidden': 128, 'layers': 1}. Valores: média ± desvio-padrão entre 5 seeds; "IC médio" = média, entre seeds, dos limites do IC de 95% por bootstrap. Sem point-adjust. A pontuação da previsora lê os últimos 6 minutos da janela (ADR 0019); "AUC-PR por dobra" = média das AUC-PR de cada dobra com falha.

## Detecção (Tarefa A, por janela)

| Método | AUC-PR | IC médio | AUC-PR por dobra | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|---|---|
| lstm_forecast | 0.35 ± 0.00 | [0.27, 0.44] | 0.38 ± 0.01 | 0.78 ± 0.00 | 0.31 ± 0.01 | 0.35 ± 0.02 | 0.33 ± 0.01 | 5.67 ± 0.36 |
| lstm_classifier | 0.22 ± 0.01 | [0.16, 0.30] | 0.26 ± 0.01 | 0.66 ± 0.01 | 0.17 ± 0.01 | 0.41 ± 0.03 | 0.24 ± 0.02 | 14.32 ± 0.56 |
| E4: zscore_3sigma | 0.10 | [0.08, 0.13] | 0.31 | 0.48 | 0.11 | 1.00 | 0.19 | 60.00 |
| E4: zscore_val | 0.10 | [0.08, 0.13] | 0.31 | 0.48 | 0.12 | 0.68 | 0.20 | 35.06 |
| E4: zscore_local | 0.33 | [0.26, 0.41] | 0.35 | 0.79 | 0.37 | 0.40 | 0.39 | 4.77 |
| E4: isolation_forest | 0.14 | [0.11, 0.21] | 0.21 | 0.57 | 0.14 | 0.62 | 0.23 | 26.87 |
| E4: lightgbm | 0.19 | [0.13, 0.27] | 0.27 | 0.51 | 0.15 | 0.33 | 0.20 | 13.34 |
| E4: prior_horario | 0.12 | [0.10, 0.16] | 0.18 | 0.55 | 0.15 | 0.44 | 0.22 | 17.63 |

AUC-PR ao acaso (taxa de positivos): 0.105.

## Localização (Tarefa B): ranqueamento nativo da LSTM

| Ranqueamento | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| LSTM: soma dos erros | 0.33 ± 0.01 | 0.58 ± 0.01 | 0.80 ± 0.01 | 0.51 ± 0.00 |
| LSTM: média dos 3 maiores erros | 0.34 ± 0.01 | 0.57 ± 0.01 | 0.74 ± 0.01 | 0.50 ± 0.01 |
| E4: referência (frequência no treino) | 0.11 | 0.27 | 0.38 | 0.26 |
| E4: z-score global | 0.32 | 0.48 | 0.68 | 0.47 |
| E4: z-score local | 0.11 | 0.47 | 0.66 | 0.34 |

## Custo (CPU, Apple M4 Pro)

| Modelo | Parâmetros | Treino por dobra (s) | Épocas | Inferência (ms/janela) |
|---|---|---|---|---|
| lstm_forecast | 893,450 | 5.62 | 9.7 | 0.2299 |
| lstm_classifier | 727,169 | 3.27 | 6.6 | — |

## Busca de hiperparâmetros (validação, 1 seed; 84 treinos, 815.3 s)

| Modelo | Configuração | AUC-PR média na validação |
|---|---|---|
| previsora | global/h64/l1 | 0.226 |
| previsora | global/h64/l2 | 0.220 |
| previsora | global/h128/l1 | 0.213 |
| previsora | global/h128/l2 | 0.204 |
| previsora | local/h64/l1 | 0.367 |
| previsora | local/h64/l2 | 0.344 |
| previsora | local/h128/l1 | 0.380 |
| previsora | local/h128/l2 | 0.363 |
| classificadora | global/h64/l1 | 0.145 |
| classificadora | global/h128/l1 | 0.178 |
| classificadora | local/h64/l1 | 0.228 |
| classificadora | local/h128/l1 | 0.280 |
