# LSTM — Telecom (E5, ADR 0015)

Configuração escolhida na validação: previsora {'hidden': 128, 'layers': 1, 'normalization': 'local'}; classificadora {'normalization': 'local', 'hidden': 128, 'layers': 1}. Valores: média ± desvio-padrão entre 5 seeds; "IC médio" = média, entre seeds, dos limites do IC de 95% por bootstrap. Sem point-adjust. A pontuação da previsora lê os últimos 6 minutos da janela (ADR 0019); "AUC-PR por dobra" = média das AUC-PR de cada dobra com falha.

## Detecção (Tarefa A, por janela)

| Método | AUC-PR | IC médio | AUC-PR por dobra | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|---|---|
| lstm_forecast | 0.52 ± 0.02 | [0.37, 0.66] | 0.56 ± 0.01 | 0.81 ± 0.01 | 0.31 ± 0.01 | 0.51 ± 0.02 | 0.38 ± 0.01 | 11.27 ± 0.97 |
| lstm_classifier | 0.48 ± 0.02 | [0.35, 0.64] | 0.60 ± 0.01 | 0.83 ± 0.01 | 0.37 ± 0.04 | 0.55 ± 0.02 | 0.45 ± 0.02 | 9.20 ± 1.29 |
| E4: zscore_3sigma | 0.24 | [0.16, 0.38] | 0.43 | 0.70 | 0.14 | 1.00 | 0.25 | 60.00 |
| E4: zscore_val | 0.24 | [0.16, 0.38] | 0.43 | 0.70 | 0.26 | 0.65 | 0.37 | 18.24 |
| E4: zscore_local | 0.39 | [0.27, 0.54] | 0.49 | 0.76 | 0.30 | 0.41 | 0.35 | 9.30 |
| E4: isolation_forest | 0.16 | [0.11, 0.26] | 0.36 | 0.60 | 0.17 | 0.62 | 0.26 | 30.35 |
| E4: lightgbm | 0.25 | [0.15, 0.41] | 0.56 | 0.59 | 0.22 | 0.50 | 0.31 | 17.50 |
| E4: prior_horario | 0.15 | [0.09, 0.27] | 0.28 | 0.44 | 0.15 | 0.48 | 0.22 | 27.15 |

AUC-PR ao acaso (taxa de positivos): 0.141.

## Localização (Tarefa B): ranqueamento nativo da LSTM

| Ranqueamento | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| LSTM: soma dos erros | 0.42 ± 0.04 | 0.67 ± 0.02 | 0.73 ± 0.01 | 0.56 ± 0.02 |
| LSTM: média dos 3 maiores erros | 0.34 ± 0.03 | 0.71 ± 0.02 | 0.80 ± 0.00 | 0.54 ± 0.01 |
| E4: referência (frequência no treino) | 0.09 | 0.18 | 0.27 | 0.19 |
| E4: z-score global | 0.13 | 0.38 | 0.56 | 0.31 |
| E4: z-score local | 0.09 | 0.49 | 0.60 | 0.33 |

## Custo (CPU, Apple M4 Pro)

| Modelo | Parâmetros | Treino por dobra (s) | Épocas | Inferência (ms/janela) |
|---|---|---|---|---|
| lstm_forecast | 989,600 | 2.37 | 16.0 | 0.2677 |
| lstm_classifier | 803,969 | 1.21 | 8.3 | — |

## Busca de hiperparâmetros (validação, 1 seed; 108 treinos, 335.0 s)

| Modelo | Configuração | AUC-PR média na validação |
|---|---|---|
| previsora | global/h64/l1 | 0.496 |
| previsora | global/h64/l2 | 0.426 |
| previsora | global/h128/l1 | 0.438 |
| previsora | global/h128/l2 | 0.436 |
| previsora | local/h64/l1 | 0.462 |
| previsora | local/h64/l2 | 0.478 |
| previsora | local/h128/l1 | 0.550 |
| previsora | local/h128/l2 | 0.466 |
| classificadora | global/h64/l1 | 0.633 |
| classificadora | global/h128/l1 | 0.647 |
| classificadora | local/h64/l1 | 0.569 |
| classificadora | local/h128/l1 | 0.705 |
