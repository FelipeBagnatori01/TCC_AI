# Baselines — Telecom (`baselines_janela_inteira`)

Gerado por `tcc-rca baselines`. Valores com IC de 95% por bootstrap (2000 reamostragens): janelas em blocos de 30 min; falhas e consultas individualmente. 45 falhas e 45 consultas de teste em 10 dobras. Sem point-adjust. Os z-scores pontuam a janela inteira (ADR 0019). "AUC-PR por dobra" = média das AUC-PR de cada dobra com falha; `prior_horario` = referência sem telemetria (só o horário da janela). Falhas de teste sem janela positiva: 3; janelas-oráculo deslocadas para não cruzar o dia: 3.

## Detecção (Tarefa A, por janela e por evento)

| Método | AUC-PR | AUC-PR por dobra | AUC-ROC | Precisão | Revocação | F1 | Revocação por evento | ao acaso | Atraso mediano (min) | Alarmes falsos/h |
|---|---|---|---|---|---|---|---|---|---|---|
| zscore_3sigma | 0.21 [0.14, 0.36] | 0.39 [0.34, 0.57] | 0.65 [0.55, 0.75] | 0.14 [0.10, 0.19] | 1.00 [1.00, 1.00] | 0.25 [0.18, 0.31] | 1.00 [1.00, 1.00] | 1.00 | -5.00 [-5.00, -5.00] | 60.00 [60.00, 60.00] |
| zscore_val | 0.21 [0.14, 0.36] | 0.39 [0.34, 0.57] | 0.65 [0.55, 0.75] | 0.19 [0.12, 0.28] | 0.71 [0.58, 0.85] | 0.30 [0.21, 0.40] | 0.78 [0.64, 0.89] | 1.00 | -5.00 [-5.00, -2.00] | 29.77 [23.20, 36.71] |
| zscore_local | 0.24 [0.15, 0.37] | 0.36 [0.32, 0.59] | 0.67 [0.58, 0.76] | 0.18 [0.11, 0.28] | 0.44 [0.30, 0.58] | 0.26 [0.16, 0.36] | 0.64 [0.51, 0.78] | 1.00 | 0.00 [-5.00, 1.00] | 19.34 [14.00, 25.10] |
| isolation_forest | 0.16 [0.11, 0.26] | 0.36 [0.29, 0.54] | 0.60 [0.50, 0.70] | 0.17 [0.11, 0.24] | 0.62 [0.48, 0.75] | 0.26 [0.18, 0.35] | 0.78 [0.64, 0.89] | 1.00 | -5.00 [-5.00, -1.00] | 30.35 [23.75, 37.16] |
| lightgbm | 0.25 [0.15, 0.41] | 0.56 [0.50, 0.70] | 0.59 [0.48, 0.68] | 0.22 [0.14, 0.33] | 0.50 [0.37, 0.64] | 0.31 [0.21, 0.41] | 0.64 [0.49, 0.78] | 1.00 | 1.00 [-1.00, 1.00] | 17.50 [12.17, 23.17] |
| prior_horario | 0.15 [0.09, 0.27] | 0.28 [0.21, 0.47] | 0.44 [0.34, 0.55] | 0.15 [0.09, 0.22] | 0.48 [0.33, 0.63] | 0.22 [0.14, 0.31] | 0.49 [0.36, 0.64] | 1.00 | -5.00 [-5.00, -5.00] | 27.15 [20.73, 33.57] |

Falhas detectadas por motivo (detectadas/total, por evento):

| Método | CPU fault | db close | db connection limit | network delay | network loss |
|---|---|---|---|---|---|
| zscore_3sigma | 15/15 | 4/4 | 6/6 | 13/13 | 7/7 |
| zscore_val | 11/15 | 4/4 | 5/6 | 11/13 | 4/7 |
| zscore_local | 8/15 | 3/4 | 4/6 | 8/13 | 6/7 |
| isolation_forest | 12/15 | 3/4 | 5/6 | 13/13 | 2/7 |
| lightgbm | 9/15 | 3/4 | 4/6 | 8/13 | 5/7 |
| prior_horario | 7/15 | 2/4 | 5/6 | 6/13 | 2/7 |

Taxa de positivos nas janelas rotuladas de teste: 0.1409 (AUC-PR de um classificador aleatório).

## Localização (Tarefa B): heurísticas de z-score

| Heurística | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| referência: frequência no treino | 0.09 [0.02, 0.18] | 0.18 [0.09, 0.29] | 0.27 [0.13, 0.40] | 0.19 [0.12, 0.29] |
| z-score global (máximo) | 0.13 [0.04, 0.24] | 0.38 [0.24, 0.51] | 0.56 [0.40, 0.71] | 0.31 [0.22, 0.41] |
| z-score local (média dos 3 maiores) | 0.09 [0.02, 0.18] | 0.49 [0.33, 0.62] | 0.60 [0.44, 0.73] | 0.33 [0.26, 0.41] |

| Motivo | Falhas | HR@1 referência | HR@1 global | HR@1 local |
|---|---|---|---|---|
| CPU fault | 15 | 0.00 | 0.13 | 0.00 |
| db close | 4 | 0.25 | 0.00 | 0.50 |
| db connection limit | 6 | 0.50 | 0.33 | 0.00 |
| network delay | 13 | 0.00 | 0.15 | 0.08 |
| network loss | 7 | 0.00 | 0.00 | 0.14 |

## Diagnóstico (Tarefa C): baseline constante, script oficial do OpenRCA

Acerto estrito (%): 11.11 [2.22, 22.22]; pontuação parcial (%): 18.49 [9.60, 28.49].

| Tarefa | Consultas | Acertos estritos | Parcial médio |
|---|---|---|---|
| task_1 | 9 | 0 | 0.00 |
| task_2 | 7 | 4 | 0.57 |
| task_3 | 4 | 1 | 0.25 |
| task_4 | 7 | 0 | 0.07 |
| task_5 | 4 | 0 | 0.00 |
| task_6 | 6 | 0 | 0.25 |
| task_7 | 8 | 0 | 0.17 |
