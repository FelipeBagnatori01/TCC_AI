# Baselines — Bank (`baselines`)

Gerado por `tcc-rca baselines`. Valores com IC de 95% por bootstrap (2000 reamostragens): janelas em blocos de 30 min; falhas e consultas individualmente. 120 falhas e 120 consultas de teste em 8 dobras. Sem point-adjust. Os z-scores pontuam os últimos 6 minutos da janela (ADR 0019). "AUC-PR por dobra" = média das AUC-PR de cada dobra com falha; `prior_horario` = referência sem telemetria (só o horário da janela). Falhas de teste sem janela positiva: 2; janelas-oráculo deslocadas para não cruzar o dia: 2.

## Detecção (Tarefa A, por janela e por evento)

| Método | AUC-PR | AUC-PR por dobra | AUC-ROC | Precisão | Revocação | F1 | Revocação por evento | ao acaso | Atraso mediano (min) | Alarmes falsos/h |
|---|---|---|---|---|---|---|---|---|---|---|
| zscore_3sigma | 0.10 [0.08, 0.13] | 0.31 [0.26, 0.37] | 0.48 [0.42, 0.54] | 0.11 [0.09, 0.13] | 1.00 [1.00, 1.00] | 0.19 [0.16, 0.23] | 1.00 [1.00, 1.00] | 1.00 | -5.00 [-5.00, -5.00] | 60.00 [60.00, 60.00] |
| zscore_val | 0.10 [0.08, 0.13] | 0.31 [0.26, 0.37] | 0.48 [0.42, 0.54] | 0.12 [0.09, 0.15] | 0.68 [0.60, 0.76] | 0.20 [0.16, 0.25] | 0.77 [0.69, 0.84] | 1.00 | -5.00 [-5.00, -5.00] | 35.06 [31.16, 38.89] |
| zscore_local | 0.33 [0.26, 0.41] | 0.35 [0.30, 0.45] | 0.79 [0.75, 0.83] | 0.37 [0.29, 0.45] | 0.40 [0.32, 0.48] | 0.39 [0.31, 0.46] | 0.66 [0.57, 0.74] | 0.95 | 0.00 [0.00, 2.00] | 4.77 [3.85, 5.79] |
| isolation_forest | 0.14 [0.11, 0.21] | 0.21 [0.17, 0.30] | 0.57 [0.51, 0.63] | 0.14 [0.11, 0.18] | 0.62 [0.54, 0.71] | 0.23 [0.18, 0.28] | 0.72 [0.63, 0.79] | 1.00 | -5.00 [-5.00, -5.00] | 26.87 [23.28, 30.58] |
| lightgbm | 0.19 [0.13, 0.27] | 0.27 [0.21, 0.35] | 0.51 [0.45, 0.58] | 0.15 [0.10, 0.20] | 0.33 [0.25, 0.41] | 0.20 [0.15, 0.26] | 0.49 [0.40, 0.57] | 1.00 | -5.00 [-5.00, -1.00] | 13.34 [10.55, 16.50] |
| prior_horario | 0.12 [0.10, 0.16] | 0.18 [0.15, 0.27] | 0.55 [0.49, 0.61] | 0.15 [0.11, 0.20] | 0.44 [0.35, 0.53] | 0.22 [0.17, 0.28] | 0.56 [0.47, 0.65] | 1.00 | -5.00 [-5.00, -5.00] | 17.63 [14.42, 21.23] |

Falhas detectadas por motivo (detectadas/total, por evento):

| Método | JVM Out of Memory (OOM) Heap | high CPU usage | high disk I/O read usage | high disk space usage | high memory usage | network latency | network packet loss |
|---|---|---|---|---|---|---|---|
| zscore_3sigma | 3/3 | 31/31 | 18/18 | 5/5 | 5/5 | 26/26 | 32/32 |
| zscore_val | 3/3 | 23/31 | 17/18 | 4/5 | 4/5 | 18/26 | 23/32 |
| zscore_local | 3/3 | 19/31 | 15/18 | 3/5 | 3/5 | 15/26 | 21/32 |
| isolation_forest | 3/3 | 20/31 | 12/18 | 4/5 | 4/5 | 18/26 | 25/32 |
| lightgbm | 2/3 | 13/31 | 6/18 | 2/5 | 3/5 | 11/26 | 22/32 |
| prior_horario | 2/3 | 19/31 | 9/18 | 4/5 | 2/5 | 17/26 | 14/32 |

Taxa de positivos nas janelas rotuladas de teste: 0.1052 (AUC-PR de um classificador aleatório).

## Localização (Tarefa B): heurísticas de z-score

| Heurística | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| referência: frequência no treino | 0.11 [0.06, 0.17] | 0.27 [0.18, 0.35] | 0.38 [0.30, 0.47] | 0.26 [0.21, 0.31] |
| z-score global (máximo) | 0.32 [0.23, 0.40] | 0.48 [0.39, 0.57] | 0.68 [0.59, 0.76] | 0.47 [0.41, 0.54] |
| z-score local (média dos 3 maiores) | 0.11 [0.06, 0.17] | 0.47 [0.38, 0.56] | 0.66 [0.57, 0.74] | 0.34 [0.29, 0.39] |

| Motivo | Falhas | HR@1 referência | HR@1 global | HR@1 local |
|---|---|---|---|---|
| JVM Out of Memory (OOM) Heap | 3 | 0.00 | 0.00 | 0.00 |
| high CPU usage | 31 | 0.06 | 0.35 | 0.06 |
| high disk I/O read usage | 18 | 0.17 | 0.56 | 0.33 |
| high disk space usage | 5 | 0.00 | 0.40 | 0.20 |
| high memory usage | 5 | 0.20 | 0.00 | 0.20 |
| network latency | 26 | 0.08 | 0.23 | 0.08 |
| network packet loss | 32 | 0.16 | 0.28 | 0.03 |

## Diagnóstico (Tarefa C): baseline constante, script oficial do OpenRCA

Acerto estrito (%): 8.33 [3.33, 13.33]; pontuação parcial (%): 13.74 [8.60, 19.16].

| Tarefa | Consultas | Acertos estritos | Parcial médio |
|---|---|---|---|
| task_1 | 20 | 1 | 0.05 |
| task_2 | 17 | 5 | 0.29 |
| task_3 | 14 | 3 | 0.21 |
| task_4 | 17 | 1 | 0.12 |
| task_5 | 16 | 0 | 0.09 |
| task_6 | 19 | 0 | 0.16 |
| task_7 | 17 | 0 | 0.06 |
