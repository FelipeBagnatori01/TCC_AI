# Baselines — Bank (`baselines_sem_isencao`)

Gerado por `tcc-rca baselines`. Valores com IC de 95% por bootstrap (2000 reamostragens): janelas em blocos de 30 min; falhas e consultas individualmente. 120 falhas e 120 consultas de teste em 8 dobras. Sem point-adjust.

## Detecção (Tarefa A, por janela e por evento)

| Método | AUC-PR | AUC-ROC | Precisão | Revocação | F1 | Revocação por evento | ao acaso | Atraso mediano (min) | Alarmes falsos/h |
|---|---|---|---|---|---|---|---|---|---|
| zscore_3sigma | 0.10 [0.08, 0.13] | 0.48 [0.42, 0.54] | 0.11 [0.09, 0.13] | 1.00 [1.00, 1.00] | 0.19 [0.16, 0.23] | 1.00 [1.00, 1.00] | 1.00 | -5.00 [-5.00, -5.00] | 60.00 [60.00, 60.00] |
| zscore_val | 0.10 [0.08, 0.13] | 0.48 [0.42, 0.54] | 0.12 [0.09, 0.15] | 0.68 [0.60, 0.76] | 0.20 [0.16, 0.25] | 0.77 [0.69, 0.84] | 1.00 | -5.00 [-5.00, -5.00] | 35.80 [31.98, 39.50] |
| zscore_local | 0.19 [0.15, 0.24] | 0.69 [0.64, 0.73] | 0.19 [0.14, 0.25] | 0.43 [0.35, 0.52] | 0.26 [0.21, 0.33] | 0.65 [0.57, 0.73] | 1.00 | 0.00 [0.00, 0.00] | 13.07 [10.70, 15.50] |
| isolation_forest | 0.14 [0.11, 0.19] | 0.60 [0.54, 0.66] | 0.14 [0.11, 0.17] | 0.56 [0.48, 0.64] | 0.22 [0.18, 0.27] | 0.69 [0.60, 0.77] | 1.00 | -5.00 [-5.00, -5.00] | 24.66 [21.03, 28.18] |
| lightgbm | 0.18 [0.12, 0.26] | 0.51 [0.44, 0.58] | 0.16 [0.12, 0.21] | 0.42 [0.33, 0.50] | 0.24 [0.18, 0.29] | 0.61 [0.52, 0.69] | 1.00 | -5.00 [-5.00, -5.00] | 14.92 [11.98, 18.17] |

Falhas detectadas por motivo (detectadas/total, por evento):

| Método | JVM Out of Memory (OOM) Heap | high CPU usage | high disk I/O read usage | high disk space usage | high memory usage | network latency | network packet loss |
|---|---|---|---|---|---|---|---|
| zscore_3sigma | 3/3 | 31/31 | 18/18 | 5/5 | 5/5 | 26/26 | 32/32 |
| zscore_val | 3/3 | 24/31 | 16/18 | 4/5 | 4/5 | 18/26 | 23/32 |
| zscore_local | 2/3 | 16/31 | 14/18 | 3/5 | 4/5 | 15/26 | 24/32 |
| isolation_forest | 2/3 | 17/31 | 12/18 | 4/5 | 3/5 | 19/26 | 26/32 |
| lightgbm | 2/3 | 16/31 | 9/18 | 3/5 | 3/5 | 14/26 | 26/32 |

Taxa de positivos nas janelas rotuladas de teste: 0.1052 (AUC-PR de um classificador aleatório).

## Localização (Tarefa B): heurísticas de z-score

| Heurística | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|
| referência: frequência no treino | 0.11 [0.06, 0.17] | 0.27 [0.18, 0.35] | 0.38 [0.30, 0.47] | 0.26 [0.21, 0.31] |
| z-score global (máximo) | 0.32 [0.23, 0.40] | 0.48 [0.39, 0.57] | 0.68 [0.59, 0.76] | 0.47 [0.41, 0.54] |
| z-score local (média dos 3 maiores) | 0.11 [0.06, 0.17] | 0.46 [0.37, 0.55] | 0.65 [0.57, 0.73] | 0.34 [0.29, 0.39] |

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
