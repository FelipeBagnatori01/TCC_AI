# Caracterização da divisão temporal (Bank)

Gerado por `tcc-rca tasks` (ADR 0007/0008). Janelas de 20 min; dias excluídos: .

| Dobra | Treino | Val. | Teste | Jan. treino (normais) | Jan. val. (pos.) | Jan. teste | Pos. | Neg. | Ign. | Falhas teste | Consultas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 dias | 2021_03_05 | 2021_03_06 | 1421 (711) | 1401 (0) | 1421 | 66 | 901 | 454 | 11 | 11 |
| 1 | 2 dias | 2021_03_06 | 2021_03_07 | 2842 (2132) | 1401 (64) | 1421 | 96 | 636 | 689 | 16 | 16 |
| 2 | 3 dias | 2021_03_07 | 2021_03_09 | 4263 (3033) | 1421 (96) | 1421 | 126 | 528 | 767 | 21 | 21 |
| 3 | 4 dias | 2021_03_09 | 2021_03_10 | 5684 (3669) | 1401 (120) | 1421 | 96 | 669 | 656 | 16 | 16 |
| 4 | 5 dias | 2021_03_10 | 2021_03_12 | 7105 (4197) | 1421 (96) | 1421 | 78 | 893 | 450 | 13 | 13 |
| 5 | 6 dias | 2021_03_12 | 2021_03_23 | 8526 (4866) | 1421 (78) | 1421 | 78 | 814 | 529 | 14 | 14 |
| 6 | 7 dias | 2021_03_23 | 2021_03_24 | 9947 (5759) | 1401 (78) | 1421 | 60 | 992 | 369 | 10 | 10 |
| 7 | 8 dias | 2021_03_24 | 2021_03_25 | 11368 (6573) | 1401 (60) | 1421 | 108 | 592 | 721 | 19 | 19 |

## Falhas testadas (120 falhas nos dias de teste)

| Motivo | Falhas |
|---|---|
| network packet loss | 32 |
| high CPU usage | 31 |
| network latency | 26 |
| high disk I/O read usage | 18 |
| high memory usage | 5 |
| high disk space usage | 5 |
| JVM Out of Memory (OOM) Heap | 3 |

| Componente | Falhas |
|---|---|
| Tomcat01 | 19 |
| MG01 | 15 |
| MG02 | 13 |
| Tomcat04 | 12 |
| IG02 | 12 |
| Tomcat03 | 11 |
| apache02 | 10 |
| Tomcat02 | 8 |
| apache01 | 8 |
| Redis02 | 4 |
| IG01 | 3 |
| Mysql01 | 2 |
| Mysql02 | 2 |
| Redis01 | 1 |

| Tipo de tarefa | Consultas |
|---|---|
| task_1 | 20 |
| task_2 | 17 |
| task_3 | 14 |
| task_4 | 17 |
| task_5 | 16 |
| task_6 | 19 |
| task_7 | 17 |
