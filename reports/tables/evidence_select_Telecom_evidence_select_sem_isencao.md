# Evidência por seleção prequencial — Telecom (ADR 0020)

45 consultas de teste (45 falhas), 10 dobras. Em cada dobra, cada etapa escolhe o método com maior acerto nas consultas dos dias anteriores ao teste (ordem: instante, componente dado o instante, motivo dados os dois). IC de 95% por bootstrap sobre consultas.

## Escolhas por dobra

| Dobra | Dia de teste | Consultas passadas | Instante | Componente | Motivo | Motivo supera R0 no passado |
|---|---|---|---|---|---|---|
| 0 | 2020_05_22 | 6 | T1 | C3 | R0 | não |
| 1 | 2020_05_23 | 10 | T1 | C3 | R0 | não |
| 2 | 2020_05_24 | 16 | T1 | C8 | R0 | não |
| 3 | 2020_05_25 | 19 | T1 | C8 | R0 | não |
| 4 | 2020_05_26 | 22 | T1 | C5 | R0 | não |
| 5 | 2020_05_27 | 29 | T1 | C8 | R3 | sim |
| 6 | 2020_05_28 | 37 | T1 | C2 | R1 | sim |
| 7 | 2020_05_29 | 39 | T1 | C2 | R3 | sim |
| 8 | 2020_05_30 | 42 | T1 | C8 | R0 | não |
| 9 | 2020_05_31 | 47 | T1 | C2 | R0 | não |

## Acerto por campo (todas as falhas de teste)

| Cadeia | instante (≤ 60 s) | componente | motivo |
|---|---|---|---|
| seletor prequencial | 0.42 [0.29; 0.56] | 0.04 [0.00; 0.11] | 0.29 [0.16; 0.42] |
| pacote atual | 0.71 [0.58; 0.84] | 0.16 [0.07; 0.27] | 0.31 [0.18; 0.44] |

## Nota oficial do OpenRCA (só os campos pedidos por consulta)

| Cadeia | Nota parcial | Acerto estrito |
|---|---|---|
| seletor prequencial | 0.29 [0.18; 0.40] | 0.16 [0.07; 0.27] |
| pacote atual | 0.42 [0.29; 0.55] | 0.31 [0.18; 0.44] |
| A0 constante (E4) | 0.18 [0.09; 0.29] | 0.11 [0.02; 0.20] |

## Comparações pareadas (seletor menos rival, mesmas consultas)

| Rival | Métrica | Diferença | Veredito |
|---|---|---|---|
| current_package | nota parcial | -0.133 [-0.282; +0.011] | empate |
| current_package | acerto estrito | -0.156 [-0.311; -0.022] | inferior |
| constant_a0 | nota parcial | +0.100 [-0.037; +0.234] | empate |
| constant_a0 | acerto estrito | +0.044 [-0.089; +0.178] | empate |
| current_package | campo occurrence_datetime | -0.289 [-0.444; -0.156] | inferior |
| current_package | campo component | -0.111 [-0.244; +0.000] | empate |
| current_package | campo reason | -0.022 [-0.200; +0.156] | empate |

## Descritivo: cada método no teste (não usado para escolher)

Instante: acerto a até 60 s. Componente: acerto com o instante escolhido pelo seletor na dobra, e com o instante rotulado (janela-oráculo, comparável à E4 a E6). Motivo: acerto com o instante e o componente escolhidos pelo seletor.

| Método | Acerto | MRR | HR@1 com instante rotulado | MRR com instante rotulado |
|---|---|---|---|---|
| T0: início da janela da consulta | 0.00 | — | — | — |
| T1: previsora LSTM: 1º minuto acima do limiar | 0.42 | — | — | — |
| T2: z-score local: 1º minuto acima do limiar | 0.40 | — | — | — |
| T3: BOCPD sobre o z-score local | 0.29 | — | — | — |
| T4: BOCPD sobre a previsora LSTM | 0.36 | — | — | — |
| C0: frequência no passado | 0.13 | 0.20 | 0.09 | 0.20 |
| C1: previsora LSTM + SHAP | 0.11 | 0.29 | 0.29 | 0.49 |
| C2: ranqueamento nativo da previsora | 0.16 | 0.33 | 0.18 | 0.40 |
| C3: z-score global | 0.09 | 0.25 | 0.16 | 0.34 |
| C4: pontuação robusta (BARO) | 0.04 | 0.20 | 0.11 | 0.29 |
| C5: N-Sigma | 0.16 | 0.31 | 0.18 | 0.44 |
| C6: desvio sazonal (dia anterior) | 0.00 | 0.13 | 0.00 | 0.15 |
| C7: PageRank no grafo de dependências | 0.07 | 0.20 | 0.07 | 0.24 |
| C8: fusão por posto (C1, C3, C4) | 0.16 | 0.32 | 0.29 | 0.50 |
| R0: mais frequente no passado | 0.33 | — | — | — |
| R1: mais frequente no passado para o componente | 0.22 | — | — | — |
| R2: regra categoria do KPI → motivo | 0.29 | — | — | — |
| R3: regressão logística | 0.42 | — | — | — |
