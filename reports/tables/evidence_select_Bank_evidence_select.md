# Evidência por seleção prequencial — Bank (ADR 0020)

120 consultas de teste (134 falhas), 8 dobras. Em cada dobra, cada etapa escolhe o método com maior acerto nas consultas dos dias anteriores ao teste (ordem: instante, componente dado o instante, motivo dados os dois). IC de 95% por bootstrap sobre consultas.

## Escolhas por dobra

| Dobra | Dia de teste | Consultas passadas | Instante | Componente | Motivo | Motivo supera R0 no passado |
|---|---|---|---|---|---|---|
| 0 | 2021_03_06 | 16 | T0 | C7 | R0 | não |
| 1 | 2021_03_07 | 27 | T3 | C8 | R2 | sim |
| 2 | 2021_03_09 | 43 | T4 | C8 | R2 | sim |
| 3 | 2021_03_10 | 64 | T4 | C8 | R2 | sim |
| 4 | 2021_03_12 | 80 | T2 | C8 | R2 | sim |
| 5 | 2021_03_23 | 93 | T2 | C8 | R2 | sim |
| 6 | 2021_03_24 | 107 | T2 | C8 | R2 | sim |
| 7 | 2021_03_25 | 117 | T2 | C8 | R3 | sim |

## Acerto por campo (todas as falhas de teste)

| Cadeia | instante (≤ 60 s) | componente | motivo |
|---|---|---|---|
| seletor prequencial | 0.38 [0.30; 0.46] | 0.34 [0.26; 0.41] | 0.29 [0.22; 0.36] |
| pacote atual | 0.40 [0.33; 0.48] | 0.28 [0.21; 0.36] | 0.35 [0.27; 0.43] |

## Nota oficial do OpenRCA (só os campos pedidos por consulta)

| Cadeia | Nota parcial | Acerto estrito |
|---|---|---|
| seletor prequencial | 0.34 [0.27; 0.41] | 0.23 [0.15; 0.30] |
| pacote atual | 0.35 [0.28; 0.42] | 0.23 [0.16; 0.31] |
| A0 constante (E4) | 0.14 [0.09; 0.20] | 0.08 [0.03; 0.13] |

## Comparações pareadas (seletor menos rival, mesmas consultas)

| Rival | Métrica | Diferença | Veredito |
|---|---|---|---|
| current_package | nota parcial | -0.011 [-0.085; +0.063] | empate |
| current_package | acerto estrito | -0.008 [-0.092; +0.067] | empate |
| constant_a0 | nota parcial | +0.203 [+0.124; +0.285] | superior |
| constant_a0 | acerto estrito | +0.142 [+0.067; +0.225] | superior |
| current_package | campo occurrence_datetime | -0.022 [-0.112; +0.065] | empate |
| current_package | campo component | +0.052 [-0.023; +0.128] | empate |
| current_package | campo reason | -0.060 [-0.152; +0.030] | empate |

## Descritivo: cada método no teste (não usado para escolher)

Instante: acerto a até 60 s. Componente: acerto com o instante escolhido pelo seletor na dobra, e com o instante rotulado (janela-oráculo, comparável à E4 a E6). Motivo: acerto com o instante e o componente escolhidos pelo seletor.

| Método | Acerto | MRR | HR@1 com instante rotulado | MRR com instante rotulado |
|---|---|---|---|---|
| T0: início da janela da consulta | 0.06 | — | — | — |
| T1: previsora LSTM: 1º minuto acima do limiar | 0.38 | — | — | — |
| T2: z-score local: 1º minuto acima do limiar | 0.46 | — | — | — |
| T3: BOCPD sobre o z-score local | 0.31 | — | — | — |
| T4: BOCPD sobre a previsora LSTM | 0.38 | — | — | — |
| C0: frequência no passado | 0.13 | 0.26 | 0.11 | 0.26 |
| C1: previsora LSTM + SHAP | 0.29 | 0.43 | 0.49 | 0.64 |
| C2: ranqueamento nativo da previsora | 0.27 | 0.41 | 0.34 | 0.52 |
| C3: z-score global | 0.25 | 0.42 | 0.32 | 0.47 |
| C4: pontuação robusta (BARO) | 0.28 | 0.42 | 0.32 | 0.51 |
| C5: N-Sigma | 0.21 | 0.38 | 0.30 | 0.54 |
| C6: desvio sazonal (dia anterior) | 0.14 | 0.31 | 0.14 | 0.33 |
| C7: PageRank no grafo de dependências | 0.23 | 0.39 | 0.22 | 0.41 |
| C8: fusão por posto (C1, C3, C4) | 0.33 | 0.47 | 0.46 | 0.62 |
| R0: mais frequente no passado | 0.18 | — | — | — |
| R1: mais frequente no passado para o componente | 0.19 | — | — | — |
| R2: regra categoria do KPI → motivo | 0.28 | — | — | — |
| R3: regressão logística | 0.35 | — | — | — |
