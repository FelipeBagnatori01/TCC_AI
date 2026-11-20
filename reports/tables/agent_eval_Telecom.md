# Avaliação do agente (E8): Telecom

45 consultas de teste; nota pelo script oficial do OpenRCA; IC de 95% por bootstrap sobre consultas.

| Cadeia | Nota parcial | Acerto estrito | Instante | Componente | Motivo |
|---|---|---|---|---|---|
| A0 | 0.18 [0.10; 0.29] | 0.11 [0.02; 0.22] | 0.00 [0.00; 0.00] | 0.18 [0.05; 0.36] | 0.32 [0.14; 0.50] |
| pacote | 0.41 [0.29; 0.54] | 0.29 [0.16; 0.42] | 0.50 [0.32; 0.68] | 0.41 [0.23; 0.64] | 0.32 [0.14; 0.50] |
| A1 | 0.60 [0.46; 0.72] | 0.51 [0.36; 0.67] | 0.71 [0.54; 0.86] | 0.68 [0.50; 0.86] | 0.46 [0.29; 0.64] |
| A2 | 0.60 [0.48; 0.73] | 0.49 [0.36; 0.64] | 0.64 [0.46; 0.82] | 0.73 [0.55; 0.91] | 0.43 [0.25; 0.61] |
| A3 | 0.58 [0.44; 0.71] | 0.49 [0.33; 0.64] | 0.68 [0.50; 0.86] | 0.73 [0.55; 0.91] | 0.43 [0.25; 0.61] |

Campos: acerto por critério do gabarito (aproximado em consultas com duas falhas).

## Comparações pareadas

| Comparação | Papel | Diferença parcial | Veredito | Melhores / piores | McNemar b / c | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| A3 - A1 | primary | -0.018 [-0.129; 0.093] | empate | 6 / 8 | 4 / 5 | 1.000 | 1.000 |
| A2 - A1 | secondary | 0.008 [-0.129; 0.148] | empate | 8 / 10 | 6 / 7 | 1.000 | 1.000 |
| A3 - A2 | secondary | -0.026 [-0.141; 0.085] | empate | 5 / 5 | 4 / 4 | 1.000 | 1.000 |
| A1 - A0 | reference | 0.411 [0.241; 0.567] | superior | 27 / 3 | 21 / 3 | < 0.001 | — |
| A1 - pacote | reference | 0.182 [0.018; 0.341] | superior | 20 / 9 | 14 / 4 | 0.031 | — |
| A2 - A0 | reference | 0.419 [0.256; 0.575] | superior | 24 / 3 | 19 / 2 | < 0.001 | — |
| A2 - pacote | reference | 0.189 [0.044; 0.340] | superior | 17 / 7 | 11 / 2 | 0.022 | — |
| A3 - A0 | reference | 0.393 [0.219; 0.557] | superior | 26 / 4 | 20 / 3 | < 0.001 | — |
| A3 - pacote | reference | 0.163 [0.022; 0.304] | superior | 15 / 5 | 11 / 2 | 0.022 | — |

## Localização (consultas de uma falha que pedem componente: 22)

| Cadeia | HR@1 | Menos o pacote | Veredito | McNemar b / c / p |
|---|---|---|---|---|
| pacote (C1) | 0.41 [0.23; 0.64] | — | — | — |
| A1 | 0.68 [0.50; 0.86] | 0.27 [-0.05; 0.59] | empate | 10 / 4 / 0.180 |
| A2 | 0.73 [0.55; 0.91] | 0.32 [0.09; 0.55] | superior | 8 / 1 / 0.039 |
| A3 | 0.73 [0.55; 0.91] | 0.32 [0.09; 0.55] | superior | 8 / 1 / 0.039 |

## Fidelidade automática (itens sinalizados / verificados)

| Variante | IDs não emitidos | Componentes desconhecidos | Identificadores desconhecidos | Números sem lastro literal |
|---|---|---|---|---|
| A1 | 0 / 519 (0 expl.) | 0 / 347 (0 expl.) | 0 / 210 (0 expl.) | 5 / 361 (4 expl.) |
| A2 | 0 / 531 (0 expl.) | 0 / 344 (0 expl.) | 0 / 223 (0 expl.) | 1 / 419 (1 expl.) |
| A3 | 0 / 774 (0 expl.) | 0 / 332 (0 expl.) | 0 / 198 (0 expl.) | 5 / 483 (4 expl.) |

## Custo

| Variante | US$ total | US$/consulta | Latência/consulta (s) | Ferramentas/consulta | Usa get_evidence | Usa find_anomalous_kpis |
|---|---|---|---|---|---|---|
| A1 | 8.69 | 0.193 ± 0.061 | 34.0 ± 9.6 | 13.2 ± 4.5 | 0 | 45 |
| A2 | 7.55 | 0.168 ± 0.079 | 29.0 ± 11.5 | 10.8 ± 5.8 | 45 | 45 |
| A3 | 7.30 | 0.162 ± 0.071 | 28.3 ± 10.7 | 9.8 ± 5.3 | 45 | 45 |

## Reprodução a partir do cache

- A1: 20261120T070324-agent_a1_replay, 0 chamadas pagas, 368 do cache, previsões iguais: True, metrics.json igual: True
- A2: 20261120T070349-agent_a2_replay, 0 chamadas pagas, 298 do cache, previsões iguais: True, metrics.json igual: True
- A3: 20261120T070410-agent_a3_replay, 0 chamadas pagas, 271 do cache, previsões iguais: True, metrics.json igual: True

## Auditoria manual

0 de 60 afirmações deste sistema classificadas.
