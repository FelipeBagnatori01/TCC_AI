# Avaliação do agente (E8): Bank

120 consultas de teste; nota pelo script oficial do OpenRCA; IC de 95% por bootstrap sobre consultas.

| Cadeia | Nota parcial | Acerto estrito | Instante | Componente | Motivo |
|---|---|---|---|---|---|
| A0 | 0.14 [0.09; 0.19] | 0.08 [0.04; 0.13] | 0.04 [0.00; 0.08] | 0.13 [0.06; 0.21] | 0.16 [0.08; 0.24] |
| pacote | 0.35 [0.28; 0.43] | 0.23 [0.16; 0.31] | 0.41 [0.32; 0.51] | 0.26 [0.17; 0.36] | 0.34 [0.23; 0.45] |
| A1 | 0.43 [0.35; 0.51] | 0.36 [0.28; 0.45] | 0.37 [0.26; 0.47] | 0.54 [0.42; 0.65] | 0.44 [0.33; 0.55] |
| A2 | 0.40 [0.33; 0.49] | 0.31 [0.23; 0.39] | 0.43 [0.32; 0.53] | 0.50 [0.38; 0.62] | 0.45 [0.34; 0.57] |
| A3 | 0.46 [0.38; 0.54] | 0.37 [0.28; 0.45] | 0.46 [0.35; 0.57] | 0.54 [0.42; 0.66] | 0.49 [0.38; 0.61] |

Campos: acerto por critério do gabarito (aproximado em consultas com duas falhas).

## Comparações pareadas

| Comparação | Papel | Diferença parcial | Veredito | Melhores / piores | McNemar b / c | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| A3 - A1 | primary | 0.029 [-0.049; 0.104] | empate | 20 / 14 | 13 / 12 | 1.000 | 1.000 |
| A2 - A1 | secondary | -0.024 [-0.095; 0.046] | empate | 17 / 17 | 8 / 14 | 0.286 | 0.689 |
| A3 - A2 | secondary | 0.052 [-0.025; 0.131] | empate | 22 / 16 | 16 / 9 | 0.230 | 0.689 |
| A1 - A0 | reference | 0.291 [0.194; 0.388] | superior | 54 / 11 | 38 / 5 | < 0.001 | — |
| A1 - pacote | reference | 0.077 [-0.032; 0.185] | empate | 41 / 30 | 31 / 16 | 0.040 | — |
| A2 - A0 | reference | 0.267 [0.176; 0.358] | superior | 53 / 10 | 32 / 5 | < 0.001 | — |
| A2 - pacote | reference | 0.053 [-0.050; 0.156] | empate | 41 / 27 | 26 / 17 | 0.222 | — |
| A3 - A0 | reference | 0.320 [0.226; 0.413] | superior | 57 / 9 | 40 / 6 | < 0.001 | — |
| A3 - pacote | reference | 0.106 [0.008; 0.203] | superior | 40 / 20 | 29 / 13 | 0.020 | — |

## Localização (consultas de uma falha que pedem componente: 56)

| Cadeia | HR@1 | Menos o pacote | Veredito | McNemar b / c / p |
|---|---|---|---|---|
| pacote (C1) | 0.25 [0.14; 0.38] | — | — | — |
| A1 | 0.55 [0.43; 0.68] | 0.30 [0.16; 0.45] | superior | 20 / 3 / < 0.001 |
| A2 | 0.45 [0.32; 0.57] | 0.20 [0.05; 0.34] | superior | 14 / 3 / 0.013 |
| A3 | 0.50 [0.38; 0.62] | 0.25 [0.12; 0.39] | superior | 16 / 2 / 0.001 |

## Fidelidade automática (itens sinalizados / verificados)

| Variante | IDs não emitidos | Componentes desconhecidos | Identificadores desconhecidos | Números sem lastro literal |
|---|---|---|---|---|
| A1 | 0 / 1221 (0 expl.) | 0 / 920 (0 expl.) | 0 / 115 (0 expl.) | 11 / 1203 (11 expl.) |
| A2 | 0 / 1519 (0 expl.) | 0 / 901 (0 expl.) | 0 / 90 (0 expl.) | 13 / 1449 (8 expl.) |
| A3 | 0 / 2018 (0 expl.) | 0 / 830 (0 expl.) | 0 / 100 (0 expl.) | 9 / 1522 (6 expl.) |

## Custo

| Variante | US$ total | US$/consulta | Latência/consulta (s) | Ferramentas/consulta | Usa get_evidence | Usa find_anomalous_kpis |
|---|---|---|---|---|---|---|
| A1 | 27.83 | 0.232 ± 0.114 | 36.4 ± 14.0 | 12.4 ± 6.6 | 0 | 120 |
| A2 | 26.02 | 0.217 ± 0.111 | 35.5 ± 14.4 | 12.1 ± 6.4 | 120 | 120 |
| A3 | 26.49 | 0.221 ± 0.107 | 36.2 ± 14.9 | 12.0 ± 6.4 | 120 | 120 |

## Reprodução a partir do cache

- A1: 20261123T200600-agent_a1_replay, 0 chamadas pagas, 931 do cache, previsões iguais: True, metrics.json igual: True
- A2: 20261123T200703-agent_a2_replay, 0 chamadas pagas, 915 do cache, previsões iguais: True, metrics.json igual: True
- A3: 20261123T200803-agent_a3_replay, 0 chamadas pagas, 889 do cache, previsões iguais: True, metrics.json igual: True

## Auditoria de fidelidade

Classificada por: Claude Opus 5.5 (LLM, mesmo modelo do agente; sem validação humana) (`auditoria_fidelidade_llm.csv`). 60 de 60 afirmações deste sistema classificadas.

| Variante | Explicações | Afirmações | sustentada | parcial | nao_sustentada | nao_verificavel |
|---|---|---|---|---|---|---|
| A1 | 5 | 20 | 0.75 [0.60; 0.90] | 0.05 [0.00; 0.15] | 0.00 [0.00; 0.00] | 0.20 [0.10; 0.25] |
| A2 | 5 | 20 | 0.90 [0.70; 1.00] | 0.05 [0.00; 0.15] | 0.00 [0.00; 0.00] | 0.05 [0.00; 0.15] |
| A3 | 5 | 20 | 0.90 [0.80; 1.00] | 0.05 [0.00; 0.15] | 0.00 [0.00; 0.00] | 0.05 [0.00; 0.15] |
