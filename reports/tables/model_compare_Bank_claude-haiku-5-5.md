# Haiku vs. Opus no agente — Bank (120 consultas; ADR 0023)

Exploratório. Modelos: `claude-opus-5-5` e `claude-haiku-5-5`; margem de não inferioridade 0.10 na nota parcial oficial.

## Nota oficial (IC 95% por bootstrap sobre consultas)

| Cadeia | Nota parcial | Acerto estrito |
|---|---|---|
| A0 | 0.14 [0.09; 0.19] | 0.08 [0.04; 0.13] |
| pacote | 0.35 [0.28; 0.43] | 0.23 [0.16; 0.31] |
| Opus A1 | 0.43 [0.35; 0.51] | 0.36 [0.28; 0.45] |
| Opus A2 | 0.40 [0.33; 0.49] | 0.31 [0.23; 0.39] |
| Opus A3 | 0.46 [0.38; 0.54] | 0.37 [0.28; 0.45] |
| Haiku A1 | 0.16 [0.11; 0.21] | 0.07 [0.03; 0.12] |
| Haiku A2 | 0.19 [0.13; 0.25] | 0.10 [0.05; 0.16] |
| Haiku A3 | 0.29 [0.22; 0.36] | 0.17 [0.10; 0.23] |

## Dentro do Haiku (família da E8; Holm no McNemar)

| Comparação | Diferença | Veredito | McNemar (só 1º / só 2º) | p Holm | Papel |
|---|---|---|---|---|---|
| Haiku A3 - Haiku A1 | 0.13 [0.07; 0.19] | superior | 13/1 | 0,005 | primary |
| Haiku A2 - Haiku A1 | 0.03 [-0.01; 0.07] | empate | 5/1 | 0,219 | secondary |
| Haiku A3 - Haiku A2 | 0.10 [0.04; 0.16] | superior | 12/4 | 0,154 | secondary |

## Haiku vs. Opus (não inferioridade, margem 0.10)

| Comparação | Diferença | Não inferioridade | Veredito usual | McNemar | p |
|---|---|---|---|---|---|
| Haiku A1 - Opus A1 | -0.27 [-0.35; -0.19] | inferior | inferior | 2/37 | < 0,001 |
| Haiku A1 - Opus A2 | -0.25 [-0.32; -0.17] | inferior | inferior | 1/30 | < 0,001 |
| Haiku A1 - Opus A3 | -0.30 [-0.38; -0.22] | inferior | inferior | 2/38 | < 0,001 |
| Haiku A2 - Opus A1 | -0.24 [-0.32; -0.16] | inferior | inferior | 3/34 | < 0,001 |
| Haiku A2 - Opus A2 | -0.22 [-0.29; -0.14] | inferior | inferior | 3/28 | < 0,001 |
| Haiku A2 - Opus A3 | -0.27 [-0.35; -0.19] | inferior | inferior | 4/36 | < 0,001 |
| Haiku A3 - Opus A1 | -0.14 [-0.23; -0.06] | inconclusivo | inferior | 6/29 | < 0,001 |
| Haiku A3 - Opus A2 | -0.12 [-0.19; -0.05] | inconclusivo | inferior | 4/21 | < 0,001 |
| Haiku A3 - Opus A3 | -0.17 [-0.24; -0.10] | inconclusivo | inferior | 4/28 | < 0,001 |

Iguala o Opus (não inferior a todas as variantes dele):

- Haiku A1: não (só contexto)
- Haiku A2: não (afirmação registrada)
- Haiku A3: não (afirmação registrada)

## Contra as referências sem LLM

| Comparação | Diferença | Veredito |
|---|---|---|
| Haiku A1 - A0 | 0.02 [-0.05; 0.09] | empate |
| Haiku A1 - pacote | -0.19 [-0.28; -0.11] | inferior |
| Haiku A2 - A0 | 0.05 [-0.02; 0.12] | empate |
| Haiku A2 - pacote | -0.16 [-0.25; -0.08] | inferior |
| Haiku A3 - A0 | 0.15 [0.06; 0.23] | superior |
| Haiku A3 - pacote | -0.06 [-0.15; 0.02] | empate |

## Localização (consultas de uma falha que pedem componente: 56)

| Cadeia | Acerto do componente |
|---|---|
| pacote | 0.25 [0.14; 0.38] |
| Opus A1 | 0.55 [0.43; 0.68] |
| Opus A2 | 0.45 [0.32; 0.57] |
| Opus A3 | 0.50 [0.38; 0.62] |
| Haiku A1 | 0.21 [0.11; 0.32] |
| Haiku A2 | 0.21 [0.11; 0.32] |
| Haiku A3 | 0.30 [0.18; 0.43] |

## Concordância com a resposta do pacote e instante no início da janela (descritivo)

| Cadeia | Instante = pacote | Componente = pacote | Motivo = pacote | Instante no início da janela |
|---|---|---|---|---|
| Opus A1 | 0.19 [0.09; 0.29] (n=58) | 0.25 [0.14; 0.38] (n=56) | 0.30 [0.19; 0.41] (n=63) | 0.26 [0.17; 0.35] (n=82) |
| Opus A2 | 0.19 [0.09; 0.29] (n=58) | 0.21 [0.11; 0.32] (n=56) | 0.25 [0.14; 0.37] (n=63) | 0.21 [0.13; 0.30] (n=82) |
| Opus A3 | 0.31 [0.19; 0.43] (n=58) | 0.29 [0.18; 0.41] (n=56) | 0.27 [0.16; 0.38] (n=63) | 0.18 [0.11; 0.27] (n=82) |
| Haiku A1 | 0.12 [0.05; 0.21] (n=58) | 0.20 [0.09; 0.30] (n=56) | 0.22 [0.13; 0.33] (n=63) | 0.68 [0.59; 0.78] (n=82) |
| Haiku A2 | 0.10 [0.03; 0.19] (n=58) | 0.21 [0.11; 0.32] (n=56) | 0.19 [0.10; 0.29] (n=63) | 0.70 [0.60; 0.79] (n=82) |
| Haiku A3 | 0.28 [0.17; 0.40] (n=58) | 0.34 [0.21; 0.46] (n=56) | 0.33 [0.22; 0.44] (n=63) | 0.62 [0.52; 0.72] (n=82) |
| pacote | — | — | — | 0.09 [0.03; 0.15] (n=82) |

## Custo

| Cadeia | Total (US$) | Por consulta (US$, média ± dp) | Respostas válidas |
|---|---|---|---|
| Haiku A1 | 0.90 | 0.008 ± 0.002 | 1.00 |
| Opus A1 | 27.83 | 0.232 ± 0.114 | 1.00 |
| Haiku A2 | 0.84 | 0.007 ± 0.002 | 1.00 |
| Opus A2 | 26.02 | 0.217 ± 0.111 | 1.00 |
| Haiku A3 | 0.78 | 0.006 ± 0.002 | 1.00 |
| Opus A3 | 26.49 | 0.221 ± 0.107 | 1.00 |

## Fidelidade automática (Haiku; sem gabarito)

| Cadeia | IDs não emitidos | Componentes desconhecidos | KPIs desconhecidos | Números sem correspondência | Controle x1,37 casado |
|---|---|---|---|---|---|
| Haiku A1 | 0/1260 | 0/790 | 5/116 | 6/581 | 0.33 |
| Haiku A2 | 0/1347 | 0/796 | 0/75 | 2/622 | 0.25 |
| Haiku A3 | 0/1423 | 0/780 | 1/93 | 4/775 | 0.27 |
