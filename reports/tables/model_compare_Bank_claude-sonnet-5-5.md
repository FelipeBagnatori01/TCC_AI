# Sonnet vs. Opus no agente — Bank (120 consultas; ADR 0023)

Exploratório. Modelos: `claude-opus-5-5` e `claude-sonnet-5-5`; margem de não inferioridade 0.10 na nota parcial oficial.

## Nota oficial (IC 95% por bootstrap sobre consultas)

| Cadeia | Nota parcial | Acerto estrito |
|---|---|---|
| A0 | 0.14 [0.09; 0.19] | 0.08 [0.04; 0.13] |
| pacote | 0.35 [0.28; 0.43] | 0.23 [0.16; 0.31] |
| Opus A1 | 0.43 [0.35; 0.51] | 0.36 [0.28; 0.45] |
| Opus A2 | 0.40 [0.33; 0.49] | 0.31 [0.23; 0.39] |
| Opus A3 | 0.46 [0.38; 0.54] | 0.37 [0.28; 0.45] |
| Sonnet A1 | 0.16 [0.11; 0.22] | 0.07 [0.03; 0.12] |
| Sonnet A2 | 0.23 [0.17; 0.30] | 0.13 [0.07; 0.20] |
| Sonnet A3 | 0.32 [0.25; 0.40] | 0.22 [0.14; 0.29] |

## Dentro do Sonnet (família da E8; Holm no McNemar)

| Comparação | Diferença | Veredito | McNemar (só 1º / só 2º) | p Holm | Papel |
|---|---|---|---|---|---|
| Sonnet A3 - Sonnet A1 | 0.16 [0.10; 0.23] | superior | 19/1 | < 0,001 | primary |
| Sonnet A2 - Sonnet A1 | 0.07 [0.03; 0.12] | superior | 9/1 | 0,026 | secondary |
| Sonnet A3 - Sonnet A2 | 0.09 [0.04; 0.15] | superior | 12/2 | 0,026 | secondary |

## Sonnet vs. Opus (não inferioridade, margem 0.10)

| Comparação | Diferença | Não inferioridade | Veredito usual | McNemar | p |
|---|---|---|---|---|---|
| Sonnet A1 - Opus A1 | -0.27 [-0.35; -0.18] | inferior | inferior | 3/38 | < 0,001 |
| Sonnet A1 - Opus A2 | -0.24 [-0.32; -0.17] | inferior | inferior | 2/31 | < 0,001 |
| Sonnet A1 - Opus A3 | -0.29 [-0.38; -0.21] | inferior | inferior | 3/39 | < 0,001 |
| Sonnet A2 - Opus A1 | -0.20 [-0.27; -0.12] | inferior | inferior | 3/30 | < 0,001 |
| Sonnet A2 - Opus A2 | -0.17 [-0.24; -0.10] | inconclusivo | inferior | 3/24 | < 0,001 |
| Sonnet A2 - Opus A3 | -0.22 [-0.30; -0.14] | inferior | inferior | 4/32 | < 0,001 |
| Sonnet A3 - Opus A1 | -0.10 [-0.19; -0.02] | inconclusivo | inferior | 8/25 | 0,005 |
| Sonnet A3 - Opus A2 | -0.08 [-0.15; -0.01] | inconclusivo | inferior | 7/18 | 0,043 |
| Sonnet A3 - Opus A3 | -0.13 [-0.20; -0.06] | inconclusivo | inferior | 4/22 | < 0,001 |

Iguala o Opus (não inferior a todas as variantes dele):

- Sonnet A1: não (só contexto)
- Sonnet A2: não (afirmação registrada)
- Sonnet A3: não (afirmação registrada)

## Contra as referências sem LLM

| Comparação | Diferença | Veredito |
|---|---|---|
| Sonnet A1 - A0 | 0.03 [-0.04; 0.09] | empate |
| Sonnet A1 - pacote | -0.19 [-0.28; -0.10] | inferior |
| Sonnet A2 - A0 | 0.10 [0.02; 0.17] | superior |
| Sonnet A2 - pacote | -0.12 [-0.20; -0.03] | inferior |
| Sonnet A3 - A0 | 0.19 [0.10; 0.27] | superior |
| Sonnet A3 - pacote | -0.03 [-0.11; 0.05] | empate |

## Localização (consultas de uma falha que pedem componente: 56)

| Cadeia | Acerto do componente |
|---|---|
| pacote | 0.25 [0.14; 0.38] |
| Opus A1 | 0.55 [0.43; 0.68] |
| Opus A2 | 0.45 [0.32; 0.57] |
| Opus A3 | 0.50 [0.38; 0.62] |
| Sonnet A1 | 0.16 [0.07; 0.27] |
| Sonnet A2 | 0.25 [0.14; 0.38] |
| Sonnet A3 | 0.32 [0.20; 0.45] |

## Concordância com a resposta do pacote e instante no início da janela (descritivo)

| Cadeia | Instante = pacote | Componente = pacote | Motivo = pacote | Instante no início da janela |
|---|---|---|---|---|
| Opus A1 | 0.19 [0.09; 0.29] (n=58) | 0.25 [0.14; 0.38] (n=56) | 0.30 [0.19; 0.41] (n=63) | 0.26 [0.17; 0.35] (n=82) |
| Opus A2 | 0.19 [0.09; 0.29] (n=58) | 0.21 [0.11; 0.32] (n=56) | 0.25 [0.14; 0.37] (n=63) | 0.21 [0.13; 0.30] (n=82) |
| Opus A3 | 0.31 [0.19; 0.43] (n=58) | 0.29 [0.18; 0.41] (n=56) | 0.27 [0.16; 0.38] (n=63) | 0.18 [0.11; 0.27] (n=82) |
| Sonnet A1 | 0.07 [0.02; 0.14] (n=58) | 0.05 [0.00; 0.12] (n=56) | 0.21 [0.11; 0.32] (n=63) | 0.55 [0.44; 0.66] (n=82) |
| Sonnet A2 | 0.14 [0.05; 0.24] (n=58) | 0.25 [0.14; 0.38] (n=56) | 0.29 [0.17; 0.40] (n=63) | 0.45 [0.35; 0.56] (n=82) |
| Sonnet A3 | 0.36 [0.24; 0.48] (n=58) | 0.41 [0.29; 0.54] (n=56) | 0.33 [0.22; 0.44] (n=63) | 0.39 [0.30; 0.49] (n=82) |
| pacote | — | — | — | 0.09 [0.03; 0.15] (n=82) |

## Custo

| Cadeia | Total (US$) | Por consulta (US$, média ± dp) | Respostas válidas |
|---|---|---|---|
| Sonnet A1 | 4.85 | 0.040 ± 0.018 | 1.00 |
| Opus A1 | 27.83 | 0.232 ± 0.114 | 1.00 |
| Sonnet A2 | 4.55 | 0.038 ± 0.014 | 1.00 |
| Opus A2 | 26.02 | 0.217 ± 0.111 | 1.00 |
| Sonnet A3 | 3.73 | 0.031 ± 0.032 | 1.00 |
| Opus A3 | 26.49 | 0.221 ± 0.107 | 1.00 |

## Fidelidade automática (Sonnet; sem gabarito)

| Cadeia | IDs não emitidos | Componentes desconhecidos | KPIs desconhecidos | Números sem correspondência | Controle x1,37 casado |
|---|---|---|---|---|---|
| Sonnet A1 | 0/385 | 0/505 | 0/45 | 2/413 | 0.15 |
| Sonnet A2 | 0/553 | 0/565 | 0/24 | 4/577 | 0.11 |
| Sonnet A3 | 0/1082 | 0/541 | 0/45 | 5/637 | 0.11 |
