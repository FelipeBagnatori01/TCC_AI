# Haiku vs. Opus no agente — Telecom (45 consultas; ADR 0023)

Exploratório. Modelos: `claude-opus-5-5` e `claude-haiku-5-5`; margem de não inferioridade 0.10 na nota parcial oficial.

## Nota oficial (IC 95% por bootstrap sobre consultas)

| Cadeia | Nota parcial | Acerto estrito |
|---|---|---|
| A0 | 0.18 [0.10; 0.29] | 0.11 [0.02; 0.22] |
| pacote | 0.41 [0.29; 0.54] | 0.29 [0.16; 0.42] |
| Opus A1 | 0.60 [0.46; 0.72] | 0.51 [0.36; 0.67] |
| Opus A2 | 0.60 [0.48; 0.73] | 0.49 [0.36; 0.64] |
| Opus A3 | 0.58 [0.44; 0.71] | 0.49 [0.33; 0.64] |
| Haiku A1 | 0.14 [0.06; 0.23] | 0.09 [0.02; 0.18] |
| Haiku A2 | 0.26 [0.15; 0.37] | 0.18 [0.07; 0.29] |
| Haiku A3 | 0.34 [0.21; 0.47] | 0.29 [0.16; 0.42] |

## Dentro do Haiku (família da E8; Holm no McNemar)

| Comparação | Diferença | Veredito | McNemar (só 1º / só 2º) | p Holm | Papel |
|---|---|---|---|---|---|
| Haiku A3 - Haiku A1 | 0.20 [0.04; 0.36] | superior | 12/3 | 0,105 | primary |
| Haiku A2 - Haiku A1 | 0.12 [-0.02; 0.26] | empate | 7/3 | 0,344 | secondary |
| Haiku A3 - Haiku A2 | 0.08 [-0.00; 0.18] | empate | 5/0 | 0,125 | secondary |

## Haiku vs. Opus (não inferioridade, margem 0.10)

| Comparação | Diferença | Não inferioridade | Veredito usual | McNemar | p |
|---|---|---|---|---|---|
| Haiku A1 - Opus A1 | -0.46 [-0.61; -0.30] | inferior | inferior | 2/21 | < 0,001 |
| Haiku A1 - Opus A2 | -0.47 [-0.61; -0.33] | inferior | inferior | 1/19 | < 0,001 |
| Haiku A1 - Opus A3 | -0.44 [-0.59; -0.28] | inferior | inferior | 2/20 | < 0,001 |
| Haiku A2 - Opus A1 | -0.34 [-0.50; -0.18] | inferior | inferior | 2/17 | < 0,001 |
| Haiku A2 - Opus A2 | -0.35 [-0.50; -0.19] | inferior | inferior | 2/16 | 0,001 |
| Haiku A2 - Opus A3 | -0.32 [-0.47; -0.17] | inferior | inferior | 2/16 | 0,001 |
| Haiku A3 - Opus A1 | -0.26 [-0.42; -0.09] | inconclusivo | inferior | 5/15 | 0,041 |
| Haiku A3 - Opus A2 | -0.26 [-0.42; -0.11] | inferior | inferior | 3/12 | 0,035 |
| Haiku A3 - Opus A3 | -0.24 [-0.39; -0.09] | inconclusivo | inferior | 4/13 | 0,049 |

Iguala o Opus (não inferior a todas as variantes dele):

- Haiku A1: não (só contexto)
- Haiku A2: não (afirmação registrada)
- Haiku A3: não (afirmação registrada)

## Contra as referências sem LLM

| Comparação | Diferença | Veredito |
|---|---|---|
| Haiku A1 - A0 | -0.05 [-0.16; 0.06] | empate |
| Haiku A1 - pacote | -0.28 [-0.42; -0.14] | inferior |
| Haiku A2 - A0 | 0.07 [-0.10; 0.23] | empate |
| Haiku A2 - pacote | -0.16 [-0.29; -0.04] | inferior |
| Haiku A3 - A0 | 0.16 [-0.02; 0.33] | empate |
| Haiku A3 - pacote | -0.07 [-0.20; 0.05] | empate |

## Localização (consultas de uma falha que pedem componente: 22)

| Cadeia | Acerto do componente |
|---|---|
| pacote | 0.41 [0.23; 0.64] |
| Opus A1 | 0.68 [0.50; 0.86] |
| Opus A2 | 0.73 [0.55; 0.91] |
| Opus A3 | 0.73 [0.55; 0.91] |
| Haiku A1 | 0.09 [0.00; 0.23] |
| Haiku A2 | 0.36 [0.18; 0.55] |
| Haiku A3 | 0.55 [0.32; 0.73] |

## Concordância com a resposta do pacote e instante no início da janela (descritivo)

| Cadeia | Instante = pacote | Componente = pacote | Motivo = pacote | Instante no início da janela |
|---|---|---|---|---|
| Opus A1 | 0.46 [0.29; 0.64] (n=28) | 0.27 [0.09; 0.45] (n=22) | 0.43 [0.25; 0.61] (n=28) | 0.07 [0.00; 0.18] (n=28) |
| Opus A2 | 0.39 [0.21; 0.57] (n=28) | 0.45 [0.23; 0.68] (n=22) | 0.32 [0.14; 0.50] (n=28) | 0.07 [0.00; 0.18] (n=28) |
| Opus A3 | 0.50 [0.32; 0.68] (n=28) | 0.50 [0.27; 0.73] (n=22) | 0.39 [0.21; 0.57] (n=28) | 0.11 [0.00; 0.21] (n=28) |
| Haiku A1 | 0.07 [0.00; 0.18] (n=28) | 0.09 [0.00; 0.23] (n=22) | 0.39 [0.21; 0.57] (n=28) | 0.75 [0.57; 0.89] (n=28) |
| Haiku A2 | 0.21 [0.07; 0.36] (n=28) | 0.32 [0.14; 0.50] (n=22) | 0.29 [0.14; 0.46] (n=28) | 0.61 [0.43; 0.79] (n=28) |
| Haiku A3 | 0.39 [0.21; 0.57] (n=28) | 0.59 [0.36; 0.77] (n=22) | 0.29 [0.14; 0.46] (n=28) | 0.50 [0.32; 0.68] (n=28) |
| pacote | — | — | — | 0.04 [0.00; 0.11] (n=28) |

## Custo

| Cadeia | Total (US$) | Por consulta (US$, média ± dp) | Respostas válidas |
|---|---|---|---|
| Haiku A1 | 0.28 | 0.006 ± 0.001 | 1.00 |
| Opus A1 | 8.69 | 0.193 ± 0.061 | 1.00 |
| Haiku A2 | 0.29 | 0.006 ± 0.001 | 1.00 |
| Opus A2 | 7.55 | 0.168 ± 0.079 | 1.00 |
| Haiku A3 | 0.26 | 0.006 ± 0.001 | 1.00 |
| Opus A3 | 7.30 | 0.162 ± 0.071 | 1.00 |

## Fidelidade automática (Haiku; sem gabarito)

| Cadeia | IDs não emitidos | Componentes desconhecidos | KPIs desconhecidos | Números sem correspondência | Controle x1,37 casado |
|---|---|---|---|---|---|
| Haiku A1 | 0/416 | 1/296 | 0/301 | 9/231 | 0.38 |
| Haiku A2 | 0/515 | 0/303 | 0/289 | 5/272 | 0.35 |
| Haiku A3 | 0/565 | 0/264 | 1/302 | 7/286 | 0.31 |

## Reprodução a partir do cache

- Haiku A1: ok (0 pagas)
- Haiku A2: ok (0 pagas)
- Haiku A3: ok (0 pagas)
