# Sonnet vs. Opus no agente — Telecom (45 consultas; ADR 0023)

Exploratório. Modelos: `claude-opus-5-5` e `claude-sonnet-5-5`; margem de não inferioridade 0.10 na nota parcial oficial.

## Nota oficial (IC 95% por bootstrap sobre consultas)

| Cadeia | Nota parcial | Acerto estrito |
|---|---|---|
| A0 | 0.18 [0.10; 0.29] | 0.11 [0.02; 0.22] |
| pacote | 0.41 [0.29; 0.54] | 0.29 [0.16; 0.42] |
| Opus A1 | 0.60 [0.46; 0.72] | 0.51 [0.36; 0.67] |
| Opus A2 | 0.60 [0.48; 0.73] | 0.49 [0.36; 0.64] |
| Opus A3 | 0.58 [0.44; 0.71] | 0.49 [0.33; 0.64] |
| Sonnet A1 | 0.17 [0.08; 0.28] | 0.11 [0.02; 0.20] |
| Sonnet A2 | 0.24 [0.14; 0.36] | 0.18 [0.07; 0.29] |
| Sonnet A3 | 0.31 [0.20; 0.44] | 0.22 [0.11; 0.36] |

## Dentro do Sonnet (família da E8; Holm no McNemar)

| Comparação | Diferença | Veredito | McNemar (só 1º / só 2º) | p Holm | Papel |
|---|---|---|---|---|---|
| Sonnet A3 - Sonnet A1 | 0.14 [-0.01; 0.29] | empate | 9/4 | 0,801 | primary |
| Sonnet A2 - Sonnet A1 | 0.07 [-0.05; 0.19] | empate | 6/3 | 1,000 | secondary |
| Sonnet A3 - Sonnet A2 | 0.07 [-0.05; 0.19] | empate | 5/3 | 1,000 | secondary |

## Sonnet vs. Opus (não inferioridade, margem 0.10)

| Comparação | Diferença | Não inferioridade | Veredito usual | McNemar | p |
|---|---|---|---|---|---|
| Sonnet A1 - Opus A1 | -0.42 [-0.56; -0.28] | inferior | inferior | 1/19 | < 0,001 |
| Sonnet A1 - Opus A2 | -0.43 [-0.56; -0.31] | inferior | inferior | 0/17 | < 0,001 |
| Sonnet A1 - Opus A3 | -0.40 [-0.53; -0.28] | inferior | inferior | 0/17 | < 0,001 |
| Sonnet A2 - Opus A1 | -0.35 [-0.50; -0.19] | inferior | inferior | 2/17 | < 0,001 |
| Sonnet A2 - Opus A2 | -0.36 [-0.50; -0.22] | inferior | inferior | 1/15 | < 0,001 |
| Sonnet A2 - Opus A3 | -0.33 [-0.47; -0.20] | inferior | inferior | 1/15 | < 0,001 |
| Sonnet A3 - Opus A1 | -0.28 [-0.44; -0.12] | inferior | inferior | 2/15 | 0,002 |
| Sonnet A3 - Opus A2 | -0.29 [-0.43; -0.16] | inferior | inferior | 0/12 | < 0,001 |
| Sonnet A3 - Opus A3 | -0.26 [-0.40; -0.14] | inferior | inferior | 0/12 | < 0,001 |

Iguala o Opus (não inferior a todas as variantes dele):

- Sonnet A1: não (só contexto)
- Sonnet A2: não (afirmação registrada)
- Sonnet A3: não (afirmação registrada)

## Contra as referências sem LLM

| Comparação | Diferença | Veredito |
|---|---|---|
| Sonnet A1 - A0 | -0.01 [-0.13; 0.11] | empate |
| Sonnet A1 - pacote | -0.24 [-0.38; -0.10] | inferior |
| Sonnet A2 - A0 | 0.06 [-0.09; 0.21] | empate |
| Sonnet A2 - pacote | -0.17 [-0.30; -0.03] | inferior |
| Sonnet A3 - A0 | 0.13 [-0.05; 0.30] | empate |
| Sonnet A3 - pacote | -0.10 [-0.22; 0.01] | empate |

## Localização (consultas de uma falha que pedem componente: 22)

| Cadeia | Acerto do componente |
|---|---|
| pacote | 0.41 [0.23; 0.64] |
| Opus A1 | 0.68 [0.50; 0.86] |
| Opus A2 | 0.73 [0.55; 0.91] |
| Opus A3 | 0.73 [0.55; 0.91] |
| Sonnet A1 | 0.09 [0.00; 0.23] |
| Sonnet A2 | 0.41 [0.23; 0.59] |
| Sonnet A3 | 0.45 [0.27; 0.68] |

## Concordância com a resposta do pacote e instante no início da janela (descritivo)

| Cadeia | Instante = pacote | Componente = pacote | Motivo = pacote | Instante no início da janela |
|---|---|---|---|---|
| Opus A1 | 0.46 [0.29; 0.64] (n=28) | 0.27 [0.09; 0.45] (n=22) | 0.43 [0.25; 0.61] (n=28) | 0.07 [0.00; 0.18] (n=28) |
| Opus A2 | 0.39 [0.21; 0.57] (n=28) | 0.45 [0.23; 0.68] (n=22) | 0.32 [0.14; 0.50] (n=28) | 0.07 [0.00; 0.18] (n=28) |
| Opus A3 | 0.50 [0.32; 0.68] (n=28) | 0.50 [0.27; 0.73] (n=22) | 0.39 [0.21; 0.57] (n=28) | 0.11 [0.00; 0.21] (n=28) |
| Sonnet A1 | 0.11 [0.00; 0.21] (n=28) | 0.09 [0.00; 0.23] (n=22) | 0.32 [0.14; 0.50] (n=28) | 0.46 [0.29; 0.64] (n=28) |
| Sonnet A2 | 0.18 [0.04; 0.32] (n=28) | 0.59 [0.36; 0.77] (n=22) | 0.25 [0.11; 0.43] (n=28) | 0.50 [0.32; 0.68] (n=28) |
| Sonnet A3 | 0.54 [0.36; 0.71] (n=28) | 0.59 [0.36; 0.77] (n=22) | 0.11 [0.00; 0.21] (n=28) | 0.32 [0.14; 0.50] (n=28) |
| pacote | — | — | — | 0.04 [0.00; 0.11] (n=28) |

## Custo

| Cadeia | Total (US$) | Por consulta (US$, média ± dp) | Respostas válidas |
|---|---|---|---|
| Sonnet A1 | 1.97 | 0.044 ± 0.020 | 1.00 |
| Opus A1 | 8.69 | 0.193 ± 0.061 | 1.00 |
| Sonnet A2 | 1.56 | 0.035 ± 0.008 | 1.00 |
| Opus A2 | 7.55 | 0.168 ± 0.079 | 1.00 |
| Sonnet A3 | 2.12 | 0.047 ± 0.020 | 1.00 |
| Opus A3 | 7.30 | 0.162 ± 0.071 | 1.00 |

## Fidelidade automática (Sonnet; sem gabarito)

| Cadeia | IDs não emitidos | Componentes desconhecidos | KPIs desconhecidos | Números sem correspondência | Controle x1,37 casado |
|---|---|---|---|---|---|
| Sonnet A1 | 0/173 | 0/181 | 0/196 | 12/131 | 0.24 |
| Sonnet A2 | 0/172 | 0/191 | 0/178 | 1/187 | 0.12 |
| Sonnet A3 | 0/398 | 0/226 | 0/175 | 4/210 | 0.17 |
