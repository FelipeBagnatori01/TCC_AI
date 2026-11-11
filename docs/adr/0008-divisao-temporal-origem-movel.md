# ADR 0008 — Divisão temporal em origem móvel, dias excluídos e máscara de período normal

- **Status:** aceito (E3 aprovada em 2026-11-11)
- **Data:** 2026-11-11
- **Etapa:** E3

## Contexto

O Telecom tem 15 dias. Onze deles têm falhas: 2020_04_11 (6) e 2020_05_22 a 2020_05_31 (45). Os quatro dias de 2020_04_20 a 2020_04_23 não têm nenhuma falha rotulada. A regra R2 exige divisão cronológica por dias, com intervalo de guarda, sem K-Fold aleatório; o plano pede origem móvel para que cada falha seja testada uma vez.

## Decisão 1 — excluir três dias que são cópias sem rótulo (decisão do autor, 2026-11-11)

Na comparação célula a célula das métricas da matriz da E2:

| Dia excluído | Gêmeo | Valores iguais |
|---|---|---|
| 2020_04_21 | 2020_05_31 (4 falhas) | 99,997% |
| 2020_04_22 | 2020_05_23 (6 falhas) | 99,995% |
| 2020_04_23 | 2020_05_25 (3 falhas) | 99,999% |

- Entre os outros pares de dias, a coincidência fica entre 42% e 48%, e vem quase toda de séries constantes.
- As poucas diferenças dentro de cada par caem logo após os inícios das falhas do gêmeo: 00:40, 01:18 e 04:52 contra falhas às 00:40, 01:16 e 04:47 em 05_23.
- Ou seja, os dias de abril contêm as mesmas falhas, sem rótulo.
- Mantê-los como normais teria dois efeitos:
  - traria a telemetria de teste para o treino, o que é vazamento;
  - ensinaria à LSTM que o comportamento de falha é normal, o que é ruído de rótulo.
- Os três dias saem de todos os papéis (`tasks.excluded_days`).
- O dia 2020_04_20 é independente (no máximo 48% de coincidência com qualquer outro) e permanece.

## Decisão 2 — bloco inicial e falhas não testadas (decisão do autor, 2026-11-11)

- 2020_04_11 é o dia mais antigo; nenhuma dobra cronológica tem treino anterior a ele.
- Ele e 2020_04_20 formam o **bloco inicial**: são só treino e validação.
- As 6 falhas de 04_11 servem para escolher limiares e hiperparâmetros e **não são testadas**.
- A avaliação das tarefas A, B e C usa as **45 falhas de maio**, cada uma testada exatamente uma vez.
- Esse desvio do critério "todas as falhas testadas" foi escolhido para preservar a R2.

## Decisão 3 — origem móvel por dia

Cada dia a partir de 2020_05_22 é o teste de uma dobra, num total de 10 dobras. Os dias anteriores utilizáveis formam o treino, exceto o último, que é a validação (`val_days: 1`). A janela de treino é expansiva.

- A dobra 0 é fraca: treina só em 04_11, com 105 janelas normais, e valida em 04_20, que não tem falhas.
  - Nessa dobra, o limiar não pode usar positivos de validação.
  - A E5 deve usar uma regra baseada só em janelas normais (por exemplo, um quantil) e reportar a dobra 0 separadamente na análise de sensibilidade.
- **Guarda:** como as janelas não atravessam dias e os dias distam pelo menos 18 h, a distância mínima entre janelas de treino ou validação e janelas de teste é muito maior que W + H = 20 min. O build falha se alguma dobra violar a guarda.

## Decisão 4 — máscara de período normal

Uma janela entra no treino da LSTM só se for normal (ADR 0007): não toca [s − max(5, H), s + 30] min de nenhum início s. Ficam de 105 (dobra 0) a 1.681 (dobra 9) janelas normais de treino.

## Caracterização

Em `reports/tables/split_characterization_Telecom.md` e `tasks_Telecom.json`, por dobra:
- janelas de treino, validação e teste;
- janelas positivas, negativas e ignoradas;
- falhas e consultas testadas, de 2 a 8 por dobra.

As falhas testadas são 15 `CPU fault`, 13 `network delay`, 7 `network loss`, 6 `db connection limit` e 4 `db close`, distribuídas por 15 componentes.

## Consequências

- Comparações estatísticas usam as 45 falhas agregadas entre dobras; nenhuma dobra isolada tem falhas suficientes.
- Os números deste trabalho não são comparáveis aos do artigo do OpenRCA, que usa as 51 consultas.
- Ameaças registradas em `docs/AMEACAS_VALIDADE.md`.
