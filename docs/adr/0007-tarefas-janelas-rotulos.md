# ADR 0007 — Formalização das tarefas, janelas e rótulos

- **Status:** aceito (E3 aprovada em 2026-11-11)
- **Data:** 2026-11-11
- **Etapa:** E3

## Contexto

O OpenRCA rotula só o **início** de cada falha (`record.csv`: instante, componente, motivo), sem duração. A matriz da E2 tem passo de 1 minuto, e cada dia do Telecom é um bloco isolado de 6 h (00:00–06:00 UTC+8). Entre inícios de falha no mesmo dia, o intervalo mínimo é de 30 min (mediana de 36 min).

## Tarefa A — detecção ou predição (LSTM)

- **Janela:** W minutos que terminam no minuto t, ou seja, as linhas [t − (W − 1), t] de um único dia. Janelas nunca atravessam dias.
- **Rótulo** de cada janela, em função da distância entre t e os inícios s:
  - **positivo (1):** algum início em [t − D, t + H]. H é o horizonte (0 = detecção; > 0 = predição) e D é a tolerância de detecção;
  - **negativo (0):** a janela é *normal*, isto é, não toca [s − max(m_antes, H), s + m_depois] para nenhum início s;
  - **ignorada (nulo):** o resto. Ela está perto demais de uma falha para ser normal, mas fora da faixa positiva. Não serve de treino normal nem de negativo na avaliação. Como as falhas não têm fim rotulado, chamar essas janelas de normais ou de positivas seria inventar um rótulo.
- **Valores:**
  - W = 20 min, menor que o intervalo mínimo de 30 min entre inícios, para que uma janela positiva corresponda a uma única falha. Com W = 30, o número de janelas normais de treino cairia de cerca de 2.000 para 1.700 (medido na preparação da E3). W = 20 também é o ponto de partida da E5, onde W pode ser variado como hiperparâmetro, apenas na validação;
  - H = 0, ou seja, detecção (ver abaixo);
  - D = 5 min;
  - m_antes = 5 min e m_depois = 30 min. O valor de 30 min é a duração da janela de consulta do OpenRCA, que é o único indício disponível da duração de uma falha.

## Sinal precursor: medido, não há

Estatística sem aprendizado (`tcc_rca.dataset.precursor`):
- A pontuação de cada minuto é a média das 10 maiores |z| robustas.
- O seletor e o escalonador foram ajustados só nos minutos normais do bloco inicial (2020_04_11 e 2020_04_20).
- Para cada antecedência L, compara-se a pontuação L minutos antes de cada uma das 45 falhas testadas (L < 0: depois do início) com a dos 2.093 minutos normais dos mesmos dias.
- A comparação usa AUROC com IC de 95% por bootstrap (2.000 reamostragens).
- Regra fixada antes da medição: existe precursor em L > 0 se o IC ficar inteiro acima de 0,5.

| L (min) | 15 | 10 | 5 | 2 | 1 | 0 | −1 | −2 | −5 |
|---|---|---|---|---|---|---|---|---|---|
| AUROC | 0,59 | 0,53 | 0,52 | 0,49 | 0,51 | 0,58 | **0,75** | **0,63** | 0,58 |
| IC 95% | 0,50–0,68 | 0,44–0,62 | 0,43–0,61 | 0,40–0,58 | 0,42–0,61 | 0,50–0,67 | 0,67–0,81 | 0,55–0,70 | 0,49–0,66 |

Fonte: `reports/tables/precursor_Telecom.csv`, que também traz o resultado por motivo.

**Decisão: o trabalho assume a tarefa de detecção precoce (H = 0)** e reporta o atraso de detecção, como o plano prevê.
- Nenhuma antecedência positiva tem IC inteiro acima de 0,5. A de 15 min encosta em 0,50 e, entre 5 antecedências testadas, não é evidência de precursor.
- O sinal aparece 1 a 2 minutos **depois** do início rotulado, e é coerente com amostras de 60 s que só registram a falha no minuto seguinte.
- Por motivo, a pontuação simples enxerga bem `db close` (0,94 em −1), `network delay` (0,86) e `CPU fault` (0,72). Não enxerga `db connection limit` nem `network loss`: os ICs incluem 0,5 em todas as antecedências. Esses dois motivos são o alvo de melhoria da LSTM em relação ao baseline.

## Tarefa B — localização (LSTM + XAI)

Para cada falha testada, produzir uma ordenação de todos os **componentes candidatos**: os 56 componentes que têm pelo menos um atributo na matriz. Os 15 componentes-gabarito do Telecom estão todos entre os candidatos (verificado por `tcc-rca tasks`). As métricas são HR@1, HR@3, HR@5 e MRR do componente-gabarito, calculadas nas 45 falhas testadas.

## Tarefa C — diagnóstico completo (agente)

Responder às consultas do `query.csv` (instante, componente, motivo, conforme o tipo de tarefa) no formato oficial, pontuadas pelo script oficial (`main.evaluate`). A avaliação usa as 45 consultas cujas janelas caem em dias de teste: `task_1` 9, `task_2` 7, `task_3` 4, `task_4` 7, `task_5` 4, `task_6` 6, `task_7` 8. As 6 consultas de 2020_04_11 ficam no bloco inicial (ADR 0008). Cada consulta do Telecom contém exatamente uma falha.

## Consequências

- Janelas e rótulos ficam em `data/processed/Telecom/windows.parquet` (não versionado). O SHA-256 de uma renderização CSV canônica fica em `reports/tables/tasks_Telecom.json` e independe da versão do codificador parquet.
- Os rótulos são gabarito: `tcc_rca.dataset`, `read_failures`, `build_cases`, `record.csv` e `windows.parquet` são proibidos no pacote do agente (teste em `tests/test_cases.py`).
