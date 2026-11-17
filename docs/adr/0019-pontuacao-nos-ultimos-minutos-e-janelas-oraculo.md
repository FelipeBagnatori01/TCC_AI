# ADR 0019 — Pontuação de janela nos últimos minutos, janelas-oráculo no mesmo dia e referências de detecção

- **Status:** aceito (implementado e reexecutado em 2026-11-17, com autorização do autor)
- **Data:** 2026-11-17
- **Etapa:** correção da E4/E5/E6 e do ADR 0018, antes das execuções de teste da E7

## Contexto

Uma revisão do projeto (2026-11-16), feita depois da E6, do ADR 0018 e do piloto da E7, mas **antes** de qualquer execução de teste do agente, encontrou dois problemas nos modelos anteriores ao LLM.

1. **Pontuação desalinhada com o rótulo.** Uma janela é positiva quando uma falha começou nos últimos D = 5 minutos (ADR 0007). A previsora LSTM e os dois z-scores pontuavam a janela pelo **máximo sobre os 20 minutos**:
   - os 14 minutos anteriores ao início só acrescentam ruído;
   - um pico isolado mantém a pontuação alta por 20 janelas seguidas.

   A classificadora LSTM, a TCN e os modelos tabulares aprendem sozinhos a olhar o fim da janela, então a comparação favorecia os supervisionados por construção.
2. **Janelas-oráculo cruzando dias.** A janela de localização termina D minutos após o início da falha. Para falhas nos primeiros 14 minutos do dia, ela pegava linhas do dia anterior na matriz, inclusive de `2020_04_23`, um dia excluído.
   - **Telecom:** 3 das 45 falhas;
   - **Bank:** 2 das 120.

   Isso afetava a localização da E4, da E5, da E6 e do ADR 0018. Os pacotes de evidência do agente não eram afetados, porque `candidate_end_rows` já checa o dia.

### Diagnóstico (modelos salvos da E5, seed 0, sem retreino)

AUC-PR com a pontuação no máximo dos últimos 6 minutos, contra a janela inteira:

| Sistema | Detector | Validação (média por dobra) | Teste (agrupado) |
|---|---|---|---|
| Telecom | LSTM previsora | 0,45 → 0,55 | 0,40 → 0,51 |
| Telecom | z-score local | 0,34 → 0,48 | 0,24 → 0,39 |
| Bank | LSTM previsora | 0,28 → 0,36 | 0,20 → 0,30 |
| Bank | z-score local | 0,20 → 0,33 | 0,19 → 0,33 |

O ganho aparece na validação e no teste.

### Cronologia (ameaça nº 21)

1. A ideia surgiu ao **ler os resultados de teste** da E5.
2. Foi confirmada na **validação**.
3. O valor 6 não foi buscado: vem da definição do rótulo (D + 1).

É uma decisão pós-teste e fica registrada como tal.

## Decisões

### 1. Pontuação nos últimos D + 1 minutos (`tasks.score_minutes: 6`)

- **Previsora LSTM:** máximo, sobre os últimos 6 passos de erro, da média dos k maiores erros normalizados (`ForecastScorer`). A função explicada pela E6 (`ForecastScoreModule`) recebe a mesma mudança, e o teste de igualdade entre as duas continua valendo.
- **z-score 3σ, z-score com limiar da validação e z-score local:** máximo sobre os últimos 6 minutos (`window_max_abs`, `last_steps`).
- **Classificadora LSTM, TCN, Isolation Forest, LightGBM e logística:** não mudam, porque já leem a janela inteira como entrada de um modelo.
- **Ranqueamento nativo da E5 e heurísticas de localização da E4:** não mudam. Mudá-los seria outra decisão pós-teste, sem diagnóstico prévio.
- **Valor:** fixo, nunca buscado. O validador exige `score_minutes ≤ W − 1`. `null` significa a janela inteira.
- **Consistência:** a E5 grava `score_minutes` no `metrics.json`. A E6 e o `pilot-evidence` recusam explicar uma execução da E5 com outro valor (execuções sem a chave contam como `null`).

### 2. Controle com a janela inteira

`configs/experiment/{baselines,lstm}_janela_inteira.yaml` (`score_minutes: null`) repetem a E4 e a E5 com o protocolo anterior. A E5 de controle tem busca própria, porque o critério de validação depende da pontuação.

Critério de aceite: os números de detecção do controle são iguais aos das tabelas anteriores a este ADR.

### 3. Janela-oráculo sempre dentro do dia (`oracle_end_row`)

- Uma única função substitui as quatro cópias da E4, E5, E6 e ADR 0018.
- Quando a janela usual cruzaria o dia, usa-se a **primeira janela completa do dia** (fim no minuto W − 1). Ela ainda contém o início da falha.
- Os denominadores continuam 45 e 120.
- A E4 relata quantas janelas foram deslocadas (`oracle_windows_shifted`: 3 no Telecom, 2 no Bank) e quantas falhas testadas não têm nenhuma janela positiva (`failures_without_positive_window`: as mesmas 3 e 2). Essas falhas entram na localização, mas não podem ser detectadas por janela.

### 4. Referências de detecção

- **`prior_horario` (E4):** detector sem telemetria. A pontuação de uma janela é a fração de janelas positivas entre as janelas rotuladas de treino cujo fim fica a até 30 min (circular) do mesmo minuto do dia (`baselines.time_prior_halfwidth_minutes`). Mede quanto um detector ganharia só com o horário. No Bank as falhas se concentram entre 14h e 20h (UTC+8): no diagnóstico, essa referência deu AUC-PR 0,16 contra 0,105 do acaso (ameaça nº 22).
- **AUC-PR média por dobra (`auc_pr_fold_mean`):** em todas as tabelas de detecção (E4, E5, ADR 0018), ao lado da AUC-PR agrupada e com IC pelos mesmos blocos de 30 min. A taxa de positivos varia de 0,05 a 0,37 entre dobras no Telecom. A agrupada mistura escalas de modelos diferentes (ameaça nº 7); a média por dobra não.

### 5. Eficiência

`Experiment.inputs` calcula a normalização local uma vez para todos os atributos e fatia as colunas por dobra. Ela é causal e não tem ajuste, então o resultado é idêntico: há teste unitário, e uma checagem nos dados reais deu igualdade byte a byte em todas as 10 dobras do Telecom e 8 do Bank. Isso evita recalcular o z local da matriz inteira a cada dobra na busca, no treino final, na E6 e no ADR 0018.

### 6. Fora do escopo

Também apareceram na revisão, mas não entram aqui. Cada item seria outra decisão pós-teste:
- reajuste final em treino + validação;
- média das seeds e fusão previsora + classificadora;
- modelo menor;
- correção para comparações múltiplas.

Ficam como limitações na monografia.

## Testes (`tests/test_score_span.py`)

- **Configuração:** a base usa D + 1, e um valor acima de W − 1 é rejeitado.
- **Pontuação:**
  - `last_steps` e `window_max_abs` calculados à mão;
  - um pico no início da janela deixa de contar, e um pico no último minuto conta igual nos dois modos;
  - `ForecastScoreModule` é igual ao `ForecastScorer` com `score_minutes` 3 e `null`, inclusive com alvo ausente.
- **Janela-oráculo:** inícios nos minutos 0, 5, 13 e 14 do dia dão a primeira janela completa do dia, sem linhas de outro dia; um dia curto demais gera erro.
- **`failures_without_positive_window`**, **`prior_horario`** (à mão e circular) e **AUC-PR média por dobra** (igual à média das AUC-PR de cada dobra, dentro do IC).
- **Cache do z local:** igual à normalização por dobra.

## Reexecuções (o autor inicia; só CPU, sem custo de API)

Ordem, por sistema:
1. `baselines`, `baselines_janela_inteira`, `baselines_sem_isencao`;
2. `lstm`, `lstm_janela_inteira`, `lstm_sem_isencao` (reaproveita a nova seleção), com `compare-e5` para as três;
3. `xai`, `xai_sem_isencao`;
4. `alt-models` tabular e TCN, com os controles `*_sem_isencao`;
5. `pilot-evidence`.

Os pacotes de evidência do agente mudam, porque as atribuições citadas vêm da previsora. Isso não custa nada: nenhuma execução A2/A3 de teste foi feita, e o piloto A1 não usa pacotes.

## Resultado (reexecução de 2026-11-17; detalhes em `docs/RESULTADOS.md`)

- **Execução:** E4, E5 (com nova busca), E6, ADR 0018 e os controles, nos dois sistemas, em cerca de 5 h 20 min em CPU, sem erros.
- **Controles:** os controles `*_janela_inteira` reproduzem **exatamente** os números de detecção das tabelas anteriores (E4 e E5, Telecom e Bank, todos os métodos). A única mudança na detecção é, portanto, a pontuação.
- **Janelas-oráculo:** 3 deslocadas no Telecom e 2 no Bank, e nenhuma janela cruza o dia.
- **Detecção (AUC-PR agrupada; antes → depois):**

  | Sistema | Previsora LSTM | z-score local | Classificadora LSTM |
  |---|---|---|---|
  | Telecom | 0,40 → 0,52 | 0,24 → 0,39 | 0,48 (não muda) |
  | Bank | 0,20 → 0,35 | 0,19 → 0,33 | 0,22 (não muda) |

- **Vereditos pareados contra o z-score local:**

  | Sistema | Previsora LSTM | Classificadora LSTM |
  |---|---|---|
  | Telecom | superior em 4/5 seeds | empate 5/5 (antes: superior 5/5) |
  | Bank | empate 5/5 | **inferior 5/5** |

- **Localização via XAI:** a previsora melhorou nos dois sistemas.
  - SHAP: HR@1 de 0,33 para 0,42 (Telecom) e de 0,36 para 0,48 (Bank).
  - No Bank, passa a ser superior ao ranqueamento nativo.
- **Referência só de horário:** AUC-PR 0,15 no Telecom (acaso 0,14) e 0,12 no Bank (acaso 0,105).
- **Modelos alternativos:** a pontuação deles não muda. Comparados com todos os candidatos:
  - nenhum é melhor que a previsora LSTM em nenhum sistema;
  - a logística L1 empata com ela no Telecom.

## Consequências

- Os números de detecção da E4 e da E5 e os de localização da E4, E5, E6 e ADR 0018 de `RESULTADOS.md` são substituídos. Os anteriores ficam no controle `janela_inteira` (detecção) e no histórico do git.
- **A leitura da RQ1 muda.** A previsora corrigida se aproxima do z-score local sem ultrapassá-lo de forma consistente: supera-o no Telecom (4/5 seeds) e empata no Bank. A classificadora deixa de ser o melhor detector. A tese deve afirmar que a LSTM previsora detecta tão bem quanto o melhor candidato em cada sistema e localiza melhor que todos, e não que a LSTM supera os baselines na detecção.
- As decisões da E7 (modelo do agente, pacotes vindos da LSTM) não mudam (ameaça nº 19).
