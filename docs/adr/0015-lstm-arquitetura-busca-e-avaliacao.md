# ADR 0015 — LSTM: arquitetura, normalização, busca de hiperparâmetros e avaliação (E5)

- **Status:** aceito (E5 aprovada em 2026-11-11)
- **Data:** 2026-11-11
- **Etapa:** E5 (Telecom e Bank)

## Contexto

O plano pede uma LSTM de previsão treinada só no período normal (formulação principal), uma LSTM classificadora supervisionada (secundária, se houver positivos), um modelo por sistema, arquitetura pequena, busca de hiperparâmetros só na validação com orçamento registrado, no mínimo 5 seeds, `score()` determinística e registro de custo. A E4 mostrou que a normalização decide o resultado no Bank: a deriva entre dias quebra escalonamentos fixos (ADR 0013/0014).

## Decisões

### 1. Formulação principal: LSTM previsora (`models/lstm`)
- **Arquitetura:** LSTM (1–2 camadas, 64–128 unidades, dropout 0,2) seguida de uma camada linear. Ela lê os passos 1..19 da janela de W = 20 e prevê o passo seguinte em cada posição.
- **Perda:** MSE só nos valores observados, ignorando os ausentes.
- **Dados:** treino só nas **janelas normais** dos dias de treino (máscara da E3). Parada antecipada pela perda nas janelas normais do dia de validação (paciência 5, até 30 épocas). Adam 1e-3, lote 128.
- **Pontuação:**
  - erro absoluto de previsão dividido pela mediana do erro daquele KPI nas janelas de treino;
  - pontuação do minuto = média dos 10 maiores erros;
  - pontuação da janela = máximo dos minutos.
  
  É a mesma agregação dos baselines. `ForecastScorer.score(window_tensor) -> float` é a função que a E6 vai explicar: modo de avaliação, sem gradiente, CPU, sem efeito colateral (testado).
- **Ranqueamento nativo:** na janela que termina 5 min após o início da falha, o maior erro de cada KPI ao longo dos passos, agregado por componente:
  - pela **soma**, como o plano especifica;
  - pela **média dos 3 maiores**, a variante da E4, menos sensível ao número de KPIs do componente.

### 2. Formulação secundária: LSTM classificadora
- Mesma família de rede; a última saída passa por uma camada linear e dá o logit de "falha iniciada nos últimos D minutos".
- Treino supervisionado nas janelas rotuladas de treino, com `pos_weight` = negativos/positivos. Parada antecipada pela perda nas janelas rotuladas da validação.
- Implementada nos **dois** sistemas por decisão do autor (2026-11-11). No Telecom há de 30 a 234 janelas positivas de treino por dobra; a amostra pequena é uma ressalva.

### 3. Normalização como hiperparâmetro
- **global:** mediana e IQR das linhas normais de treino da dobra, |z| limitado a 10.
- **local:** z causal contra os 60 minutos anteriores do mesmo dia (`dataset.precursor.local_z`), sem ajuste e limitado a 50.

Os ausentes entram como 0, o centro da normalização.

### 4. Busca (orçamento registrado)
- **Previsora:** 8 configurações {64, 128} × {1, 2} camadas × {global, local}.
- **Classificadora:** {64, 128} × 1 camada, testada nas duas normalizações; o tamanho é escolhido dentro da normalização vencedora da previsora.
- **Seed e critério:** 1 seed; critério = **AUC-PR média na validação** das dobras cuja validação tem falhas (9 no Telecom, 7 no Bank). Empate: o modelo menor.
- **Custo da busca:** Telecom, 108 treinos em 343 s; Bank, 84 treinos em 881 s.
- **Escolhas:**

| Sistema | Previsora | Classificadora |
|---|---|---|
| Telecom | local, 128, 1 camada (validação 0,447; global/64/1 teve 0,422) | local, 128, 1 camada (0,705) |
| Bank | local, 128, 2 camadas (0,280; a melhor global teve 0,192) | local, 128, 1 camada (0,280) |

- **Finais:** 5 seeds × todas as dobras. Limiar escolhido na validação pela regra da E4.
- **Controle sem a isenção de disponibilidade:** reaproveita a seleção (`lstm.reuse_selection_from`), sem nova busca.

### 5. Comparação pareada com os baselines (`evaluation/compare.py`, `tcc-rca compare-e5`)
- A diferença de AUC-PR (e de HR@1, HR@3, MRR) entre LSTM e rival é calculada **nas mesmas janelas e falhas**, com bootstrap conjunto (blocos de 30 min para janelas, falhas para postos), por seed.
- **Veredito:** "superior" ou "inferior" quando o IC de 95% da diferença exclui 0; "empate" caso contrário.
- É assim que o critério "supera — ou não" é decidido, e não pela sobreposição de ICs separados.

### 6. Execução e ambiente
- CPU (Apple M4 Pro), algoritmos determinísticos. O MPS foi mais lento no modelo pequeno e não garante determinismo.
- **Conflito de OpenMP:** depois de um `fit` do LightGBM no mesmo processo, o PyTorch multithread aborta no macOS. O LightGBM passou a ser importado só dentro da função que o usa; os comandos `baselines` e `lstm` rodam em processos separados; os testes da LSTM usam 1 thread.

## Resultados (média ± dp entre 5 seeds; pareado = diferença com IC por bootstrap)

| | Telecom | Telecom sem isenção | Bank |
|---|---|---|---|
| Previsora, AUC-PR | 0,40 ± 0,02 | 0,31 ± 0,02 | 0,20 ± 0,00 |
| Classificadora, AUC-PR | **0,48 ± 0,02** | **0,49 ± 0,02** | 0,22 ± 0,01 |
| Melhor baseline E4, AUC-PR | 0,25 (LightGBM) | 0,26 (LightGBM) | 0,19 (z-score local) |
| Classificadora vs. melhores baselines | superior em 5/5 seeds | superior em 5/5 | empate em 5/5 |
| Previsora vs. melhores baselines | superior ao z local (5/5); empate com LightGBM (4/5) | empate | empate |
| Localização, soma, HR@1 | **0,44 ± 0,03** | 0,22 ± 0,02 | 0,33 ± 0,01 |
| Melhor heurística E4, HR@1 | 0,13 (z global) | 0,13 | 0,32 (z global) |
| Localização vs. melhor heurística | superior em 5/5 | empate | empate com o z global; superior ao local e à referência |

- **RQ1 (detecção):**
  - no Telecom, a LSTM **classificadora** supera os baselines com folga, com ou sem a isenção, e com 9 alarmes falsos por hora contra 17,5 do LightGBM;
  - no Bank, as LSTMs empatam com o z-score local (e com o LightGBM). A deriva e a heterogeneidade do Bank limitam o ganho.
- **RQ2 (localização):**
  - a LSTM supera todas as heurísticas no Telecom, **mas o ganho depende da isenção de disponibilidade**: sem ela, HR@1 cai de 0,44 para 0,22 e vira empate. É a ameaça nº 1 se materializando: os KPIs de disponibilidade explicam cerca de metade do HR@1 do Telecom;
  - no Bank, a LSTM empata com o z-score global em HR@1 e é melhor em profundidade (HR@5 0,79 contra 0,68).
- **Variância entre seeds:** pequena (dp ≤ 0,04 em todas as métricas principais).
- **Custo:**
  - previsora com cerca de 1 milhão de parâmetros, treino de 2,4 s (Telecom) e 8,4 s (Bank) por dobra e seed, inferência de 0,26 ms por janela;
  - classificadora com 0,7–0,8 milhão de parâmetros e 1,2–3,3 s por treino;
  - pico de memória de 3,6 GB (Telecom) e 6,4 GB (Bank).

## Consequências

- A E6 explica `ForecastScorer.score`. Os modelos ficam salvos por dobra e seed em `experiments/<run>/artifacts/models/`: pesos, escala dos erros, atributos e normalização.
- Como a classificadora é o melhor detector, a E6 pode explicar também `classifier_scores`, por exemplo com SHAP. A decisão fica para o plano da E6.
- A comparação pareada já está pronta para a E8.

## Adendo (2026-11-17, ADR 0019)

- A pontuação da previsora passa a ler só os últimos D + 1 = 6 passos de erro (`tasks.score_minutes`, gravado no `metrics.json`). A busca é refeita, porque o critério de validação depende dela. O protocolo anterior continua como controle `lstm_janela_inteira`.
- A classificadora e o ranqueamento nativo não mudam.
- As janelas-oráculo do ranqueamento nativo não cruzam o dia (`Experiment.oracle_row`).
- A normalização local é calculada uma vez por execução (resultado idêntico, testado).
