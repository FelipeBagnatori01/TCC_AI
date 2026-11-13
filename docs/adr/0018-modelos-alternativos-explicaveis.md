# ADR 0018 — Modelos alternativos explicáveis por SHAP/LIME (complemento da E5/E6)

- **Status:** aceito (executado em 2026-11-13 com autorização do autor; aprovado em 2026-11-13)
- **Data:** 2026-11-13
- **Etapa:** complemento da E5/E6, antes do piloto da E7

## Contexto

A E5 comparou a LSTM com baselines estatísticos e tabulares só na detecção, e o LightGBM da E4 não teve busca de hiperparâmetros. A E6 explicou só a LSTM. Antes do piloto do agente, o autor quis saber se o resultado vem da LSTM ou se outros modelos explicáveis fariam o mesmo, com o mesmo conjunto de dados, as mesmas dobras e as mesmas métricas (RQ1 e RQ2). O plano já previa o LightGBM como "referência de XAI tabular (TreeSHAP)" (E4, item 2).

O autor escolheu quatro modelos em 2026-11-13.

## Decisões

### 1. Modelos e explicadores

| Modelo | Entrada | Explicação SHAP | Escala explicada pelo SHAP |
|---|---|---|---|
| LightGBM | estatísticas da janela | TreeSHAP intervencional (exato) | log-odds |
| Isolation Forest | estatísticas da janela, só janelas normais de treino | TreeSHAP intervencional (exato), sinal invertido | comprimento esperado do caminho |
| Regressão logística L1 | estatísticas da janela | SHAP linear intervencional (exato) | log-odds |
| TCN classificadora | janela 20 × F (a mesma tensora da LSTM) | GradientExplainer (como a E6) | probabilidade |

- **LIME:** todos usam `lime_groups` da E6 sem mudança. "Desligar" uma série `componente::kpi` troca a série inteira pela referência normal da dobra; nos modelos tabulares, as estatísticas são recalculadas a partir da janela perturbada. A definição é a mesma para todos os modelos.
- **Fidelidade:** deleção e inserção, iguais à E6, usam a pontuação de detecção (probabilidade, ou `−score_samples` no Isolation Forest).
- **Estatísticas da janela:** média, desvio, mínimo, máximo e último valor, as mesmas da E4 (`tensor_statistics`, que tem teste de igualdade com `window_statistics`). As 5 atribuições de uma série são somadas, o que dá uma importância por série. Assim os rankings por componente são calculados como na E6 (`component_scores`).
- **Isolation Forest:** o shap explica o comprimento esperado do caminho, que cresce quanto mais normal a janela. O sinal é invertido para que positivo signifique "mais anômalo", como nas demais explicações (testado).
- **TCN:** dois a três blocos residuais de convoluções causais dilatadas (1, 2, 4), *max-pool* no tempo e cabeça linear. É treinada com a receita da `ClassifierLSTM`: `pos_weight`, Adam, parada antecipada na perda de validação, e taxa de aprendizado, lote, épocas, paciência e *dropout* de `lstm`. O treino foi generalizado em `fit_window_classifier`; `train_classifier` continua idêntico para a LSTM.
- **Só classificadora:** a TCN entra como par da `ClassifierLSTM`, o melhor detector da E5. Uma TCN previsora dobraria o custo e ficou de fora.

### 2. Protocolo (igual à E5/E6)

- **Dados:** mesmas janelas (W = 20, H = 0, D = 5), dobras rolling-origin, purga e máscaras (`evaluation/lstm.py::Experiment`, `_masks`).
- **Entrada:** todos os modelos leem a mesma matriz normalizada da LSTM, com ausente = 0. Isso difere do LightGBM da E4, que usava escala robusta e `NaN`.
- **Normalização** (`global`/`local`): é um hiperparâmetro buscado, como na LSTM.
- **Busca:** só na validação, com 1 seed e o critério da E5 (AUC-PR média de validação nas dobras cuja validação tem falha). São 8 configurações por modelo, o orçamento da previsora LSTM; em empate vence a menor:

  | Modelo | Grade (× `global`/`local`) |
  |---|---|
  | LightGBM | `num_leaves` {15, 31} × `n_estimators` {300, 600}; demais parâmetros da E4 |
  | Isolation Forest | `n_estimators` {100, 300} × `max_samples` {256, 1024} |
  | Logística L1 | C {0,01; 0,1; 1; 10}, classes balanceadas, liblinear |
  | TCN | canais {32, 64} × blocos {2, 3}, núcleo 3 |

- **Treino final:** seeds `lstm.seeds` (0–4) em todas as dobras. A logística é determinística (liblinear) e roda só com a seed explicada.
- **Limiar:** escolhido no dia de validação (`choose_threshold`).
- **Métricas de detecção:** as da E4 (`detection_metrics`).
- **XAI:**
  - janelas-oráculo da E6 (as mesmas falhas testadas), seed explicada 0;
  - fundo de 100 janelas normais de treino, sorteadas com a mesma semente da E6;
  - referência = mediana das janelas normais de treino.
- **Qualidade das explicações:**
  - fidelidade contra 5 ordens aleatórias;
  - estabilidade entre a seed 0 e as seeds 1–4, e entre janelas vizinhas (±1 min);
  - concordância SHAP × LIME (Jaccard e Kendall nos 10 maiores; mesmo componente no topo).
  - Não há IG: é uma verificação cruzada da E6 e não se aplica a árvores.

### 3. Comparações pareadas (`evaluation/compare.py`)

- **Detecção:** AUC-PR de cada modelo contra a previsora e a classificadora LSTM da E5, nas mesmas janelas de teste e com a mesma seed, por bootstrap em blocos de 30 min.
- **Localização:** HR@1 e MRR (agregação soma) contra as mesmas falhas:
  - o SHAP/LIME da LSTM na E6 (mesmo explicador);
  - o ranqueamento nativo da E5;
  - os z-scores global e local da E4.
- **Execuções de referência:** são localizadas por `reports/tables/{lstm,xai,baselines}_<sistema>_<nome>.json`, cujos nomes vêm de `alt_models.{lstm,xai,baselines}_name`.

### 4. Execução

- **Dois processos:** `family: tabular` (LightGBM, Isolation Forest, logística) e `family: tcn`. O OpenMP do LightGBM não pode conviver com torch multithread no mesmo processo (ADR 0010/0015). O LightGBM é importado de forma preguiçosa e roda em 1 thread.
- **Comando:** `tcc-rca alt-models`, com as configurações `alt_tabular`, `alt_tcn` e os controles `*_sem_isencao`. Os controles reaproveitam a seleção da execução principal e comparam com as execuções `*_sem_isencao` da E4/E5/E6.
- **Saídas:**
  - `reports/tables/alt_models_<sistema>_<nome>.{json,md}`;
  - na execução: `search.json`, `runs.json`, `paired.json`, `test_window_scores.parquet`, `xai_ranks.csv` e `xai_quality.csv`.
  - `metrics.json` não leva tempos, para continuar reprodutível.
- **Fumaça:** `alt_models.max_folds` limita as dobras. Serve só para medir o tempo; o valor `null` é o oficial.

### 5. O agente não muda

Os pacotes de evidência da E7 continuam vindo da LSTM (ADR 0016/0017). Escolher o modelo das evidências pelo resultado no teste seria seleção no teste (ameaça nº 19). Um alternativo que vença é relatado como achado.

## Testes (`tests/test_alt_models.py`)

- `tensor_statistics` é igual a `window_statistics`, inclusive com ausentes; a soma por série é calculada à mão.
- **Aditividade:**
  - TreeSHAP do LightGBM: a soma mais o valor esperado dá o log-odds;
  - SHAP linear: a soma mais o valor esperado dá `decision_function`;
  - nos dois, a série deslocada recebe a maior atribuição.
- **Isolation Forest:** a série anômala recebe a maior atribuição, com sinal positivo.
- **LIME** num modelo tabular: aponta a série da qual o modelo depende.
- **Grade:** 8 configurações por modelo, sem repetição, ordem da menor para a maior, ida e volta pelo JSON.
- **Vazamento:** o ajuste lê só as janelas de treino (normais, no Isolation Forest). A TCN lê também a validação, mas só para a parada antecipada, como a LSTM.
- **TCN:** reprodutível com a mesma seed, aprende o padrão, a SHAP aponta a série certa e a convolução é causal.

## Resultado (2026-11-13; detalhes em `docs/RESULTADOS.md` e `notebooks/analise_modelos_alternativos.ipynb`)

- **Detecção:** a vantagem da LSTM não é exclusiva.
  - No Telecom, a regressão logística L1 empata com as duas LSTMs (AUC-PR 0,51 contra 0,48).
  - No Bank, o LightGBM e a TCN empatam com elas.
  - O Isolation Forest perde nos dois sistemas.
- **Localização via XAI:** os modelos tabulares (HR@1 ≤ 0,15) são inferiores à XAI da LSTM previsora e ao ranqueamento nativo em todas as comparações. A TCN fica perto da LSTM.
- **Qualidade:** as explicações dos modelos tabulares são mais fiéis, mas não localizam. As do Isolation Forest não são fiéis nem estáveis. A previsora LSTM tem as explicações mais estáveis entre seeds.
- **Reprodutibilidade:** duas execuções iguais geram `metrics.json` idênticos byte a byte, nas duas famílias.

## Consequências

- A RQ1 e a RQ2 passam a ter comparação com modelos explicáveis de outras famílias, com as mesmas métricas e testes pareados.
- **Custo:** as execuções completas são longas (busca de 8 configurações × dobras por modelo, mais 5 seeds e XAI) e são iniciadas pelo autor. O tempo por modelo fica na tabela de custo.
