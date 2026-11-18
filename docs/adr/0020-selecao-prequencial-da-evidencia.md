# ADR 0020 — Escolha da evidência do agente por seleção prequencial (instante, componente, motivo)

- **Status:** aceito e executado (2026-11-18). Decisão final do autor: os pacotes do agente **não mudam** (opção 3; a decisão 4 não é aplicada)
- **Data:** 2026-11-18
- **Etapa:** complemento da E4–E6, antes das execuções de teste da E7

## Contexto

O autor quer que o agente receba a evidência mais precisa possível, venha ela de uma LSTM ou não. Até aqui, os modelos anteriores ao LLM foram comparados com métricas que não são as do agente.

1. **Detecção não é o gargalo.** A consulta do OpenRCA já diz que houve falha na janela. O script oficial cobra três campos:
   - **instante:** acerta se ficar a até 60 s do real;
   - **componente:** 43 candidatos no Telecom, 14 no Bank;
   - **motivo:** 5 candidatos no Telecom, 8 no Bank.
2. **Nenhum modelo estima o instante.** O campo entra em 4 dos 7 tipos de tarefa. O pacote atual traz só a janela de 20 min e o pico de cada KPI.
3. **A localização foi medida em janelas-oráculo,** que já conhecem o início rotulado. O pacote real usa a janela escolhida pela classificadora LSTM, e essa cadeia nunca foi medida contra o gabarito. Depois do ADR 0019, a classificadora perde para o z-score local no Bank.
4. **Motivo:** só existe a categoria sugerida (soma do SHAP por categoria). Ela supera o prior de treino só no Bank.
5. **Validação = teste.** Nas dobras rolling-origin, o dia de validação da dobra k é o dia de teste da dobra k − 1 (Telecom: 05_22 a 05_30; Bank: 03_06 a 03_24). Para um método sem treino, a nota na validação é **a mesma** nota de teste daquelas falhas. "Escolher na validação", portanto, não protege contra seleção no teste. Isso também vale, em grau menor, para as buscas da E5 e do ADR 0018: a configuração foi escolhida uma vez para todas as dobras, com falhas que são de teste em outras dobras (nova ameaça nº 23).

## Decisão proposta

### 1. Métrica comum: a resposta só da evidência

Para cada consulta de teste, uma cadeia sem LLM produz instante, componente e motivo. A cadeia usa só o que o agente vê:
- a janela e o texto da consulta;
- a telemetria do dia;
- modelos treinados nos dias anteriores;
- rótulos de dias anteriores, como qualquer treino.

Ela é avaliada de duas formas:
- **Acerto por campo**, em todas as falhas testadas: as três respostas são conhecidas para todas elas (Telecom 45, Bank 120).
  - instante a até 60 s;
  - componente: HR@1, HR@3 e MRR entre os candidatos do OpenRCA;
  - motivo: acerto.

  Consultas com 2 falhas usam a melhor associação, como o script oficial. Essa é a métrica de seleção, porque tem mais poder que a nota por tarefa.
- **Nota oficial do OpenRCA** por consulta, só com os campos pedidos. Ela é comparada com:
  - o A0 constante;
  - a cadeia equivalente ao pacote atual: pico do K1, C1 e motivo pela categoria sugerida.

  Ela também fica como referência que o agente precisa superar.

### 2. Lista fechada de candidatos (fixada antes de qualquer medição)

**Instante (T)**, dentro da janela da consulta:

| Id | Método |
|---|---|
| T0 | início da janela da consulta (o que o A0 faz) |
| T1 | primeiro minuto em que o erro por minuto da previsora LSTM passa do limiar da validação; se nunca passa, o máximo |
| T2 | igual ao T1, sobre o z-score local agregado (média dos k maiores \|z\| por minuto) |
| T3 | ponto de mudança (BOCPD) sobre o sinal do T2, inspirado no BARO (Pham et al., FSE 2024) |
| T4 | ponto de mudança (BOCPD) sobre o sinal do T1 |

**Componente (C)**, na janela que termina D minutos depois do instante estimado (`oracle_end_row` com o instante estimado no lugar do rotulado):

| Id | Método |
|---|---|
| C0 | frequência no treino (referência da E4) |
| C1 | previsora LSTM + SHAP, soma por componente (o C# atual) |
| C2 | ranqueamento nativo da previsora (o N# atual) |
| C3 | z-score global, máximo (E4; o melhor heurístico do Bank) |
| C4 | pontuação robusta do BARO: por KPI, (x − mediana) / IQR do trecho antes do instante (mesmo dia, até 60 min; sem trecho, os minutos normais de treino); máximo depois do instante; máximo por componente |
| C5 | N-Sigma: igual ao C4, com média e desvio-padrão |
| C6 | desvio sazonal: referência nos mesmos minutos do dia utilizável anterior (os dias do Telecom são quase duplicados) |
| C7 | PageRank personalizado no grafo de dependências (arestas da E2), com o vetor do C4 |
| C8 | fusão por posto recíproco de C1, C3 e C4 |

**Motivo (R)**, dado o componente escolhido:

| Id | Método |
|---|---|
| R0 | motivo mais frequente nas falhas passadas (o A0) |
| R1 | motivo mais frequente nas falhas passadas do mesmo componente; sem histórico, R0 |
| R2 | regra fixa: categoria do KPI mais anômalo do componente → motivos cujo `explain.reason_categories` a contém (fixado em 2026-11-12); desempate pelo R1 |
| R3 | regressão logística multinomial, treinada nas falhas passadas, sobre o desvio máximo por categoria de KPI do componente + o tipo do componente |

- **Consultas com 2 falhas:** o segundo instante é o melhor candidato a pelo menos W = 20 min do primeiro. Se não houver, repete-se o instante com o 2º componente.
- **Fora desta rodada:** autoencoder LSTM, USAD e TranAD (decisão do autor: ficam para uma próxima rodada).

### 3. Seleção prequencial, sem olhar o teste

Para a dobra k, o "passado" é o bloco inicial mais os dias de teste das dobras anteriores. Cada dia passado é pontuado pelos modelos da dobra em que ele foi teste, portanto fora da amostra. Só o bloco inicial usa os modelos da dobra 0, dentro da amostra; isso é declarado.

Em cada dobra, a escolha é feita em ordem fixa, com desempate pela ordem das tabelas (o mais simples primeiro):
1. T\* = maior acerto de instante no passado;
2. C\* = maior HR@1 (depois MRR), dado T\*;
3. R\* = maior acerto de motivo, dados T\* e C\*.

O resultado de teste relatado é o do **seletor**, aplicado dobra a dobra. Ele é honesto por construção: cada escolha só usa o passado. Também resolve o problema das comparações múltiplas, porque o que se mede é o procedimento de escolha, e não o melhor candidato visto no teste. As notas de teste de cada candidato aparecem só como tabela descritiva, marcada como tal.

Comparações pareadas por falha (`evaluation/compare.py`) do seletor contra:
- a cadeia do pacote atual;
- o A0;
- C0/R0.

### 4. Consequência para a E7, fixada antes de medir

Os pacotes passam a seguir as escolhas do seletor em cada dobra, **qualquer que seja o resultado de teste**:
- a janela explicada passa a terminar D minutos após o instante T\*, no lugar da janela escolhida pela classificadora. Isso resolve a questão da classificadora;
- o ranqueamento N# passa a vir do C\*;
- o instante estimado entra no pacote.

O A3 continua com o SHAP/LIME da previsora LSTM sobre essa janela, porque a RQ3 é sobre explicações de modelo (ameaça nº 19).

O motivo sugerido entra no pacote **só** nas dobras em que o método de motivo escolhido superou o R0 (mais frequente) no passado da dobra; nas demais, o pacote não traz motivo, como hoje.

## Decisões do autor (2026-11-18)

1. Autoencoder LSTM, USAD e TranAD ficam para uma próxima rodada.
2. O motivo sugerido entra no pacote só quando o método escolhido superou o R0 no passado da dobra (decisão 4).
3. O BARO é reimplementado (pontuação robusta e BOCPD), sem a dependência `fse-baro`.
4. A ameaça nº 23 (validação = teste nas buscas da E5 e do ADR 0018) é registrada já.

## Detalhes fixados antes de medir

- **Sinais por minuto:**
  - previsora: média dos k maiores erros normalizados do último passo da janela que termina no minuto;
  - z local: média dos k maiores \|z local\| do minuto (`minute_scores`).

  Os limiares do T1 e do T2 são os da validação da dobra, regra da E4/E5 (`choose_threshold`): o da previsora vem da E5 (seed explicada) e o do z local é recalculado com a mesma regra. Minuto acima do limiar ⇔ janela acima do limiar, porque a pontuação da janela é o máximo dos últimos minutos.
- **BOCPD** (Adams e MacKay, 2007): normal com média e variância desconhecidas (priori normal-gama: μ0 = 0, κ0 = 1, α0 = 1, β0 = 1) e risco constante 1/30 por minuto.
  - O sinal é padronizado pela mediana e pelo MAD do próprio trecho, que vai de até 60 min antes da consulta até o fim dela, no mesmo dia.
  - O instante é o início do novo regime (minuto − comprimento de corrida MAP) na queda de comprimento MAP com o início dentro da consulta; com várias, a maior queda.
  - Sem nenhuma, vale o máximo do sinal na consulta. O mesmo fallback vale para T1/T2 quando o sinal nunca passa do limiar.
- **Segunda falha:** o primeiro candidato do método a pelo menos 20 min do primeiro instante; senão, o máximo do sinal a pelo menos 20 min; senão, o mesmo instante. Cada causa recebe o componente mais bem ranqueado ainda não usado por uma causa anterior.
- **Janela do componente:** termina D minutos após o instante estimado. Fica limitada ao fim do dia e, se começaria no dia anterior, é trocada pela primeira janela completa do dia, como `oracle_end_row`.
  - C4–C6 leem os minutos do instante ao fim da janela;
  - C1–C3 leem a janela inteira, como na E4/E6.
- **C4/C5:** valores brutos, referência de até 60 min antes do instante no mesmo dia (mínimo de 10 min; senão, os minutos normais de treino da dobra). Escala zero vira 1, como o `RobustScaler` do scikit-learn usado pelo BARO.
- **C6:** escala = IQR do KPI nos minutos normais de treino (zero vira 1). Sem dia anterior, a pontuação é 0.
- **C7:** arestas `calls`, `hosted_on`, `member_of` e `runs_in` com peso 1 no sentido origem → destino (de quem sofre para quem causa) e 0,2 no sentido contrário; amortecimento 0,85; personalização = C4.
- **C8:** fusão por posto recíproco com k = 60.
- **Ranqueamento:** sobre os componentes candidatos do OpenRCA (os do prompt do agente), com posto pessimista nos empates.
- **R2:** "KPI mais anômalo" = maior pontuação do C4 entre os KPIs do componente.
- **R3:** atributos = log(1 + maior pontuação do C4) por categoria de KPI do componente + tipo do componente (nome sem dígitos); `LogisticRegression` do scikit-learn com C = 1. Treina nas consultas passadas com uma só falha. Com menos de duas classes no passado, vale o R0.
- **Passado da dobra 0:** os dias do bloco inicial, pontuados com os modelos da dobra 0 (dentro da amostra para a LSTM e para a escala do z global; os sinais não supervisionados não têm ajuste).
- **A0:** é o mesmo da E4 (`constant_predictions`), para conferir a nota contra a tabela da E4.
- **Conferência do avaliador:** a nota oficial por consulta deve ser igual à fração dos campos pedidos acertados pela melhor associação calculada aqui; se diferir, a execução para.

## Implementação

- **Código:**
  - `src/tcc_rca/models/rca/`:
    - `onset.py`: limiar por minuto e BOCPD;
    - `localize.py`: C4–C8;
    - `reason.py`: R0–R3.
  - `src/tcc_rca/evaluation/evidence_select.py`: candidatos sem rótulos (`_candidates`), rótulos anexados depois (`attach_gold`), seleção prequencial (`prequential_selections`), métricas, tabelas e nota oficial.
  - `evaluation/openrca.py`: `MultiPrediction`, que escreve várias causas como objetos numerados.
  - Comando `tcc-rca evidence-select`.
  - Configurações: seção `evidence_select` em `config.py`/`base.yaml` (só as constantes; as listas de métodos ficam fixas no código), `configs/experiment/evidence_select.yaml` e `evidence_select_sem_isencao.yaml`.
- **Reuso:**
  - `Experiment` (dobras, máscaras, modelos salvos da E5) e `_FoldContext` da E6 (SHAP com o mesmo fundo e a mesma seed);
  - `_fit_fold_transforms` da E4 (z global);
  - `constant_predictions` da E4 (A0);
  - `evaluation/openrca.py` (script oficial, sem reimplementar).
- **Testes** (`tests/test_evidence_select.py`, 17):
  - BOCPD: a posteriori soma 1 e acha um degrau sintético; fallback sem mudança;
  - limiar, segunda falha e janela estimada calculados à mão;
  - C4/C5/C6 à mão, inclusive escala zero e referência vazia; PageRank de dois nós à mão; fusão por posto;
  - R0–R2 seguem as regras fixas; R3 com uma classe ou sem dados cai na constante e separa classes separáveis;
  - melhor associação igual à regra do script oficial; `MultiPrediction` avaliada pelo script oficial com nota 1;
  - **vazamento:** trocar os rótulos do dia de teste e dos seguintes não muda a escolha da dobra (e trocar um dia passado muda, como controle);
  - regra da cadeia do pacote atual.
- **Conferências nos dados reais (fumaça em 2 dobras, 2026-11-18):**
  - Telecom em 7 s e Bank em 15 s;
  - a nota oficial bateu com a calculada aqui em todas as consultas, inclusive as do Bank com 2 falhas;
  - `metrics.json` byte a byte igual em duas execuções;
  - com o instante rotulado, C1, C2 e C3 reproduzem a ordem dos postos da E6 (previsora SHAP), da E5 (nativo, seed 0) e da E4 (z global) nas mesmas falhas. Os postos ficam iguais ou menores porque aqui o ranqueamento é sobre os 43 candidatos do OpenRCA, e não sobre os 56 componentes com telemetria.
- **Documentação:**
  - este ADR;
  - ameaças nº 23 e nº 24;
  - `CLAUDE.md`;
  - `RESULTADOS.md` e notebook depois das execuções completas.

## Resultado (execução de 2026-11-18; detalhes em `docs/RESULTADOS.md`)

- **Sanidade:**
  - o A0 reproduz a nota da E4 nos dois sistemas;
  - a nota oficial bateu com a calculada em todas as consultas;
  - com o instante rotulado, a previsora SHAP reproduz o HR@1 da E6.
- **Nota oficial parcial do seletor contra o pacote atual:**
  - **Bank:** 0,34 contra 0,28 (empate). No instante (+0,10) e no componente (+0,13), o seletor é superior.
  - **Telecom:** 0,35 contra 0,50, **inferior**. O instante perde 0,31: 0,44 contra 0,76.
- **Causa no Telecom:** o pacote estima o instante pelo pico do KPI mais atribuído (K1) dentro da janela escolhida pela classificadora. Esse estimador **existia antes deste ADR, mas ficou fora da lista de candidatos** (omissão do plano). Os métodos de "primeiro minuto acima do limiar" e o BOCPD são enganados por alarmes anteriores dentro da consulta: adiantam 6 min ou mais em 9 a 15 das 45 consultas, contra 2 do pico do K1. No Bank, o pico do K1 é pior que o z local (0,30 contra 0,46).
- **Outros achados:**
  - BARO, N-Sigma, desvio sazonal e PageRank não superam a previsora SHAP na localização;
  - a fusão por posto é a escolha estável do Bank;
  - a logística de motivo (R3) é a melhor nos dois sistemas no descritivo.
- **Pendente:** a decisão 4 manda os pacotes seguirem o seletor qualquer que seja o resultado. No Telecom, isso pioraria a evidência do agente. O autor decide entre:
  1. cumprir a regra;
  2. corrigir a omissão: incluir a cadeia do pacote atual como candidata e repetir a seleção prequencial (decisão pós-teste, registrada na ameaça nº 24);
  3. manter os pacotes atuais.

## Decisão final do autor (2026-11-18): opção 3, manter os pacotes atuais

Os pacotes de evidência da E7 continuam os da E6 (ADR 0016/0017, regerados com o ADR 0019), nos dois sistemas:
- a classificadora LSTM escolhe a janela e dá pontuação e alarme;
- a previsora LSTM fornece o SHAP/LIME (`C#`, `K#`) e o ranqueamento nativo (`N#`);
- não há instante estimado nem motivo sugerido.

A decisão 4 deste ADR (pacotes seguem o seletor) **não é aplicada**.

- **Por quê:** seguir o seletor pioraria a evidência do Telecom, o cenário principal (nota parcial −0,15, inferior), em troca de um empate a favor no Bank (+0,07).
- **Custo metodológico:** abandonar a regra foi decidido depois de ver o teste (ameaça nº 24). Mesmo assim, o que o agente recebe é exatamente o que estava fixado antes deste ADR: nenhuma escolha sobre a entrada do agente foi ajustada aos resultados de teste.
- **O que fica deste ADR:**
  - a comparação como achado (RQ2/RQ3): a cadeia do pacote atual, sem LLM, já tem nota oficial parcial de 0,50 no Telecom e 0,28 no Bank, e serve de referência que o agente precisa superar;
  - os métodos clássicos de RCA não superam a previsora LSTM;
  - a logística de motivo supera o prior;
  - nada disso entra nos pacotes.
- **Classificadora como detector da janela:** depois do ADR 0019, ela perde para o z-score local na detecção do Bank. Continua escolhendo a janela por ser a configuração fixada antes; isso fica registrado na ameaça nº 11.

## Ordem

1. Aprovação deste plano e das perguntas em aberto.
2. Implementação, testes e fumaça em uma dobra (eu rodo, minutos).
3. Execução completa nos dois sistemas e no controle sem isenção (o autor inicia).
4. Análise e relato.
5. Regeração dos pacotes pela regra da decisão 4.
6. Execuções de teste da E7.
