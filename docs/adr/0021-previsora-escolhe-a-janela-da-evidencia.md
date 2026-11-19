# ADR 0021 — A previsora LSTM também escolhe a janela dos pacotes de evidência

- **Status:** aceito e executado (2026-11-19). Nenhuma execução de teste da E7 antes desta decisão
- **Data:** 2026-11-19
- **Etapa:** complemento da E6, antes das execuções de teste da E7
- **Altera:** ADR 0016 (decisão 6: classificadora como detector dos pacotes) e a decisão final do ADR 0020 (opção 3)

## Contexto

Até aqui, cada pacote de evidência da E7 vinha de dois modelos:
- **classificadora LSTM:** escolhia a janela de 20 min dentro da consulta (maior pontuação) e dava a pontuação, o limiar e o alarme;
- **previsora LSTM:** fornecia as atribuições SHAP/LIME citáveis (`C#`, `K#`) e o ranqueamento nativo (`N#`) dessa janela.

A classificadora como detector foi fixada antes do teste, quando era o melhor detector da E5. Depois do ADR 0019, que fez a pontuação da previsora ler só os últimos 6 minutos, isso mudou:

| Sistema | AUC-PR previsora | AUC-PR classificadora | Classificadora vs. z-score local |
|---|---|---|---|
| Telecom | 0,52 ± 0,02 | 0,48 ± 0,02 | empate 5/5 |
| Bank | 0,35 ± 0,00 | 0,22 ± 0,01 | **inferior 5/5** |

A divisão também deixava o pacote incoerente. O prompt da A3 diz que o SHAP/LIME explica "dessa pontuação", mas a pontuação era da classificadora e as explicações eram da previsora.

## Decisão

1. **A previsora escolhe a janela.** A janela é a de maior pontuação da previsora (`ForecastScorer.score_windows`) entre as janelas de 20 min que terminam dentro da consulta. O alarme é pontuação ≥ limiar de validação da previsora na mesma dobra e seed (`runs.json` da E5, `explain.explained_seed`).
2. **O resto do pacote não muda:** explicações SHAP/LIME, ranqueamento nativo, pico de cada KPI, `top_components`/`top_kpis`, e nenhum instante estimado nem motivo sugerido.
3. **Implementação:** só configuração, `explain.evidence_detector: lstm_forecast` em `configs/base.yaml`. O caminho já existia em `evaluation/xai.py::write_packages`. Os pacotes são regerados com o comando `xai` (principal e controle sem isenção); as métricas da E6 não dependem dessa chave.
4. **A decisão vale qualquer que seja o resultado.** Ela é fixada **antes** de medir os novos pacotes. A nota oficial da cadeia sem LLM (`evidence-select`, bloco "pacote atual") será recalculada e relatada como a nova referência que o agente precisa superar. Se ela piorar, os pacotes **não** voltam para a classificadora: voltar escolhendo pelo teste seria mais um grau de liberdade (ameaça nº 24).
5. **Piloto:** os pacotes do piloto (`xai_piloto_insample`) foram usados só para conferir a mecânica da A3. Eles não são regerados, e nenhuma escolha do piloto dependeu do detector.

## Consequências

- O pacote passa a vir de um único modelo: a pontuação, o alarme e as explicações se referem à mesma função.
- A afirmação do ADR 0020 (opção 3), de que a entrada do agente era exatamente a fixada antes do teste, deixa de valer. A troca do detector usa resultados de teste da E5 posteriores ao ADR 0019 (ameaça nº 11). Atenuantes:
  - a regra (maior pontuação do melhor detector LSTM) é a mesma fixada no plano; mudou só qual LSTM é a melhor;
  - nenhuma execução de teste do agente existe;
  - a decisão não depende da nota dos novos pacotes (decisão 4).
- A referência sem LLM (0,50 Telecom, 0,28 Bank) é substituída pelo valor medido com os novos pacotes. Os valores antigos ficam registrados no ADR 0020 como os da configuração anterior.

## Resultado (2026-11-19)

- **Pacotes regerados:** execuções `xai` `20261119T165528` e `20261119T165646` (Telecom), `20261119T165803` e `20261119T170105` (Bank). As métricas da E6 ficaram idênticas às anteriores. Em todos os 165 pacotes, a janela é a de maior pontuação da previsora (seed 0) entre as candidatas da consulta, e a pontuação é igual à da E5.
- **Alarme:** 29 de 45 consultas no Telecom (antes 34) e 62 de 120 no Bank (antes 80).
- **Cadeia sem LLM** (nota oficial parcial):
  - Telecom: 0,50 → **0,41**. Diferença pareada −0,09 [−0,19; +0,01], empate. O instante cai de 0,76 para 0,56: o pico do K1 passa a adiantar seis minutos ou mais em 10 de 45 consultas (antes, 2).
  - Bank: 0,28 → **0,35**. Diferença pareada +0,08 [0,00; +0,15], empate no limite. O instante sobe de 0,28 para 0,40 e o componente de 0,20 para 0,28.
- **Conforme a decisão 4,** a troca fica, apesar da piora (sem significância) no Telecom. A nova referência que o agente precisa superar é 0,41 no Telecom e 0,35 no Bank (`docs/RESULTADOS.md`).

## Ordem

1. Configuração e documentação (este ADR, ameaça nº 11, `CLAUDE.md`).
2. O autor regera os pacotes (`xai` e `xai_sem_isencao`, nos dois sistemas).
3. O autor reexecuta o `evidence-select` (principal e sem isenção). Eu analiso e atualizo o `RESULTADOS.md`: alarme por consulta e nota parcial oficial da cadeia, antes e depois.
4. Execuções de teste da E7 com os novos pacotes.
