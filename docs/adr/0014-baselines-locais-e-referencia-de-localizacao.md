# ADR 0014 — z-score local e referência de frequência nos baselines (E4 do Bank)

- **Status:** aceito (E4 do Bank aprovada em 2026-11-11)
- **Data:** 2026-11-11
- **Etapa:** Plano B, E4 do Bank (vale para os dois sistemas)

## Contexto

A E3 do Bank mostrou que o z-score com escalonamento fixo do treino satura por causa da deriva entre dias (ADR 0013). Os baselines de z-score da E4 (ADR 0010) ficariam no acaso no Bank por defeito da régua, não por falta de sinal. Um piso fraco demais tornaria fácil e enganoso qualquer ganho da LSTM.

## Decisões

1. **`zscore_local`** (decisão do autor, 2026-11-11), nos dois sistemas:
   - **Detecção:** |z| causal de cada KPI contra a mediana e o desvio dos 60 minutos anteriores do mesmo dia (`dataset.precursor.local_abs_z`, o mesmo da E3). A pontuação do minuto é a média das 10 maiores |z|; a da janela é o máximo dos minutos. O limiar vem da validação, como nos demais.
   - **Localização:** cada componente é pontuado pela média dos seus 3 KPIs com maior |z| local no máximo da janela. Como o |z| local é limitado a 50, um máximo simples empataria muitos componentes no teto.
   - Não há ajuste: a pontuação usa só o passado de cada minuto; a seleção de atributos é a da dobra.
2. **Referência de localização "frequência no treino"** (`prior`): ranqueia os candidatos pelo número de vezes que cada um foi causa raiz nas falhas de treino e validação da dobra, sem ler telemetria. É o análogo, na localização, do baseline constante A0. Ela mostrou que uma heurística pode parecer razoável e não passar de chutar os componentes mais frequentes (z-score local no Bank, HR@1).
3. **Os métodos anteriores não mudaram:** nos dois sistemas, os `metrics.json` novos contêm todos os valores antigos sem alteração; só há chaves novas (`detection.zscore_local`, `localization_local`, `localization_prior`).
4. **Ameaça à validade:** os dois acréscimos vieram depois de ver os dados do Bank (mesma natureza da ameaça nº 8). O efeito é conservador: tornam o piso mais difícil de superar, nunca mais fácil.

## Resultados

Ver `docs/RESULTADOS.md`:
- **Detecção no Bank:** z-score global no acaso (AUROC 0,48); z-score local é o melhor baseline (AUC-PR 0,19 contra 0,105 do acaso; 13 alarmes falsos por hora).
- **Localização no Bank:** o z-score global é o melhor (HR@1 0,32, referência 0,11), e o local não passa da referência em HR@1.
- **Telecom:** o z-score local acerta `db close` (HR@1 0,75 contra 0 do global).
- **Consequência para a E5 e a E6:** a normalização que serve à detecção (local) e a que serve à localização (global) podem ser diferentes, e o desenho da LSTM e do XAI deve testar as duas.

## Execução

- Por sistema e configuração: `uv run tcc-rca baselines --config configs/base.yaml [configs/system/bank.yaml] configs/experiment/baselines[_sem_isencao].yaml`.
- Tempo: cerca de 70 s no Telecom e 130 s no Bank (pico de 5,9 GB).
- Os nomes das execuções não levam o sistema; o `run_dir` de cada tabela fica registrado no JSON dela, e o notebook passou a usá-lo.
