# ADR 0010 — Baselines e protocolo de avaliação

- **Status:** aceito (E4 aprovada em 2026-11-11)
- **Data:** 2026-11-11
- **Etapa:** E4

## Contexto

O plano pede baselines medidos na mesma tarefa, divisão e métricas que a LSTM, o XAI e o agente usarão (R4), e pede que a avaliação completa da E8 seja implementada já na E4. A divisão e os rótulos são os da E3 (ADR 0007/0008). A execução confere o hash das janelas com `tasks_Telecom.json`.

## Decisões

### Ajuste por dobra

Seleção de atributos, z clássico (média e desvio) e escala robusta são ajustados, em cada dobra, só nos minutos normais dos dias de treino. A validação nunca entra no ajuste, e o teste nunca entra em nada.

### Detecção (Tarefa A)

| Método | Pontuação da janela | Limiar |
|---|---|---|
| `zscore_3sigma` | máximo de \|z\| entre KPIs e minutos da janela | fixo em 3 (regra 3σ) |
| `zscore_val` | a mesma | escolhido na validação |
| `isolation_forest` | −`score_samples`. Treinado nas janelas normais de treino sobre média, desvio, mínimo, máximo e último valor de cada KPI escalado (cerca de 7.900 colunas); ausente = 0, que é a mediana de treino | escolhido na validação |
| `lightgbm` | P(falha). Treinado nas janelas rotuladas de treino, com as mesmas estatísticas e `scale_pos_weight` = negativos/positivos; hiperparâmetros fixos e pequenos (300 árvores, 15 folhas, 20% das colunas) | escolhido na validação |

- **Limiar escolhido na validação:** o que maximiza F1 nas janelas rotuladas do dia de validação. Se esse dia não tem falha (dobra 0), usa-se o quantil 0,99 das pontuações das janelas normais da validação.
- **Hiperparâmetros do LightGBM:** não houve busca, porque a dobra 0 não tem positivos de validação e uma busca só nas demais dobras mudaria o protocolo entre dobras. A E5 fará a busca da LSTM com orçamento registrado; se for comparar com um LightGBM ajustado, este receberá o mesmo orçamento.
- **LightGBM em 1 thread:** com o `libomp` do PyTorch já carregado (o `Run` fixa a seed do torch), o LightGBM multithread aborta no macOS com `OMP: pthread_mutex_init failed` (reproduzido isoladamente). Com 1 thread funciona em qualquer ordem de importação e é determinístico.

### Localização (Tarefa B)

- Para cada falha testada, a janela de 20 min que termina 5 min após o início do gabarito.
- Componente pontuado pelo maior |z| entre seus KPIs e minutos.
- Ranqueamento entre os 56 candidatos; um componente sem atributo selecionado na dobra recebe 0.
- **Posto pessimista:** empatados com o gabarito contam à frente dele, para que nenhum acerto venha da ordem de desempate.

### Diagnóstico (Tarefa C), A0 constante

- Por dobra: componente e motivo mais frequentes entre as falhas de treino e validação (empate em ordem alfabética), e instante igual ao início da janela da consulta.
- **Avaliação pelo script oficial** (`main.evaluate`, submódulo), sem reimplementação:
  - o script exige uma predição por linha do arquivo de consultas, então um `query.csv` filtrado com as 45 consultas testadas é gravado no diretório do experimento (fora do git);
  - o script engole exceções por arquivo, então o wrapper confere o número de linhas do relatório.

### Métricas e dispersão (pacote `evaluation`, cada uma com teste calculado à mão)

- **Por janela:**
  - AUC-PR (prioritária; o valor ao acaso é a taxa de positivos, 0,14), AUC-ROC, precisão, revocação e F1, somadas entre dobras (*pooled*);
  - IC por bootstrap em blocos de 30 min (janelas vizinhas se sobrepõem).
- **Por evento:**
  - uma falha é detectada pelo primeiro alarme em [início − 5, início + 30] min, e o atraso é esse alarme menos o início (negativo = antes);
  - também se reportam alarmes falsos por hora em janelas normais;
  - e a **revocação por evento ao acaso**, isto é, a de alarmes aleatórios com a mesma taxa: 1 − (1 − p)^36.
- **Sem point-adjust.**
- **Localização:** HR@1/3/5 e MRR, com IC por bootstrap sobre falhas.
- **Diagnóstico:** acerto estrito e pontuação parcial, com IC por bootstrap sobre consultas.
- **McNemar exato:** pronto para as variantes do agente (E7/E8).

## Consequências

- Um comando por configuração, com execução registrada (`experiments/…-baselines`). Duas execuções com o mesmo código geraram `metrics.json` idênticos byte a byte.
- Controle de sensibilidade (ADR 0005): `configs/experiment/baselines_sem_isencao.yaml`.
- `scikit-learn` não traz tipos: o mypy ignora só a importação de `sklearn.*` (`pyproject.toml`); o modo estrito continua no restante.

## Adendo (2026-11-17, ADR 0019)

- Os z-scores (3σ, limiar da validação e local) passam a pontuar só os últimos D + 1 = 6 minutos da janela. O protocolo anterior continua como controle `baselines_janela_inteira`.
- A janela-oráculo da localização nunca cruza o dia: se cruzaria, usa-se a primeira janela completa do dia (3 falhas no Telecom, 2 no Bank).
- Novos itens nas tabelas: o detector de referência `prior_horario` (só o horário) e a AUC-PR média por dobra.
