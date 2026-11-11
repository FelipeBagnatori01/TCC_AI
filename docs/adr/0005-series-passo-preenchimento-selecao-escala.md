# ADR 0005 — Séries multivariadas: passo, agregação, preenchimento, seleção e escala

- **Status:** aceito (isenção de KPIs de disponibilidade aceita pelo autor em 2026-11-11, com controle de sensibilidade)
- **Data:** 2026-11-10
- **Etapa:** E2

## Contexto

A E1 mostrou passo modal de 60 s em todas as séries de métricas, dias de 6 h (00:00–06:00 UTC+8), 141 séries com mais de um `itemid` (discos, interfaces, tablespaces) e cerca de 20% das séries constantes.

## Decisões

1. **Passo de 1 minuto.** Cada amostra vai para o minuto que a contém (`truncate`). A grade de cada dia vai do primeiro ao último minuto com telemetria naquele dia, sem presumir 6 h. No Telecom são 360 minutos por dia, mais 1 minuto em 2020_04_11 por causa de uma amostra do `metric_app` exatamente às 06:00, num total de 5.401 linhas.
2. **Várias amostras no mesmo minuto:** fica a última de cada item (`itemid`).
3. **Vários itens do mesmo KPI no mesmo componente:** combinados pela regra `item_agg` do catálogo de KPIs. É `max` para `Disk_io_util` e `New_Tbs_Used_Pct` (pior disco ou tablespace), `min` para `New_Tbs_Free_Gb`, `sum` para tráfego, pacotes e erros por interface, e `mean` no resto. Neste último caso só há troca de `itemid` ao longo do tempo, sem itens simultâneos.
4. **`metric_app`:** linhas do mesmo minuto são somadas (`num`, `succee_num`). `avg_time` é a média ponderada por `num`; `succee_rate` é recalculado.
5. **Traces:** por minuto, há agregados para o componente que executa o span (`cmdb_id`) e para o banco chamado (`dsName`): chamadas, latência média, p95, p99 e taxa de erro. Um minuto sem spans num dia com traces vale **0 chamadas** (é uma observação); latência e taxa de erro ficam ausentes.
6. **Preenchimento:** só para frente, até **5 passos**, dentro do mesmo dia. Nunca para trás e nunca por interpolação, porque ambos usam valores futuros. O que sobra continua ausente. Uma máscara `observed` registra onde houve amostra real.
7. **Seleção de atributos** (`FeatureSelector`), ajustada **só nas linhas de treino**, nesta ordem:
   - descarta se estiver ausente em mais de 50% das linhas;
   - descarta se for constante;
   - descarta se a moda cobrir 99,9% ou mais das linhas.
8. **Normalização** (`RobustScaler`), ajustada só nas linhas de treino: `(x − mediana) / IQR`. Se o IQR for zero, usa o desvio-padrão; se este também for zero, usa 1.
9. **Relatório de descartados da E2** (`reports/tables/feature_drop_provisional_Telecom.csv`): é **provisório**. Foi ajustado nos 5 primeiros dias, que serão o primeiro bloco de treino. Os seletores usados pelos modelos são reajustados por dobra na E3.
10. **Categorias de KPI:** às 7 do plano somam-se "carga", "disponibilidade" e "outro" (decisão do autor em 2026-11-10).

## Isenção dos KPIs de disponibilidade do descarte por constância

### Decisão

A categoria "disponibilidade" é isenta das regras de constância e quase constância da decisão 7 (`series.selection.constant_exempt_categories: [disponibilidade]` em `configs/base.yaml`). A regra de excesso de ausentes continua valendo para ela. A isenção vale para a categoria inteira, nunca para atributos escolhidos individualmente. Uma categoria isenta que não exista no catálogo de KPIs faz o build falhar, para que um erro de digitação não desligue a isenção em silêncio.

Justificativa de domínio: indicadores de disponibilidade (ping, estado ligado/desligado, travamento) são constantes **por definição** em operação normal; a variância no treino não mede a sua utilidade, que está justamente em mudar quando há falha.

### Cronologia (rastreabilidade)

A ordem dos acontecimentos é registrada porque a decisão **não** foi tomada às cegas:

1. O plano define a seleção por variância ajustada só no treino; a E2 a implementou assim, sem isenção.
2. Ao inspecionar a E2, o autor do código consultou **todos os 15 dias e os rótulos** (`record.csv`) e constatou:
   - `db_*::On_Off_State` muda (0/1) em todas as 5 falhas `db close` e numa `db connection limit`;
   - `os_*::ICMP_ping` muda em 11 falhas `network delay` ou `network loss`.

   Esses KPIs são constantes nos dias de treino; a regra os descartava.
3. Só depois disso a isenção foi proposta e aceita pelo autor do TCC (2026-11-11).

### Ameaça à validade e mitigação

A escolha foi informada por dados de avaliação: é *data snooping* (grau de liberdade do pesquisador), embora o código continue sem vazamento (o seletor é ajustado só no treino, verificado em `tests/test_series_transforms.py`). Os sinais recuperados aparecem em até 17 das 51 falhas (5 + 1 + 11), então o efeito sobre as métricas pode ser grande. Mitigações:

1. **Declaração** em `docs/AMEACAS_VALIDADE.md`, para a seção correspondente da monografia.
2. **Análise de sensibilidade obrigatória:** toda métrica principal de detecção (E3/E4) e de explicabilidade/agente que dependa da seleção é reportada também na configuração de controle `configs/experiment/sensitivity_sem_isencao.yaml`, que só desliga a isenção. Se a conclusão se mantiver nas duas, a ameaça fica neutralizada na prática; se mudar, a diferença é ela própria um resultado.
3. **Auditoria:** `FeatureSelector.rescued` registra os atributos mantidos só pela isenção, com o motivo pelo qual seriam descartados. A seleção de controle é exatamente `kept` menos `rescued` (teste `test_rescued_features_reproduce_the_control_selection`). No relatório provisório: 1.586 mantidos, dos quais 78 só pela isenção (`reports/tables/feature_exempt_provisional_Telecom.csv`); controle com 1.508.
