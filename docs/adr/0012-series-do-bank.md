# ADR 0012 — Séries multivariadas do Bank (Plano B, E2)

- **Status:** aceito (E2 do Bank aprovada em 2026-11-11, incluindo o preenchimento adaptativo)
- **Data:** 2026-11-11
- **Etapa:** Plano B, E2 do Bank

## Contexto

O Bank tem métricas em formato longo (`cmdb_id, kpi_name, value`), 497 nomes de KPI, serviços em `metric_app`, traces sem indicador de sucesso e logs (`docs/DICIONARIO_DADOS.md`). A matriz segue o ADR 0005 (passo de 1 min, `componente::kpi`, máscara de amostras reais, seleção e escala só no treino), com as adaptações abaixo.

## Decisões

1. **Bloco inicial: 2021_03_04 e 2021_03_05** (aprovado pelo autor em 2026-11-11). O 03/04 é o primeiro dia e não pode ser testado em ordem cronológica; o 03/05 não tem falhas. Tudo o que é *ajustado* na E2 usa só esses dias (`series.initial_block_days`): templates de log, passo nativo de amostragem e relatório provisório de descarte. A E3 do Bank deve testar de 2021_03_06 em diante.
2. **Catálogo por regras** (`configs/data/kpi_catalog_bank.yaml`):
   - Escrever 497 KPIs à mão não é viável. O catálogo aceita, além de entradas explícitas, regras de expressão regular ordenadas; a primeira que casa com o nome inteiro vence, e grupos nomeados entram na descrição (ex.: "Ocupação do disco sdb").
   - Um KPI sem entrada nem regra faz o build falhar, como antes.
   - Categorias, unidades e descrições continuam inferidas e marcadas como tal; "desconhecida" quando não há segurança (contadores do MySQL e do Redis).
3. **KPIs de contêiner:**
   - O nome traz o hash do contêiner, que muda a cada reinício (4 a 9 hashes por host). O hash e o pod são removidos (`dockerA1::Container_CpuPercent`); cada host roda um único pod.
   - Quando contêiner velho e novo coexistem num minuto (até 11% dos minutos do `dockerA2`), a combinação segue o `item_agg` do catálogo: soma para CPU, memória usada e rede; máximo para limite e percentual de memória.
4. **`metric_app`:** chamadas somadas por minuto; `mrt`, `sr` e `rr` ponderados pelas chamadas.
5. **Traces:**
   - Por componente e minuto: chamadas e duração média, p95 e p99. Não há taxa de erro, porque não existe campo de sucesso.
   - Arestas chamador→chamado vêm de `parent_id`→`span_id` dentro do mesmo `trace_id`.
   - Não há planilha de implantação: os componentes são os que têm atributos, e não há arestas de implantação.
6. **Logs** (`series/logs.py`):
   - **Expressões regulares onde há estrutura:**
     - logs de acesso: requisições, erros (status ≥ 400) e tempo médio de resposta. O penúltimo número da linha está em ms no Tomcat e em µs no apache (inferido das magnitudes);
     - `gc`: eventos, Full GC e soma das pausas `real=`;
     - `slow` do MySQL: consultas, `Query_time` médio e `Rows_examined` médio;
     - `error` do MySQL: contagem.
   - **drain3 só no texto livre** (`catalina`, `localhost`, primeira linha de cada registro, com números, hex e argumentos mascarados):
     - contagem por nível e por template;
     - **o miner é ajustado só no bloco inicial**, e só os 10 templates mais frequentes por tipo de log viram atributos; o resto, inclusive templates surgidos depois, vai para `tpl_other`.
     - O texto dos templates fica em `data/processed/Bank/log_templates.csv`, fora do git por ser derivado dos logs.
7. **Contagens valem zero quando não há evento** (spans, requisições, linhas de log, GC). Um minuto sem evento é uma observação, como as chamadas de trace no ADR 0005.
8. **Recorte ao dia de calendário da pasta** (UTC+8, [00:00, 24:00)). Os traces do Bank transbordam até 4 h para o dia anterior, o que duplicaria minutos entre dias. No Telecom o recorte não muda nada: a matriz ficou idêntica (mesmo hash).
9. **Preenchimento adaptativo — desvio do ADR 0005 no Bank:**
   - O passo de amostragem varia de 1 a 60 min conforme a família de KPI (contêineres de 6 a 38 min, sistemas de arquivos 60 min).
   - Com o limite fixo de 5 passos, essas séries ficariam majoritariamente ausentes e seriam descartadas.
   - No Bank (`series.adaptive_ffill: true`), o limite de cada atributo é o seu passo nativo (mediana do intervalo entre amostras reais **no bloco inicial**), entre 5 e 60 passos. Na prática, mantém o último valor até a próxima amostra esperada; continua só para frente e dentro do dia.
   - Os limites ficam em `reports/tables/fill_limits_Bank.csv`. O Telecom mantém o limite fixo de 5.

## Resultados (nos dados reais)

- **Matriz:** 14.400 minutos × 1.942 atributos de 29 componentes (18 de infraestrutura e 11 serviços), em 22 s com pico de 6,3 GB.
- **Atributos por fonte:** métricas 1.749, logs 101, traces 48, `metric_app` 44.
- **Templates:** 15 (10 do `catalina`, 5 do `localhost`).
- **Arestas:** 32, formando a topologia IG → Tomcat → MG ↔ docker.
- **Seleção provisória (bloco inicial):** 1.269 mantidos (34 só pela isenção de disponibilidade) e 673 descartados (493 constantes, 87 quase constantes, 93 com excesso de ausentes). Sem o preenchimento adaptativo, seriam 157 descartes por ausência, contra 93.
- **Ida e volta nos dados reais:** 4.262 de 4.262 valores iguais, em 4 famílias de KPI.
