# Dicionário de dados — OpenRCA, sistema Telecom

Descreve o que foi **encontrado** nos arquivos (inventário da etapa E1, `reports/tables/inventory_Telecom.json`, gerado por `tcc-rca inventory`). Nada aqui foi presumido a partir da documentação. Só o Telecom foi baixado e inventariado; Bank e Market não.

Dados lidos em: `data/raw/openrca/Telecom/` (cópia/symlink do download, não versionado). Commit do OpenRCA usado: ver ADR 0003.

## Estrutura

```
Telecom/
|-- query.csv        51 consultas (colunas: task_index, instruction, scoring_points)
|-- record.csv       51 falhas rotuladas (colunas: level, reason, component, timestamp, datetime)
`-- telemetry/
    `-- AAAA_MM_DD/  15 dias: 2020-04-11, 04-20..04-23, 05-22..05-31
        |-- metric/  metric_app, metric_container, metric_middleware, metric_node, metric_service (.csv)
        `-- trace/   trace_span.csv
```

- Total no disco: 92 arquivos, 17,66 GB (`data/MANIFEST.csv`).
- **Cada dia cobre só 6 horas**: de 00:00 a 06:00 em UTC+8 (16:00–22:00 UTC do dia anterior). O nome da pasta é o dia civil em UTC+8. Não há telemetria fora dessa faixa.
- Todos os arquivos têm o mesmo esquema em todos os dias (verificado).
- Não há logs (como esperado para o Telecom).
- Não há `.csv` de topologia no download. O arquivo `AppTelecomDeploymentList.xlsx` (fora do repositório, em `TCC_Project/`) lista serviços e contêineres, mas não foi usado na E1.

## Fuso horário

- `record.csv`: `timestamp` (epoch em segundos) e `datetime` (texto em **UTC+8**) concordam em todas as 51 linhas quando `datetime` é lido como UTC+8 (checado pelo leitor). Exemplo: `1586534700` = `2020-04-11 00:05:00` (UTC+8) = `2020-04-10 16:05:00` UTC.
- `query.csv`: as janelas de 30 minutos estão em texto livre em inglês dentro de `instruction`, em UTC+8 (ex.: "April 11, 2020, from 00:00 to 00:30"). O leitor extrai data e as duas horas e converte para UTC.
- Telemetria: colunas de tempo em epoch **milissegundos**; convertidas para UTC na coluna `time_utc` (ADR 0002).

## Rótulos e consultas

- 51 consultas, cada uma com exatamente 1 falha; as 51 falhas de `record.csv` caem cada uma em exatamente uma janela (nenhuma órfã, nenhuma consulta sem falha) — `tcc-rca cases`.
- `record.csv`: níveis `node` (20), `pod` (19), `service` (12); motivos `CPU fault` (19), `network delay` (13), `db connection limit` (7), `network loss` (7), `db close` (5); 15 componentes distintos (`db_*`, `docker_*`, `os_*`).
- Tipos de tarefa em `query.csv`: task_1 (10), task_2 (10), task_3 (4), task_4 (8), task_5 (4), task_6 (6), task_7 (9).
- **`scoring_points` é o gabarito** e está no mesmo arquivo que a instrução que o agente verá. O código separa as duas coisas: `ingestion/cases.py` não expõe `scoring_points`; só `ingestion/labels.py` o lê (R3).

## Métricas

Esquema comum de `metric_container`, `metric_middleware`, `metric_node`, `metric_service`:

| Coluna | Tipo | Descrição |
|---|---|---|
| itemid | Int64 | identificador do item de monitoramento |
| name | String | nome do KPI |
| bomc_id | String | identificador do coletor (nulo em 34.702 linhas de `metric_service`) |
| timestamp | Int64 | epoch em milissegundos |
| value | Float64 | valor do KPI |
| cmdb_id | String | componente |

| Arquivo | Componentes (`cmdb_id`) | KPIs | Séries (componente×KPI) | Séries constantes | Linhas (15 dias) |
|---|---|---|---|---|---|
| metric_container | `docker_001..008` (7–8 por dia) | 9 | 72 | 56 | 356.260 |
| metric_middleware | `redis_001..012` | 22 | 264 | 102 | 1.408.622 |
| metric_node | `os_001..022` (14–22 por dia; 14 em 2020_05_22) | 51 | 1.122 | 220 | 4.280.977 |
| metric_service | `db_001..013` | 46 (47 em 2 dias) | 611 | 135 | 2.549.996 |

- **Granularidade:** passo modal de 60 s por série em todos os arquivos e dias.
- `metric_app` (agregado do serviço): colunas `serviceName`(String), `startTime`(Int64, ms), `avg_time`(Float64), `num`, `succee_num`(Int64), `succee_rate`(Float64). Um único serviço (`osb_001`), 5.926 linhas no total; sem nulos.
- Exceto `bomc_id`, não há nulos nas métricas.
- KPIs por arquivo — container: `container_cpu_used`, `container_mem_used`, `container_fgc`, `container_fgct`, `container_session_used`, `container_thread_{idle,running,total,used_pct}`. Middleware (Redis): contadores de clientes, comandos, chaves, memória e CPU. Node: CPU, memória, disco, rede, processos, carga, swap. Service (banco de dados Oracle): sessões, tablespaces, PGA, TPS, esperas, CPU, memória. A lista completa está no JSON de inventário e é consolidada em catálogo na E2.

## Traces

`trace_span.csv`, uma linha por span:

| Coluna | Tipo | Descrição |
|---|---|---|
| callType | String | `OSB`, `CSF`, `LOCAL`, `FlyRemote`, `RemoteProcess`, `JDBC` |
| startTime | Int64 | epoch em milissegundos |
| elapsedTime | Float64 | duração do span |
| success | Boolean | sucesso da chamada |
| traceId, id, pid | String | trace, span e span pai |
| cmdb_id | String | componente que executa o span: `docker_001..008`, `os_021`, `os_022` (9–11 por dia); nulo em 322 spans |
| dsName | String | banco acessado (`db_003`, `db_007`, `db_009`); nulo em 30.276.568 spans |
| serviceName | String | serviço chamado (`osb_001`, `csf_*`, `local_method_*`, `fly_remote_001`); nulo em 65.355.386 spans |

- 145.907.050 spans em 15 dias, 7,3–10,6 milhões por dia, 0,86–1,24 GB por dia em CSV.
- Os traces só identificam `docker_*`, `os_021`, `os_022` (via `cmdb_id`) e três bancos (via `dsName`). Não há traces para os demais `os_*` e `db_*`.
- O par `(cmdb_id → dsName/serviceName)` fornece arestas chamador→chamado para a E2.

## Cobertura das falhas (relatório de qualidade)

Em `reports/tables/quality_Telecom.{md,json}`:

- 51 de 51 falhas têm métricas do componente rotulado nos 30 min antes e nos 30 min depois do início.
- Nenhum componente rotulado está sem dados de métricas.
- 14 falhas (componentes `os_001`, `os_009`, `os_017`, `os_018`, `os_020`) não têm nenhum trace do componente, por limitação do dataset (ver acima).
- Como o dia começa às 00:00 e termina às 06:00 (UTC+8), 5 falhas têm menos de 30 min de histórico antes do início e 2 têm menos de 30 min depois.

## Observações relevantes para as próximas etapas

- **Dias quase duplicados.** Três pares de dias têm exatamente o mesmo número de spans (2020_04_22×05_23, 04_23×05_25, 04_21×05_31). No par 04_22×05_23, `metric_node` tem as mesmas chaves, deslocamento de tempo constante e apenas 158 de 301.617 valores diferentes. Os outros dois pares só foram comparados pela contagem de linhas. O comportamento "normal" parece repetido entre dias, o que afeta a divisão temporal da E3 e deve constar como ameaça à validade.
- Muitos KPIs são constantes (cerca de 20% das séries); a seleção de atributos da E2 deve usar só o treino.
- O sinal da falha é visível em alguns KPIs, mas na amostra inspecionada (`docker_003`, CPU fault) o `container_cpu_used` sobe depois do instante rotulado, e não antes. A análise de sinal precursor é da E3.

## Lista de implantação (`AppTelecomDeploymentList.xlsx`)

Planilha do Drive oficial do OpenRCA, com colunas `Type, Number, Host, Name, From (containers)`. Ela lista 61 componentes em grupos:
- `OSB` (`osb_001`) e `CSF` (`csf_001..005`, dos quais `csf_001` e `csf_002` vêm dos grupos de contêineres `container_001` e `container_002`);
- `oracle_11G` (`db_001..013`);
- `redis` (`redis_001..012`, hospedados em `os_003..005`);
- `container_001` (`docker_001..004`) e `container_002` (`docker_005..008`), hospedados em `os_017..020`;
- `Virtual Machine` (`os_001..022`).

Leitura e arestas: ADR 0006 e `reports/tables/dependency_edges_Telecom.csv`.

## Séries multivariadas (E2)

- Matriz em `data/processed/Telecom/series.parquet` (não versionada): 5.401 minutos × 2.138 atributos `componente::kpi`, mais `observed.parquet` com a máscara de amostras reais.
- Catálogo por atributo: `reports/tables/feature_catalog_Telecom.csv`, gerado a partir do catálogo de KPIs versionado `configs/data/kpi_catalog_telecom.yaml`. Categoria, unidade e descrição são **inferidas do nome**; as siglas `ACS`, `AIOS` e `AWS` não estão documentadas.
- Regras de agregação e preenchimento: ADR 0005.
- Seleção provisória (5 primeiros dias): `reports/tables/feature_drop_provisional_Telecom.csv` lista os 552 descartados com motivo. `reports/tables/feature_exempt_provisional_Telecom.csv` lista os 78 atributos de disponibilidade mantidos **só** pela isenção (`Agent_ping` 22, `ICMP_ping` 19, `Hang` 13, `On_Off_State` 12, `redis_ping` 12, todos constantes no treino); sem eles, obtém-se a seleção de controle (1.508).

## Janelas, rótulos e divisão (E3)

- `data/processed/Telecom/windows.parquet` (não versionado; contém rótulos, portanto é gabarito e proibido ao agente): uma linha por janela de 20 min com `row` (linha final na matriz), `day`, `start_utc`, `end_utc`, `minutes_to_next_onset`, `minutes_since_last_onset`, `normal` e `label` (1, 0 ou nulo, ADR 0007). São 4.093 janelas nos 12 dias utilizáveis.
- Dias 2020_04_21, 04_22 e 04_23 excluídos: cópias sem rótulo de 05_31, 05_23 e 05_25 (ADR 0008).
- Dobras, contagens e hash: `reports/tables/tasks_Telecom.json` e `split_characterization_Telecom.md`. Medição de precursor: `reports/tables/precursor_Telecom.csv`.

## Memória

Converter um dia para parquet (leitura lazy e escrita em streaming) teve pico de **1.706 MB** de RSS no dia mais pesado (2020_04_20, 1,24 GB de trace) e 1.681 MB em 2020_04_11, medidos em processo isolado por `tcc-rca convert --days <dia>`, na máquina de 24 GB de RAM. Os 15 dias convertidos ocupam 2,7 GB em `data/interim/`.

# Bank (Plano B)

Inventário em `reports/tables/inventory_Bank.json` (`tcc-rca inventory --config configs/base.yaml configs/system/bank.yaml`); qualidade em `reports/tables/quality_Bank.{md,json}`; similaridade entre dias em `reports/tables/day_similarity_Bank.csv`. O layout de arquivos é próprio do Bank (ADR 0011): nada foi herdado do Telecom.

## Estrutura

- 10 dias (2021_03_04, 05, 06, 07, 09, 10, 12, 23, 24, 25), cada um com **o dia inteiro** (00:00–23:59 UTC+8), ao contrário das 6 h do Telecom. 25 GB de CSV, 3,2 GB em parquet.
- Arquivos por dia: `metric/metric_container.csv`, `metric/metric_app.csv`, `trace/trace_span.csv`, `log/log_service.csv`. Os cabeçalhos são iguais nos 10 dias.

| Arquivo | Colunas | Época | Conteúdo |
|---|---|---|---|
| `metric_container` | `timestamp, cmdb_id, kpi_name, value` | segundos | 18 componentes, 348–374 KPIs por dia, passo modal de 60 s. Famílias pelo prefixo do KPI: `OSLinux`, `Mysql`, `Tomcat`, `redis`, `JVM`, `Container` |
| `metric_app` | `timestamp, rr, sr, cnt, mrt, tc` | segundos | 11 serviços `ServiceTest1..11` (`tc`); `cnt` chamadas, `mrt` tempo médio de resposta, `rr`/`sr` taxas em % (significados inferidos dos nomes) |
| `trace_span` | `timestamp, cmdb_id, parent_id, span_id, trace_id, duration` | **ms até 2021_03_12; segundos de 2021_03_23 em diante** | 12 componentes emitem spans (IG, MG, Tomcat, dockerA/B); Redis, Mysql e apache não. Sem `dsName` nem indicador de sucesso |
| `log_service` | `log_id, timestamp, cmdb_id, log_name, value` | segundos | 7 tipos de log: `gc`, `catalina`, `localhost`, `localhost_access_log` (Tomcat), `apache_access_log` (apache), `error` e `slow` (Mysql). `value` é texto livre e pode ocupar várias linhas |

- **Mudança de unidade nos traces.** Em 23, 24 e 25/03 o `timestamp` do trace tem 10 dígitos (segundos), contra 13 nos outros dias. Lido como milissegundos, cairia em 20/01/1970. A unidade é detectada por arquivo, e a conversão falha se um arquivo misturar as duas (ADR 0011). Nesses três dias o instante dos spans tem só resolução de segundo.
- Componentes com telemetria: `IG01-02`, `MG01-02`, `Mysql01-02`, `Redis01-02`, `Tomcat01-04`, `apache01-02`, `dockerA1-2`, `dockerB1-2`.

## Rótulos e consultas

- 136 falhas em `record.csv` (`timestamp` em segundos escrito como `1614841020.0`) e 136 consultas.
- 14 componentes e 8 motivos: `high CPU usage` 33, `network packet loss` 32, `network latency` 27, `high disk I/O read usage` 19, `high memory usage` 10, `JVM Out of Memory (OOM) Heap` 7, `high disk space usage` 5, `high JVM CPU load` 3.
- **Consultas com duas falhas.** 9 janelas de 30 min contêm duas falhas e cada uma é usada por duas consultas de tipos diferentes; essas 18 consultas pedem 2 falhas no gabarito (conferido: 0 divergências). Por isso o mapeamento tem 154 pares consulta–falha.
- Falhas por dia: 2021_03_05 não tem nenhuma; os outros 9 dias têm de 10 a 21.
- **Motivo e tipo de componente não estão ligados** (diferente do Telecom): `high CPU usage`, por exemplo, ocorre em Redis, Mysql, IG, Tomcat, apache e MG.

## Qualidade

- Nenhuma coluna com nulos; 515 das 1.889 séries de métricas são constantes.
- Todas as 154 falhas têm métricas do componente rotulado antes e depois do início. 38 não têm spans do componente (Redis, Mysql e apache não emitem traces).
- **Sem dias duplicados:** o par de dias mais parecido coincide em 53% dos valores (o Telecom tinha três pares acima de 99,99%).

## Séries multivariadas do Bank (E2)

- Matriz em `data/processed/Bank/series.parquet` e `observed.parquet` (não versionadas): 14.400 minutos (10 dias × 1.440) × 1.942 atributos `componente::kpi`. Construção e decisões: ADR 0012.
- **Passo de amostragem heterogêneo:**
  - 60 s: JVM, Tomcat, Redis;
  - 120 s: CPU do SO e MySQL, na mediana;
  - 300 s: memória do Tomcat e NTP;
  - 6 a 38 min: contêineres;
  - 60 min: sistemas de arquivos.
  
  Por isso, o limite de preenchimento é ajustado por atributo (`reports/tables/fill_limits_Bank.csv`).
- Catálogo por atributo em `reports/tables/feature_catalog_Bank.csv`, gerado a partir das regras de `configs/data/kpi_catalog_bank.yaml`.
- Atributos de log: `access_*` (Tomcat, apache), `gc_*` (Tomcat), `slow_*` e `mysql_error_count` (MySQL), `log_<catalina|localhost>_{count,level_*,tpl_*}` (Tomcat). O texto dos templates fica em `data/processed/Bank/log_templates.csv` (não versionado).
- Dependências: `reports/tables/dependency_edges_Bank.csv`. São 32 arestas observadas nos traces, IG01-02 → Tomcat01-04 → MG01-02 ↔ dockerA1-2/dockerB1-2; Redis, MySQL e apache não emitem spans.
