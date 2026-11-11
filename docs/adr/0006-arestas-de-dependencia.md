# ADR 0006 — Arestas de dependência: traces e lista de implantação

- **Status:** aceito
- **Data:** 2026-11-10
- **Etapa:** E2

## Contexto

O agente (E7) precisa do contexto de dependências entre componentes. Há duas fontes: os traces (chamadas observadas) e o `AppTelecomDeploymentList.xlsx`, que vem do Drive oficial do OpenRCA, segundo o autor.

## Decisões

- **Planilha:** fica em `data/raw/openrca/` (symlink, não versionada) e entra no `data/MANIFEST.csv`. Ela tem a mesma licença CC BY-NC 4.0 da telemetria. Só a tabela pequena derivada dela, citando o OpenRCA, é versionada em `reports/tables/`.
- **Leitura:** colunas `Type, Number, Host, Name, From (containers)`. `Type` e `Number` só aparecem na primeira linha de cada grupo. O leitor confere o tamanho de cada grupo contra `Number` e falha se divergir.
- **Tabela única** `reports/tables/dependency_edges_Telecom.csv` (`src, dst, relation, source, weight`), com estas relações:
  - `calls` (`traces`): componente do span pai → componente do span filho, via `pid` → `id` no mesmo trace, e componente → banco em `dsName`. `weight` = número de chamadas nos 15 dias.
  - `hosted_on`: `redis_*` → `os_003..005` e `docker_*` → `os_017..020`.
  - `member_of`: `docker_001..004` → `container_001` e `docker_005..008` → `container_002`.
  - `runs_in`: `csf_001` → `container_001` e `csf_002` → `container_002`.
- O relacionamento é guardado como está; a interpretação (ex.: falha em `os_017` afeta `docker_001` e `docker_005`) fica a cargo do agente e da análise.

## Consequências

- O grafo observado tem 32 arestas `calls` e 30 de implantação:
  - `os_021`/`os_022` → `docker_001..004`;
  - `docker_001..004` → `docker_005..008`, `db_007`, `db_009`;
  - `docker_005..008` → `db_003`.
- Os bancos `db_*` e os hosts `os_021`/`os_022` não têm host na planilha.
- `components_Telecom.csv` lista os 61 componentes da planilha com grupo e host.
