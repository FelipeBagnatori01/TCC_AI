# ADR 0011 — Ingestão de mais de um sistema (Plano B: Bank)

- **Status:** aceito (E1 do Bank aprovada em 2026-11-11)
- **Data:** 2026-11-11
- **Etapa:** Plano B, E1 do Bank

## Contexto

A ingestão da E1 (ADR 0004) foi escrita a partir do inventário do Telecom: lista de arquivos e esquemas fixos. O inventário do Bank mostrou outros arquivos (há logs), outras colunas, épocas em segundos e uma mudança de unidade no meio da série dos traces (`docs/DICIONARIO_DADOS.md`).

## Decisões

1. **Um layout por sistema** (`ingestion/telemetry.py`: `TELECOM_FILES`, `BANK_FILES`). Cada arquivo declara esquema imposto, unidade da época, colunas de componente e coluna de KPI. Um sistema sem layout inventariado é erro (`files_for`). O parquet intermediário preserva as colunas brutas mais `time_utc`; a normalização entre sistemas fica para a E2.
2. **Unidade da época detectada por arquivo nos traces do Bank** (`time_unit="auto"`): 13 dígitos são ms, 10 dígitos são s, e um arquivo que misture os dois falha.
3. **Verificação de faixa de datas na conversão, para todos os sistemas:** os instantes convertidos devem cair no dia da pasta, com 1 dia de folga para cada lado. Teria detectado a mudança de unidade automaticamente; se falhar, o arquivo parcial é apagado.
4. **`record.csv` com época em ponto flutuante** é aceito só se o valor for inteiro.
5. **Relatórios com nomes neutros:** o relatório de qualidade passa a usar `component`/`kpi` em vez de `cmdb_id`/`name`; o do Telecom foi regenerado (só as chaves mudaram).
6. **Manifesto com os dois sistemas** (`data_systems: [Telecom, Bank]`); configuração do Bank em `configs/system/bank.yaml`.
7. **Similaridade entre dias como artefato** (`tcc-rca similarity`): a checagem de dias copiados, feita à parte na E3, agora gera `reports/tables/day_similarity_<sistema>.csv`. No Telecom ela reproduz os três pares de cópias a partir dos parquets da E1.
8. **Market fora do escopo** (adendo do ADR 0009): não tem layout e o tipo de configuração não o aceita mais.

## Consequências

- O Telecom não mudou: reconvertido com o código novo, os 6 arquivos de um dia ficaram idênticos aos da E1, e as janelas da E3 mantêm o mesmo hash.
- O sistema sintético dos testes passou a se chamar `Telecom` (é o layout dele); há um sistema sintético no formato do Bank, com época em segundos e log em várias linhas.
