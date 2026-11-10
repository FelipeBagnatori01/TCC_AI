# ADR 0004 — Ingestão: parquet por dia, esquemas explícitos e leitura lazy

- **Status:** aceito
- **Data:** 2026-11-10
- **Etapa:** E1

## Contexto

O Telecom tem 17,7 GB de CSV (até 1,24 GB por dia só de traces). É preciso ler por dia, sem materializar arquivos inteiros, e falhar alto se o formato mudar.

## Decisões

- **Formato intermediário:** um parquet (zstd) por arquivo e por dia em `data/interim/<sistema>/<dia>/<arquivo>.parquet`, escrito em streaming (`scan_csv(...).sink_parquet`), com escrita atômica (arquivo temporário e rename) e conversão idempotente.
- **Esquemas explícitos:** colunas e tipos de cada arquivo foram fixados a partir do inventário real (ver `DICIONARIO_DADOS.md`). A leitura compara o cabeçalho e rejeita o arquivo se ele diferir, em vez de inferir tipos.
- **Consultas lazy:** relatórios de qualidade e EDA usam `scan_parquet` com glob sobre os dias e filtros empurrados para a leitura.
- **Janela da consulta:** extraída por expressão regular do texto livre de `instruction`; falha com erro se não achar data e dois horários. Falhas são associadas à consulta pelo intervalo semiaberto `[início, fim)`.
- **Gabarito isolado (R3):** `query.csv` mistura instrução e `scoring_points`. O tipo `Query` não tem o gabarito; `ingestion/labels.py` é o único ponto de leitura e um teste garante que `tcc_rca.agent` não o importa.

## Consequências

- Pico de memória medido: cerca de 1,7 GB por dia no Telecom (máquina de 24 GB).
- Se o Bank/Market entrarem (Plano B), os esquemas dos logs e de outros arquivos precisam ser inventariados e adicionados; a leitura falhará até lá.
