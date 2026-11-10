# ADR 0002 — Fuso horário: UTC em todo o pipeline

- **Status:** aceito
- **Data:** 2026-11-10
- **Etapa:** E1

## Contexto

Os rótulos (`record.csv`), as janelas das consultas (`query.csv`) e os instantes que o avaliador oficial espera estão em UTC+8. A telemetria usa epoch em milissegundos (sem fuso). Esquecer a conversão desloca todos os rótulos em 8 horas e invalida tudo adiante.

## Decisão

- Dentro do pipeline, todo instante é um `datetime` com fuso **UTC**.
- A conversão acontece em um único módulo, `ingestion/timeutil.py`, e só nas bordas: leitura de `record.csv`/`query.csv` (UTC+8 → UTC) e escrita de respostas para o avaliador oficial (UTC → UTC+8, `utc_to_local_str`).
- O leitor de `record.csv` recusa linhas em que o epoch e o texto `datetime` (lido como UTC+8) discordam.
- Telemetria: colunas de epoch em ms viram `time_utc` na leitura; os parquets guardam UTC.
- Os nomes das pastas de dia (`AAAA_MM_DD`) continuam sendo o dia civil em UTC+8, como no download.

## Consequências

- Testes em `tests/test_timeutil.py` e `tests/test_cases.py` cobrem o deslocamento de 8 h, o ida-e-volta e a janela de 30 min.
- Figuras e relatórios mostram horários em UTC; o dia de telemetria em UTC começa às 16:00 do dia anterior.
- Agrupar por "dia" na E3 deve usar o dia UTC+8 (as pastas), não o dia UTC, para não partir uma janela de 6 h em dois dias.
