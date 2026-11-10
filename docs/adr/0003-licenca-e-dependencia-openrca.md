# ADR 0003 — Licença CC BY-NC 4.0 e o repositório OpenRCA como dependência externa

- **Status:** aceito
- **Data:** 2026-11-10
- **Etapa:** E1

## Contexto

A telemetria do OpenRCA é distribuída sob **CC BY-NC 4.0**; o código do repositório, sob MIT. O trabalho precisa do script oficial de avaliação (`main/evaluate.py`) e das consultas.

## Decisões

1. **Dados:** uso estritamente não comercial (este TCC), com citação de XU, J. et al., *OpenRCA: Can Large Language Models Locate the Root Cause of Software Failures?*, ICLR 2025. A telemetria não é versionada nem redistribuída; `data/raw`, `data/interim` e `data/processed` ficam no `.gitignore` e o repositório guarda apenas `data/MANIFEST.csv` (caminho, tamanho, SHA-256). O download é manual, a partir do link do README oficial.
2. **Implicação para uso industrial:** resultados e código podem ser reaproveitados, mas os dados do OpenRCA não podem ser usados em contexto comercial (por exemplo, treino de um produto). O documento de recomendações da E9 deve considerar isso.
3. **Dependência externa:** repositório OpenRCA como submódulo git em `external/OpenRCA`, fixado no commit `2a481ae63cb045c3eadbeca9eddaa10ec1151979` (Merge PR #42, 2026-10-22). Atualizar o commit exige novo ADR ou nota, porque o avaliador define a métrica oficial.
4. **Dependências do avaliador:** o `requirements.txt` do OpenRCA fixa versões antigas de vários pacotes (anthropic, openai, pandas 1.5.3, etc.). Em `main/evaluate.py` só `pandas` é importado em tempo de execução (`file_evaluate`, `report`); o restante é biblioteca padrão. O projeto instala apenas `pandas` (versão atual, travada no `uv.lock`) e roda o avaliador sem alterá-lo, com `python -m main.evaluate` a partir de `external/OpenRCA` (`uv run --project <repo>`). Verificado em 2026-11-10 com uma predição de teste sobre o `query.csv` do Telecom. Se alguma versão futura de pandas quebrar o avaliador, a alternativa é um ambiente separado com `pandas==1.5.3`.
5. O ruff exclui `external/` (o submódulo não é código do projeto).

## Consequências

- Reprodução por terceiros: clonar com `git clone --recurse-submodules`, baixar o Telecom do Drive e conferir com `tcc-rca verify`.
- Os autores não publicam checksums; o manifesto registra o que foi baixado neste trabalho e só detecta corrupção/alteração local, não prova equivalência com a fonte.
