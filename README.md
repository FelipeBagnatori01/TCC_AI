# TCC — Causa raiz de falhas em telecomunicações (OpenRCA)

LSTM para detecção/predição de falhas, SHAP e LIME para explicabilidade e agente de LLM para explicar a causa raiz.


## Início rápido

```bash
make setup   # uv sync, cria .env a partir de .env.example, instala pre-commit
make lint    # ruff + mypy (estrito em src/)
make test    # pytest
make e2e     # execução de fumaça da infraestrutura -> experiments/<timestamp>-e2e/
```

Requer [uv](https://docs.astral.sh/uv/). A telemetria do OpenRCA **não** é versionada (CC BY-NC 4.0, uso não comercial); ver `data/raw/`.

Estado: etapa E0 (infraestrutura). As demais etapas são implementadas uma por vez.
