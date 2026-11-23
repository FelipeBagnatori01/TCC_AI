# Rastreabilidade dos resultados (E9, critério 2)

**Regra:** todo número quantitativo da monografia sai de um artefato versionado em `reports/`, gerado por um comando registrado. Nada é digitado à mão.

- **Tabelas e figuras da monografia:** `reports/monografia/`, gerado por `uv run tcc-rca thesis-assets`.
  - `MANIFEST.csv` dá, para cada tabela e figura, os arquivos de origem, o comando e o commit.
  - `numeros.csv` dá os números citados no texto, cada um com o arquivo JSON e a chave de onde sai.
- **Números de `docs/RESULTADOS.md`:** a tabela abaixo liga cada seção às tabelas de `reports/tables/` e ao comando que as gera.
  - `<s>` = `Telecom` ou `Bank`.
  - Os sistemas além do Telecom levam `configs/system/bank.yaml` no comando.

| Seção de `RESULTADOS.md` | Artefatos | Comando |
|---|---|---|
| Achados sobre os dados (E1–E3) | `inventory_<s>.json`, `quality_<s>.{json,md}`, `day_similarity_<s>.csv`, `components_<s>.csv`, `feature_catalog_<s>.csv`, `dependency_edges_<s>.csv`, `tasks_<s>.json`, `precursor_<s>.csv`, `split_characterization_<s>.md`; figuras `eda_*.png` | `inventory`, `quality`, `similarity`, `series`, `tasks`, `eda` |
| E4: baselines (detecção, localização, A0) | `baselines_<s>_baselines.json` (e `_sem_isencao`, `_janela_inteira`) | `baselines` + `configs/experiment/baselines*.yaml` |
| E5: LSTM | `lstm_<s>_lstm.json` (e `_sem_isencao`, `_janela_inteira`); `lstm_vs_baselines_<s>_*.json`; figuras `lstm_curves_*.png` | `lstm`, `compare-e5` |
| E6: explicabilidade | `xai_<s>_xai.json` (e `_sem_isencao`) | `xai` |
| Modelos alternativos e comparação de candidatos | `alt_models_<s>_alt_{tabular,tcn}{,_sem_isencao}.json` | `alt-models` |
| Seleção prequencial e cadeia do pacote | `evidence_select_<s>_evidence_select{,_sem_isencao}.json` | `evidence-select` |
| E7: execuções do agente | `agent_<s>_agent_a{1,2,3}_<modelo>.json`; piloto `agent_<s>_agent_piloto*_<modelo>.json` | `agent` (pago); replay pelo cache: `reproduce --level agente-cache` |
| E8: nota oficial, comparações, localização, fidelidade, custo, auditoria e reprodução | `agent_eval_<s>.{json,md}`; `reports/audit/auditoria_fidelidade_llm.csv` | `evaluate-agent` |
| E8: curvas PR/ROC (RQ1) | `reports/figures/rq1_curvas_pr_roc.png` | `rq1-curves` |
| ADR 0023: Haiku e Sonnet | `model_compare_<s>_claude-{haiku,sonnet}-5-5.{json,md}` | `compare-models` + `model_compare{,_sonnet}.yaml` |
| E9: histórico do Git | `reports/auditoria_git.md` | `repo-audit` |

**Como conferir:**
- `make reproduce` (e `make reproduce-agent`) regenera as tabelas.
- `make reproduce-check` mostra que elas batem com as versionadas.
- **Se aparecer na monografia um número que não esteja num destes arquivos,** o número está fora da regra e deve ser removido ou ganhar um artefato.

**Exceção registrada:** o custo do Bank A3 do Sonnet soma US$ 2,70 da execução interrompida por falta de crédito (ADR 0023). A soma é feita em código (`INTERRUPTED_USD` em `src/tcc_rca/thesis_assets.py` e no notebook `analise_modelos_llm.ipynb`), com a origem citada.
