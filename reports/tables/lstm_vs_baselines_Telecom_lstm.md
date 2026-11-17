# LSTM versus baselines da E4 — Telecom (`lstm`), diferenças pareadas

Diferença = LSTM menos rival, nas mesmas janelas de teste (detecção) ou nas mesmas falhas (localização); IC de 95% por bootstrap pareado (2000 reamostragens). Médias entre as 5 seeds; a contagem diz em quantas seeds o IC da diferença ficou inteiro acima de 0 (superior), contém 0 (empate) ou abaixo (inferior).

| Tarefa | Métrica | LSTM | Rival | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | auc_pr | lstm_forecast | zscore_local | +0.125 | [+0.009, +0.229] | 4 / 1 / 0 |
| detecção | auc_pr | lstm_forecast | lightgbm | +0.269 | [+0.089, +0.404] | 5 / 0 / 0 |
| detecção | auc_pr | lstm_classifier | zscore_local | +0.091 | [-0.042, +0.227] | 0 / 5 / 0 |
| detecção | auc_pr | lstm_classifier | lightgbm | +0.235 | [+0.093, +0.358] | 5 / 0 / 0 |
| localização | hr@1 | lstm_sum | zscore_global | +0.289 | [+0.111, +0.462] | 5 / 0 / 0 |
| localização | hr@3 | lstm_sum | zscore_global | +0.289 | [+0.071, +0.507] | 5 / 0 / 0 |
| localização | mrr | lstm_sum | zscore_global | +0.248 | [+0.090, +0.407] | 5 / 0 / 0 |
| localização | hr@1 | lstm_sum | zscore_local | +0.333 | [+0.187, +0.475] | 5 / 0 / 0 |
| localização | hr@3 | lstm_sum | zscore_local | +0.178 | [-0.018, +0.373] | 0 / 5 / 0 |
| localização | mrr | lstm_sum | zscore_local | +0.233 | [+0.110, +0.357] | 5 / 0 / 0 |
| localização | hr@1 | lstm_sum | prior | +0.333 | [+0.178, +0.489] | 5 / 0 / 0 |
| localização | hr@3 | lstm_sum | prior | +0.489 | [+0.338, +0.644] | 5 / 0 / 0 |
| localização | mrr | lstm_sum | prior | +0.366 | [+0.237, +0.495] | 5 / 0 / 0 |
| localização | hr@1 | lstm_top3 | zscore_global | +0.209 | [+0.044, +0.369] | 5 / 0 / 0 |
| localização | hr@3 | lstm_top3 | zscore_global | +0.329 | [+0.138, +0.515] | 5 / 0 / 0 |
| localização | mrr | lstm_top3 | zscore_global | +0.231 | [+0.089, +0.376] | 5 / 0 / 0 |
| localização | hr@1 | lstm_top3 | zscore_local | +0.253 | [+0.102, +0.409] | 5 / 0 / 0 |
| localização | hr@3 | lstm_top3 | zscore_local | +0.218 | [+0.035, +0.391] | 4 / 1 / 0 |
| localização | mrr | lstm_top3 | zscore_local | +0.216 | [+0.094, +0.339] | 5 / 0 / 0 |
| localização | hr@1 | lstm_top3 | prior | +0.253 | [+0.084, +0.409] | 5 / 0 / 0 |
| localização | hr@3 | lstm_top3 | prior | +0.529 | [+0.373, +0.684] | 5 / 0 / 0 |
| localização | mrr | lstm_top3 | prior | +0.349 | [+0.222, +0.475] | 5 / 0 / 0 |
