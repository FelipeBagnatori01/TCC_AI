# LSTM versus baselines da E4 — Telecom (`lstm`), diferenças pareadas

Diferença = LSTM menos rival, nas mesmas janelas de teste (detecção) ou nas mesmas falhas (localização); IC de 95% por bootstrap pareado (2000 reamostragens). Médias entre as 5 seeds; a contagem diz em quantas seeds o IC da diferença ficou inteiro acima de 0 (superior), contém 0 (empate) ou abaixo (inferior).

| Tarefa | Métrica | LSTM | Rival | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | auc_pr | lstm_forecast | zscore_local | +0.159 | [+0.043, +0.284] | 5 / 0 / 0 |
| detecção | auc_pr | lstm_forecast | lightgbm | +0.149 | [-0.019, +0.300] | 1 / 4 / 0 |
| detecção | auc_pr | lstm_classifier | zscore_local | +0.244 | [+0.119, +0.374] | 5 / 0 / 0 |
| detecção | auc_pr | lstm_classifier | lightgbm | +0.235 | [+0.093, +0.358] | 5 / 0 / 0 |
| localização | hr@1 | lstm_sum | zscore_global | +0.307 | [+0.124, +0.484] | 5 / 0 / 0 |
| localização | hr@3 | lstm_sum | zscore_global | +0.333 | [+0.116, +0.533] | 5 / 0 / 0 |
| localização | mrr | lstm_sum | zscore_global | +0.269 | [+0.112, +0.427] | 5 / 0 / 0 |
| localização | hr@1 | lstm_sum | zscore_local | +0.329 | [+0.182, +0.471] | 5 / 0 / 0 |
| localização | hr@3 | lstm_sum | zscore_local | +0.178 | [-0.018, +0.373] | 0 / 5 / 0 |
| localização | mrr | lstm_sum | zscore_local | +0.228 | [+0.105, +0.352] | 5 / 0 / 0 |
| localização | hr@1 | lstm_sum | prior | +0.351 | [+0.196, +0.511] | 5 / 0 / 0 |
| localização | hr@3 | lstm_sum | prior | +0.489 | [+0.338, +0.644] | 5 / 0 / 0 |
| localização | mrr | lstm_sum | prior | +0.376 | [+0.246, +0.504] | 5 / 0 / 0 |
| localização | hr@1 | lstm_top3 | zscore_global | +0.231 | [+0.067, +0.400] | 5 / 0 / 0 |
| localização | hr@3 | lstm_top3 | zscore_global | +0.382 | [+0.196, +0.560] | 5 / 0 / 0 |
| localização | mrr | lstm_top3 | zscore_global | +0.255 | [+0.113, +0.399] | 5 / 0 / 0 |
| localização | hr@1 | lstm_top3 | zscore_local | +0.253 | [+0.102, +0.409] | 5 / 0 / 0 |
| localização | hr@3 | lstm_top3 | zscore_local | +0.227 | [+0.053, +0.396] | 5 / 0 / 0 |
| localização | mrr | lstm_top3 | zscore_local | +0.214 | [+0.092, +0.338] | 5 / 0 / 0 |
| localização | hr@1 | lstm_top3 | prior | +0.276 | [+0.107, +0.436] | 5 / 0 / 0 |
| localização | hr@3 | lstm_top3 | prior | +0.538 | [+0.378, +0.693] | 5 / 0 / 0 |
| localização | mrr | lstm_top3 | prior | +0.363 | [+0.233, +0.490] | 5 / 0 / 0 |
