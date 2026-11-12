# LSTM versus baselines da E4 — Bank (`lstm`), diferenças pareadas

Diferença = LSTM menos rival, nas mesmas janelas de teste (detecção) ou nas mesmas falhas (localização); IC de 95% por bootstrap pareado (2000 reamostragens). Médias entre as 5 seeds; a contagem diz em quantas seeds o IC da diferença ficou inteiro acima de 0 (superior), contém 0 (empate) ou abaixo (inferior).

| Tarefa | Métrica | LSTM | Rival | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | auc_pr | lstm_forecast | zscore_local | +0.014 | [-0.016, +0.069] | 0 / 5 / 0 |
| detecção | auc_pr | lstm_forecast | lightgbm | +0.012 | [-0.063, +0.100] | 0 / 5 / 0 |
| detecção | auc_pr | lstm_classifier | zscore_local | +0.032 | [-0.034, +0.108] | 0 / 5 / 0 |
| detecção | auc_pr | lstm_classifier | lightgbm | +0.030 | [-0.045, +0.105] | 0 / 5 / 0 |
| localização | hr@1 | lstm_sum | zscore_global | +0.012 | [-0.095, +0.122] | 0 / 5 / 0 |
| localização | hr@3 | lstm_sum | zscore_global | +0.117 | [+0.013, +0.223] | 5 / 0 / 0 |
| localização | mrr | lstm_sum | zscore_global | +0.041 | [-0.038, +0.119] | 0 / 5 / 0 |
| localização | hr@1 | lstm_sum | zscore_local | +0.220 | [+0.133, +0.310] | 5 / 0 / 0 |
| localização | hr@3 | lstm_sum | zscore_local | +0.142 | [+0.052, +0.233] | 5 / 0 / 0 |
| localização | mrr | lstm_sum | zscore_local | +0.174 | [+0.112, +0.237] | 5 / 0 / 0 |
| localização | hr@1 | lstm_sum | prior | +0.220 | [+0.120, +0.322] | 5 / 0 / 0 |
| localização | hr@3 | lstm_sum | prior | +0.333 | [+0.217, +0.447] | 5 / 0 / 0 |
| localização | mrr | lstm_sum | prior | +0.253 | [+0.174, +0.334] | 5 / 0 / 0 |
| localização | hr@1 | lstm_top3 | zscore_global | +0.008 | [-0.103, +0.118] | 0 / 5 / 0 |
| localização | hr@3 | lstm_top3 | zscore_global | +0.093 | [-0.002, +0.188] | 2 / 3 / 0 |
| localização | mrr | lstm_top3 | zscore_global | +0.023 | [-0.056, +0.102] | 0 / 5 / 0 |
| localização | hr@1 | lstm_top3 | zscore_local | +0.217 | [+0.130, +0.305] | 5 / 0 / 0 |
| localização | hr@3 | lstm_top3 | zscore_local | +0.118 | [+0.013, +0.223] | 5 / 0 / 0 |
| localização | mrr | lstm_top3 | zscore_local | +0.156 | [+0.091, +0.223] | 5 / 0 / 0 |
| localização | hr@1 | lstm_top3 | prior | +0.217 | [+0.120, +0.313] | 5 / 0 / 0 |
| localização | hr@3 | lstm_top3 | prior | +0.310 | [+0.190, +0.428] | 5 / 0 / 0 |
| localização | mrr | lstm_top3 | prior | +0.235 | [+0.156, +0.315] | 5 / 0 / 0 |
