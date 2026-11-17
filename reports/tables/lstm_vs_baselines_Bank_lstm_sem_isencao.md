# LSTM versus baselines da E4 — Bank (`lstm_sem_isencao`), diferenças pareadas

Diferença = LSTM menos rival, nas mesmas janelas de teste (detecção) ou nas mesmas falhas (localização); IC de 95% por bootstrap pareado (2000 reamostragens). Médias entre as 5 seeds; a contagem diz em quantas seeds o IC da diferença ficou inteiro acima de 0 (superior), contém 0 (empate) ou abaixo (inferior).

| Tarefa | Métrica | LSTM | Rival | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | auc_pr | lstm_forecast | zscore_local | +0.014 | [-0.029, +0.067] | 0 / 5 / 0 |
| detecção | auc_pr | lstm_forecast | lightgbm | +0.165 | [+0.069, +0.261] | 5 / 0 / 0 |
| detecção | auc_pr | lstm_classifier | zscore_local | -0.116 | [-0.196, -0.034] | 0 / 0 / 5 |
| detecção | auc_pr | lstm_classifier | lightgbm | +0.035 | [-0.044, +0.106] | 0 / 5 / 0 |
| localização | hr@1 | lstm_sum | zscore_global | +0.013 | [-0.095, +0.123] | 0 / 5 / 0 |
| localização | hr@3 | lstm_sum | zscore_global | +0.095 | [-0.015, +0.203] | 0 / 5 / 0 |
| localização | mrr | lstm_sum | zscore_global | +0.039 | [-0.042, +0.118] | 0 / 5 / 0 |
| localização | hr@1 | lstm_sum | zscore_local | +0.222 | [+0.133, +0.308] | 5 / 0 / 0 |
| localização | hr@3 | lstm_sum | zscore_local | +0.112 | [+0.020, +0.203] | 5 / 0 / 0 |
| localização | mrr | lstm_sum | zscore_local | +0.169 | [+0.108, +0.229] | 5 / 0 / 0 |
| localização | hr@1 | lstm_sum | prior | +0.222 | [+0.122, +0.325] | 5 / 0 / 0 |
| localização | hr@3 | lstm_sum | prior | +0.312 | [+0.193, +0.430] | 5 / 0 / 0 |
| localização | mrr | lstm_sum | prior | +0.252 | [+0.172, +0.334] | 5 / 0 / 0 |
| localização | hr@1 | lstm_top3 | zscore_global | +0.020 | [-0.088, +0.128] | 0 / 5 / 0 |
| localização | hr@3 | lstm_top3 | zscore_global | +0.080 | [-0.017, +0.177] | 0 / 5 / 0 |
| localização | mrr | lstm_top3 | zscore_global | +0.029 | [-0.051, +0.107] | 0 / 5 / 0 |
| localização | hr@1 | lstm_top3 | zscore_local | +0.228 | [+0.145, +0.313] | 5 / 0 / 0 |
| localização | hr@3 | lstm_top3 | zscore_local | +0.097 | [-0.007, +0.198] | 0 / 5 / 0 |
| localização | mrr | lstm_top3 | zscore_local | +0.158 | [+0.095, +0.224] | 5 / 0 / 0 |
| localização | hr@1 | lstm_top3 | prior | +0.228 | [+0.128, +0.327] | 5 / 0 / 0 |
| localização | hr@3 | lstm_top3 | prior | +0.297 | [+0.177, +0.418] | 5 / 0 / 0 |
| localização | mrr | lstm_top3 | prior | +0.241 | [+0.159, +0.324] | 5 / 0 / 0 |
