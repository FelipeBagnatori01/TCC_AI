# LSTM versus baselines da E4 — Bank (`lstm`), diferenças pareadas

Diferença = LSTM menos rival, nas mesmas janelas de teste (detecção) ou nas mesmas falhas (localização); IC de 95% por bootstrap pareado (2000 reamostragens). Médias entre as 5 seeds; a contagem diz em quantas seeds o IC da diferença ficou inteiro acima de 0 (superior), contém 0 (empate) ou abaixo (inferior).

| Tarefa | Métrica | LSTM | Rival | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | auc_pr | lstm_forecast | zscore_local | +0.021 | [-0.024, +0.074] | 0 / 5 / 0 |
| detecção | auc_pr | lstm_forecast | lightgbm | +0.157 | [+0.063, +0.256] | 5 / 0 / 0 |
| detecção | auc_pr | lstm_classifier | zscore_local | -0.107 | [-0.190, -0.021] | 0 / 0 / 5 |
| detecção | auc_pr | lstm_classifier | lightgbm | +0.030 | [-0.045, +0.105] | 0 / 5 / 0 |
| localização | hr@1 | lstm_sum | zscore_global | +0.015 | [-0.093, +0.127] | 0 / 5 / 0 |
| localização | hr@3 | lstm_sum | zscore_global | +0.097 | [-0.012, +0.205] | 1 / 4 / 0 |
| localização | mrr | lstm_sum | zscore_global | +0.041 | [-0.040, +0.121] | 0 / 5 / 0 |
| localização | hr@1 | lstm_sum | zscore_local | +0.223 | [+0.137, +0.310] | 5 / 0 / 0 |
| localização | hr@3 | lstm_sum | zscore_local | +0.113 | [+0.023, +0.205] | 5 / 0 / 0 |
| localização | mrr | lstm_sum | zscore_local | +0.171 | [+0.111, +0.231] | 5 / 0 / 0 |
| localização | hr@1 | lstm_sum | prior | +0.223 | [+0.123, +0.325] | 5 / 0 / 0 |
| localização | hr@3 | lstm_sum | prior | +0.313 | [+0.193, +0.430] | 5 / 0 / 0 |
| localização | mrr | lstm_sum | prior | +0.254 | [+0.172, +0.335] | 5 / 0 / 0 |
| localização | hr@1 | lstm_top3 | zscore_global | +0.027 | [-0.080, +0.137] | 0 / 5 / 0 |
| localização | hr@3 | lstm_top3 | zscore_global | +0.090 | [-0.008, +0.190] | 1 / 4 / 0 |
| localização | mrr | lstm_top3 | zscore_global | +0.033 | [-0.046, +0.112] | 0 / 5 / 0 |
| localização | hr@1 | lstm_top3 | zscore_local | +0.235 | [+0.150, +0.320] | 5 / 0 / 0 |
| localização | hr@3 | lstm_top3 | zscore_local | +0.107 | [+0.005, +0.210] | 3 / 2 / 0 |
| localização | mrr | lstm_top3 | zscore_local | +0.163 | [+0.098, +0.229] | 5 / 0 / 0 |
| localização | hr@1 | lstm_top3 | prior | +0.235 | [+0.137, +0.335] | 5 / 0 / 0 |
| localização | hr@3 | lstm_top3 | prior | +0.307 | [+0.185, +0.423] | 5 / 0 / 0 |
| localização | mrr | lstm_top3 | prior | +0.246 | [+0.164, +0.328] | 5 / 0 / 0 |
