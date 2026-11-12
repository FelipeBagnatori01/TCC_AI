# LSTM versus baselines da E4 — Bank (`lstm_sem_isencao`), diferenças pareadas

Diferença = LSTM menos rival, nas mesmas janelas de teste (detecção) ou nas mesmas falhas (localização); IC de 95% por bootstrap pareado (2000 reamostragens). Médias entre as 5 seeds; a contagem diz em quantas seeds o IC da diferença ficou inteiro acima de 0 (superior), contém 0 (empate) ou abaixo (inferior).

| Tarefa | Métrica | LSTM | Rival | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | auc_pr | lstm_forecast | zscore_local | +0.017 | [-0.014, +0.075] | 0 / 5 / 0 |
| detecção | auc_pr | lstm_forecast | lightgbm | +0.029 | [-0.049, +0.115] | 0 / 5 / 0 |
| detecção | auc_pr | lstm_classifier | zscore_local | +0.023 | [-0.039, +0.095] | 0 / 5 / 0 |
| detecção | auc_pr | lstm_classifier | lightgbm | +0.035 | [-0.044, +0.106] | 0 / 5 / 0 |
| localização | hr@1 | lstm_sum | zscore_global | +0.005 | [-0.105, +0.113] | 0 / 5 / 0 |
| localização | hr@3 | lstm_sum | zscore_global | +0.118 | [+0.013, +0.223] | 4 / 1 / 0 |
| localização | mrr | lstm_sum | zscore_global | +0.036 | [-0.043, +0.113] | 0 / 5 / 0 |
| localização | hr@1 | lstm_sum | zscore_local | +0.213 | [+0.128, +0.298] | 5 / 0 / 0 |
| localização | hr@3 | lstm_sum | zscore_local | +0.143 | [+0.055, +0.233] | 5 / 0 / 0 |
| localização | mrr | lstm_sum | zscore_local | +0.169 | [+0.109, +0.230] | 5 / 0 / 0 |
| localização | hr@1 | lstm_sum | prior | +0.213 | [+0.113, +0.315] | 5 / 0 / 0 |
| localização | hr@3 | lstm_sum | prior | +0.335 | [+0.220, +0.447] | 5 / 0 / 0 |
| localização | mrr | lstm_sum | prior | +0.248 | [+0.168, +0.330] | 5 / 0 / 0 |
| localização | hr@1 | lstm_top3 | zscore_global | +0.005 | [-0.103, +0.112] | 0 / 5 / 0 |
| localização | hr@3 | lstm_top3 | zscore_global | +0.090 | [-0.005, +0.187] | 2 / 3 / 0 |
| localização | mrr | lstm_top3 | zscore_global | +0.021 | [-0.058, +0.099] | 0 / 5 / 0 |
| localização | hr@1 | lstm_top3 | zscore_local | +0.213 | [+0.128, +0.300] | 5 / 0 / 0 |
| localização | hr@3 | lstm_top3 | zscore_local | +0.115 | [+0.010, +0.220] | 3 / 2 / 0 |
| localização | mrr | lstm_top3 | zscore_local | +0.154 | [+0.089, +0.221] | 5 / 0 / 0 |
| localização | hr@1 | lstm_top3 | prior | +0.213 | [+0.113, +0.315] | 5 / 0 / 0 |
| localização | hr@3 | lstm_top3 | prior | +0.307 | [+0.185, +0.425] | 5 / 0 / 0 |
| localização | mrr | lstm_top3 | prior | +0.233 | [+0.153, +0.314] | 5 / 0 / 0 |
