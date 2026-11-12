# LSTM versus baselines da E4 — Telecom (`lstm_sem_isencao`), diferenças pareadas

Diferença = LSTM menos rival, nas mesmas janelas de teste (detecção) ou nas mesmas falhas (localização); IC de 95% por bootstrap pareado (2000 reamostragens). Médias entre as 5 seeds; a contagem diz em quantas seeds o IC da diferença ficou inteiro acima de 0 (superior), contém 0 (empate) ou abaixo (inferior).

| Tarefa | Métrica | LSTM | Rival | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | auc_pr | lstm_forecast | zscore_local | +0.076 | [-0.022, +0.204] | 1 / 4 / 0 |
| detecção | auc_pr | lstm_forecast | lightgbm | +0.050 | [-0.115, +0.226] | 0 / 5 / 0 |
| detecção | auc_pr | lstm_classifier | zscore_local | +0.255 | [+0.134, +0.381] | 5 / 0 / 0 |
| detecção | auc_pr | lstm_classifier | lightgbm | +0.229 | [+0.096, +0.357] | 5 / 0 / 0 |
| localização | hr@1 | lstm_sum | zscore_global | +0.085 | [-0.067, +0.240] | 0 / 5 / 0 |
| localização | hr@3 | lstm_sum | zscore_global | +0.240 | [+0.035, +0.444] | 4 / 1 / 0 |
| localização | mrr | lstm_sum | zscore_global | +0.125 | [-0.006, +0.260] | 1 / 4 / 0 |
| localização | hr@1 | lstm_sum | zscore_local | +0.107 | [-0.013, +0.231] | 0 / 5 / 0 |
| localização | hr@3 | lstm_sum | zscore_local | +0.085 | [-0.089, +0.253] | 0 / 5 / 0 |
| localização | mrr | lstm_sum | zscore_local | +0.092 | [-0.003, +0.190] | 2 / 3 / 0 |
| localização | hr@1 | lstm_sum | prior | +0.129 | [-0.004, +0.267] | 1 / 4 / 0 |
| localização | hr@3 | lstm_sum | prior | +0.396 | [+0.235, +0.556] | 5 / 0 / 0 |
| localização | mrr | lstm_sum | prior | +0.232 | [+0.125, +0.345] | 5 / 0 / 0 |
| localização | hr@1 | lstm_top3 | zscore_global | +0.022 | [-0.111, +0.156] | 0 / 5 / 0 |
| localização | hr@3 | lstm_top3 | zscore_global | +0.253 | [+0.071, +0.440] | 5 / 0 / 0 |
| localização | mrr | lstm_top3 | zscore_global | +0.104 | [-0.015, +0.225] | 1 / 4 / 0 |
| localização | hr@1 | lstm_top3 | zscore_local | +0.044 | [-0.076, +0.164] | 0 / 5 / 0 |
| localização | hr@3 | lstm_top3 | zscore_local | +0.098 | [-0.062, +0.262] | 0 / 5 / 0 |
| localização | mrr | lstm_top3 | zscore_local | +0.071 | [-0.025, +0.166] | 0 / 5 / 0 |
| localização | hr@1 | lstm_top3 | prior | +0.067 | [-0.067, +0.200] | 0 / 5 / 0 |
| localização | hr@3 | lstm_top3 | prior | +0.409 | [+0.231, +0.582] | 5 / 0 / 0 |
| localização | mrr | lstm_top3 | prior | +0.211 | [+0.096, +0.324] | 5 / 0 / 0 |
