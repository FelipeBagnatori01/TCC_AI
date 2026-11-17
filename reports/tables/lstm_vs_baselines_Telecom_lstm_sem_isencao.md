# LSTM versus baselines da E4 — Telecom (`lstm_sem_isencao`), diferenças pareadas

Diferença = LSTM menos rival, nas mesmas janelas de teste (detecção) ou nas mesmas falhas (localização); IC de 95% por bootstrap pareado (2000 reamostragens). Médias entre as 5 seeds; a contagem diz em quantas seeds o IC da diferença ficou inteiro acima de 0 (superior), contém 0 (empate) ou abaixo (inferior).

| Tarefa | Métrica | LSTM | Rival | Diferença média | IC médio | Superior / empate / inferior |
|---|---|---|---|---|---|---|
| detecção | auc_pr | lstm_forecast | zscore_local | +0.045 | [-0.064, +0.158] | 0 / 5 / 0 |
| detecção | auc_pr | lstm_forecast | lightgbm | +0.169 | [-0.006, +0.327] | 2 / 3 / 0 |
| detecção | auc_pr | lstm_classifier | zscore_local | +0.105 | [-0.026, +0.240] | 1 / 4 / 0 |
| detecção | auc_pr | lstm_classifier | lightgbm | +0.229 | [+0.096, +0.357] | 5 / 0 / 0 |
| localização | hr@1 | lstm_sum | zscore_global | +0.067 | [-0.071, +0.218] | 0 / 5 / 0 |
| localização | hr@3 | lstm_sum | zscore_global | +0.196 | [-0.013, +0.404] | 1 / 4 / 0 |
| localização | mrr | lstm_sum | zscore_global | +0.103 | [-0.026, +0.236] | 0 / 5 / 0 |
| localização | hr@1 | lstm_sum | zscore_local | +0.111 | [-0.009, +0.236] | 1 / 4 / 0 |
| localização | hr@3 | lstm_sum | zscore_local | +0.085 | [-0.089, +0.253] | 0 / 5 / 0 |
| localização | mrr | lstm_sum | zscore_local | +0.096 | [+0.001, +0.195] | 3 / 2 / 0 |
| localização | hr@1 | lstm_sum | prior | +0.111 | [-0.013, +0.240] | 0 / 5 / 0 |
| localização | hr@3 | lstm_sum | prior | +0.396 | [+0.235, +0.556] | 5 / 0 / 0 |
| localização | mrr | lstm_sum | prior | +0.221 | [+0.117, +0.330] | 5 / 0 / 0 |
| localização | hr@1 | lstm_top3 | zscore_global | +0.004 | [-0.129, +0.138] | 0 / 5 / 0 |
| localização | hr@3 | lstm_top3 | zscore_global | +0.204 | [+0.018, +0.400] | 3 / 2 / 0 |
| localização | mrr | lstm_top3 | zscore_global | +0.082 | [-0.034, +0.201] | 0 / 5 / 0 |
| localização | hr@1 | lstm_top3 | zscore_local | +0.049 | [-0.071, +0.169] | 0 / 5 / 0 |
| localização | hr@3 | lstm_top3 | zscore_local | +0.093 | [-0.067, +0.258] | 0 / 5 / 0 |
| localização | mrr | lstm_top3 | zscore_local | +0.075 | [-0.019, +0.170] | 0 / 5 / 0 |
| localização | hr@1 | lstm_top3 | prior | +0.049 | [-0.080, +0.178] | 0 / 5 / 0 |
| localização | hr@3 | lstm_top3 | prior | +0.404 | [+0.227, +0.578] | 5 / 0 / 0 |
| localização | mrr | lstm_top3 | prior | +0.200 | [+0.089, +0.310] | 5 / 0 / 0 |
