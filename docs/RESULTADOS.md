# Resultados

Preenchido a partir da etapa E4. Nenhum número é reportado sem medida de dispersão. IC de 95% por bootstrap (2.000 reamostragens). Avaliação nas **45 falhas e 45 consultas de maio** do Telecom, em 10 dobras de origem móvel (ADR 0008). Os números não são comparáveis aos do artigo do OpenRCA, que usa as 51 consultas e outro protocolo.

## Achados sobre os dados (E1–E3)

Análise visual completa, com gráficos e tabelas: `notebooks/analise_telecom_e1_e4.ipynb`.

| Achado | Evidência | Consequência |
|---|---|---|
| Três dias de abril são cópias sem rótulo de dias de maio com falhas | 99,997% (04/21–05/31), 99,995% (04/22–05/23) e 99,999% (04/23–05/25) dos valores de métricas iguais; os demais pares ficam entre 40% e 48%. Em 04/22 vs 05/23, as diferenças caem logo após **todas as seis** falhas de 05/23 | Os três dias são excluídos (ADR 0008); mantê-los vazaria telemetria de teste para o treino |
| O dia mais antigo (04/11) tem 6 falhas e nada antes dele | Divisão cronológica | Fica no bloco inicial; avaliação nas 45 falhas de maio |
| Não há sinal precursor | AUROC antes do início: todos os ICs incluem 0,5; +1 min após o início: 0,75 [0,67; 0,81] | Tarefa A = detecção precoce (ADR 0007) |
| Visibilidade depende do motivo | AUROC em +1 min: `db close` 0,94, `network delay` 0,86, `CPU fault` 0,72; `db connection limit` e `network loss` com IC incluindo 0,5 | São os casos difíceis para a LSTM e o XAI |
| Classes desbalanceadas | `CPU fault` 19/51; 15 componentes, vários com 1–2 falhas; cada motivo ocorre num único tipo de componente: `CPU fault` só em `docker_*`, `network delay/loss` só em `os_*`, `db close/connection limit` só em `db_*` | O tipo do componente restringe o motivo a no máximo duas opções (útil ao agente, E7); padrão deste conjunto, não regra geral |
| O fim das falhas não é rotulado | Só o início em `record.csv` | Janelas "ignoradas" e margem de 30 min (ameaça nº 6) |

## E4 — Baselines (ADR 0010)

Configuração principal (isenção de disponibilidade, ADR 0005). Tabelas completas, incluindo os resultados por tipo de tarefa e por motivo: `reports/tables/baselines_Telecom_baselines.md` e, para o controle sem isenção, `reports/tables/baselines_Telecom_baselines_sem_isencao.md`.

### Detecção (RQ1) — por janela

Na AUC-PR, o valor ao acaso é a taxa de positivos: **0,14**.

| Método | AUC-PR | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|
| z-score, 3σ fixo | 0,21 [0,14; 0,36] | 0,65 [0,55; 0,75] | 0,14 [0,10; 0,19] | 1,00 [1,00; 1,00] | 0,25 [0,18; 0,31] | 60,0 [60,0; 60,0] |
| z-score, limiar da validação | 0,21 [0,14; 0,36] | 0,65 [0,55; 0,75] | 0,19 [0,12; 0,28] | 0,71 [0,58; 0,85] | 0,30 [0,21; 0,40] | 29,8 [23,2; 36,7] |
| Isolation Forest | 0,16 [0,11; 0,26] | 0,60 [0,50; 0,70] | 0,17 [0,11; 0,24] | 0,62 [0,48; 0,75] | 0,26 [0,18; 0,35] | 30,4 [23,8; 37,2] |
| LightGBM | 0,25 [0,15; 0,41] | 0,59 [0,48; 0,68] | 0,22 [0,14; 0,33] | 0,50 [0,37; 0,64] | 0,31 [0,21; 0,41] | 17,5 [12,2; 23,2] |

**Leitura:**
- Todos os baselines ficam pouco acima do acaso. Os ICs da AUC-PR encostam em 0,14, exceto o do LightGBM, que começa em 0,15.
- O 3σ fixo alarma em todas as janelas: o máximo de |z| entre cerca de 1.500 KPIs sempre passa de 3.
- **A revocação por evento (64–100%) não tem valor informativo.** Com 17 a 60 alarmes falsos por hora, alarmes aleatórios com a mesma taxa cairiam em 100% dos intervalos de 36 min de busca (coluna "ao acaso" das tabelas completas). Por isso o atraso de detecção destes baselines também não é interpretável.
- O piso real que a LSTM precisa superar é o das métricas **por janela**, em especial a AUC-PR (0,25 com o LightGBM), e com **menos alarmes falsos por hora**.

### Localização (RQ2) — heurística |z|

Sobre 56 componentes candidatos.

| HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|
| 0,13 [0,04; 0,24] | 0,33 [0,20; 0,47] | 0,53 [0,38; 0,69] | 0,30 [0,22; 0,40] |

| Motivo | Falhas | HR@1 |
|---|---|---|
| CPU fault | 15 | 0,13 |
| db close | 4 | 0,00 |
| db connection limit | 6 | 0,33 |
| network delay | 13 | 0,15 |
| network loss | 7 | 0,00 |

`db close` tem HR@1 = 0 mesmo com `On_Off_State` disponível. O KPI de disponibilidade é constante no treino, então seu desvio de referência vira 1 e o |z| dele vale no máximo 1. Já KPIs com desvio de treino minúsculo produzem |z| na casa dos milhares e vencem o máximo. É uma limitação do z clássico, que o XAI da E6 deve superar. Isso não foi corrigido no baseline, porque a correção seria escolhida olhando o teste.

### Diagnóstico (RQ3) — A0 constante, script oficial do OpenRCA

| Acerto estrito | Pontuação parcial |
|---|---|
| 11,1% [2,2; 22,2] (5/45) | 18,5% [9,6; 28,5] |

Os 5 acertos estritos são 4 `task_2` (só o motivo; `CPU fault` é o motivo mais frequente) e 1 `task_3` (só o componente).

### Sensibilidade à isenção de disponibilidade (ADR 0005)

| Métrica | Com isenção | Sem isenção |
|---|---|---|
| AUC-PR, z-score | 0,21 [0,14; 0,36] | 0,21 [0,14; 0,36] |
| AUC-PR, Isolation Forest | 0,16 [0,11; 0,26] | 0,18 [0,12; 0,32] |
| AUC-PR, LightGBM | 0,25 [0,15; 0,41] | 0,26 [0,15; 0,41] |
| HR@1, heurística \|z\| | 0,13 [0,04; 0,24] | 0,13 [0,04; 0,24] |
| Acerto estrito, A0 | 11,1% | 11,1% (não lê telemetria) |

A isenção não altera o z-score nem a localização, pelo motivo explicado acima. Nos modelos aprendidos, as diferenças ficam dentro dos ICs. Até aqui, a ameaça à validade da isenção não afeta nenhuma conclusão.
