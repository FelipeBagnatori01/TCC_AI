# Resultados

Preenchido a partir da etapa E4. Nenhum número é reportado sem medida de dispersão. IC de 95% por bootstrap (2.000 reamostragens). Avaliação por sistema: **Telecom**, 45 falhas e 45 consultas de maio em 10 dobras de origem móvel (ADR 0008); **Bank**, 120 falhas e 120 consultas em 8 dobras (ADR 0013). Os números não são comparáveis aos do artigo do OpenRCA, que usa todas as consultas e outro protocolo.

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

## Achados sobre os dados do Bank (E1–E3, Plano B)

| Achado | Evidência | Consequência |
|---|---|---|
| Sem dias duplicados | Par mais parecido com 53% de valores iguais (`day_similarity_Bank.csv`) | Nenhum dia excluído |
| Unidade dos traces muda em 23/03 (ms → s) | Inventário; checagem de faixa de datas | Unidade detectada por arquivo (ADR 0011) |
| Motivo não define o tipo de componente | `high CPU usage` em 6 tipos de componente | A pista do Telecom não generaliza (E7) |
| Cadência de amostragem de 1 a 60 min | `fill_limits_Bank.csv` | Preenchimento adaptativo (ADR 0012) |
| **Deriva entre dias** | Pontuação global nos minutos normais: de 85 (03/04) a 760 milhões (03/24) | Régua global satura; método local (ADR 0013) |
| Não há precursor; o sinal surge no início | AUROC local: 0,81 [0,76; 0,85] no início; antes do início, todos os ICs incluem 0,5 | Tarefa A = detecção precoce também no Bank |

## E4 — Baselines (ADR 0010 e 0014)

Configuração principal (isenção de disponibilidade, ADR 0005). Tabelas completas, com resultados por tipo de tarefa e por motivo: `reports/tables/baselines_<sistema>_baselines.md`; controle sem isenção em `baselines_<sistema>_baselines_sem_isencao.md`. O `zscore_local` e a referência de frequência na localização foram acrescentados depois da E3 do Bank (ADR 0014); os números dos demais métodos não mudaram.

### Detecção (RQ1) — por janela

Na AUC-PR, o valor ao acaso é a taxa de positivos: **0,14 no Telecom** e **0,105 no Bank**.

| Sistema | Método | AUC-PR | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|---|
| Telecom | z-score, 3σ fixo | 0,21 [0,14; 0,36] | 0,65 [0,55; 0,75] | 0,14 [0,10; 0,19] | 1,00 [1,00; 1,00] | 0,25 [0,18; 0,31] | 60,0 [60,0; 60,0] |
| Telecom | z-score, limiar da validação | 0,21 [0,14; 0,36] | 0,65 [0,55; 0,75] | 0,19 [0,12; 0,28] | 0,71 [0,58; 0,85] | 0,30 [0,21; 0,40] | 29,8 [23,2; 36,7] |
| Telecom | z-score local | 0,24 [0,15; 0,37] | 0,67 [0,58; 0,76] | 0,18 [0,11; 0,28] | 0,44 [0,30; 0,58] | 0,26 [0,16; 0,36] | 19,3 [14,0; 25,1] |
| Telecom | Isolation Forest | 0,16 [0,11; 0,26] | 0,60 [0,50; 0,70] | 0,17 [0,11; 0,24] | 0,62 [0,48; 0,75] | 0,26 [0,18; 0,35] | 30,4 [23,8; 37,2] |
| Telecom | LightGBM | 0,25 [0,15; 0,41] | 0,59 [0,48; 0,68] | 0,22 [0,14; 0,33] | 0,50 [0,37; 0,64] | 0,31 [0,21; 0,41] | 17,5 [12,2; 23,2] |
| Bank | z-score, 3σ fixo | 0,10 [0,08; 0,13] | 0,48 [0,42; 0,54] | 0,11 [0,09; 0,13] | 1,00 [1,00; 1,00] | 0,19 [0,16; 0,23] | 60,0 [60,0; 60,0] |
| Bank | z-score, limiar da validação | 0,10 [0,08; 0,13] | 0,48 [0,42; 0,54] | 0,12 [0,09; 0,15] | 0,68 [0,60; 0,76] | 0,20 [0,16; 0,25] | 35,8 [32,0; 39,5] |
| Bank | **z-score local** | **0,19 [0,15; 0,24]** | **0,69 [0,64; 0,73]** | 0,19 [0,14; 0,25] | 0,43 [0,35; 0,52] | 0,26 [0,21; 0,33] | **13,1 [10,7; 15,5]** |
| Bank | Isolation Forest | 0,14 [0,11; 0,21] | 0,57 [0,51; 0,63] | 0,14 [0,11; 0,18] | 0,62 [0,54; 0,71] | 0,23 [0,18; 0,28] | 26,9 [23,3; 30,6] |
| Bank | LightGBM | 0,19 [0,13; 0,27] | 0,51 [0,45; 0,58] | 0,15 [0,10; 0,20] | 0,33 [0,25; 0,41] | 0,20 [0,15; 0,26] | 13,3 [10,6; 16,5] |

**Leitura:**
- **Telecom:** todos os baselines ficam pouco acima do acaso; o LightGBM (0,25) e o z-score local (0,24) são os únicos com IC da AUC-PR inteiro acima de 0,14.
- **Bank:** o z-score global fica **no acaso** (AUROC 0,48), como a E3 previu: a distribuição deriva entre dias e o escalonamento fixo envelhece (ADR 0013). O **z-score local** é o melhor baseline: AUC-PR 0,19 (quase o dobro do acaso), AUROC 0,69 e o menor número de alarmes falsos.
- **Revocação por evento** (de 49% a 100% nos dois sistemas): sem valor informativo, porque o nível do acaso para a mesma taxa de alarmes é 1,00 em todos os métodos. O atraso de detecção também não é interpretável.
- **O piso que a LSTM precisa superar:** AUC-PR de 0,25 no Telecom e 0,19 no Bank, sem passar dos alarmes falsos dos melhores baselines (17,5 por hora no Telecom e 13 no Bank).

### Localização (RQ2)

Candidatos: 56 componentes no Telecom e 29 no Bank (inclui os 11 serviços). A referência "frequência no treino" ranqueia pelo número de vezes que cada componente foi causa raiz nas falhas de treino e validação da dobra, sem ler telemetria.

| Sistema | Heurística | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|---|
| Telecom | referência: frequência no treino | 0,09 [0,02; 0,18] | 0,18 [0,09; 0,29] | 0,27 [0,13; 0,40] | 0,19 [0,12; 0,29] |
| Telecom | z-score global (máximo) | 0,13 [0,04; 0,24] | 0,33 [0,20; 0,47] | 0,53 [0,38; 0,69] | 0,30 [0,22; 0,40] |
| Telecom | z-score local (média dos 3 maiores) | 0,11 [0,02; 0,20] | 0,49 [0,33; 0,62] | 0,60 [0,44; 0,73] | 0,34 [0,26; 0,42] |
| Bank | referência: frequência no treino | 0,11 [0,06; 0,17] | 0,27 [0,18; 0,35] | 0,38 [0,30; 0,47] | 0,26 [0,21; 0,31] |
| Bank | **z-score global (máximo)** | **0,32 [0,23; 0,40]** | 0,48 [0,39; 0,57] | **0,68 [0,59; 0,76]** | **0,47 [0,41; 0,54]** |
| Bank | z-score local (média dos 3 maiores) | 0,11 [0,06; 0,17] | 0,46 [0,37; 0,55] | 0,65 [0,57; 0,73] | 0,34 [0,29; 0,39] |

**Leitura:**
- **Telecom:** as heurísticas superam a referência sobretudo em HR@3 e HR@5. A variante local acerta `db close` em 3 de 4 casos (HR@1 0,75), contra 0 da global: o `On_Off_State` só vira sinal quando comparado ao passado recente (ADR 0010).
- **Bank:** aqui o z-score **global** é o melhor, apesar da deriva.
  - Em 50 das 120 falhas ele aponta Mysql01/02 (contadores em deriva) e erra todas.
  - Fora disso, acerta bem (Tomcat01 10/19, apache02 7/8).
  - O z-score local concentra-se em MG01 e IG02, componentes com muitos KPIs voláteis, e não passa da referência em HR@1.
  
  Detecção e localização pedem normalizações diferentes no Bank, o que é relevante para o XAI (E6).

### Diagnóstico (RQ3) — A0 constante, script oficial do OpenRCA

| Sistema | Acerto estrito | Pontuação parcial |
|---|---|---|
| Telecom | 11,1% [2,2; 22,2] (5/45) | 18,5% [9,6; 28,5] |
| Bank | 8,3% [3,3; 13,3] (10/120) | 13,7% [8,6; 19,2] |

- **Telecom:** os acertos são 4 em `task_2` (só o motivo) e 1 em `task_3` (só o componente).
- **Bank:** 5 em `task_2`, 3 em `task_3`, 1 em `task_1` e 1 em `task_4`. As 18 consultas com duas falhas recebem zero automaticamente, porque o A0 dá uma única resposta e o script oficial exige uma por falha.

### Sensibilidade à isenção de disponibilidade (ADR 0005)

| Sistema | Métrica | Com isenção | Sem isenção |
|---|---|---|---|
| Telecom | AUC-PR, z-score / local / IF / LightGBM | 0,21 / 0,24 / 0,16 / 0,25 | 0,21 / 0,24 / 0,18 / 0,26 |
| Telecom | HR@1, z-score global / local | 0,13 / 0,11 | 0,13 / 0,11 |
| Bank | AUC-PR, z-score / local / IF / LightGBM | 0,10 / 0,19 / 0,14 / 0,19 | 0,10 / 0,19 / 0,14 / 0,18 |
| Bank | HR@1, z-score global / local | 0,32 / 0,11 | 0,32 / 0,11 |

Nenhuma conclusão muda em nenhum dos dois sistemas: as diferenças ficam dentro dos ICs (tabelas completas com IC nos arquivos de controle).

### Reavaliação do Plano B (ADR 0009)

Com 120 falhas, os ICs do Bank são cerca de metade da largura dos do Telecom. Exemplos: AUC-PR do melhor baseline com largura de 0,09 contra 0,22–0,26; HR@1 de 0,17 contra 0,20. A comparação da LSTM com os baselines (RQ1 e RQ2) passa a ter poder estatístico razoável no Bank; no Telecom, só diferenças grandes serão significativas.

## E5 — LSTM (ADR 0015)

Configurações escolhidas na validação; 5 seeds por dobra; média ± desvio-padrão entre seeds. A comparação com os baselines é **pareada**: diferença na mesma janela ou falha, com IC por bootstrap em cada seed. "Superior" significa IC da diferença inteiro acima de 0. Tabelas completas: `reports/tables/lstm_<sistema>_<nome>.md` e `lstm_vs_baselines_<sistema>_<nome>.md`; curvas de treino em `reports/figures/lstm_curves_<sistema>_<nome>.png`.

### Detecção (RQ1)

| Sistema | Modelo | AUC-PR | AUC-ROC | Alarmes falsos/h | Vs. z-score local (E4) | Vs. LightGBM (E4) |
|---|---|---|---|---|---|---|
| Telecom | LSTM previsora | 0,40 ± 0,02 | 0,74 ± 0,01 | 18,0 ± 3,6 | +0,16, superior 5/5 | +0,15, empate 4/5 |
| Telecom | **LSTM classificadora** | **0,48 ± 0,02** | **0,83 ± 0,01** | **9,2 ± 1,3** | **+0,24, superior 5/5** | **+0,24, superior 5/5** |
| Bank | LSTM previsora | 0,20 ± 0,00 | 0,70 ± 0,00 | 10,5 ± 1,7 | +0,01, empate 5/5 | +0,01, empate 5/5 |
| Bank | LSTM classificadora | 0,22 ± 0,01 | 0,66 ± 0,01 | 14,3 ± 0,6 | +0,03, empate 5/5 | +0,03, empate 5/5 |

AUC-PR ao acaso: 0,14 (Telecom), 0,105 (Bank). Melhores baselines da E4: 0,25 (Telecom, LightGBM) e 0,19 (Bank, z-score local).

### Localização (RQ2): ranqueamento nativo da LSTM

| Sistema | Ranqueamento | HR@1 | HR@3 | HR@5 | MRR | HR@1 vs. melhor heurística da E4 |
|---|---|---|---|---|---|---|
| Telecom | soma dos erros | **0,44 ± 0,03** | 0,67 ± 0,02 | 0,73 ± 0,01 | 0,57 ± 0,02 | +0,31 sobre o z global, superior 5/5 |
| Telecom | média dos 3 maiores | 0,36 ± 0,03 | 0,72 ± 0,02 | 0,80 ± 0,00 | 0,56 ± 0,01 | +0,23, superior 5/5 |
| Bank | soma dos erros | 0,33 ± 0,01 | 0,60 ± 0,01 | 0,79 ± 0,01 | 0,51 ± 0,00 | +0,01 sobre o z global, empate 5/5 |
| Bank | média dos 3 maiores | 0,33 ± 0,02 | 0,58 ± 0,01 | 0,72 ± 0,01 | 0,49 ± 0,01 | +0,01, empate 5/5 |

No Bank, a LSTM supera o z-score local e a referência de frequência em todas as seeds (HR@1 +0,22), e empata com o z-score global.

### Sensibilidade à isenção de disponibilidade

| Sistema | Métrica | Com isenção | Sem isenção |
|---|---|---|---|
| Telecom | Previsora, AUC-PR | 0,40 ± 0,02 | 0,31 ± 0,02 (empate com os baselines) |
| Telecom | Classificadora, AUC-PR | 0,48 ± 0,02 | 0,49 ± 0,02 (continua superior 5/5) |
| Telecom | Localização (soma), HR@1 | 0,44 ± 0,03 | **0,22 ± 0,02** (empate com as heurísticas) |
| Bank | Previsora / classificadora, AUC-PR | 0,20 / 0,22 | 0,20 / 0,21 |
| Bank | Localização (soma), HR@1 | 0,33 | 0,32 |

**No Telecom, a isenção muda conclusões da LSTM** (ameaça nº 1). A vantagem da classificadora na detecção se mantém sem ela. Já a vantagem da previsora e a da localização dependem dos KPIs de disponibilidade (`On_Off_State`, `ICMP_ping`), que a isenção mantém: sem eles, o HR@1 cai pela metade e vira empate. No Bank, a isenção não faz diferença.

### Leitura

- **Telecom:** a LSTM é claramente melhor que os baselines. A classificadora supervisionada é o melhor detector, com AUC-PR 0,48 e metade dos alarmes falsos do melhor baseline. A localização nativa triplica o HR@1, mas esse ganho depende da isenção.
- **Bank:** a LSTM empata com o melhor baseline na detecção e na localização (HR@1), e só ganha em profundidade (HR@5 0,79 contra 0,68). A deriva entre dias e a maior heterogeneidade limitam o ganho. Note que a normalização local venceu com folga na busca do Bank (0,28 contra 0,19 na validação).
- **Variância entre seeds:** pequena (dp ≤ 0,04).
- **Custo:**
  - previsora com cerca de 1 milhão de parâmetros, 2,4 s (Telecom) a 8,4 s (Bank) de treino por dobra e seed, 0,26 ms de inferência por janela;
  - execução completa (busca + 5 seeds): 9,5 min no Telecom e 24 min no Bank, em CPU.
