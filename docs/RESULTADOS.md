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

## E4 — Baselines (ADR 0010, 0014 e 0019)

Configuração principal (isenção de disponibilidade, ADR 0005). Tabelas completas, com resultados por motivo: `reports/tables/baselines_<sistema>_baselines.md`. Controles:
- `baselines_<sistema>_baselines_sem_isencao.md`: sem a isenção;
- `baselines_<sistema>_baselines_janela_inteira.md`: pontuação na janela inteira, o protocolo anterior ao ADR 0019. Seus números de detecção são idênticos aos das tabelas anteriores (verificado nos dois sistemas).

O que mudou com o ADR 0019:
- **Pontuação:** os z-scores pontuam só os últimos D + 1 = 6 minutos da janela, o trecho a que o rótulo se refere.
- **Janelas-oráculo:** a janela da localização não cruza mais o dia. Ela foi deslocada para a primeira janela completa do dia em 3 falhas do Telecom e 2 do Bank. Essas mesmas falhas não têm nenhuma janela positiva, então contam na localização, mas não podem ser detectadas por janela.
- **Referência nova:** `prior_horario`, um detector sem telemetria que pontua a janela pela fração de positivos no mesmo horário dos dias de treino.

### Detecção (RQ1) — por janela

O valor ao acaso é a taxa de positivos: **0,14 no Telecom** e **0,105 no Bank** na AUC-PR agrupada. Na média por dobra, o acaso é a média das taxas por dobra: 0,16 no Telecom e 0,11 no Bank.

| Sistema | Método | AUC-PR | AUC-PR por dobra | AUC-ROC | Precisão | Revocação | F1 | Alarmes falsos/h |
|---|---|---|---|---|---|---|---|---|
| Telecom | z-score, 3σ fixo | 0,24 [0,16; 0,38] | 0,43 [0,38; 0,60] | 0,70 [0,61; 0,79] | 0,14 | 1,00 | 0,25 | 60,0 |
| Telecom | z-score, limiar da validação | 0,24 [0,16; 0,38] | 0,43 [0,38; 0,60] | 0,70 [0,61; 0,79] | 0,26 | 0,65 | 0,37 | 18,2 [12,3; 24,2] |
| Telecom | **z-score local** | **0,39 [0,27; 0,54]** | 0,49 [0,42; 0,69] | **0,76 [0,68; 0,83]** | 0,30 | 0,41 | 0,35 | **9,3 [6,6; 12,4]** |
| Telecom | Isolation Forest | 0,16 [0,11; 0,26] | 0,36 [0,29; 0,54] | 0,60 [0,50; 0,70] | 0,17 | 0,62 | 0,26 | 30,4 [23,8; 37,2] |
| Telecom | LightGBM | 0,25 [0,15; 0,41] | 0,56 [0,50; 0,70] | 0,59 [0,48; 0,68] | 0,22 | 0,50 | 0,31 | 17,5 [12,2; 23,2] |
| Telecom | referência: só horário | 0,15 [0,09; 0,27] | 0,28 [0,21; 0,47] | 0,44 [0,34; 0,55] | 0,15 | 0,48 | 0,22 | 27,2 [20,7; 33,6] |
| Bank | z-score, 3σ fixo | 0,10 [0,08; 0,13] | 0,31 [0,26; 0,37] | 0,48 [0,42; 0,54] | 0,11 | 1,00 | 0,19 | 60,0 |
| Bank | z-score, limiar da validação | 0,10 [0,08; 0,13] | 0,31 [0,26; 0,37] | 0,48 [0,42; 0,54] | 0,12 | 0,68 | 0,20 | 35,1 [31,2; 38,9] |
| Bank | **z-score local** | **0,33 [0,26; 0,41]** | **0,35 [0,30; 0,45]** | **0,79 [0,75; 0,83]** | 0,37 | 0,40 | 0,39 | **4,8 [3,9; 5,8]** |
| Bank | Isolation Forest | 0,14 [0,11; 0,21] | 0,21 [0,17; 0,30] | 0,57 [0,51; 0,63] | 0,14 | 0,62 | 0,23 | 26,9 [23,3; 30,6] |
| Bank | LightGBM | 0,19 [0,13; 0,27] | 0,27 [0,21; 0,35] | 0,51 [0,45; 0,58] | 0,15 | 0,33 | 0,20 | 13,3 [10,6; 16,5] |
| Bank | referência: só horário | 0,12 [0,10; 0,16] | 0,18 [0,15; 0,27] | 0,55 [0,49; 0,61] | 0,15 | 0,44 | 0,22 | 17,6 [14,4; 21,2] |

**Leitura:**
- **z-score local:** com a pontuação alinhada ao rótulo, é o melhor baseline nos dois sistemas. No Telecom, a AUC-PR sobe de 0,24 para 0,39; no Bank, de 0,19 para 0,33. Os alarmes falsos caem pela metade ou mais.
- **z-score global:** no Bank continua no acaso pela AUC-PR agrupada (AUROC 0,48): a distribuição deriva entre dias e o escalonamento fixo envelhece (ADR 0013).
- **AUC-PR agrupada × por dobra:** a média por dobra (0,31) fica bem acima da agrupada (0,10) para o z-score global no Bank. A escala dele muda de dobra para dobra, e juntar as pontuações mistura essas escalas (ameaça nº 7). Por isso as duas medidas são relatadas lado a lado.
- **Referência só de horário:** no Telecom fica no acaso. No Bank fica um pouco acima dele (0,12 contra 0,105 na AUC-PR; AUROC 0,55), o que indica um atalho modesto pelo horário (ameaça nº 22). O Isolation Forest da E4 fica no mesmo nível dela nos dois sistemas.
- **Revocação por evento:** sem valor informativo, porque o acaso para a mesma taxa de alarmes fica entre 0,95 e 1,00 em todos os métodos.
- **O piso que a LSTM precisa superar:** o z-score local, com AUC-PR 0,39 no Telecom e 0,33 no Bank, e 9 e 5 alarmes falsos por hora.

### Localização (RQ2)

Candidatos: 56 componentes no Telecom e 29 no Bank (inclui os 11 serviços). A referência "frequência no treino" ranqueia pelo número de vezes que cada componente foi causa raiz nas falhas de treino e validação da dobra, sem ler telemetria. As heurísticas de localização não mudaram com o ADR 0019; só as janelas-oráculo foram corrigidas.

| Sistema | Heurística | HR@1 | HR@3 | HR@5 | MRR |
|---|---|---|---|---|---|
| Telecom | referência: frequência no treino | 0,09 [0,02; 0,18] | 0,18 [0,09; 0,29] | 0,27 [0,13; 0,40] | 0,19 [0,12; 0,29] |
| Telecom | z-score global (máximo) | 0,13 [0,04; 0,24] | 0,38 [0,24; 0,51] | 0,56 [0,40; 0,71] | 0,31 [0,22; 0,41] |
| Telecom | z-score local (média dos 3 maiores) | 0,09 [0,02; 0,18] | 0,49 [0,33; 0,62] | 0,60 [0,44; 0,73] | 0,33 [0,26; 0,41] |
| Bank | referência: frequência no treino | 0,11 [0,06; 0,17] | 0,27 [0,18; 0,35] | 0,38 [0,30; 0,47] | 0,26 [0,21; 0,31] |
| Bank | **z-score global (máximo)** | **0,32 [0,23; 0,40]** | 0,48 [0,39; 0,57] | **0,68 [0,59; 0,76]** | **0,47 [0,41; 0,54]** |
| Bank | z-score local (média dos 3 maiores) | 0,11 [0,06; 0,17] | 0,47 [0,38; 0,56] | 0,66 [0,57; 0,74] | 0,34 [0,29; 0,39] |

**Leitura:**
- **Telecom:** as heurísticas superam a referência sobretudo em HR@3 e HR@5. A variante local acerta `db close` em 2 de 4 casos, contra 0 da global: o `On_Off_State` só vira sinal quando comparado ao passado recente (ADR 0010).
- **Bank:** aqui o z-score **global** é o melhor, apesar da deriva.
  - Em 50 das 120 falhas ele aponta Mysql01/02 (contadores em deriva) e erra todas.
  - Fora disso, acerta bem (Tomcat01 10/19, apache02 7/10).
  - O z-score local concentra-se em MG01 e IG02, componentes com muitos KPIs voláteis, e não passa da referência em HR@1.

  Detecção e localização pedem normalizações diferentes no Bank.

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
| Telecom | AUC-PR, z-score / local / IF / LightGBM | 0,24 / 0,39 / 0,16 / 0,25 | 0,24 / 0,38 / 0,18 / 0,26 |
| Telecom | HR@1, z-score global / local | 0,13 / 0,09 | 0,13 / 0,09 |
| Bank | AUC-PR, z-score / local / IF / LightGBM | 0,10 / 0,33 / 0,14 / 0,19 | 0,10 / 0,33 / 0,14 / 0,18 |
| Bank | HR@1, z-score global / local | 0,32 / 0,11 | 0,32 / 0,11 |

Nenhuma conclusão muda em nenhum dos dois sistemas: as diferenças ficam dentro dos ICs.

### Reavaliação do Plano B (ADR 0009)

Com 120 falhas, os ICs do Bank são cerca de metade da largura dos do Telecom. Exemplos: AUC-PR do melhor baseline com largura de 0,15 contra 0,27; HR@1 de 0,17 contra 0,20. A comparação da LSTM com os baselines (RQ1 e RQ2) tem poder estatístico razoável no Bank; no Telecom, só diferenças grandes são significativas.

## E5 — LSTM (ADR 0015 e 0019)

Configurações escolhidas na validação; 5 seeds por dobra; média ± desvio-padrão entre seeds. A comparação com os baselines é **pareada**: diferença na mesma janela ou falha, com IC por bootstrap em cada seed. "Superior" significa IC da diferença inteiro acima de 0; a contagem é de seeds (superior / empate / inferior). As cinco seeds compartilham as mesmas janelas de teste, então "5/5" mostra que o resultado não depende da seed, e não equivale a cinco confirmações independentes.

Tabelas completas: `reports/tables/lstm_<sistema>_<nome>.md` e `lstm_vs_baselines_<sistema>_<nome>.md`. Curvas de treino: `reports/figures/lstm_curves_<sistema>_<nome>.png`.

Com o ADR 0019, a previsora pontua só os últimos 6 passos de erro, e a busca foi refeita.
- **Telecom:** escolheu a mesma configuração (local, 128 unidades, 1 camada).
- **Bank:** passou de 2 camadas para 1.
- **Classificadora:** é a mesma de antes, em configuração e números.

### Detecção (RQ1)

| Sistema | Modelo | AUC-PR | AUC-PR por dobra | AUC-ROC | Alarmes falsos/h | Vs. z-score local (E4) | Vs. LightGBM (E4) |
|---|---|---|---|---|---|---|---|
| Telecom | **LSTM previsora** | **0,52 ± 0,02** | 0,56 ± 0,01 | 0,81 ± 0,01 | 11,3 ± 1,0 | **+0,13, superior 4/5** | +0,27, superior 5/5 |
| Telecom | LSTM classificadora | 0,48 ± 0,02 | **0,60 ± 0,01** | **0,83 ± 0,01** | **9,2 ± 1,3** | +0,09, empate 5/5 | +0,24, superior 5/5 |
| Bank | **LSTM previsora** | **0,35 ± 0,00** | **0,38 ± 0,01** | 0,78 ± 0,00 | 5,7 ± 0,4 | +0,02, empate 5/5 | +0,16, superior 5/5 |
| Bank | LSTM classificadora | 0,22 ± 0,01 | 0,26 ± 0,01 | 0,66 ± 0,01 | 14,3 ± 0,6 | **−0,11, inferior 5/5** | +0,03, empate 5/5 |

AUC-PR ao acaso: 0,14 (Telecom), 0,105 (Bank). Melhor baseline da E4 nos dois sistemas: z-score local (0,39 e 0,33).

**Antes do ADR 0019** (controle `lstm_janela_inteira`), a previsora tinha AUC-PR 0,40 no Telecom e 0,20 no Bank. Nessa versão, a classificadora parecia superior ao z-score local no Telecom (+0,24, 5/5), mas só porque o z-score local também pontuava a janela inteira.

### Localização (RQ2): ranqueamento nativo da LSTM

| Sistema | Ranqueamento | HR@1 | HR@3 | HR@5 | MRR | HR@1 vs. heurísticas da E4 |
|---|---|---|---|---|---|---|
| Telecom | soma dos erros | **0,42 ± 0,04** | 0,67 ± 0,02 | 0,73 ± 0,01 | 0,56 ± 0,02 | superior 5/5 às três (+0,29 a +0,33) |
| Telecom | média dos 3 maiores | 0,34 ± 0,03 | 0,71 ± 0,02 | 0,80 ± 0,00 | 0,54 ± 0,01 | superior 5/5 às três |
| Bank | soma dos erros | 0,33 ± 0,01 | 0,58 ± 0,01 | 0,80 ± 0,01 | 0,51 ± 0,00 | empate com o z global; +0,22, superior 5/5 ao z local e à referência |
| Bank | média dos 3 maiores | 0,34 ± 0,01 | 0,57 ± 0,01 | 0,74 ± 0,01 | 0,50 ± 0,01 | idem |

O ranqueamento nativo não mudou com o ADR 0019. A pequena queda no Telecom (HR@1 de 0,44 para 0,42) vem das 3 janelas-oráculo corrigidas.

### Sensibilidade à isenção de disponibilidade

| Sistema | Métrica | Com isenção | Sem isenção |
|---|---|---|---|
| Telecom | Previsora, AUC-PR | 0,52 ± 0,02 | 0,43 ± 0,02 (empate com o z-score local 5/5) |
| Telecom | Classificadora, AUC-PR | 0,48 ± 0,02 | 0,49 ± 0,02 (empate com o z-score local em 4/5) |
| Telecom | Localização (soma), HR@1 | 0,42 ± 0,04 | **0,20 ± 0,02** (empate com as heurísticas em HR@1 na maioria das seeds; MRR ainda superior à referência de frequência) |
| Bank | Previsora / classificadora, AUC-PR | 0,35 / 0,22 | 0,34 / 0,21 |
| Bank | Localização (soma), HR@1 | 0,33 | 0,33 |

**No Telecom, a isenção muda conclusões da LSTM** (ameaça nº 1): sem os KPIs de disponibilidade, a vantagem da previsora na detecção e a da localização desaparecem. No Bank, a isenção não faz diferença.

### Leitura

- **Telecom:** a previsora LSTM é o único detector que supera o melhor baseline (z-score local), e só em 4 das 5 seeds, por +0,13. A classificadora empata com ele. A localização nativa da LSTM triplica o HR@1 das heurísticas, mas esse ganho depende da isenção.
- **Bank:** as LSTMs **não** superam o z-score local na detecção: a previsora empata e a classificadora perde. Na localização, a LSTM empata com o z-score global em HR@1 e ganha em profundidade (HR@5 0,80 contra 0,68).
- **Previsora × classificadora:** com a pontuação alinhada ao rótulo, a previsora, treinada só com janelas normais, passa a ser o melhor detector LSTM nos dois sistemas pela AUC-PR agrupada. A classificadora continua à frente pela média por dobra e pelos alarmes falsos no Telecom.
- **Variância entre seeds:** pequena (dp ≤ 0,04).
- **Custo** (CPU, Apple M4 Pro):
  - previsora com 0,9 a 1 milhão de parâmetros: 2,4 s (Telecom) a 5,6 s (Bank) de treino por dobra e seed; 0,23 a 0,27 ms de inferência por janela;
  - execução completa (busca + 5 seeds): cerca de 10 min no Telecom e 22 min no Bank.

## E6 — Explicabilidade (ADR 0016 e 0019)

SHAP (GradientExplainer), LIME por série `componente::kpi` e IG (captum) sobre as LSTMs da E5, seed 0.
- **Medição:** usa janelas-oráculo, que terminam 5 min após o início de cada falha testada, ou são a primeira janela completa do dia quando a falha começa no início do dia (ADR 0019).
- **Pacotes de evidência:** usam a janela de maior pontuação do detector dentro da consulta.
- **IC e vereditos:** IC de 95% por bootstrap sobre falhas; "superior", "inferior" e "empate" vêm da diferença pareada.
- **O que mudou com o ADR 0019:** a função explicada da previsora passou a ler só os últimos 6 minutos. Isso concentrou as atribuições no trecho da falha.

Tabelas completas: `reports/tables/xai_<sistema>_<nome>.md`.

### Localização via XAI (RQ2), agregação por soma

| Sistema | Explicação | HR@1 | HR@3 | MRR | Vs. nativo da E5 (HR@1 / MRR) | Vs. z global (HR@1) | Vs. z local (HR@1) |
|---|---|---|---|---|---|---|---|
| Telecom | **previsora / SHAP** | **0,42** | **0,78** | **0,61** | empate / empate | superior | superior |
| Telecom | **previsora / LIME** | **0,44** | 0,76 | 0,60 | empate / empate | superior | superior |
| Telecom | classificadora / SHAP | 0,27 | 0,51 | 0,42 | empate / inferior | empate | superior |
| Telecom | classificadora / LIME | 0,13 | 0,33 | 0,29 | inferior / inferior | empate | empate |
| Bank | **previsora / SHAP** | **0,48** | 0,66 | **0,61** | **superior / superior** | **superior** | superior |
| Bank | **previsora / LIME** | **0,45** | 0,64 | 0,59 | **superior / superior** | **superior** | superior |
| Bank | classificadora / SHAP | 0,44 | 0,73 | 0,61 | empate / superior | superior | superior |
| Bank | classificadora / LIME | 0,22 | 0,52 | 0,42 | inferior / inferior | empate | superior |

Valores de referência (seed 0): nativo da E5, HR@1 0,42 no Telecom e 0,34 no Bank; z global, 0,13 e 0,32.

Antes do ADR 0019, a previsora tinha HR@1 0,33 (SHAP) e 0,38 (LIME) no Telecom, e 0,36 e 0,30 no Bank. A mudança veio da função explicada: ao ler só o fim da janela, ela deixa de atribuir peso a desvios anteriores à falha.

**Sem a isenção de disponibilidade:**
- **Telecom:** a previsora cai para HR@1 0,24 (SHAP) e 0,22 (LIME). Ela ainda é superior ao z local em HR@1 com SHAP, e superior em MRR ao z global (SHAP) e ao z local (SHAP e LIME). Ganho menor, mas não desaparece (ameaça nº 1).
- **Bank:** nada muda. A previsora com SHAP tem HR@1 0,50 e continua superior a todas as referências.

### Qualidade das explicações (médias por janela; previsora / classificadora)

| Medida | Telecom | Bank |
|---|---|---|
| Deleção SHAP (menor = mais fiel) | 0,54 / 0,76 | 0,62 / 0,70 |
| Deleção LIME | 0,53 / 0,76 | 0,62 / 0,71 |
| Deleção em ordem aleatória | 0,99 / 1,00 | 0,99 / 1,00 |
| Inserção SHAP (maior = mais fiel) | 0,69 / 0,81 | 0,66 / — |
| Inserção LIME | 0,67 / 0,87 | 0,65 / — |
| Inserção em ordem aleatória | 0,04 / 0,33 | 0,06 / — |
| SHAP × LIME, Jaccard top-10 | 0,73 / 0,45 | 0,76 / 0,48 |
| SHAP × LIME, Kendall top-10 | 0,74 / 0,17 | 0,71 / 0,25 |
| SHAP × LIME, mesmo componente no topo | 0,98 / 0,58 | 0,79 / 0,56 |
| IG × SHAP, Jaccard top-10 | 0,90 / 0,64 | 0,83 / 0,51 |
| Janelas vizinhas (±1 min), Jaccard | 0,75 / 0,45 | 0,70 / 0,54 |
| Entre seeds, Jaccard top-10 | **0,85 / 0,20** | **0,86 / 0,26** |
| Entre seeds, Kendall top-10 | 0,79 / −0,26 | 0,80 / −0,14 |

- **Fidelidade:** SHAP e LIME apontam atributos que realmente movem a saída. Retirando-os aos poucos (k = 1 a 50), a pontuação fica em média em 53% a 76% da original; em ordem aleatória, praticamente não muda (99% a 100%).
- **Inserção da classificadora no Bank (sem valor na tabela):** a métrica não serve aqui. A classificadora dá à janela de referência "normal" uma probabilidade maior que a de metade das janelas de falha, e a curva de inserção, relativa à janela completa, passa de 1. A deleção não tem esse problema.
- **Estabilidade e concordância:** as explicações da previsora ficaram ainda mais estáveis entre seeds (Jaccard de 0,81–0,82 para 0,85–0,86) e mais concordantes entre SHAP e LIME. A estabilidade entre minutos vizinhos caiu um pouco (0,86–0,89 para 0,70–0,75): ao ler só os últimos minutos, a explicação acompanha mais o que muda de um minuto para o outro. A classificadora segue instável entre seeds (efeito Rashomon, ameaça nº 12), e nela SHAP e LIME divergem.
- **Divergência SHAP × LIME:** pequena na previsora e grande na classificadora. As causas prováveis:
  - os dois métodos medem coisas diferentes (gradientes contra um fundo, no SHAP; troca de séries inteiras pela referência e ajuste linear, no LIME);
  - a probabilidade da classificadora satura e interage entre atributos;
  - a referência normal não é neutra para a classificadora no Bank.

### Categoria de motivo sugerida

| Sistema | Previsora SHAP | Previsora LIME | Classificadora SHAP | Classificadora LIME | Prior de treino | Uniforme |
|---|---|---|---|---|---|---|
| Telecom | 0,09 | 0,07 | 0,44 | 0,22 | 0,33 | 0,17 |
| Bank | **0,54** | 0,26 | 0,33 | 0,19 | 0,38 | 0,16 |

Comparação pareada com o prior de treino (diferença no acerto, mesmas falhas):

| Sistema | Previsora SHAP | Previsora LIME | Classificadora SHAP | Classificadora LIME |
|---|---|---|---|---|
| Telecom | −0,24, inferior | −0,27, inferior | +0,11, empate | −0,11, empate |
| Bank | **+0,17, superior** | −0,12, empate | −0,05, empate | −0,18, inferior |

- **Bank:** com a pontuação nos últimos minutos, a categoria sugerida pelo SHAP da previsora passa a superar o prior de treino. É o primeiro caso em que isso acontece.
- **Telecom:** a mesma sugestão continua inferior ao prior.

A soma por categoria favorece categorias com muitos KPIs (ameaça nº 13). Como o resultado não se repete no Telecom, a categoria sugerida continua fora do que o agente vê (ADR 0017). O agente deve raciocinar sobre os KPIs.

### Pacotes de evidência

São 45 pacotes no Telecom e 120 no Bank, um por consulta testada, regerados com o ADR 0019.
- **Montagem (desde o ADR 0021, 2026-11-19):** a previsora escolhe a janela (maior pontuação dentro da consulta), dá pontuação e alarme e fornece as atribuições citáveis (`C#`, `K#`) e o ranqueamento nativo (`N#`). O alarme dispara em 64% das consultas no Telecom (29/45) e em 52% no Bank (62/120). Conferido: em todos os 165 pacotes, a pontuação é a da E5 (seed 0) e é a maior entre as janelas candidatas da consulta. Até o ADR 0021, a classificadora escolhia a janela e dava o alarme (76% e 67% das consultas).
- **Ausência de rótulos:**
  - nenhum pacote tem campo de causa, motivo ou rótulo: o esquema rejeita campos extras, e há teste de importação;
  - a janela é escolhida pelo detector, nunca pelo início rotulado.
- **Rastreabilidade (conferida em um exemplo):** na consulta de 2020-05-22 00:00–00:30, o KPI `db_004::Call_Per_Sec` tem valor bruto 264,8 às 00:29 (UTC+8), contra mediana normal de 14,2. Na telemetria convertida, o registro é exatamente esse: 264,8 em 2020-05-21 16:29:37 UTC.

### Leitura

- **RQ2:** a XAI da previsora LSTM é agora a melhor localização medida nos dois sistemas.
  - **Telecom:** empata com o ranqueamento nativo e supera as heurísticas.
  - **Bank:** supera o ranqueamento nativo, o z global e o z local, em HR@1 e MRR.
  - **Sem a isenção:** a vantagem diminui no Telecom e não muda no Bank.
- **Qualidade:** as explicações da previsora são fiéis, estáveis entre seeds e consistentes entre métodos. A escolha dela para os pacotes do agente (ADR 0016, ameaça nº 11) sai reforçada: além de mais estável, agora ela também localiza melhor que a classificadora no Bank.
- **Atribuição não é causalidade:** os pacotes dizem a que o modelo reagiu, não o que causou a falha (ameaça nº 14).
- **Custo:** cerca de 75 s no Telecom e 175 s no Bank por execução, em CPU.

## Complemento da E5/E6 — modelos alternativos explicáveis (ADR 0018 e 0019)

**Pergunta:** o resultado vem da LSTM, ou outros modelos explicáveis por SHAP/LIME fariam o mesmo?

**Modelos comparados:**
- LightGBM e Isolation Forest, com TreeSHAP exato;
- regressão logística L1, com SHAP linear exato;
- uma TCN classificadora, com GradientExplainer.

Todos usam também o LIME da E6.

**Protocolo:** o mesmo da E5/E6:
- mesmas janelas, dobras e janelas-oráculo;
- 8 configurações por modelo, escolhidas só na validação, com a normalização inclusa;
- 5 seeds (a logística é determinística e roda com 1);
- comparações pareadas com as LSTMs da E5/E6 e, desde o ADR 0019, com **todos** os baselines da E4 na detecção e na localização.

A pontuação dos modelos alternativos não mudou com o ADR 0019, porque já leem a janela inteira como entrada de um modelo supervisionado. Os números mudam só onde as referências mudaram e nas janelas-oráculo corrigidas.

**Fontes:**
- tabelas: `reports/tables/alt_models_<sistema>_alt_{tabular,tcn}[_sem_isencao].md`;
- análise e gráficos: `notebooks/analise_modelos_alternativos.ipynb`.

### Detecção (RQ1): AUC-PR, média entre seeds; veredito pareado (seeds superior/empate/inferior)

| Sistema | Modelo | AUC-PR | AUC-PR por dobra | Alarmes falsos/h | Vs. LSTM previsora | Vs. LSTM classificadora | Vs. z-score local | Vs. só horário |
|---|---|---|---|---|---|---|---|---|
| Telecom | LightGBM | 0,20 | 0,55 | 20,9 | 0/0/5 | 0/0/5 | 0/0/5 | 0/5/0 |
| Telecom | Isolation Forest | 0,17 | 0,35 | 31,0 | 0/0/5 | 0/0/5 | 0/0/5 | 0/5/0 |
| Telecom | **Logística L1** | **0,51** | **0,64** | 8,9 | empate | empate | empate | superior |
| Telecom | TCN | 0,34 | 0,53 | 16,0 | 0/5/0 | 0/4/1 | 0/5/0 | 5/0/0 |
| Bank | LightGBM | 0,18 | 0,22 | 22,9 | 0/0/5 | 0/4/1 | 0/0/5 | 2/3/0 |
| Bank | Isolation Forest | 0,14 | 0,22 | 26,0 | 0/0/5 | 0/0/5 | 0/0/5 | 0/5/0 |
| Bank | Logística L1 | 0,10 | 0,19 | 18,7 | inferior | inferior | inferior | empate |
| Bank | TCN | 0,16 | 0,20 | 18,7 | 0/0/5 | 0/4/1 | 0/0/5 | 0/5/0 |

- **Telecom:** a regressão logística L1 empata com as duas LSTMs e com o z-score local. A TCN empata com a previsora e com o z-score local. LightGBM e Isolation Forest perdem para as duas LSTMs e para o z-score local, e não superam a referência só de horário.
- **Bank:** nenhum alternativo é competitivo. Todos perdem para a previsora LSTM e para o z-score local, e só o LightGBM supera a referência só de horário, em 2 das 5 seeds. A logística fica no acaso (a busca escolheu C = 0,1 no Bank e C = 10 no Telecom).

### Localização via XAI (RQ2): HR@1 (soma por componente, janelas-oráculo, seed 0)

| Modelo | Telecom SHAP / LIME | Bank SHAP / LIME |
|---|---|---|
| LSTM previsora (E6) | **0,42 / 0,44** | **0,48 / 0,45** |
| LSTM classificadora (E6) | 0,27 / 0,13 | 0,44 / 0,22 |
| LightGBM | 0,04 / 0,00 | 0,12 / 0,02 |
| Isolation Forest | 0,11 / 0,09 | 0,12 / 0,03 |
| Logística L1 | 0,13 / 0,07 | 0,15 / 0,08 |
| TCN | 0,27 / 0,22 | 0,25 / 0,18 |

| Referência | Telecom | Bank |
|---|---|---|
| Nativo LSTM (E5) | 0,42 | 0,34 |
| z global (E4) | 0,13 | 0,32 |
| z local (E4) | 0,09 | 0,11 |
| Frequência no treino (E4) | 0,09 | 0,11 |

- **Modelos tabulares:** são **inferiores** em HR@1 à XAI da LSTM previsora e ao ranqueamento nativo em todas as comparações, nos dois sistemas. No Telecom empatam com as heurísticas da E4, exceto o LIME do LightGBM, que perde até para elas. No Bank perdem até para o z global e empatam com a referência de frequência, que não lê telemetria, ou perdem para ela (LIME do LightGBM e do Isolation Forest). As estatísticas da janela diluem o desvio de cada KPI, e um bom detector não precisa olhar para o componente da falha, como mostra a logística no Telecom.
- **TCN:**
  - **Telecom:** com SHAP, empata com a XAI da previsora LSTM e com o nativo; com LIME, perde para os dois. É superior ao z local e à referência de frequência com SHAP.
  - **Bank:** perde para a XAI das duas LSTMs com SHAP e é superior ao z local.

### Qualidade das explicações (médias por janela; Telecom / Bank)

| Medida | LSTM previsora | LSTM classificadora | LightGBM | Isolation Forest | Logística L1 | TCN |
|---|---|---|---|---|---|---|
| Deleção SHAP (menor = mais fiel; aleatório ≈ 1,0) | 0,54 / 0,62 | 0,76 / 0,70 | 0,28 / 0,37 | 0,97 / 0,97 | 0,32 / 0,20 | 0,59 / 0,54 |
| Entre seeds, Jaccard top-10 | **0,85 / 0,86** | 0,20 / 0,26 | 0,62 / 0,59 | 0,03 / 0,03 | — | 0,33 / 0,26 |
| SHAP × LIME, Jaccard top-10 | 0,73 / 0,76 | 0,45 / 0,48 | 0,25 / 0,20 | 0,37 / 0,38 | 0,52 / 0,56 | 0,66 / 0,54 |
| Janelas vizinhas, Jaccard | 0,75 / 0,70 | 0,45 / 0,54 | 0,77 / 0,86 | 0,67 / 0,73 | 0,81 / 0,88 | 0,80 / 0,78 |

- **Fidelidade:** as explicações de LightGBM e logística são as mais fiéis. Esses modelos dependem de poucos atributos, e o SHAP exato os encontra. Mas fidelidade alta não trouxe boa localização.
- **Isolation Forest:** as explicações quase não são fiéis (deleção 0,97) e são instáveis entre seeds (Jaccard 0,03).
- **Estabilidade:** a previsora LSTM tem as explicações mais estáveis entre seeds.

### Sensibilidade à isenção de disponibilidade

Os modelos alternativos quase não mudam sem a isenção:
- **Telecom:** logística 0,50; TCN 0,37; LightGBM 0,21; Isolation Forest 0,17.
- **Bank:** iguais aos da configuração principal até a segunda casa, exceto a TCN, retreinada, cujo HR@1 com SHAP vai de 0,25 a 0,33, dentro da variação entre seeds.

Quem muda é a referência: sem os KPIs de disponibilidade, a LSTM perde a vantagem no Telecom (E5/E6).

### Reprodutibilidade e custo

Duas execuções de fumaça idênticas (Telecom, 2 dobras) deram `metrics.json` iguais byte a byte, tanto no tabular quanto na TCN.

Tempo em CPU (Apple M4 Pro), por execução completa, com busca:

| Execução | Telecom | Bank |
|---|---|---|
| Tabular | 24 min | 40 min |
| TCN | 24 min | 52 min |

O SHAP exato custa de 0,4 s a 4 s por janela explicada, incluindo vizinhas e seeds.

## Comparação de todos os candidatos (ADR 0018 e 0019)

Uma visão única, nas mesmas janelas e falhas de teste, dos modelos anteriores ao agente: baselines da E4, LSTMs da E5/E6 e alternativos do ADR 0018. Valores da configuração principal; os vereditos vêm das comparações pareadas acima.

### Detecção (AUC-PR agrupada; entre parênteses, a média por dobra)

| Candidato | Telecom | Bank | Leitura |
|---|---|---|---|
| **LSTM previsora** | **0,52 (0,56)** | **0,35 (0,38)** | Melhor ou empatada com a melhor nos dois sistemas; supera o z-score local só no Telecom (4/5 seeds) |
| Logística L1 | 0,51 (0,64) | 0,10 (0,19) | Empata com as LSTMs no Telecom; acaso no Bank |
| LSTM classificadora | 0,48 (0,60) | 0,22 (0,26) | Empata com o z-score local no Telecom; perde para ele no Bank |
| z-score local (E4) | 0,39 (0,49) | 0,33 (0,35) | Melhor baseline; empata com a previsora no Bank |
| TCN | 0,34 (0,53) | 0,16 (0,20) | Empata no Telecom; perde no Bank |
| LightGBM (busca, ADR 0018) | 0,20 (0,55) | 0,18 (0,22) | Não supera a referência só de horário no Telecom |
| Isolation Forest (busca, ADR 0018) | 0,17 (0,35) | 0,14 (0,22) | Idem |
| Referência só de horário | 0,15 (0,28) | 0,12 (0,18) | Piso sem telemetria |

### Localização (HR@1; janelas-oráculo)

| Candidato | Telecom | Bank |
|---|---|---|
| **LSTM previsora + SHAP** | **0,42** | **0,48** |
| **LSTM previsora + LIME** | **0,44** | 0,45 |
| LSTM classificadora + SHAP | 0,27 | 0,44 |
| Ranqueamento nativo da LSTM (seed 0) | 0,42 | 0,34 |
| TCN + SHAP | 0,27 | 0,25 |
| z-score global (E4) | 0,13 | 0,32 |
| Logística / LightGBM / Isolation Forest + SHAP | 0,13 / 0,04 / 0,11 | 0,15 / 0,12 / 0,12 |
| z-score local (E4) | 0,09 | 0,11 |
| Frequência no treino (E4) | 0,09 | 0,11 |

### Conclusão

- **A LSTM previsora é a única família que fica entre as melhores nos dois sistemas e nas duas tarefas.**
  - **Detecção:** tem a maior AUC-PR agrupada nos dois sistemas e nunca é inferior a outro candidato. Supera o melhor baseline só no Telecom; no Bank empata com o z-score local.
  - **Localização via XAI:** é a melhor nos dois sistemas e supera, no Bank, até o ranqueamento nativo.
- **A LSTM não é necessária para detectar:**
  - no Telecom, a logística L1 e a classificadora empatam com ela;
  - no Bank, o z-score local, sem aprendizado, empata.

  A tese deve dizer "a LSTM previsora detecta tão bem quanto o melhor candidato em cada sistema e localiza melhor que todos", não "a LSTM supera os baselines na detecção".
- **A localização via XAI favorece modelos sequenciais.** Os modelos tabulares, mesmo quando detectam bem, não apontam o componente da falha.
- **Ressalvas:**
  - no Telecom, as vantagens da LSTM dependem da isenção de disponibilidade (ameaça nº 1);
  - a pontuação nos últimos minutos foi decidida depois de ver o teste (ameaça nº 21), embora fixada pela definição do rótulo e aplicada igualmente aos baselines;
  - nada disso muda o agente: os pacotes de evidência continuam vindo da LSTM, como decidido antes destas comparações (ameaça nº 19).

## Evidência do agente por seleção prequencial (ADR 0020)

Pergunta: qual cadeia, sem LLM, devolve o instante, o componente e o motivo mais precisos para o agente?
- **Candidatos:** lista fechada antes de medir (5 de instante, 9 de componente, 4 de motivo; ADR 0020).
- **Seletor:** em cada dobra, escolhe o método de cada etapa pelo acerto nas consultas dos dias **anteriores** ao teste.
- **Rivais:**
  - o "pacote atual": o que os pacotes da E6 sugerem, ou seja, pico do K1, componente C1 e motivo pela categoria sugerida. As tabelas desta seção são as dos pacotes **com a classificadora** escolhendo a janela; com a previsora (ADR 0021), veja a seção seguinte;
  - o A0 constante da E4.

  A nota do A0 reproduz exatamente a da E4 nos dois sistemas.
- **Execução:** 26 s (Telecom) e 60 s (Bank), em CPU (execuções `20261118T210915` e `20261118T210943`).

### Resultado principal (teste; IC de 95% por bootstrap sobre consultas)

| Sistema | Cadeia | Instante (≤ 60 s) | Componente | Motivo | Nota oficial parcial | Acerto estrito |
|---|---|---|---|---|---|---|
| Telecom | seletor prequencial | 0,44 | 0,24 | 0,33 | 0,35 [0,24; 0,47] | 0,20 |
| Telecom | **pacote atual** | **0,76** | **0,33** | 0,29 | **0,50 [0,38; 0,62]** | **0,36** |
| Telecom | A0 constante | 0,00 | — | — | 0,18 | 0,11 |
| Bank | **seletor prequencial** | **0,38** | **0,34** | 0,29 | **0,34 [0,27; 0,41]** | **0,23** |
| Bank | pacote atual | 0,28 | 0,20 | 0,31 | 0,28 [0,21; 0,35] | 0,16 |
| Bank | A0 constante | 0,06 | — | — | 0,14 | 0,08 |

Comparação pareada, seletor menos pacote atual:

| Sistema | Nota parcial | Instante | Componente | Motivo |
|---|---|---|---|---|
| Telecom | −0,15, **inferior** | −0,31, **inferior** | −0,09, empate | +0,04, empate |
| Bank | +0,07, empate | +0,10, **superior** | +0,13, **superior** | −0,02, empate |

Contra o A0, o seletor é superior na nota parcial nos dois sistemas (+0,17 no Telecom, +0,20 no Bank).

### Escolhas do seletor

| Sistema | Instante | Componente | Motivo |
|---|---|---|---|
| Telecom | BOCPD sobre o z local nas dobras 0–1; depois, 1º minuto acima do limiar da previsora LSTM (T1) | z global (0–2), fusão (3), nativo da previsora (4–9) | R0 em 6 dobras; R1 ou R3 nas outras |
| Bank | início da janela (0), BOCPD (1–3), 1º minuto acima do limiar do z local (4–7) | PageRank (0), fusão por posto de SHAP + z global + BARO (1–7) | regra categoria → motivo (1–6), logística (7) |

- As primeiras dobras têm de 6 a 27 consultas passadas e escolhem mal: é o custo de aprender só com o passado.
- No Bank, a escolha estabiliza cedo na fusão por posto. No Telecom, estabiliza na previsora LSTM: limiar para o instante, ranqueamento nativo para o componente.

### Cada método no teste (descritivo, não usado para escolher)

| Etapa | Telecom | Bank |
|---|---|---|
| Instante: melhor candidato da lista | T1 previsora 0,51 | T2 z local 0,46 |
| Instante: pico do K1 do pacote (fora da lista) | **0,76** | 0,30 |
| Componente, instante rotulado: previsora SHAP / fusão / BARO | 0,42 / 0,29 / 0,11 | 0,49 / 0,46 / 0,32 |
| Componente, instante do seletor: melhor | nativo da previsora 0,33 | fusão 0,33 |
| Motivo: R0 / R2 / R3 | 0,33 / 0,31 / **0,44** | 0,18 / 0,28 / **0,35** |

- **Coerência com a E6:** com o instante rotulado, a previsora SHAP reproduz o HR@1 da E6 (0,42 no Telecom; 0,49 contra 0,48 no Bank, pela lista de candidatos menor).
- **BARO, N-Sigma, sazonal e PageRank:** sozinhos, nenhum supera a previsora SHAP em nenhum sistema. A fusão por posto chega perto dela no Bank.
- **Motivo:** a logística (R3) é a melhor nos dois sistemas no descritivo, mas raramente foi escolhida no Telecom, onde o passado é pequeno.

### Diagnóstico do instante no Telecom

Erros com sinal, em minutos (estimado menos rotulado; ≤ −6 = seis minutos ou mais cedo):

| Método | Acerto (±1 min) | Cedo demais (≤ −6) | Tarde demais (≥ +6) |
|---|---|---|---|
| T1: 1º minuto acima do limiar da previsora | 0,51 | 9 de 45 | 7 |
| T2: 1º minuto acima do limiar do z local | 0,44 | 13 | 4 |
| T3/T4: BOCPD | 0,36 / 0,38 | 15 / 14 | 5 / 7 |
| Pico do K1 do pacote | **0,76** | **2** | 4 |

Os métodos de "primeiro minuto acima do limiar" são enganados por alarmes anteriores dentro da janela da consulta. O pico do KPI mais atribuído, dentro da janela escolhida pela classificadora, raramente adianta.

No Bank isso não se repete: o pico do K1 acerta 0,30, contra 0,46 do z local, e erra cedo tanto quanto os outros.

Esse estimador já existia nos pacotes antes do ADR 0020, mas **não entrou na lista de candidatos**. Foi uma omissão do plano.

### Sensibilidade à isenção de disponibilidade

| Sistema | Seletor: componente | Pacote atual: componente | Seletor menos pacote (nota parcial) |
|---|---|---|---|
| Telecom | 0,24 → **0,04** | 0,33 → 0,16 | empate (−0,13) |
| Bank | 0,34 → 0,25 | 0,20 → 0,26 | empate (−0,01) |

- **Telecom:** sem a isenção, a localização desaba, mais uma vez (ameaça nº 1).
- **Bank:** no controle, a vantagem do seletor no componente desaparece.

### Leitura

- **Não há um vencedor único.**
  - **Bank:** o seletor dá uma evidência melhor que a dos pacotes atuais (instante +0,10 e componente +0,13, superiores; nota parcial com empate a favor).
  - **Telecom:** dá uma evidência pior (nota parcial −0,15, inferior), por causa do instante.
- **Métodos clássicos de RCA:** BARO, N-Sigma, PageRank e desvio sazonal não superam a previsora LSTM na localização. A combinação deles com a LSTM (fusão por posto) é a escolha estável no Bank.
- **Motivo:** a logística sobre o perfil de anomalia por categoria supera o prior nos dois sistemas. É o primeiro sinal de que a telemetria ajuda a acertar o motivo.
- **Decisão (autor, 2026-11-18):** os pacotes do agente **não mudam** (ADR 0020, opção 3), porque seguir o seletor pioraria a evidência do Telecom.
- **Referência para o agente (E8):** com os pacotes desta seção (classificadora), a cadeia sem LLM tinha nota oficial parcial de 0,50 no Telecom e 0,28 no Bank. Com os pacotes atuais (ADR 0021), a referência passa a ser 0,41 e 0,35 (seção seguinte).

## Pacotes com a previsora escolhendo a janela (ADR 0021)

Em 2026-11-19, a previsora LSTM passou a escolher a janela dos pacotes e a dar a pontuação e o alarme. A troca foi fixada antes de medir e não seria revertida pelo resultado. Os pacotes foram regerados (`xai`); as métricas da E6 saíram idênticas. Depois, o `evidence-select` foi reexecutado (execuções `20261119T170412` e `20261119T170508`). O seletor não depende dos pacotes, e suas notas não mudaram.

### Cadeia do pacote, sem LLM (teste; IC de 95% por bootstrap sobre consultas)

| Sistema | Detector dos pacotes | Instante (≤ 60 s) | Componente | Motivo | Nota oficial parcial | Acerto estrito |
|---|---|---|---|---|---|---|
| Telecom | classificadora (até o ADR 0021) | 0,76 | 0,33 | 0,29 | 0,50 [0,38; 0,62] | 0,36 |
| Telecom | **previsora** | 0,56 | 0,33 | 0,29 | **0,41 [0,29; 0,54]** | 0,29 |
| Bank | classificadora (até o ADR 0021) | 0,28 | 0,20 | 0,31 | 0,28 [0,21; 0,35] | 0,16 |
| Bank | **previsora** | 0,40 | 0,28 | 0,35 | **0,35 [0,28; 0,42]** | 0,23 |

Diferença pareada da nota parcial (previsora menos classificadora, mesma consulta; bootstrap sobre consultas):
- **Telecom:** −0,09 [−0,19; +0,01], empate; 6 consultas pioram, 2 melhoram e 37 não mudam;
- **Bank:** +0,08 [0,00; +0,15], empate no limite; 26 melhoram, 13 pioram e 81 não mudam.

Seletor menos pacote: no Telecom, −0,06, empate (antes −0,15, inferior); no Bank, −0,01, empate (antes +0,07).

### Por que o instante muda

- **Telecom:** o pico do K1 passa a cair seis minutos ou mais antes do início rotulado em 10 das 45 consultas (antes, 2). A janela de maior pontuação da previsora é às vezes a de um desvio anterior dentro da consulta, o mesmo defeito dos métodos de "primeiro minuto acima do limiar" (T1, 9 de 45).
- **Bank:** o pico do K1 acerta 0,40 (antes 0,30), perto do melhor candidato do ADR 0020 (T2, z local: 0,46).

### Sensibilidade à isenção de disponibilidade

| Sistema | Nota parcial com a classificadora | Com a previsora |
|---|---|---|
| Telecom | 0,42 | 0,35 |
| Bank | 0,30 | 0,36 |

Sem a isenção, o sentido das mudanças é o mesmo. No Bank, o seletor passa a ser inferior ao pacote (−0,07).

### Leitura

- A troca **não melhora a evidência em geral**. Ela piora o instante no cenário principal (Telecom, −0,20 no campo; nota parcial −0,09, empate) e melhora o Bank (+0,08, empate no limite). Nenhuma das duas diferenças é significativa.
- O pacote ganha coerência: pontuação, alarme e explicações passam a vir do mesmo modelo, como o prompt da A3 descreve.
- **Referência para o agente (E8):** a cadeia do pacote, sem LLM, tem nota oficial parcial de **0,41 no Telecom e 0,35 no Bank**, contra 0,18 e 0,14 do A0. As variantes A2/A3 precisam superá-la para mostrar que o LLM acrescenta algo à evidência.

## E7 e E8 — Agente de LLM no teste (ADR 0017, 0021 e 0022)

- **Execuções da E7 (2026-11-19):** Claude Opus 5.5, esforço `medium`, 1 repetição, todas as consultas testadas (45 no Telecom e 120 no Bank), pacotes de evidência do ADR 0021.
  - 100% das respostas válidas no esquema; de 1 a 15 consultas por execução precisaram de nova tentativa.
  - Custo total de US$ 103,88, com teto de US$ 50 por execução.
- **Avaliação da E8 (2026-11-20):** `evaluate-agent`, com a nota do script oficial do OpenRCA, IC por bootstrap sobre consultas e diferenças pareadas.
  - **Tabelas:** `reports/tables/agent_eval_<sistema>.{json,md}`.
  - **Notebook:** `notebooks/analise_e7_agente.ipynb`.

### Diagnóstico (RQ3): nota oficial

| Cadeia | Telecom: parcial | Telecom: estrito | Bank: parcial | Bank: estrito |
|---|---|---|---|---|
| A0 constante | 0,18 [0,10; 0,29] | 0,11 | 0,14 [0,09; 0,19] | 0,08 |
| pacote sem LLM (ADR 0021) | 0,41 [0,29; 0,54] | 0,29 | 0,35 [0,28; 0,43] | 0,23 |
| A1: telemetria | 0,60 [0,46; 0,72] | 0,51 | 0,43 [0,35; 0,51] | 0,36 |
| A2: + pontuação e ranqueamento LSTM | 0,60 [0,48; 0,73] | 0,49 | 0,40 [0,33; 0,49] | 0,31 |
| A3: + SHAP/LIME | 0,58 [0,44; 0,71] | 0,49 | 0,46 [0,38; 0,54] | 0,37 |

**Comparações pareadas da nota parcial.** As colunas de McNemar valem para o acerto estrito: b = só a primeira cadeia acerta, c = só a segunda. O p entre variantes é corrigido por Holm.

| Comparação | Telecom | Bank |
|---|---|---|
| **A3 − A1 (principal)** | −0,02 [−0,13; +0,09], empate; b/c 4/5, p = 1,0 | +0,03 [−0,05; +0,10], empate; b/c 13/12, p = 1,0 |
| A2 − A1 | +0,01, empate; p = 1,0 | −0,02, empate; p (Holm) = 0,69 |
| A3 − A2 | −0,03, empate; p = 1,0 | +0,05, empate; p (Holm) = 0,69 |
| A1 / A2 / A3 − A0 | +0,41 / +0,42 / +0,39, superiores | +0,29 / +0,27 / +0,32, superiores |
| A1 / A2 / A3 − pacote | +0,18 / +0,19 / +0,16, superiores | +0,08 / +0,05 empates; **A3 +0,11, superior** |

**Acerto por campo** (aproximado; cada critério do gabarito lido separadamente):

| Cadeia | Telecom: instante / componente / motivo | Bank: instante / componente / motivo |
|---|---|---|
| pacote | 0,50 / 0,41 / 0,32 | 0,41 / 0,26 / 0,34 |
| A1 | 0,71 / 0,68 / 0,46 | 0,37 / 0,54 / 0,44 |
| A2 | 0,64 / 0,73 / 0,43 | 0,43 / 0,50 / 0,45 |
| A3 | 0,68 / 0,73 / 0,43 | 0,46 / 0,54 / 0,49 |

### Localização pelo agente (RQ2)

Esta tabela usa as consultas de uma falha que pedem componente (22 no Telecom e 56 no Bank). Ela compara o componente respondido com o C1 do pacote, nas mesmas consultas.

| Cadeia | Telecom: HR@1 | Menos o C1 | Bank: HR@1 | Menos o C1 |
|---|---|---|---|---|
| pacote (C1) | 0,41 | — | 0,25 | — |
| A1 | 0,68 | +0,27, empate (b/c 10/4) | 0,55 | +0,30, **superior** (20/3) |
| A2 | 0,73 | +0,32, **superior** (8/1) | 0,45 | +0,20, **superior** (14/3) |
| A3 | 0,73 | +0,32, **superior** (8/1) | 0,50 | +0,25, **superior** (16/2) |

Quando o agente e o C1 discordam, quem acerta é quase sempre o agente. O HR@1 da E6 (0,42 e 0,48) foi medido nas janelas-oráculo, com o início rotulado. Aqui a janela é a escolhida pelo detector dentro da consulta, e os números não são diretamente comparáveis.

### Fidelidade (RQ4)

**Verificação automática** de todas as explicações, contra as saídas das ferramentas da própria transcrição, sem gabarito:

| Checagem | Telecom (A1 / A2 / A3) | Bank (A1 / A2 / A3) |
|---|---|---|
| IDs citados no texto e não emitidos | 0 / 0 / 0 | 0 / 0 / 0 |
| Componentes inexistentes | 0 / 0 / 0 | 0 / 0 / 0 |
| Identificadores de KPI desconhecidos | 0 / 0 / 0 | 0 / 0 / 0 |
| Números sem correspondência | 1,4% / 0,2% / 1,0% | 0,9% / 0,9% / 0,6% |
| Controle de acaso (números × 1,37 com correspondência) | 27% / 21% / 22% | 27% / 25% / 22% |

- **Regra dos números:** correspondência até uma unidade da última casa escrita, ou 5% quando o texto marca aproximação.
- **O que entra na taxa:** números derivados (diferenças, razões) também contam, então a taxa é um teto para valores inventados.
- **O controle:** mostra que a regra discrimina. Por coincidência, ela aceitaria cerca de 1 em cada 4 números; os números reais passam em cerca de 99%.

**Auditoria por LLM (adendo do ADR 0022):** 120 afirmações de 30 explicações, 5 por sistema e variante.
- **Quem classificou:** o Claude Opus 5.5, por decisão do autor, com a rubrica de `docs/AUDITORIA_FIDELIDADE.md`.
- **Sem validação humana:** o auditor é o mesmo modelo do agente (ameaça nº 28). Os números são de uma auditoria por LLM, não manual.
- **Formato da tabela:** fração das afirmações de cada rótulo, com IC por bootstrap sobre explicações; 20 afirmações por célula.

| Rótulo | Telecom A1 / A2 / A3 | Bank A1 / A2 / A3 |
|---|---|---|
| sustentada | 0,65 / 0,80 / 0,80 | 0,75 / 0,90 / 0,90 |
| parcial | 0,10 / 0,15 / 0,10 | 0,05 / 0,05 / 0,05 |
| não sustentada | 0 / 0 / 0 | 0 / 0 / 0 |
| não verificável | 0,25 / 0,05 / 0,10 | 0,20 / 0,05 / 0,05 |

- **Nenhuma afirmação contradiz as saídas ou inventa dado.** Das 106 que trazem dado, 96 são sustentadas (91%) e 10 parciais.
- **Tipos de parcial:**
  - generalização além do dado ("em cada caso", "sinais fracos");
  - um ID trocado;
  - atribuição de ranking errada;
  - horário inexato.
- **A1 tem mais "não verificável"** (títulos, conclusões, "não consultei X") e menos "sustentada". Com 5 explicações por célula, é descritivo, sem teste.

### Custo (RQ4)

| Variante | Telecom: US$/consulta | Telecom: latência (s) | Telecom: ferramentas | Bank: US$/consulta | Bank: latência (s) | Bank: ferramentas |
|---|---|---|---|---|---|---|
| A1 | 0,193 ± 0,061 | 34,0 ± 9,6 | 13,2 ± 4,5 | 0,232 ± 0,114 | 36,4 ± 14,0 | 12,4 ± 6,6 |
| A2 | 0,168 ± 0,079 | 29,0 ± 11,5 | 10,8 ± 5,8 | 0,217 ± 0,111 | 35,5 ± 14,4 | 12,1 ± 6,4 |
| A3 | 0,162 ± 0,071 | 28,3 ± 10,7 | 9,8 ± 5,3 | 0,221 ± 0,107 | 36,2 ± 14,9 | 12,0 ± 6,4 |

### Análise de erros (descritiva, grupos pequenos)

- **Rede é o ponto fraco.**
  - **Bank:** latência de rede tem acerto de motivo de 0 a 0,14 em todas as cadeias (14 critérios); perda de pacotes fica entre 0,12 e 0,33, abaixo do pacote (0,50; 24 critérios).
  - **Telecom:** perda de rede fica entre 0 e 0,2.
- **CPU é o ponto forte.** No Bank, "high CPU usage" tem acerto de 0,89 a 0,95 no agente, contra 0,37 do pacote.
- **Componentes:**
  - **Telecom:** o agente acerta os contêineres (`docker`, 0,8 a 1,0), onde o C1 do pacote erra todos.
  - **Bank:** os gateways `IG` vão de 0,1 (pacote) a 0,7 a 1,0 (agente).

### Reprodução (critério da E7)

As seis execuções foram refeitas a partir do cache com teto de US$ 0 (`configs/experiment/agent_a*_replay.yaml`): 0 chamadas pagas, `predictions_s0.csv` e `metrics.json` idênticos byte a byte aos originais. Duas execuções seguidas do `evaluate-agent` produzem o mesmo `metrics.json`.

### Detecção (RQ1): curvas PR e ROC

`reports/figures/rq1_curvas_pr_roc.png` (`rq1-curves`) mostra as curvas dos detectores no teste: LSTMs na seed 0, z-score local, LightGBM e referência só de horário da E4. As AUCs recalculadas batem com as tabelas da E4 e da E5.

### Leitura

- **RQ3:** o agente com LLM acerta muito mais que o A0 e, no geral, mais que a evidência sozinha (pacote sem LLM). Mas a **evidência LSTM/XAI não melhora o agente de forma detectável**: A3 × A1 empata nos dois sistemas, pela diferença pareada e pelo McNemar.
- **Por quê, provavelmente:**
  - todas as variantes têm a varredura heurística `find_anomalous_kpis`, usada em 100% das consultas (ameaça nº 17);
  - o agente usa a evidência (a A3 cita `C#`/`K#` em quase todas as respostas), mas, quando ela aponta o componente errado, ele a corrige com a telemetria.
- **RQ2:** como localizador, o agente supera o C1 do pacote em 5 das 6 comparações. A localização da E6 continua sendo a melhor entre os métodos sem LLM.
- **RQ4:** a verificação automática não acha IDs, componentes ou KPIs inventados, e só cerca de 1% de números sem correspondência; na auditoria por LLM das 120 afirmações amostradas (sem validação humana), nenhuma contradiz as saídas, 91% das que trazem dado são sustentadas e 9% parciais. O custo fica entre US$ 0,16 e 0,23 por consulta, e a evidência não o aumenta.
- **Ameaças:**
  - contaminação (nº 15);
  - varredura heurística (nº 17);
  - uma repetição e poder limitado (nº 25): com 45 consultas, efeitos de até cerca de 0,13 passam por empate;
  - auditoria feita pelo mesmo modelo do agente, sem validação humana (nº 28).

## Modelo mais barato com as mesmas ferramentas: Haiku 5.5 × Opus 5.5 (ADR 0023, exploratório)

Esta pergunta surgiu depois de ver a E8 (ameaça nº 27). O plano foi fixado antes de rodar o Haiku: mesmas variantes, prompts e ferramentas; 1 repetição; margem de não inferioridade de 0,10.

Execuções em 2026-11-20: 100% de respostas válidas, US$ 3,35 no total.

**Tabelas:** `reports/tables/model_compare_<sistema>_<modelo>.{json,md}`.

**Notebook:** `notebooks/analise_modelos_llm.ipynb` (Haiku, Sonnet e Opus juntos).

| Cadeia (nota parcial) | Telecom | Bank |
|---|---|---|
| A0 / pacote sem LLM | 0,18 / 0,41 | 0,14 / 0,35 |
| Opus A1 / A2 / A3 | 0,60 / 0,60 / 0,58 | 0,43 / 0,40 / 0,46 |
| Haiku A1 / A2 / A3 | 0,14 / 0,26 / 0,34 | 0,16 / 0,19 / 0,29 |
| Haiku A3 − Haiku A1 | +0,20 [0,04; 0,36], superior (McNemar Holm p = 0,105) | +0,13 [0,07; 0,19], superior (p = 0,005) |
| Instante no início da janela (Haiku A1 → A3; Opus) | 0,75 → 0,50; Opus de 0,07 a 0,11 | 0,68 → 0,62; Opus de 0,18 a 0,26 |
| Localização (Haiku A1 → A3; Opus A3) | 0,09 → 0,55; 0,73 | 0,21 → 0,30; 0,50 |
| Custo por consulta | US$ 0,006 (Opus: 0,16 a 0,19) | US$ 0,006 a 0,008 (Opus: 0,22 a 0,23) |

- **A evidência ajuda o modelo mais fraco.** O Haiku A3 supera o Haiku A1 nos dois sistemas, sobretudo porque acerta mais o componente. No Opus, A3 × A1 empatou.
- **O Haiku não iguala o Opus.**
  - Todas as variantes do Haiku são inferiores a todas as do Opus pelo veredito pareado, com diferenças de −0,12 a −0,47.
  - Nenhuma é não inferior com margem de 0,10: o Haiku A3 fica `inconclusivo` ou `inferior`, com estimativas pontuais já além da margem.
- **O Haiku A3 empata com a cadeia do pacote sem LLM** nos dois sistemas. Ou seja, com estas ferramentas, o modelo barato não acrescenta nada detectável à evidência lida diretamente.
- **A falha típica do Haiku continua:** de 50% a 75% dos instantes respondidos ficam no início da janela.
- **Fidelidade automática:** sem IDs inventados; de 0,3% a 3,9% de números sem correspondência.
- **Detalhes e leitura completa:** ADR 0023.

### Adendo: Sonnet 5.5 (mesmo protocolo; 2026-11-20)

| Nota parcial | Telecom | Bank |
|---|---|---|
| Sonnet A1 / A2 / A3 | 0,17 / 0,24 / 0,31 | 0,16 / 0,23 / 0,32 |
| Sonnet A3 − Sonnet A1 | +0,14 [−0,01; 0,29], empate | +0,16 [0,10; 0,23], superior (McNemar Holm p < 0,001) |
| Custo por consulta | US$ 0,035 a 0,047 | US$ 0,038 a 0,054 |

- **Contra o Opus:** o Sonnet não o iguala.
  - **Telecom:** inferior em tudo.
  - **Bank:** o Sonnet A3 fica `inconclusivo` contra as três variantes do Opus (−0,08 a −0,13), mas inferior pelo veredito pareado.
- **Contra o pacote sem LLM:** o Sonnet A3 empata nos dois sistemas.
- **Custo:** US$ 21,48 no total. O Bank A3 foi interrompido por falta de crédito e refeito; o cache reaproveitou as 50 consultas já feitas.
- **Leitura conjunta, descritiva:**
  - os dois modelos mais baratos ficam no nível da cadeia do pacote;
  - a evidência melhora os dois (A3 × A1 superior em 3 dos 4 casos), sem chegar ao Opus;
  - no Opus, a evidência não teve efeito detectável.
- **Detalhes:** ADR 0023, adendo.
