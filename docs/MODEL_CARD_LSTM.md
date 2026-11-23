# Model card: LSTMs de detecção de falhas (E5/E6)

Formato inspirado em MITCHELL, M. et al. *Model Cards for Model Reporting* (FAT\*, 2019). Os números vêm de `reports/tables/lstm_<sistema>_lstm.json` e `xai_<sistema>_xai.json`.

## Identificação

| Item | Valor |
|---|---|
| Modelos | **LSTM previsora** (`ForecastLSTM`) e **LSTM classificadora** (`ClassifierLSTM`), uma por sistema e por dobra |
| Código | `src/tcc_rca/models/lstm/`, `src/tcc_rca/evaluation/lstm.py` |
| Decisões | ADR 0015 (arquitetura, busca, avaliação), ADR 0019 (pontuação nos últimos 6 min), ADR 0016/0021 (uso nas evidências do agente) |
| Arquitetura escolhida na validação | 128 unidades, 1 camada, normalização causal local, nos dois sistemas e nos dois modelos |
| Parâmetros (Telecom / Bank) | previsora: 989.600 / 893.450; classificadora: 803.969 / 727.169 |
| Sementes finais | 5 (0 a 4) por dobra; os resultados são a média entre sementes, com o desvio-padrão |
| Hardware de treino | Apple M4 Pro, só CPU, `torch.use_deterministic_algorithms(True)` |

## Uso pretendido

- **Para que serve:** detectar, por janela de 20 minutos, se uma falha começa nos últimos minutos da janela, e dar o score que o SHAP/LIME explicam para apontar componentes e KPIs suspeitos.
- **Uso no agente:** a previsora é a fonte das evidências da A2/A3 (pontuação, limiar, alarme, ranqueamento nativo, atribuições).
- **Não serve para:**
  - **Prever falhas com antecedência:** não houve precursor detectável (ADR 0007), então a tarefa é detecção (H = 0).
  - **Operar sem calibração:** o limiar vem do dia de validação de cada dobra.
  - **Uso fora do OpenRCA sem retreino:** o catálogo de KPIs e a normalização são específicos de cada sistema.

## Dados

- **Telecom:**
  - 15 dias, 6 h por dia, passo de 60 s;
  - 51 falhas, das quais 45 testadas em 10 dobras de origem móvel;
  - catálogo de 2.138 séries `componente::KPI`.
- **Bank:**
  - 10 dias;
  - 120 falhas testadas em 8 dobras, a partir de 2021-03-06;
  - catálogo de 1.942 séries, com logs agregados em templates drain3.
- **Treino:**
  - A previsora treina só em janelas normais; a classificadora usa `pos_weight`.
  - Seleção de KPIs e escala são ajustadas só nos dias de treino; a parada antecipada usa a validação.
  - Não há vazamento temporal (testado em `tests/`).
- **Isenção da categoria `disponibilidade`** do descarte por constância: foi decidida depois de ver os rótulos (ameaça nº 1). Todo resultado principal tem o controle sem isenção ao lado.

## Desempenho no teste

| Métrica (média de 5 sementes) | Telecom previsora | Telecom classificadora | Bank previsora | Bank classificadora |
|---|---|---|---|---|
| AUC-PR agrupada | 0,52 (dp 0,015) | 0,48 (dp 0,024) | 0,35 (dp 0,004) | 0,22 (dp 0,014) |
| AUC-PR média por dobra | 0,56 | 0,60 | 0,38 | 0,26 |
| AUC-ROC | 0,81 | 0,83 | 0,78 | 0,66 |
| Alarmes falsos por hora (limiar da validação) | 11,3 | 9,2 | 5,7 | 14,3 |
| AUC-PR sem a isenção (controle) | 0,43 | — | 0,34 | — |

**Leitura:**
- A previsora tem a maior AUC-PR agrupada nos dois sistemas, mas empata com o melhor candidato em cada um:
  - **Telecom:** logística L1 e classificadora;
  - **Bank:** z-score local, sem aprendizado.
- **Localização via SHAP (HR@1):** 0,42 no Telecom e 0,48 no Bank, a melhor entre todos os localizadores.
- **Detalhes e comparações pareadas:** `docs/RESULTADOS.md`.

**Qualidade das explicações da previsora** (janelas-oráculo, Telecom / Bank):

| Medida | Telecom | Bank |
|---|---|---|
| Deleção (SHAP / LIME / aleatório; menor = melhor) | 0,54 / 0,53 / 0,99 | 0,62 / 0,62 / 0,99 |
| Inserção (SHAP / LIME / aleatório; maior = melhor) | 0,69 / 0,67 / 0,04 | 0,66 / 0,65 / 0,06 |
| Concordância SHAP × LIME (Jaccard top-k) | 0,73 | 0,76 |
| Estabilidade entre sementes (Jaccard) | 0,85 | 0,86 |

## Custo

- **Treino por semente e dobra** (Telecom / Bank):
  - previsora: 2,4 s / 5,6 s;
  - classificadora: 1,2 s / 3,3 s.
- **Busca na validação:** 108 treinos em 335 s (Telecom) e 84 treinos em 815 s (Bank).
- **Inferência:** cerca de 0,25 ms por janela.
- **XAI completo** (SHAP, LIME e IG nas janelas-oráculo, mais os pacotes): 75 s (Telecom) e 178 s (Bank).

## Limitações e riscos

- **Amostra pequena:** 45 falhas testadas no Telecom, com ICs largos (ameaças nº 4 e 25).
- **Atribuição não é causalidade:** o componente mais atribuído pode ser o mais afetado, não a causa.
- **Categoria de motivo sugerida:** só superou a referência uma vez (SHAP da previsora no Bank), por isso fica oculta do agente.
- **Decisões tomadas depois de ver o teste:**
  - a isenção de disponibilidade (nº 1);
  - a pontuação nos últimos minutos (nº 21);
  - a previsora escolhendo a janela das evidências (ADR 0021).

  Todas têm controle relatado.
- **Alarmes falsos:** entre 6 e 14 por hora no limiar escolhido. Inviável como alarme operacional direto, sem agregação ou supressão.
- **Ambiente:** o benchmark tem falhas injetadas e poucos dias, e não captura deriva de longo prazo de um ambiente de produção.
