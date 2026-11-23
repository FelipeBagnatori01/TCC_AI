# ADR 0022 — Avaliação integrada do agente (E8): nota oficial, testes pareados, fidelidade e custo

- **Status:** aceito (2026-11-20) e executado. A auditoria de fidelidade foi feita por LLM e adotada pelo autor em 2026-11-23 (adendo abaixo)
- **Data:** 2026-11-20
- **Etapa:** E8

## Contexto

A E7 rodou as variantes A1, A2 e A3 no teste em 2026-11-19, com Claude Opus 5.5, 1 repetição, 45 consultas no Telecom e 120 no Bank. O custo foi de US$ 103,88. A E8 precisa:
- responder à RQ3 pela métrica oficial do OpenRCA, com a comparação A3 × A1 como resultado central e o McNemar, como previsto no plano;
- medir a fidelidade das explicações (RQ4), que até aqui só tinha a garantia de esquema da E7: todo ID citado foi emitido;
- relatar o custo;
- fechar o critério da E7 que ainda não tinha evidência: refazer as execuções a partir do cache sem chamada paga.

As RQ1 e RQ2 já foram medidas nas E4 a E6 e nos ADRs 0018 a 0021. Na E8 elas ganham as curvas PR/ROC e a localização pelo agente.

## Decisões

A comparação principal (A3 × A1) e o McNemar vêm do plano original (RQ3), anterior a qualquer execução da E7. O resto foi definido no plano da E8, aprovado pelo autor em 2026-11-20, depois que as notas preliminares da E7 já tinham aparecido no notebook. Nenhuma decisão escolhe entre variantes, e todas valem para as três igualmente.

1. **Nota oficial, sem reimplementação.**
   - As respostas de cada execução (`predictions_s0.csv`) passam pelo `score_with_official_script`.
   - O A0 e a cadeia do pacote vêm dos relatórios oficiais do `evidence-select` (ADR 0021), nas mesmas consultas; o código confere que batem com as tabelas versionadas.
   - A nota parcial é a média da nota oficial por consulta; o acerto estrito é a fração com nota 1.
2. **Comparação principal: A3 × A1**, como no plano (RQ3 e item 20 do acompanhamento). A2 × A1 e A3 × A2 são secundárias. As comparações com o A0 e com o pacote são referências.
3. **Estatística:**
   - **IC:** bootstrap de percentil sobre consultas (10 000 reamostragens).
   - **Nota parcial:** diferença pareada, "superior" ou "inferior" quando o IC exclui 0 (mesma regra de `compare.verdict`).
   - **Acerto estrito:** McNemar exato (binomial bilateral sobre os pares discordantes).
   - **Correção:** Holm só na família das três comparações entre variantes de cada sistema.
   - O tamanho do efeito (diferença com IC) é sempre relatado ao lado do p-valor.
4. **Acerto por campo:** cada critério do gabarito no relatório oficial é lido separadamente. É uma leitura **descritiva e aproximada** em consultas com duas falhas, porque o script oficial escolhe a melhor associação entre causas.
5. **Localização (RQ2) pelo agente:** nas consultas de uma falha que pedem componente, compara-se o componente do agente com o C1 do pacote, nas mesmas consultas. O HR@1 da E6 vem das janelas-oráculo e não é diretamente comparável.
6. **Fidelidade automática, sem gabarito:** cada explicação é conferida com as saídas das ferramentas da própria transcrição.
   - **IDs:** IDs não emitidos.
   - **Componentes:** componentes com o formato do sistema que não existem.
   - **Identificadores:** identificadores de KPI que não aparecem no catálogo (abreviação permitida), nas saídas ou entre as ferramentas.
   - **Números:** números sem correspondência. A tolerância é uma unidade da última casa escrita (arredondamento ou truncamento), ou 5% quando o texto marca aproximação ("cerca de", "~"). Ficam de fora horários, datas, IDs e inteiros menores que 100.
   - **Controle de acaso:** os mesmos números multiplicados por 1,37 passam pela mesma regra.
   - **Calibração, feita sem o gabarito:** uma tolerância de 5% para todos os números foi testada e descartada, porque casava 67% dos números do controle. A tolerância de uma unidade casa cerca de 25%.
7. **Auditoria manual (autor, sozinho):**
   - **Amostra:** 30 explicações, 5 por sistema × variante, com semente 0; de cada uma, 4 frases verificáveis sorteadas, num total de 120 afirmações.
   - **Cegamento parcial:** a planilha mostra só o código da explicação, e a chave fica num arquivo separado.
   - **Rótulos e regras:** em `docs/AUDITORIA_FIDELIDADE.md`.
   - **Planilha:** gerada uma única vez e nunca reescrita. O resumo usa IC por bootstrap sobre explicações.
8. **Variância do LLM:** não é medida. Foi feita só 1 repetição, por decisão do autor (2026-11-20), e isso fica registrado como ameaça (nº 25).
9. **Reprodução (critério da E7):** as seis execuções foram refeitas a partir do cache com `llm.budget_usd: 0` (`configs/experiment/agent_a*_replay.yaml`). O critério exige 0 chamadas pagas e `predictions_s0.csv` e `metrics.json` idênticos byte a byte. O `evaluate-agent` confere isso.
10. **Juiz LLM:** não foi usado. Seria pago, e um modelo da mesma família avaliando a si mesmo tende a ser enviesado.

## Implementação

- **`evaluation/agent_eval.py`:** o comando `evaluate-agent`, com `configs/experiment/agent_eval.yaml` e a seção `agent_eval` do `base.yaml`.
  - **Saídas:** `reports/tables/agent_eval_<sistema>.{json,md}` e, na pasta da execução, os relatórios oficiais e `fidelity_details.csv`.
  - **Auditoria:** a planilha e a chave ficam em `reports/audit/`. A evidência de cada explicação fica em `experiments/auditoria_fidelidade/`, fora do git, porque contém saídas de telemetria.
- **`evaluation/curves.py`:** o comando `rq1-curves` gera as curvas PR/ROC dos detectores no teste (`reports/figures/rq1_curvas_pr_roc.png`). As AUCs recalculadas batem com as tabelas da E4 e da E5.
- **Testes:** `tests/test_agent_eval.py`, com valores calculados à mão para:
  - McNemar e Holm;
  - leitura do relatório oficial, diferença pareada, localização e quebra de erros;
  - números em português, correspondência com tolerância, checagens de fidelidade;
  - amostra e resumo da auditoria, verificação da reprodução;
  - curvas PR/ROC.
- **Notebook:** `notebooks/analise_e7_agente.ipynb`, que lê só as tabelas da E8.

## Resultado (2026-11-20)

| | Telecom (45) | Bank (120) |
|---|---|---|
| A0 / pacote sem LLM | 0,18 / 0,41 | 0,14 / 0,35 |
| A1 / A2 / A3 (nota parcial) | 0,60 / 0,60 / 0,58 | 0,43 / 0,40 / 0,46 |
| **A3 − A1** | −0,02 [−0,13; +0,09], empate; McNemar 4/5, p = 1,0 | +0,03 [−0,05; +0,10], empate; McNemar 13/12, p = 1,0 |

- **Variantes contra as referências:**
  - as três variantes são superiores ao A0 nos dois sistemas;
  - no Telecom, as três superam o pacote; no Bank, só a A3.
- **Localização:** nas consultas de uma falha, o agente localiza melhor que o C1 do pacote em 5 das 6 comparações (a exceção é a A1 no Telecom, que empata).
- **Fidelidade automática:**
  - 0 IDs não emitidos, 0 componentes e 0 identificadores de KPI desconhecidos;
  - de 0,2% a 1,4% dos números sem correspondência, contra 73% a 79% sem correspondência no controle de acaso.
- **Reprodução:** 0 chamadas pagas nas seis execuções, com previsões e `metrics.json` idênticos.
- **Custo:** de US$ 0,16 a 0,23 por consulta.
- **Detalhes:** `docs/RESULTADOS.md`.

## Adendo: auditoria de fidelidade feita por LLM (2026-11-23)

- **Decisão do autor:** a auditoria manual foi substituída por uma auditoria feita pelo Claude Opus 5.5, adotada como oficial. Motivo: o volume e a complexidade das evidências (30 transcrições com milhares de linhas de saída de ferramentas).
  - A validação por uma amostra rotulada pelo autor foi proposta e dispensada pelo autor.
  - **Não há validação humana.**
- **Como foi feita:** a mesma amostra (30 explicações, 120 afirmações), a mesma rubrica e as mesmas regras de desempate de `docs/AUDITORIA_FIDELIDADE.md`.
  - Cada afirmação foi conferida contra as saídas das ferramentas daquela transcrição, e a coluna `observacao` registra o dado conferido e onde ele está.
  - **Regras usadas nos casos de fronteira:**
    - frase de conclusão que traz um dado: classifica-se o dado;
    - frase sobre o que o agente não consultou, títulos e juízos: `nao_verificavel`;
    - qualificadores como "pouco" ou "pequeno": aceitos quando os valores mostrados os justificam.
- **Arquivos:**
  - `reports/audit/auditoria_fidelidade_llm.csv` é a planilha oficial, apontada por `agent_eval.audit_labels_sheet`.
  - A planilha original do autor, `auditoria_fidelidade.csv`, fica vazia como modelo.
  - O `evaluate-agent` confere que as duas têm as mesmas afirmações, resume por sistema e variante e registra quem classificou (`agent_eval.audit_rater`) nas tabelas.
- **Limitações:**
  - **Autoavaliação (ameaça nº 28):** o auditor é o mesmo modelo que escreveu as explicações e pode ser leniente com os próprios tipos de erro.
  - **Cegamento:** a ferramenta `get_evidence` aparece nas transcrições e revela A2/A3. A chave só foi aberta depois de rotular tudo.
  - **Como relatar:** os números devem ser lidos como "auditoria por LLM", nunca como auditoria manual.
- **Resultado (120 afirmações):**
  - 96 sustentadas, 10 parciais, 0 não sustentadas e 14 não verificáveis, ou seja, 91% sustentadas entre as que trazem dado.
  - **Parciais:**
    - generalizações além do dado ("em cada caso", "sinais fracos", "sem tempo ocioso");
    - um ID trocado;
    - atribuições de ranking erradas;
    - horários inexatos.
  - **Por sistema e variante:** em `reports/tables/agent_eval_<sistema>.{json,md}` e em `docs/RESULTADOS.md`.
