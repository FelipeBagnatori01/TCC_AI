# Ficha do agente de LLM para causa raiz (E7/E8)

Os números vêm de `reports/tables/agent_eval_<sistema>.json` e `model_compare_<sistema>_<modelo>.json`.

**Decisões:**
- ADR 0017: ferramentas, isolamento e variantes;
- ADR 0021: evidências da previsora;
- ADR 0022: avaliação;
- ADR 0023: modelos mais baratos.

## Identificação

| Item | Valor |
|---|---|
| Modelo principal | **Claude Opus 5.5** (`claude-opus-5-5`, Anthropic), escolhido no piloto (ADR 0017) entre Opus, Sonnet 5.5 e Haiku 5.5 |
| Parâmetros de chamada | Esforço `medium`; `max_tokens` 16.000 por chamada (raciocínio incluso); sem temperatura (o modelo não aceita parâmetros de amostragem) |
| Repetições | 1 por consulta |
| Limites por consulta | Até 20 chamadas de ferramenta e 30 turnos; até 2 novas tentativas quando a resposta é inválida, com os erros de validação |
| Preços usados no custo | US$ 4 de entrada e 20 de saída por milhão de tokens; leitura de cache a 0,20 e escrita a 5 (preços de 2026-11-03, `configs/llm/claude_opus_5_5.yaml`) |
| Prompts | `configs/prompts/`: sistema, usuário, variante, nova tentativa e candidatos, com hash registrado em cada execução |
| Código | `src/tcc_rca/agent/` (ferramentas, executor, validação), `src/tcc_rca/llm/` (cliente, cache, provedores) |

## Ferramentas (somente leitura) e variantes

| Variante | Ferramentas |
|---|---|
| A1 | `list_components`, `list_kpis`, `query_kpi`, `find_anomalous_kpis` (varredura heurística de desvio, ameaça nº 17), `get_trace_summary`, `get_dependencies` |
| A2 | A1 + `get_evidence`: pontuação, limiar e alarme da LSTM previsora e o ranqueamento nativo (`N#`) |
| A3 | A2 + componentes (`C#`) e KPIs (`K#`) mais atribuídos por SHAP e LIME |

**Isolamento** (testado):
- As ferramentas só alcançam as séries processadas, o catálogo, as arestas de dependência e os pacotes de evidência do sistema.
- `record.csv`, `query.csv` e as janelas rotuladas ficam inalcançáveis, e o código do agente não importa nada do gabarito.

**Resposta:**
- JSON validado por esquema, com as chaves `causas`, `explicacao` e `evidencias_citadas`.
- É conferido contra os campos pedidos, o número de falhas, a janela, os candidatos e os IDs efetivamente emitidos.
- A explicação é escrita em português.

## Desempenho (teste; nota oficial do OpenRCA)

| Cadeia | Telecom (45 consultas) | Bank (120 consultas) |
|---|---|---|
| A0 constante | 0,18 | 0,14 |
| Pacote de evidências sem LLM | 0,41 | 0,35 |
| A1 / A2 / A3 (nota parcial) | 0,60 / 0,60 / 0,58 | 0,43 / 0,40 / 0,46 |
| A3 − A1 (comparação principal) | −0,02 [−0,13; 0,09], empate | +0,03 [−0,05; 0,10], empate |

**Leitura:**
- O agente supera o A0 e, em geral, a cadeia do pacote sem LLM.
- A evidência LSTM/XAI **não** melhora o Opus de forma detectável.
- Com modelos mais baratos (ADR 0023, exploratório), a evidência melhora a nota (A3 − A1 de +0,13 a +0,20), mas nenhum deles alcança o Opus.

## Fidelidade

- **Verificação automática** (100% das explicações):
  - 0 IDs citados sem terem sido emitidos;
  - 0 componentes e 0 KPIs inexistentes;
  - de 0,2% a 1,4% de números sem correspondência nas saídas das ferramentas, contra cerca de 75% no controle de acaso.
- **Auditoria por LLM** (120 afirmações; Claude Opus 5.5, o mesmo modelo do agente, sem validação humana, ameaça nº 28):
  - 0 não sustentadas;
  - 91% sustentadas entre as que trazem dado;
  - 9% parciais: generalizações além do dado, um ID trocado, ranking e horário inexatos.

## Custo e latência (Opus, por consulta)

| Variante | Telecom: custo / latência | Bank: custo / latência | Chamadas de ferramenta (média, Telecom / Bank) |
|---|---|---|---|
| A1 | US$ 0,193 / 34 s | US$ 0,232 / 36 s | 13,2 / 12,4 |
| A2 | US$ 0,168 / 29 s | US$ 0,217 / 36 s | 10,8 / 12,1 |
| A3 | US$ 0,162 / 28 s | US$ 0,221 / 36 s | 9,8 / 12,0 |

- **Totais:** US$ 103,88 nas seis execuções de teste do Opus; US$ 3,35 no Haiku (cerca de 30× menos por consulta); US$ 21,48 no Sonnet.
- **Reprodução:** toda chamada fica em cache, e as execuções são refeitas sem custo (`make reproduce-agent`, ADR 0024).

## Limitações e riscos

- **Contaminação** (ameaça nº 15): o OpenRCA é público, e o modelo pode ter visto o benchmark no treino.
- **Variância** (nº 25): uma repetição; efeitos de até cerca de 0,13 passam por empate no Telecom.
- **Varredura heurística** (nº 17): está em todas as variantes, o que pode mascarar o efeito da evidência.
- **Um provedor só** (nº 27): Haiku e Sonnet foram testados de forma exploratória, e outros fornecedores não foram avaliados.
- **Envio de telemetria** para um serviço externo de LLM: em produção, exige avaliação de privacidade e contrato (ver `docs/RECOMENDACOES_INDUSTRIA.md`).
- **A explicação não é uma prova:** mesmo sustentada pelos dados, a conclusão causal é inferência do modelo e deve ser conferida por um engenheiro.
