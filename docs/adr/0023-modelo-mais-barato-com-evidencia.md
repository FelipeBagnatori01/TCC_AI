# ADR 0023 — Modelo mais barato com as mesmas ferramentas e evidências (exploratório)

- **Status:** aceito (2026-11-20), antes de qualquer execução do Haiku no teste. Execuções pendentes; o autor as inicia
- **Data:** 2026-11-20
- **Etapa:** complemento da E8 (RQ3/RQ4)

## Contexto

Na E8 (ADR 0022), com o Claude Opus 5.5, a A3 empatou com a A1 nos dois sistemas: a evidência LSTM/XAI não teve efeito detectável. Uma explicação possível é que o Opus já encontra sozinho o instante e o componente.

O piloto (ADR 0017, 2026-11-16, só A1) mostrou a falha típica dos modelos mais baratos: o Haiku 5.5 respondeu o início da janela em 10 de 14 instantes, contra 1 de 14 do Opus. Com isso, acertou 0 de 6 no Telecom e 2 de 16 no Bank. A A2 e a A3 entregam justamente o instante e o componente sugeridos pelo pacote.

O autor perguntou, em 2026-11-20, se um modelo mais barato com as ferramentas da A2/A3 consegue se igualar ao Opus.

**A pergunta surgiu depois de ver os resultados do teste da E7/E8.** Por isso é uma **análise exploratória**: tudo abaixo foi fixado antes de rodar o Haiku no teste, e nada é ajustado depois.

## Decisões

1. **Modelo candidato: só o Claude Haiku 5.5** (`configs/llm/claude_haiku_5_5.yaml`, `effort: medium`, como no piloto). Decisão do autor: o Sonnet fica para depois, se houver motivo.
2. **Mesmas condições da E7:**
   - mesmas variantes, ferramentas, prompts, pacotes de evidência e consultas;
   - 1 repetição;
   - nenhum ajuste de prompt ou ferramenta para o Haiku, porque ajustar no teste seria seleção no teste.
   - **Teto:** US$ 5 por execução (`configs/experiment/agent_haiku_cap.yaml`). O preço 5× do Haiku acima de 100 mil tokens de prompt não deve ocorrer: nas execuções do Opus, o maior prompt teve 57 mil tokens.
3. **Comparação principal dentro do Haiku: A3 × A1.** É a mesma família da E8 (A3 × A1, A2 × A1, A3 × A2), com diferença pareada da nota parcial, McNemar exato e correção de Holm. A pergunta é se a evidência ajuda um modelo mais fraco.
4. **"Igualar o Opus" — não inferioridade, margem 0,10 na nota parcial oficial** (aprovada pelo autor em 2026-11-20):
   - uma variante do Haiku é **não inferior** a uma variante do Opus quando o limite inferior do IC de 95% da diferença pareada (Haiku − Opus) fica acima de −0,10;
   - é **inferior** quando o limite superior fica abaixo de −0,10;
   - nos demais casos, **inconclusivo**.
   - **Afirmação registrada:** "Haiku A2 (ou A3) iguala o Opus" só vale se a variante for não inferior a **todas** as variantes do Opus (A1, A2 e A3). Por ser um teste de interseção-união, não precisa de correção de multiplicidade. O Haiku A1 aparece só como contexto, para mostrar a distância sem evidência.
5. **Referências:** cada variante do Haiku contra o A0 e contra a cadeia do pacote sem LLM, com a mesma regra da E8.
6. **Diagnósticos descritivos (sem teste):**
   - **Concordância com o pacote:** nas respostas de uma causa, a fração com o mesmo instante (até 1 minuto), o mesmo componente ou o mesmo motivo que a resposta do pacote. Mostra se o modelo raciocina ou só copia o pacote.
   - **Instante no início da janela:** a fração de instantes respondidos até 1 minuto do início da janela da consulta, a falha do piloto.
   - **Também entram:** a localização (componente nas consultas de uma falha), a quebra de erros por motivo e tipo de componente, a fidelidade automática (as mesmas checagens da E8, sem gabarito) e o custo.
7. **Fonte das notas:**
   - as respostas do Haiku passam pelo script oficial;
   - as do Opus vêm dos relatórios oficiais da execução da E8 registrada em `reports/tables/agent_eval_<sistema>.json`, e o código confere que batem com essa tabela;
   - o A0 e o pacote vêm do `evidence-select`, como na E8.
8. **Sem auditoria manual do Haiku.** A planilha da E8 é única e nunca reescrita.

## Calibração do critério (feita antes das execuções, sem dados do Haiku)

O comando foi rodado com o próprio Opus no lugar do candidato, como teste de fumaça. O resultado reproduziu todos os números da E8.

| Comparação | Telecom | Bank |
|---|---|---|
| Opus A2 contra todas as variantes do Opus | não iguala | não iguala |
| Opus A3 contra todas as variantes do Opus | não iguala | iguala |

O critério é exigente. **No Telecom (45 consultas), nem o Opus consegue mostrar que não é inferior às próprias variantes**: as variantes empatam, mas os ICs pareados têm meia-largura de cerca de 0,11 a 0,14.

Consequência para a leitura do resultado: no Telecom, o resultado provável é "inconclusivo" ou "inferior", e "inconclusivo" ali não é evidência contra o Haiku. O Bank, com 120 consultas, é onde a pergunta pode ter resposta.

O critério não foi mudado por causa disso. Mudá-lo para caber no resultado seria escolher a régua depois de medir.

## Implementação

- **`evaluation/model_compare.py`:** comando `compare-models`, com a seção `model_compare` do `base.yaml` e `configs/experiment/model_compare.yaml`.
  - Reaproveita o protocolo da E8: `reference_chains` e `score_agent_run`, extraídos de `agent_eval.py` sem mudar o comportamento.
  - **Saída:** `reports/tables/model_compare_<sistema>_<modelo>.{json,md}`.
- **Testes:** `tests/test_model_compare.py`, com valores calculados à mão para:
  - não inferioridade e interseção-união;
  - leitura das respostas numeradas e simples;
  - tolerância de 1 minuto;
  - concordância com o pacote;
  - instante no início da janela;
  - configuração.

## Execução (o autor inicia; chamadas pagas)

Para cada sistema (o Bank com `configs/system/bank.yaml`) e cada variante:

```
uv run tcc-rca agent --config configs/base.yaml [configs/system/bank.yaml] configs/experiment/agent_aN.yaml configs/experiment/agent_1rep.yaml configs/llm/claude_haiku_5_5.yaml configs/experiment/agent_haiku_cap.yaml
uv run tcc-rca compare-models --config configs/base.yaml [configs/system/bank.yaml] configs/experiment/model_compare.yaml
```

## Resultado

Pendente.
