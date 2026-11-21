# ADR 0023 — Modelo mais barato com as mesmas ferramentas e evidências (exploratório)

- **Status:** aceito (2026-11-20), antes de qualquer execução do Haiku no teste. Executado em 2026-11-20 (seis execuções autorizadas pelo autor); resultado abaixo
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

## Resultado (2026-11-20)

**Execuções:** seis, uma por variante e sistema, todas com 100% de respostas válidas, nenhuma parada pelo teto. Custo total de US$ 3,35, contra US$ 103,88 do Opus.
- **Tabelas:** `reports/tables/agent_<sistema>_agent_aN_claude-haiku-5-5.{json,md}` (execuções) e `reports/tables/model_compare_<sistema>_claude-haiku-5-5.{json,md}` (comparação).
- **Reprodução a partir do cache:** não foi feita para o Haiku (não estava prevista neste ADR).

### Nota parcial oficial (IC de 95% por bootstrap sobre consultas)

| Cadeia | Telecom (45) | Bank (120) |
|---|---|---|
| A0 / pacote sem LLM | 0,18 / 0,41 | 0,14 / 0,35 |
| Opus A1 / A2 / A3 | 0,60 / 0,60 / 0,58 | 0,43 / 0,40 / 0,46 |
| Haiku A1 | 0,14 [0,06; 0,23] | 0,16 [0,11; 0,21] |
| Haiku A2 | 0,26 [0,15; 0,37] | 0,19 [0,13; 0,25] |
| Haiku A3 | 0,34 [0,21; 0,47] | 0,29 [0,22; 0,36] |

### Comparação principal: Haiku A3 × Haiku A1

| Sistema | Diferença parcial | Veredito | McNemar (só A3 / só A1) | p Holm |
|---|---|---|---|---|
| Telecom | +0,20 [0,04; 0,36] | superior | 12/3 | 0,105 |
| Bank | +0,13 [0,07; 0,19] | superior | 13/1 | 0,005 |

- **No Haiku, a evidência LSTM/XAI ajuda**, ao contrário do Opus, em que A3 × A1 empatou. No acerto estrito, o McNemar só é significativo no Bank.
- **A2 × A1** empata nos dois sistemas. **A3 × A2** é superior no Bank (+0,10 [0,04; 0,16]) e empata no Telecom (+0,08 [−0,00; 0,18]). O ganho vem principalmente das atribuições SHAP/LIME, não só da pontuação e do ranqueamento.

### O Haiku iguala o Opus? Não

- **Pelo veredito pareado usual,** todas as variantes do Haiku são inferiores a todas as do Opus nos dois sistemas: diferenças de −0,12 a −0,47, todos os ICs abaixo de zero.
- **Pela não inferioridade (margem 0,10),** nenhuma combinação é `nao_inferior`:
  - Haiku A1 e A2 são `inferior` contra todas as variantes do Opus nos dois sistemas.
  - Haiku A3 é `inconclusivo` contra Opus A1 e A3 no Telecom e `inferior` contra Opus A2. No Bank é `inconclusivo` contra as três.
  - Aqui `inconclusivo` não vem da falta de poder prevista na calibração. As estimativas pontuais (−0,12 a −0,26) já estão além da margem, e só o limite superior do IC (−0,05 a −0,10 nos casos inconclusivos) a toca.
- A afirmação registrada ("Haiku A2 ou A3 iguala o Opus") **não vale** em nenhum sistema.

### Contra as referências sem LLM

- **Haiku A3:** empata com a cadeia do pacote nos dois sistemas (−0,07 no Telecom e −0,06 no Bank). Supera o A0 no Bank e empata no Telecom.
- **Haiku A1 e A2:** perdem para a cadeia do pacote e empatam com o A0.
- Ou seja, o Haiku com toda a evidência fica no nível da evidência lida sem LLM.

### Diagnósticos descritivos

- **Instante no início da janela, a falha do piloto:** continua.
  - **Haiku:** 0,75 / 0,61 / 0,50 dos instantes no Telecom (A1/A2/A3) e 0,68 / 0,70 / 0,62 no Bank.
  - **Opus:** de 0,07 a 0,11 no Telecom e de 0,18 a 0,26 no Bank.
  - **Pacote:** 0,04 e 0,09.
  - A evidência reduz essa falha só em parte.
- **Localização,** nas consultas de uma falha que pedem componente: o maior ganho do Haiku com a evidência.
  - **Telecom:** 0,09 → 0,36 → 0,55, contra 0,41 do pacote e de 0,68 a 0,73 do Opus.
  - **Bank:** 0,21 → 0,21 → 0,30, contra 0,25 do pacote e de 0,45 a 0,55 do Opus.
- **Concordância com o pacote,** nas respostas de uma causa:
  - O Haiku A3 repete o componente do pacote mais que o Opus A3: 0,59 contra 0,50 no Telecom e 0,34 contra 0,29 no Bank.
  - No instante, o Haiku A3 concorda menos com o pacote que o Opus A3: 0,39 contra 0,50 no Telecom e 0,28 contra 0,31 no Bank.
  - Leitura: o Haiku aproveita a evidência sobretudo para o componente, e o instante continua sendo o seu ponto fraco.
- **Fidelidade automática:**
  - nenhum ID não emitido;
  - componentes desconhecidos: 1 de cerca de 300 (Telecom A1);
  - KPIs desconhecidos: no máximo 5 de 116 (Bank A1);
  - números sem correspondência: de 0,3% a 3,9%.
  - O controle ×1,37 casa em 25% a 38% dos casos, acima do Opus (21% a 27%), então a checagem de números discrimina menos no Haiku.
- **Custo por consulta:** US$ 0,006 a 0,008, contra US$ 0,16 a 0,23 do Opus, cerca de 30 vezes menos.

### Leitura

- **Para a RQ3:** a evidência LSTM/XAI tem efeito detectável quando o modelo é mais fraco. O Haiku A3 supera o Haiku A1 nos dois sistemas na nota parcial, sobretudo porque passa a acertar o componente. Mesmo assim, não chega perto do Opus sem evidência e só empata com a cadeia do pacote sem LLM.
- **Para a RQ4:** o Haiku custa cerca de 30 vezes menos por consulta, mas mesmo o Haiku A3 fica de 0,12 a 0,26 abaixo do Opus na nota parcial. Na prática, com estas ferramentas, a troca não compensa: o pacote sozinho, sem custo de LLM, entrega o mesmo que o Haiku A3.
- **Por ser exploratório** (ameaça nº 27), o efeito da evidência no Haiku é uma hipótese a confirmar, por exemplo com repetições ou outro modelo pequeno, e não uma conclusão principal do trabalho.

## Adendo: Claude Sonnet 5.5 (2026-11-20, antes de rodar o Sonnet no teste)

- **Origem:** o autor pediu o Sonnet depois de ver o resultado do Haiku. Na decisão 1, o Sonnet tinha ficado "para depois, se houver motivo". O motivo: o Haiku não alcançou o Opus, e o Sonnet é o degrau intermediário de preço.
- **Protocolo:** idêntico ao do Haiku.
  - Mesmas variantes, ferramentas, prompts, consultas e 1 repetição.
  - Mesmo esforço (`medium`).
  - Mesma comparação principal (Sonnet A3 × Sonnet A1, Holm).
  - Mesma margem de 0,10, mesma regra de interseção-união e mesmos diagnósticos.
  - Nenhum ajuste para o Sonnet.
- **Preço:** US$ 2 / 10 por MTok, leitura de cache a US$ 0,10 e escrita a 2,50, conferidos em 2026-11-20; sem acréscimo para prompt longo. É o que está em `configs/llm/claude_sonnet_5_5.yaml`.
  - **Teto:** US$ 15 por execução (`configs/experiment/agent_sonnet_cap.yaml`).
  - **Custo estimado:** cerca de US$ 17 no total, pelo custo do piloto vezes a razão teste/piloto de cerca de 7 observada no Opus e no Haiku.
- **Piloto (ADR 0017):** o Sonnet teve a mesma falha do Haiku, com o início da janela em 9 de 14 instantes.
- **Comparação Sonnet × Haiku:** não é feita como teste. Os dois aparecem lado a lado só de forma descritiva, porque essa pergunta não foi fixada antes.
- **Comandos:** os mesmos da seção "Execução", com `configs/llm/claude_sonnet_5_5.yaml configs/experiment/agent_sonnet_cap.yaml` no lugar dos arquivos do Haiku, e `compare-models` com `configs/experiment/model_compare_sonnet.yaml`.
- **Saída:** `reports/tables/model_compare_<sistema>_claude-sonnet-5-5.{json,md}`.

### Resultado do Sonnet (2026-11-20)

**Execuções:** seis, todas com 100% de respostas válidas, nenhuma parada pelo teto. Custo total de US$ 21,48.
- **Bank A3:** os créditos da conta acabaram na 51ª de 120 consultas (`experiments/20261120T213243-agent_a3`, US$ 2,70). Depois da recarga, a execução foi refeita (`20261121T015443-agent_a3`). As 50 primeiras consultas saíram do cache: mesmas requisições, mesmas respostas, custo zero. O custo real do Bank A3 é US$ 3,73 + 2,70 = US$ 6,43 (US$ 0,054 por consulta). A tabela `model_compare` mostra só os US$ 3,73.

| Nota parcial | Telecom (45) | Bank (120) |
|---|---|---|
| Opus A1 / A2 / A3 | 0,60 / 0,60 / 0,58 | 0,43 / 0,40 / 0,46 |
| Sonnet A1 / A2 / A3 | 0,17 / 0,24 / 0,31 | 0,16 / 0,23 / 0,32 |
| Haiku A1 / A2 / A3 (acima) | 0,14 / 0,26 / 0,34 | 0,16 / 0,19 / 0,29 |
| Sonnet A3 − Sonnet A1 | +0,14 [−0,01; 0,29], empate (McNemar 9/4, p Holm 0,80) | +0,16 [0,10; 0,23], superior (19/1, p Holm < 0,001) |

- **Comparação principal:**
  - **Bank:** a evidência melhora o Sonnet, com as três comparações da família superiores (A2 × A1 +0,07, A3 × A2 +0,09).
  - **Telecom:** mesma direção, mas o IC encosta no zero.
- **O Sonnet não iguala o Opus:**
  - **Telecom:** todas as variantes são `inferior` também pela não inferioridade (−0,26 a −0,43).
  - **Bank:** o Sonnet A3 é `inconclusivo` contra as três variantes do Opus (−0,08 a −0,13) e inferior pelo veredito usual. É o mais perto do Opus que um modelo mais barato chegou, mas a afirmação registrada não vale.
- **Contra as referências:**
  - O Sonnet A3 empata com a cadeia do pacote sem LLM nos dois sistemas (−0,10 no Telecom, −0,03 no Bank).
  - O Sonnet A3 supera o A0 no Bank e empata no Telecom.
  - O Sonnet A1 perde para o pacote nos dois sistemas.
- **Diagnósticos:**
  - **Instante no início da janela, A1 / A2 / A3:** 0,46 / 0,50 / 0,32 no Telecom e 0,55 / 0,45 / 0,39 no Bank. É menos que o Haiku e mais que o Opus.
  - **Localização, A1 → A3:** 0,09 → 0,45 no Telecom e 0,16 → 0,32 no Bank.
  - **Concordância do Sonnet A3 com o pacote:** componente 0,59 / 0,41 e instante 0,54 / 0,36 (Telecom / Bank), acima do Opus A3. Como no Haiku, o ganho vem em boa parte de adotar a evidência.
- **Fidelidade automática:**
  - nenhum ID, componente ou KPI inventado;
  - números sem correspondência: no máximo 2% em quase todas as execuções, mas 9% (12/131) no Sonnet A1 do Telecom;
  - o controle ×1,37 casa em 11% a 24%.
- **Custo por consulta:** US$ 0,035 a 0,054, cerca de 4 a 5 vezes menos que o Opus e de 5 a 8 vezes mais que o Haiku.

**Leitura conjunta (Haiku e Sonnet, descritiva):**
- Os dois modelos mais baratos ficam no mesmo patamar: A3 entre 0,29 e 0,34, empatando com a cadeia do pacote sem LLM e bem abaixo do Opus.
- Nos dois, a evidência LSTM/XAI aumenta a nota da A1 para a A3. A diferença pareada é superior em 3 dos 4 casos (o McNemar estrito só em 2: os dois no Bank); no Opus, não houve efeito.
- A evidência parece compensar parte do que falta ao modelo mais fraco (componente e instante), mas não o leva ao nível do Opus.
- Neste estudo, pagar mais pelo modelo trouxe mais ganho que dar mais evidência a um modelo barato.
