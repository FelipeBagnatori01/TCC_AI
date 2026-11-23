# Recomendações para aplicação em ambiente de telecomunicações (E9)

Este trabalho foi feito com recursos próprios e os dados públicos do OpenRCA. Não houve acesso ao ambiente de nenhuma operadora ou fornecedor. As recomendações abaixo valem para um ambiente de telecomunicações em geral.

- **Os números citados** são os medidos aqui (`docs/RESULTADOS.md`, `docs/FICHA_AGENTE.md`, `docs/MODEL_CARD_LSTM.md`), num benchmark com falhas injetadas (ameaça nº 30). Servem de ordem de grandeza, não de promessa de desempenho em produção.

## 1. O que o trabalho indica

- **Detecção:** um detector sobre séries de KPIs por minuto detecta o início de falhas com AUC-PR de 0,35 a 0,52.
  - Um z-score com linha de base local, sem aprendizado, empata em um dos dois sistemas. **Comece pelo baseline simples** e só adote a LSTM onde ela ganhar na sua telemetria.
- **Localização:** a explicação SHAP da LSTM previsora é o melhor localizador de componente, acima do ranqueamento nativo e das heurísticas (HR@1 de 0,42 a 0,48).
- **Agente:** um agente de LLM com ferramentas de leitura da telemetria acerta bem mais que a evidência sozinha (nota oficial de 0,58 a 0,60 no Telecom, contra 0,41).
  - O ganho vem do modelo forte. Dar evidências LSTM/XAI ao modelo forte não mudou o resultado.
  - Modelos mais baratos se beneficiam da evidência, mas ficam no nível da evidência sem LLM.
- **Fidelidade:** com IDs de evidência e validação da resposta, o agente não citou componentes, KPIs ou IDs inexistentes. Cerca de 1% dos números declarados não tinha correspondência nos dados consultados.

## 2. Dados necessários

| Dado | Para quê | Requisitos mínimos |
|---|---|---|
| Métricas de KPIs por componente (nós, contêineres, bancos, rede) | Séries do detector e ferramentas do agente | Passo de até 1 minuto; relógios sincronizados (UTC); nomes estáveis de componente e KPI; catálogo com unidade e categoria de cada KPI |
| Traces ou registros de chamadas entre serviços | Latência, volume e erros por serviço; arestas de dependência | Identificação de origem e destino; agregação por minuto |
| Topologia e inventário de implantação | Onde cada serviço roda; quem chama quem | Atualizado a cada mudança de topologia |
| Registro de incidentes | Rótulos para treinar e avaliar | Início, componente e motivo de cada incidente, em vocabulário controlado (o OpenRCA tem 5 a 8 motivos por sistema) |
| Logs (opcional) | Contagens de templates (drain3), como no Bank | Volume controlado; templates aprendidos só no período de treino |

**Quantidade:**
- **Retreino:** semanas de dados normais por domínio.
- **Avaliação:** dezenas a centenas de incidentes rotulados. Com 45 falhas, os ICs deste trabalho ficaram largos (± 0,13 na nota).
- **Divisão:** sempre cronológica (treino antes do teste), nunca aleatória.

## 3. Integração

**Pipeline sugerido:**
1. **Ingestão:** plataforma de observabilidade, depois séries por minuto por componente e KPI.
2. **Detecção** (baseline ou LSTM) por domínio, com limiar calibrado num período de validação.
3. **Agregação de alarmes:** supressão de repetição e correlação por topologia. Neste trabalho, o detector sozinho deu de 6 a 14 alarmes falsos por hora, inviável sem essa etapa.
4. **Pacote de evidências** para o incidente aberto: janela, pontuação e componentes/KPIs mais atribuídos.
5. **Agente de LLM** com ferramentas **somente leitura**: consulta de KPIs, varredura de anomalias, traces, dependências e evidências.
6. **Saída estruturada** (instante, componente, motivo, explicação com evidências citadas) anexada ao ticket do sistema de gestão de incidentes, **para revisão de um engenheiro**.

**Pontos de integração:**
- **API de consulta da observabilidade:** as ferramentas do agente.
- **Inventário e topologia:** as dependências.
- **Gestão de incidentes:** gatilho e destino da resposta.
- **Cofre de segredos:** a chave do provedor de LLM.

**Garantias que vale manter:**
- O agente nunca tem ferramenta de escrita ou de ação. O isolamento foi testado neste trabalho.
- A resposta é validada contra um esquema e contra os IDs emitidos, com nova tentativa quando inválida.
- Toda chamada fica registrada: prompt, resposta, tokens, custo e latência.

## 4. Custos medidos (por incidente investigado)

| Modelo | Custo por consulta | Latência média | Nota (Telecom / Bank, melhor variante) |
|---|---|---|---|
| Claude Opus 5.5 | US$ 0,16 a 0,23 | 28 a 36 s | 0,60 / 0,46 |
| Claude Sonnet 5.5 | US$ 0,035 a 0,054 | — | 0,31 / 0,32 |
| Claude Haiku 5.5 | US$ 0,006 a 0,008 | — | 0,34 / 0,29 |
| Pacote de evidências sem LLM | 0 | segundos | 0,41 / 0,35 |

**Leituras:**
- **Exemplo de escala:** 100 incidentes por dia custam cerca de US$ 20 por dia com o modelo grande, ou cerca de US$ 600 por mês, aos preços de novembro de 2026. Os preços mudam; reavalie.
- **O cache** de chamadas permite reauditar decisões sem custo.
- **Modelo barato:** aqui, ele não substituiu o grande. Se o orçamento for o limite, **o pacote de evidências sem LLM já entrega boa parte do valor**, a custo zero de LLM.
- **Treino e inferência da LSTM:** baratos. Alguns segundos por modelo em CPU e 0,25 ms por janela. O XAI leva de 1 a 3 minutos por sistema.

## 5. Riscos e como tratá-los

| Risco | Tratamento recomendado |
|---|---|
| Telemetria e topologia enviadas a um serviço externo de LLM | Avaliação de privacidade e de contrato (retenção, região de processamento); minimizar o que vai no prompt; considerar modelos hospedados em ambiente controlado (não avaliados aqui) |
| Explicação convincente, mas errada | Exigir evidências citadas por ID; checar automaticamente nomes e números contra as saídas das ferramentas; revisão humana antes de qualquer ação |
| Alarmes falsos e fadiga | Agregação e supressão; limiar por domínio; acionar o agente só em incidentes confirmados |
| Deriva (novos componentes, mudança de carga) | Normalização causal local, que foi a escolhida aqui; retreino periódico com divisão temporal; monitorar a taxa de alarmes |
| Dependência de um fornecedor de LLM | Camada própria de abstração do provedor (troca por configuração); avaliação periódica de alternativas no mesmo protocolo |
| Resultado de benchmark não se transfere | Piloto em sombra com métricas próprias (seção 6) antes de qualquer uso operacional |
| Custo fora do controle | Teto de gasto por execução, checado antes de cada chamada paga (implementado aqui); cache |

## 6. Roteiro de piloto

1. **Dados (2 a 4 semanas):**
   - inventariar KPIs, traces e topologia;
   - montar o catálogo de KPIs;
   - reunir incidentes passados rotulados;
   - medir qualidade e cobertura (como a E1 e a E2 deste trabalho).
2. **Detector em sombra:**
   - rodar o baseline local e a LSTM sem alertar ninguém;
   - comparar com os incidentes reais por AUC-PR, alarmes por hora e atraso de detecção.
3. **Agente em sombra:**
   - para cada incidente, gerar o diagnóstico sem mostrá-lo à equipe;
   - comparar depois com o pós-incidente: acerto de componente e motivo, tempo e custo;
   - auditar a fidelidade numa amostra, **com revisão humana**.
4. **Uso assistido:**
   - mostrar o diagnóstico como sugestão no ticket;
   - medir se o tempo até o diagnóstico cai.
5. **Critério para seguir:** fixe antes de cada fase o ganho mínimo sobre o baseline sem LLM e o custo máximo por incidente, e não ajuste o critério depois de ver o resultado (a lição das ameaças nº 21, 24 e 27).
