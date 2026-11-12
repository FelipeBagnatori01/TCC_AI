# Ameaças à validade

Registro incremental das ameaças identificadas ao longo das etapas, para a seção correspondente da monografia (E9). Cada item diz quando e como foi identificado, o efeito esperado e a mitigação adotada.

## Identificadas

### 1. Isenção dos KPIs de disponibilidade escolhida após ver os rótulos (E2)

- **O que:** a categoria "disponibilidade" (`ICMP_ping`, `Agent_ping`, `On_Off_State`, `Hang`, `redis_ping`) é isenta do descarte por constância. A isenção foi proposta depois de constatar, com todos os dias e os rótulos, que esses KPIs mudam em até 17 das 51 falhas (`db close`, `db connection limit`, `network delay/loss`).
- **Tipo:** *data snooping* (grau de liberdade do pesquisador). O código não vaza dados: o seletor é ajustado só no treino. O vazamento está na decisão humana.
- **Efeito esperado:** otimismo nas métricas de detecção e de causa raiz dessas falhas, e menor garantia de generalização para um sistema novo e sem rótulos.
- **Efeito medido (E5):** confirmado no Telecom, em parte. A isenção responde por cerca de metade do HR@1 da LSTM (0,44 contra 0,22) e pela vantagem da previsora; a classificadora não depende dela. No Bank, efeito nulo.
- **Mitigação:** regra aplicada à categoria inteira, com justificativa de domínio (indicadores de disponibilidade são constantes por definição em operação normal); análise de sensibilidade obrigatória com `configs/experiment/sensitivity_sem_isencao.yaml`, reportada ao lado de cada métrica principal (na E4, nenhuma conclusão muda; **na E5 do Telecom, muda**: sem a isenção, o HR@1 da localização da LSTM cai de 0,44 para 0,22 e a previsora deixa de superar os baselines; só a classificadora mantém a vantagem; ver `docs/RESULTADOS.md`); cronologia e auditoria dos 78 atributos afetados no ADR 0005.

### 2. Dias quase duplicados no Telecom (E1, resolvido na E3)

- **O que:** 2020_04_21, 04_22 e 04_23 não têm falhas rotuladas, mas são cópias com 99,99% dos valores de métricas iguais a 05_31, 05_23 e 05_25, inclusive nos minutos das falhas (ADR 0008).
- **Efeito esperado se mantidos:** vazamento da telemetria de teste para o treino e falhas ensinadas à LSTM como normais.
- **Mitigação:** os três dias são excluídos de todos os papéis. Fica a dúvida sobre a origem dessas cópias; a hipótese, não verificada, é que pertenciam a outra fase do AIOps Challenge.

### 3. Só 45 das 51 falhas são testadas (E3)

- **O que:** as 6 falhas de 2020_04_11, o dia mais antigo, ficam no bloco inicial de treino e validação, porque nenhuma dobra cronológica tem treino anterior a elas (ADR 0008).
- **Efeito esperado:** menor amostra de teste, e resultados não comparáveis diretamente aos do artigo do OpenRCA, que usa as 51 consultas.
- **Mitigação:** escolha deliberada para preservar a R2. Todos os resultados indicam o denominador (45).

### 4. Amostra pequena e dobras desiguais (E3)

- **O que:** 2 a 8 falhas por dobra. A dobra 0 treina num único dia (105 janelas normais) e valida num dia sem falhas.
- **Efeito esperado:** ICs largos (cerca de ±15 pontos para um HR@1 de 50% com 45 casos) e uma dobra 0 instável.
- **Mitigação:** métricas agregadas sobre as 45 falhas com IC por bootstrap; dobra 0 reportada à parte; reavaliação do Plano B (Bank) no fim da E4 (ADR 0009).

### 5. A formulação da tarefa A foi decidida com as falhas de teste (E3)

- **O que:** a medição de sinal precursor, que levou a detecção em vez de predição, usou as 45 falhas de maio, como o plano pede para a E3.
- **Efeito esperado:** pequeno. A medição escolhe a formulação, não ajusta parâmetros do modelo. O seletor e o escalonador dessa medição foram ajustados só no bloco inicial, e a regra de decisão foi fixada antes da medição (ADR 0007).
- **Mitigação:** declaração; nenhum hiperparâmetro ou limiar dos modelos vem dessa medição.

### 6. Fim das falhas desconhecido (E3)

- **O que:** o OpenRCA rotula só o início. A margem de 30 min após o início, usada na máscara normal e nas janelas ignoradas, é uma suposição baseada na duração da janela de consulta.
- **Efeito esperado:** falhas mais longas que 30 min contaminariam o treino normal com comportamento de falha. Falhas mais curtas só reduzem os dados normais.
- **Mitigação:** a margem é configurável; a sensibilidade à margem pode ser medida na E5.

### 7. Métricas de janela somadas entre dobras e baselines sem busca de hiperparâmetros (E4)

- **O que:** AUC-PR e AUC-ROC juntam (*pooled*) pontuações de modelos diferentes, um por dobra, cujas escalas podem diferir. Os hiperparâmetros do LightGBM e do Isolation Forest foram fixados sem busca (ADR 0010).
- **Efeito esperado:** a junção pode subestimar a separabilidade dentro de cada dobra. Baselines sem busca podem ficar abaixo do seu melhor possível, o que favoreceria a LSTM.
- **Mitigação:** os limiares são escolhidos por dobra, na validação. A E5 fará a busca da LSTM com orçamento registrado; se a LSTM for comparada a um baseline ajustado, este terá o mesmo orçamento.

### 8. Método local de precursor acrescentado depois de ver o Bank (E3 do Bank)

- **O que:** a pontuação global, fixada antes de medir, saturou no Bank por causa da deriva entre dias. O método local (referência causal nos 60 min anteriores) foi acrescentado depois desse resultado (ADR 0013).
- **Efeito esperado:** escolher a régua depois de ver o resultado pode favorecer a conclusão "as falhas são visíveis".
- **Mitigação:**
  - os dois métodos são reportados;
  - o método local foi validado no Telecom, onde reproduz o sinal já conhecido;
  - a decisão sobre precursor (nenhum nas antecedências positivas) é a mesma nos dois métodos;
  - nenhum hiperparâmetro de modelo vem dessa medição.

### 9. Janela de 20 min com falhas próximas no Bank (E3 do Bank)

- **O que:** 10 pares de falhas do Bank estão a menos de 20 min um do outro (mínimo de 11 min), e 9 janelas de consulta têm duas falhas. Com W = 20, mantido para comparar sistemas, uma janela positiva pode conter também a falha anterior.
- **Efeito esperado:** atribuição ambígua em poucos casos na detecção, e evidências misturadas na localização e no agente.
- **Mitigação:** rótulos e máscara usam todos os inícios; os casos de duas falhas serão analisados à parte na análise de erros (E8).

### 10. Deriva da distribuição no Bank (E3 do Bank)

- **O que:** ao longo das três semanas, KPIs de capacidade e contadores acumulados derivam. Um escalonamento ajustado no início fica cada vez menos representativo.
- **Efeito esperado:** desempenho que cai com a distância entre treino e teste; baselines com normalização fixa penalizados.
- **Mitigação:** origem móvel (o treino cresce até o dia anterior ao teste); método local na medição de precursor; a E5 deve tratar a normalização de forma causal.

### 11. Modelo explicado nos pacotes de evidência escolhido depois de medir (E6)

- **O que:** o plano fixava a LSTM classificadora como detector e como modelo explicado. Depois de medir, as atribuições citáveis passaram para a previsora, porque as da classificadora mudam quase todas entre seeds (ADR 0016, decisão 6). A classificadora continua escolhendo a janela.
- **Tipo:** grau de liberdade do pesquisador (decisão pós-teste). A escolha se apoiou em métricas de qualidade da explicação (estabilidade e concordância), não em métricas do agente, que ainda não existe.
- **Efeito esperado:** no Bank, a localização sugerida pelos pacotes fica um pouco pior (HR@1 0,36 contra 0,44 da classificadora). As citações do agente ficam mais reprodutíveis.
- **Mitigação:** a decisão foi tomada antes de qualquer execução do agente e está registrada. As métricas dos dois modelos ficam reportadas na E6.

### 12. Explicações da classificadora instáveis entre seeds (E6)

- **O que:** cinco classificadoras com o mesmo desempenho de detecção apontam KPIs diferentes. O Jaccard dos 10 mais importantes entre seeds fica em 0,20–0,26, e o Kendall é negativo. O efeito Rashomon aparece aqui com clareza.
- **Efeito esperado:** qualquer conclusão do tipo "a classificadora usa o KPI X" vale para uma seed, não para o modelo.
- **Mitigação:** os pacotes usam a previsora (item 11). A monografia deve reportar a instabilidade, e não exemplos escolhidos de uma seed.

### 13. Categoria de motivo por soma de atribuições (E6)

- **O que:** a categoria sugerida é a de maior soma de atribuições positivas. Categorias com muitos KPIs, como "disco", acumulam mais peso. A categoria sugerida **não supera** a categoria mais frequente do treino em nenhum caso, e a da previsora fica abaixo dela.
- **Efeito esperado:** se o agente confiar na categoria sugerida, o diagnóstico de motivo (RQ3) piora.
- **Mitigação:** a agregação **não** foi trocada depois de ver o resultado (média por categoria, por exemplo), para não ajustar no teste. O pacote mantém os pesos por categoria, mas o agente deve raciocinar sobre os KPIs individuais. O prompt da E7 deve dizer isso.

### 14. Atribuição não é causalidade (E6)

- **O que:** SHAP, LIME e IG dizem a que o modelo reagiu, não o que causou a falha. Um KPI de um componente vizinho, afetado pela falha, pode receber mais atribuição que a causa raiz.
- **Mitigação:** a monografia separa "o modelo reagiu a este KPI" de "este componente causou a falha", e cita descoberta causal (Ikram et al., 2022) como abordagem complementar. A localização via XAI é medida contra o gabarito, não presumida.

## Previstas pelo plano (a detalhar na E9)

- Tamanho amostral: 51 casos no Telecom (45 testados; ver item 4).
- Contaminação do LLM: o modelo pode ter visto o artigo ou dados do OpenRCA no treino (comparação A1 × A3 com o mesmo modelo controla parte disso).
- Dependência de um único modelo de LLM.
- Diferença entre o benchmark e ambientes de produção.
