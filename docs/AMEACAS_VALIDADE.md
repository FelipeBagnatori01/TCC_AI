# Ameaças à validade

Registro incremental das ameaças identificadas ao longo das etapas, para a seção correspondente da monografia (E9). Cada item diz quando e como foi identificado, o efeito esperado e a mitigação adotada.

## Identificadas

### 1. Isenção dos KPIs de disponibilidade escolhida após ver os rótulos (E2)

- **O que:** a categoria "disponibilidade" (`ICMP_ping`, `Agent_ping`, `On_Off_State`, `Hang`, `redis_ping`) é isenta do descarte por constância. A isenção foi proposta depois de constatar, com todos os dias e os rótulos, que esses KPIs mudam em até 17 das 51 falhas (`db close`, `db connection limit`, `network delay/loss`).
- **Tipo:** *data snooping* (grau de liberdade do pesquisador). O código não vaza dados: o seletor é ajustado só no treino. O vazamento está na decisão humana.
- **Efeito esperado:** otimismo nas métricas de detecção e de causa raiz dessas falhas, e menor garantia de generalização para um sistema novo e sem rótulos.
- **Efeito medido (E5):** confirmado no Telecom, em parte. A isenção responde por cerca de metade do HR@1 da LSTM e pela vantagem da previsora; a classificadora não depende dela. No Bank, efeito nulo. Com o ADR 0019 (2026-11-17), os números passaram a ser: HR@1 de 0,42 com isenção contra 0,20 sem ela; a previsora vai de superior ao z-score local (4/5 seeds) a empate.
- **Mitigação:** regra aplicada à categoria inteira, com justificativa de domínio (indicadores de disponibilidade são constantes por definição em operação normal); análise de sensibilidade obrigatória com `configs/experiment/sensitivity_sem_isencao.yaml`, reportada ao lado de cada métrica principal (na E4, nenhuma conclusão muda; **na E5 do Telecom, muda**: sem a isenção, o HR@1 da localização da LSTM cai de 0,42 para 0,20 e a previsora deixa de superar o z-score local; ver `docs/RESULTADOS.md`); cronologia e auditoria dos 78 atributos afetados no ADR 0005.

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
- **Mitigação:** os limiares são escolhidos por dobra, na validação. Desde o ADR 0019, toda tabela de detecção traz também a AUC-PR média por dobra, com IC. A E5 fará a busca da LSTM com orçamento registrado; se a LSTM for comparada a um baseline ajustado, este terá o mesmo orçamento.

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
- **Efeito medido após o ADR 0019:** a desvantagem desapareceu. Com a pontuação nos últimos minutos, a previsora passou a localizar melhor que a classificadora no Bank (HR@1 0,48 contra 0,44 com SHAP) e igual ou melhor no Telecom (0,42 contra 0,27).
- **Mitigação:** a decisão foi tomada antes de qualquer execução do agente e está registrada. As métricas dos dois modelos ficam reportadas na E6.

### 12. Explicações da classificadora instáveis entre seeds (E6)

- **O que:** cinco classificadoras com o mesmo desempenho de detecção apontam KPIs diferentes. O Jaccard dos 10 mais importantes entre seeds fica em 0,20–0,26, e o Kendall é negativo. O efeito Rashomon aparece aqui com clareza.
- **Efeito esperado:** qualquer conclusão do tipo "a classificadora usa o KPI X" vale para uma seed, não para o modelo.
- **Mitigação:** os pacotes usam a previsora (item 11). A monografia deve reportar a instabilidade, e não exemplos escolhidos de uma seed.

### 13. Categoria de motivo por soma de atribuições (E6)

- **O que:** a categoria sugerida é a de maior soma de atribuições positivas. Categorias com muitos KPIs, como "disco", acumulam mais peso. Até o ADR 0019, a categoria sugerida **não superava** a categoria mais frequente do treino em nenhum caso. Depois dele, a do SHAP da previsora passou a superá-la no Bank (+0,17), mas continua inferior no Telecom (−0,24).
- **Efeito esperado:** se o agente confiar na categoria sugerida, o diagnóstico de motivo (RQ3) piora.
- **Mitigação:** a agregação **não** foi trocada depois de ver o resultado (média por categoria, por exemplo), para não ajustar no teste. O pacote mantém os pesos por categoria, mas o agente deve raciocinar sobre os KPIs individuais. O prompt da E7 deve dizer isso. O resultado do Bank pós-ADR 0019 não muda essa decisão: ele não se repete no Telecom, e a categoria foi retirada do que o agente vê antes do teste (ADR 0017).

### 14. Atribuição não é causalidade (E6)

- **O que:** SHAP, LIME e IG dizem a que o modelo reagiu, não o que causou a falha. Um KPI de um componente vizinho, afetado pela falha, pode receber mais atribuição que a causa raiz.
- **Mitigação:** a monografia separa "o modelo reagiu a este KPI" de "este componente causou a falha", e cita descoberta causal (Ikram et al., 2022) como abordagem complementar. A localização via XAI é medida contra o gabarito, não presumida.

### 15. Contaminação do LLM (E7)

- **O que:** o OpenRCA é público. O modelo pode ter visto o artigo, os dados ou respostas do benchmark no treino.
- **Efeito esperado:** acerto inflado em todas as variantes, sem relação com as evidências.
- **Mitigação:** a comparação A1 × A3 com o mesmo modelo controla a parte da contaminação comum às variantes. A contaminação continua sendo uma ameaça aos números absolutos.

### 16. Modelos sem temperatura (E7)

- **O que:** os modelos atuais dos dois provedores rejeitam `temperature`, então não há execução determinística; a R6 previa temperatura 0 "quando suportado".
- **Efeito esperado:** duas execuções novas do mesmo caso podem divergir.
- **Mitigação:**
  - a reprodutibilidade vem do cache: uma nova execução sai inteira dele, sem chamada paga (testado);
  - a variância é medida com 3 repetições independentes por caso no Telecom;
  - no Bank há 1 repetição (escopo aprovado), sem medida de variância.

### 17. A variante A1 inclui uma varredura heurística (E7)

- **O que:** todas as variantes têm `find_anomalous_kpis`, que ordena os KPIs por desvio local, como o z-score local da E4. A ferramenta foi definida antes de qualquer execução real, para que a A1 não precise consultar às cegas cerca de 2 mil KPIs.
- **Efeito esperado:** a A1 fica mais forte que um "LLM só com telemetria bruta", então o ganho de A2/A3 é marginal em relação a essa base, e não ao LLM puro.
- **Mitigação:** a ressalva entra no texto, e a comparação de todas as variantes com A0 continua valendo.

### 18. Modelo e prompt escolhidos num piloto pequeno (E7)

- **O que:** o piloto tem 22 consultas de dias anteriores ao teste (6 do Telecom e 16 do Bank), só com a variante A1.
- **Efeito esperado:** a escolha pode não ser a melhor para o teste, que tem outra distribuição de motivos e componentes; mas não há *snooping*, porque nenhuma consulta de teste é usada.
- **Mitigação:** o piloto e o critério de escolha ficam registrados, e o modelo escolhido é fixado para todas as variantes.

### 19. Modelos alternativos medidos depois de ver os resultados da LSTM (complemento E5/E6)

- **O que:** a comparação com LightGBM, Isolation Forest, regressão logística L1 e TCN foi decidida depois dos resultados da E5/E6 no teste (2026-11-13).
- **Efeito esperado:** nenhum modelo foi ajustado no teste: todos usam o protocolo da E5, com busca só na validação e orçamento de 8 configurações por modelo, fixado antes de rodar. O risco é outro: se um alternativo superar a LSTM, trocar o modelo das evidências do agente seria escolher pelo teste.
- **Mitigação:** os pacotes de evidência da E7 continuam vindo da LSTM (ADR 0016/0017), qualquer que seja o resultado. Se um alternativo vencer, isso é relatado como achado (a LSTM não é necessária), não usado para reconfigurar o agente.

### 20. Comparação justa entre famílias de modelos (complemento E5/E6)

- **O que:**
  - as grades têm o mesmo tamanho (8 configurações, com a normalização incluída), mas cobrem eixos de natureza diferente;
  - a TCN entra só como classificadora, par da `ClassifierLSTM`, e não como previsora;
  - a logística é determinística e roda com 1 seed, sem medida de estabilidade entre seeds;
  - os modelos tabulares leem as mesmas janelas normalizadas que a LSTM (ausente = 0), não as estatísticas com escala robusta e ausentes `NaN` do LightGBM da E4, então os números diferem dos da E4.
- **Efeito esperado:** um modelo pode estar sub ou sobreajustado em relação aos outros.
- **Mitigação:** a busca completa (AUC-PR de validação de cada configuração) fica no relatório. O LightGBM da E4, sem busca, continua como referência.

### 21. Pontuação de janela mudada depois de ver o teste (ADR 0019)

- **O que:** a previsora LSTM e os z-scores passaram a pontuar só os últimos D + 1 = 6 minutos da janela, em vez da janela inteira. A ideia surgiu numa revisão que leu os resultados de teste da E5 (2026-11-16).
- **Tipo:** grau de liberdade do pesquisador (decisão pós-teste).
- **Efeito esperado:** otimismo na detecção dos métodos afetados. O ganho foi grande também na validação (por exemplo, previsora no Telecom de 0,45 para 0,55), então não é ajuste ao teste.
- **Efeito medido (reexecução de 2026-11-17):** previsora LSTM de 0,40 para 0,52 (Telecom) e de 0,20 para 0,35 (Bank); z-score local de 0,24 para 0,39 e de 0,19 para 0,33. Os controles `*_janela_inteira` reproduzem exatamente as tabelas anteriores. A conclusão da RQ1 ficou **menos** favorável à LSTM: no Bank, nenhuma LSTM supera o z-score local, e a classificadora passa a perder para ele.
- **Mitigação:**
  - o valor 6 não foi buscado: vem da definição do rótulo (ADR 0007);
  - a mudança vale para todos os métodos não supervisionados ao mesmo tempo, baselines e LSTM, o que favorece os rivais da LSTM tanto quanto ela;
  - o protocolo anterior é reportado como controle (`*_janela_inteira`);
  - a mudança foi feita antes de qualquer execução de teste do agente;
  - o ranqueamento nativo e as heurísticas de localização não mudaram.

### 22. Possível atalho pelo horário no Bank (ADR 0019)

- **O que:** as falhas do Bank se concentram entre 14h e 20h (UTC+8). Uma referência só de horário, sem telemetria, separa um pouco as janelas (AUC-PR 0,16 contra 0,105 do acaso, no diagnóstico), e a classificadora LSTM tem AUROC 0,59 nos 5 minutos antes do início, onde a E3 não achou sinal.
- **Efeito esperado:** parte do desempenho dos detectores supervisionados pode vir do horário (por KPIs com ciclo diário), não da falha.
- **Mitigação:** o detector de referência `prior_horario` é reportado na E4 ao lado dos demais; um detector que não o supera não demonstra uso da telemetria. A monografia discute o atalho como limitação.

## Previstas pelo plano (a detalhar na E9)

- Tamanho amostral: 51 casos no Telecom (45 testados; ver item 4).
- Contaminação do LLM: o modelo pode ter visto o artigo ou dados do OpenRCA no treino (comparação A1 × A3 com o mesmo modelo controla parte disso).
- Dependência de um único modelo de LLM.
- Diferença entre o benchmark e ambientes de produção.
