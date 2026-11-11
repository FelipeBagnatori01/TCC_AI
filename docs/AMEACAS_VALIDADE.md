# Ameaças à validade

Registro incremental das ameaças identificadas ao longo das etapas, para a seção correspondente da monografia (E9). Cada item diz quando e como foi identificado, o efeito esperado e a mitigação adotada.

## Identificadas

### 1. Isenção dos KPIs de disponibilidade escolhida após ver os rótulos (E2)

- **O que:** a categoria "disponibilidade" (`ICMP_ping`, `Agent_ping`, `On_Off_State`, `Hang`, `redis_ping`) é isenta do descarte por constância. A isenção foi proposta depois de constatar, com todos os dias e os rótulos, que esses KPIs mudam em até 17 das 51 falhas (`db close`, `db connection limit`, `network delay/loss`).
- **Tipo:** *data snooping* (grau de liberdade do pesquisador). O código não vaza dados: o seletor é ajustado só no treino. O vazamento está na decisão humana.
- **Efeito esperado:** otimismo nas métricas de detecção e de causa raiz dessas falhas, e menor garantia de generalização para um sistema novo e sem rótulos.
- **Mitigação:** regra aplicada à categoria inteira, com justificativa de domínio (indicadores de disponibilidade são constantes por definição em operação normal); análise de sensibilidade obrigatória com `configs/experiment/sensitivity_sem_isencao.yaml`, reportada ao lado de cada métrica principal; cronologia e auditoria dos 78 atributos afetados no ADR 0005.

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

## Previstas pelo plano (a detalhar na E9)

- Tamanho amostral: 51 casos no Telecom (45 testados; ver item 4).
- Contaminação do LLM: o modelo pode ter visto o artigo ou dados do OpenRCA no treino (comparação A1 × A3 com o mesmo modelo controla parte disso).
- Dependência de um único modelo de LLM.
- Diferença entre o benchmark e ambientes de produção.
