# Ameaças à validade

Registro incremental das ameaças identificadas ao longo das etapas, para a seção correspondente da monografia (E9). Cada item diz quando e como foi identificado, o efeito esperado e a mitigação adotada.

## Identificadas

### 1. Isenção dos KPIs de disponibilidade escolhida após ver os rótulos (E2)

- **O que:** a categoria "disponibilidade" (`ICMP_ping`, `Agent_ping`, `On_Off_State`, `Hang`, `redis_ping`) é isenta do descarte por constância. A isenção foi proposta depois de constatar, com todos os dias e os rótulos, que esses KPIs mudam em até 17 das 51 falhas (`db close`, `db connection limit`, `network delay/loss`).
- **Tipo:** *data snooping* (grau de liberdade do pesquisador). O código não vaza dados: o seletor é ajustado só no treino. O vazamento está na decisão humana.
- **Efeito esperado:** otimismo nas métricas de detecção e de causa raiz dessas falhas, e menor garantia de generalização para um sistema novo e sem rótulos.
- **Mitigação:** regra aplicada à categoria inteira, com justificativa de domínio (indicadores de disponibilidade são constantes por definição em operação normal); análise de sensibilidade obrigatória com `configs/experiment/sensitivity_sem_isencao.yaml`, reportada ao lado de cada métrica principal; cronologia e auditoria dos 78 atributos afetados no ADR 0005.

### 2. Dias quase duplicados no Telecom (E1)

- **O que:** três pares de dias têm o mesmo número de spans; no par 2020_04_22 × 2020_05_23, `metric_node` difere em só 158 de 301.617 valores (`docs/DICIONARIO_DADOS.md`).
- **Efeito esperado:** treino e teste de dias diferentes podem conter o mesmo comportamento normal, o que infla métricas de detecção.
- **Mitigação:** a definir na E3 (divisão temporal e tratamento dos pares).

## Previstas pelo plano (a detalhar na E9)

- Tamanho amostral: 51 casos no Telecom.
- Contaminação do LLM: o modelo pode ter visto o artigo ou dados do OpenRCA no treino (comparação A1 × A3 com o mesmo modelo controla parte disso).
- Dependência de um único modelo de LLM.
- Diferença entre o benchmark e ambientes de produção.
