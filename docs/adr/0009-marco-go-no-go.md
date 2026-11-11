# ADR 0009 — Marco go/no-go da E3

- **Status:** aceito — GO aprovado pelo autor em 2026-11-11
- **Data:** 2026-11-11
- **Etapa:** E3

## Pergunta

Os dados do Telecom sustentam as tarefas A, B e C com avaliação minimamente defensável, ou é preciso acionar o Plano B?

## Evidências

| Gatilho do Plano B | Medida | Situação |
|---|---|---|
| Poucas falhas por dobra | 45 falhas testadas em 10 dobras (2 a 8 por dobra) | Por dobra, sim. A análise agrega as 45 falhas entre dobras, mas os ICs serão largos: com 45 casos, um HR@1 de 50% tem IC de 95% de cerca de ±15 pontos. |
| Falhas pouco visíveis | AUROC da pontuação simples 1 min após o início: 0,75 (IC 0,67–0,81) | Visíveis no conjunto. Por motivo, `db connection limit` e `network loss` (13 das 45) não se distinguem do normal com essa pontuação. |
| Hardware | Pico de 3,6 GB na E2 e 0,8 GB na E3, numa máquina de 24 GB | Não aciona. |
| Sinal precursor | Nenhuma antecedência positiva com IC acima de 0,5 | Não há predição: a tarefa A vira detecção precoce (ADR 0007). |

## Recomendação: GO, com detecção precoce e Plano B adiado

- **GO no Telecom** com a tarefa A formulada como detecção precoce (H = 0) e o atraso de detecção reportado.
- O sinal existe e está concentrado nos primeiros minutos após o início.
- **O Plano B (Bank) não é acionado agora.**
  - Ponto de reavaliação: o fim da E4.
  - Se os baselines e a pontuação de detecção não separarem `db connection limit` e `network loss`, ou se os ICs da comparação principal (RQ1 e RQ2) ficarem largos demais para qualquer conclusão, o Bank (136 casos) entra como segundo sistema, como o Plano B prevê.
- A amostra pequena já fica registrada como ameaça à validade.

## Consequências

- A E4 implementa a avaliação completa sobre as 10 dobras, com IC por bootstrap sobre as 45 falhas.
- A dobra 0 é tratada à parte (ADR 0008).

## Adendo (2026-11-11, fim da E4): Plano B acionado

- **Critério atingido.** A reavaliação prevista para o fim da E4 confirmou as duas condições:
  - a detecção por evento dos baselines está no nível do acaso, e a localização acerta 0/7 em `network loss`;
  - o IC do HR@1 vai de 0,04 a 0,24 (`docs/RESULTADOS.md`).
- **Decisão do autor: acionar o Plano B.** O Bank (136 casos, com métricas, traces e logs) entra como segundo sistema, com modelo próprio e resultados reportados por sistema. As etapas E1 a E4 serão refeitas para ele, incluindo os logs desde já.
- **Expectativa.** Os dados do Bank não treinam os modelos do Telecom, porque há um modelo por sistema. O ganho esperado é amostral (ICs mais estreitos) e de generalização (as conclusões valem nos dois sistemas?).
- **Market fora do escopo.** O autor decidiu não usar o Market (2 dias por *cloudbed*, o que inviabiliza a divisão cronológica por dias) e removê-lo do disco. Se for necessário, será baixado de novo.
