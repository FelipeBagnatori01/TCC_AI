# ADR 0013 — Tarefas, divisão temporal e marco go/no-go do Bank (Plano B, E3)

- **Status:** aceito (E3 do Bank e GO aprovados em 2026-11-11)
- **Data:** 2026-11-11
- **Etapa:** Plano B, E3 do Bank

## Contexto

O Bank tem 10 dias de 24 h contíguos, 136 falhas (até 21 por dia), intervalo mínimo de 11 min entre inícios e 9 janelas de consulta com duas falhas. A E2 fixou o bloco inicial 2021_03_04–05 (ADR 0012).

## Decisões

1. **Mesma definição de tarefa do Telecom** (ADR 0007): W = 20, D = 5, margens de 5/30 min, rótulos 1 / 0 / ignorado.
   - Decisão do autor (2026-11-11): manter W = 20 para que os resultados sejam comparáveis entre sistemas, mesmo com 10 pares de falhas a menos de 20 min. Nesses casos, uma janela positiva pode conter o fim da falha anterior (ameaça nº 9).
   - Rótulos e máscara normal usam todos os inícios, inclusive os das janelas com duas falhas e os que atravessam a meia-noite.
2. **Dobras:**
   - bloco inicial 03/04–05; 8 dobras de origem móvel, testando 03/06, 07, 09, 10, 12, 23, 24 e 25;
   - o dia anterior ao teste é a validação;
   - 120 das 136 falhas testadas. As 16 de 03/04 ficam no bloco inicial; as 3 de `high JVM CPU load` caem todas nele, então esse motivo nunca é testado.
   - A execução verifica que o bloco inicial da E3 é exatamente o `series.initial_block_days` da E2.
3. **Guarda por expurgo (vale para os dois sistemas):**
   - Janelas de treino ou validação que terminam até W+H minutos antes do início do teste perdem o papel. A sobreposição real com o teste não é expurgada, para que a verificação de vazamento continue funcionando.
   - No Bank, saem as últimas 20 janelas do dia de validação nas 5 dobras com dias consecutivos; a distância mínima passa a 21 min (> W+H).
   - No Telecom não sai nada (18 h entre dias); janelas, rótulos e baselines ficaram idênticos.
4. **Sinal precursor: dois métodos.**
   - A pontuação definida na E3 do Telecom ("global": média das 10 maiores |z| com escalonador ajustado no bloco inicial) **satura no Bank**. A distribuição deriva ao longo de três semanas: sistemas de arquivos que crescem devagar, com IQR minúsculo, e contadores acumulados. A mediana da pontuação nos minutos **normais** sobe de 85 (03/04) para 2.000 (03/12) e para 760 milhões (03/24). Com isso, a AUROC fica em 0,50 em todas as antecedências, inclusive depois do início.
   - Por isso foi acrescentado o método **"local"**: |z| de cada KPI contra a mediana e o desvio-padrão dos **60 minutos anteriores do mesmo dia**. É causal e sem rótulos; o desvio tem um piso relativo à magnitude do KPI, e o |z| é limitado a 50.
   - O método foi acrescentado **depois** de ver o resultado global do Bank (ameaça nº 8). Ele foi validado no Telecom, onde reproduz o sinal conhecido (0,78 em +1 min contra 0,75 no global). Os dois métodos são reportados (`reports/tables/precursor_<sistema>.csv`, coluna `method`); os resultados globais do Telecom não mudaram.

| Bank (120 falhas) | 15 min antes | 5 min antes | início | +1 min | +2 min |
|---|---|---|---|---|---|
| global | 0,49 [0,43; 0,54] | 0,51 [0,45; 0,56] | 0,51 | 0,52 | 0,50 |
| local | 0,51 [0,45; 0,56] | 0,52 [0,47; 0,57] | **0,81 [0,76; 0,85]** | **0,79 [0,75; 0,83]** | 0,74 |

   Por motivo, no método local e no início ou em +1 min: `JVM OOM` 0,98 (3 falhas), `high disk I/O read` 0,94, `high CPU` 0,86, `packet loss` 0,84, `high memory` 0,82, `disk space` 0,81, `network latency` 0,70.

5. **Decisão: detecção precoce (H = 0) também no Bank.** Nenhuma antecedência positiva tem IC inteiro acima de 0,5, em nenhum dos dois métodos.

## Marco go/no-go do Bank

- **Recomendação: GO.**
  - As falhas são visíveis logo no início (AUROC local de 0,81), e, tomando o melhor dos dois minutos, os motivos vão de 0,696 (`network latency`) a 0,98 (`JVM OOM`).
  - Há 10 a 21 falhas por dobra (contra 2 a 8 no Telecom) e 120 falhas no total.
- **Consequência para a E4 e a E5:** com a deriva entre dias, escalonamentos fixos ajustados no treino envelhecem. Os baselines globais da E4 devem ir mal no Bank por esse motivo, e não por falta de sinal. A E5 deve tratar a normalização: referência local ou reajuste frequente, sempre causal.

## Caracterização

`reports/tables/split_characterization_Bank.md` e `tasks_Bank.json`:
- 14.210 janelas: 804 positivas, 8.157 normais e 5.249 ignoradas;
- de 1.401 a 1.421 janelas de teste por dobra e de 60 a 126 positivas;
- motivos testados: `network packet loss` 32, `high CPU usage` 31, `network latency` 26, `high disk I/O read usage` 18, `high memory usage` 5, `high disk space usage` 5, `JVM OOM` 3.
