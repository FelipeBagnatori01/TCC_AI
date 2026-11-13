# ADR 0017 — Agente de LLM: ferramentas fixas, isolamento, variantes e reprodutibilidade (E7)

- **Status:** proposto (implementação pronta; provedor e modelo a escolher pelo autor)
- **Data:** 2026-11-12
- **Etapa:** E7 (Telecom e Bank)

## Contexto

A E7 constrói um agente que recebe a consulta do OpenRCA e, conforme a variante, as evidências da E6. Ele responde instante, componente e motivo no formato oficial, com uma explicação em português que cita as evidências. O plano exige:

- ferramentas somente leitura, sem acesso ao gabarito (R3);
- quatro variantes com o mesmo modelo e o mesmo orçamento;
- prompts versionados;
- temperatura 0 quando suportada, cache e registro integral (R6);
- teto de custo (R7);
- conversão para o script oficial.

O autor aprovou o plano em 2026-11-12, com escopo e recomendações, e deixou a escolha do provedor para depois do piloto.

## Decisões

### 1. Ferramentas fixas, não um executor de Python
O RCA-agent do OpenRCA gera e executa código Python sobre os CSVs brutos. É mais flexível, mas um executor acessa o sistema de arquivos inteiro, `record.csv` inclusive. Isolá-lo exigiria um sandbox, e cada execução gera código diferente, o que dificulta a reprodução.

Ferramentas fixas resolvem os dois problemas: o acesso fica restrito por construção, cada chamada é registrada com argumentos e resultado, e um resultado em cache se reproduz byte a byte.

**Custo:** o agente só faz as análises que as ferramentas oferecem.

### 2. As ferramentas (`agent/tools.py`)
Todas são somente leitura, devolvem um resumo em JSON (nunca a série bruta inteira) e aceitam só intervalos no dia da consulta, em UTC+8.

| Ferramenta | O que devolve |
|---|---|
| `list_components` | Componentes com telemetria e número de KPIs |
| `list_kpis(componente, categoria?)` | KPIs com categoria, unidade, descrição e origem. Acima de 60 KPIs, sem categoria, devolve só a contagem por categoria. Uma lista longa por categoria vem sem descrições. Assim o pior caso cai de 42 mil para 10 mil caracteres |
| `query_kpi(componente, kpi, início, fim)` | n, mín., máx., média, mediana, primeiro e último valor, medianas a cada 5 min, desvio máximo em relação à referência local, instante do pico e primeiro instante com desvio acima de 3 |
| `find_anomalous_kpis(início, fim, componente?)` | Os 15 KPIs com maior desvio |
| `get_trace_summary(componente, início, fim)` | Desvio dos KPIs de trace do componente |
| `get_dependencies(componente)` | Arestas de chamada e de implantação da E2 |
| `get_evidence()` | Pacote da E6 filtrado pela variante (A2 e A3) |

- **Referência local:** é causal e não usa rótulo — os 60 min anteriores ao início no mesmo dia. Se houver menos de 20 minutos antes (os dias do Telecom começam às 00:00), usa os 60 min seguintes ao fim, e o resultado diz qual das duas foi usada.
- **Desvio:** (valor − mediana) / escala robusta (IQR/1,349, com piso relativo), limitado a 50, o mesmo teto do z local da E3/E5. No ensaio, sem o limite, KPIs com referência toda zero chegavam a 10⁹ e dominavam o ranking. Entre empates no teto, vem primeiro o que desviou mais cedo.
- **`find_anomalous_kpis` vai além das quatro ferramentas do plano.** Sem ela, a A1 teria de consultar às cegas cerca de 2 mil KPIs com 20 chamadas, e a comparação A1 × A3 exageraria o valor das evidências. Com ela, a A1 é "LLM + telemetria + varredura heurística", equivalente ao z-score local da E4. A2 e A3 medem o ganho marginal da LSTM e da XAI sobre essa base (ameaça nº 17).
- **Identificadores citáveis:** cada resultado recebe um id `T<n>`. Os itens do pacote mantêm `C#`, `K#` e `N#`.

### 3. Isolamento (R3)
- **Lista fechada:** `TelemetryStore` só abre quatro coisas de um sistema:
  - `data/processed/<sistema>/series.parquet`;
  - `reports/tables/feature_catalog_<sistema>.csv`;
  - `reports/tables/dependency_edges_<sistema>.csv`;
  - os JSON do diretório de evidências.
  
  Todo caminho é resolvido, com `..` e links simbólicos, antes da comparação. As ferramentas não recebem caminhos: componente e KPI são conferidos no catálogo.
- **Testes:**
  - abrir `record.csv`, `query.csv`, `windows.parquet`, `../..` ou um link simbólico do diretório de evidências para o gabarito falha;
  - nenhuma saída de ferramenta contém termos de rótulo;
  - a guarda de importação continua valendo para todo o pacote `tcc_rca.agent`.
- **Consultas:** são lidas por `read_queries`, sem `scoring_points`.
- **Candidatos:** as listas de componentes e motivos são as dos prompts do próprio OpenRCA, com teste de igualdade.

### 4. Variantes
| Variante | O que tem |
|---|---|
| A0 | Baseline constante da E4 (já medido) |
| A1 | Ferramentas de telemetria |
| A2 | A1 + `get_evidence` com pontuação, limiar e alarme do detector e ranqueamento nativo `N#` |
| A3 | A2 + componentes `C#` e KPIs `K#` de SHAP/LIME, com valor bruto, pico e mediana normal |

- **Igual em todas:** modelo, template, ferramentas de telemetria, limite de 20 chamadas, 30 turnos e 2 novas tentativas.
- **Categoria sugerida:** sai de todas as variantes. A E6 mostrou que ela não supera o prior de treino (ameaça 13), e o autor aprovou a remoção em 2026-11-12.
- **A3-SHAP e A3-LIME separados:** ficam para depois, se sobrar orçamento.

### 5. Esquema e validação (`agent/schema.py`)
- **Resposta:** `{"causas": [{"instante", "componente", "motivo"}], "explicacao", "evidencias_citadas"}`. É enviada como saída estruturada, um JSON Schema estrito, e validada com pydantic.
- **Regras:**
  - número de causas igual ao de falhas;
  - campos pedidos preenchidos;
  - instante no formato `YYYY-MM-DD HH:MM:SS` e dentro da janela;
  - componente e motivo candidatos;
  - explicação não vazia;
  - pelo menos uma citação, e só de ids que de fato apareceram na investigação.
- **Nova tentativa:** uma resposta inválida volta ao modelo com os erros, até 2 vezes.
- **Tarefa e número de falhas:** os campos vêm do `task_index` (task_1 = instante … task_7 = os três). O número de falhas sai do texto ("two failures", "two system failures"…). Os dois foram conferidos fora do pacote do agente contra os `scoring_points` das 187 consultas (51 Telecom + 136 Bank), com **0 divergências**.
- **Formato oficial:** `{"1": {...}, "2": {...}}`, só com os campos pedidos e as chaves na ordem do regex do avaliador. Há teste com o `main.evaluate` do OpenRCA, com duas falhas e nota 1,0.

### 6. Cliente de LLM e reprodutibilidade (R6, R7)
- **Tipos neutros** (`llm/types.py`): ferramentas, chamadas e resultados. As mensagens do assistente guardam os itens brutos do provedor, que o mesmo adaptador reenvia sem alteração — exigência para o raciocínio preservado entre turnos nos dois provedores.
- **Chave do cache:** inclui ferramentas, esquema, esforço e o índice da repetição (`sample`).
- **Temperatura:** os modelos atuais dos dois provedores **rejeitam `temperature`**. A R6 diz "quando suportado", então:
  - nenhum parâmetro de amostragem é enviado;
  - a reprodutibilidade vem do cache: uma nova execução sai inteira dele, com 0 chamada paga (testado);
  - a variância vem de repetições independentes (`sample` = 0, 1, 2) (ameaça nº 16).
- **Esforço:** o esforço de raciocínio é fixado para o estudo inteiro (`medium`).
- **Custo:** o cálculo separa entrada cheia, leitura de cache, escrita de cache e saída (com o raciocínio).
  - **Teto por execução:** US$ 50 nas variantes e US$ 10 no piloto. Ele é conferido antes de cada chamada paga, e uma execução interrompida guarda as repetições completas.
  - **Relatório:** custo total e por execução de caso em `reports/tables/agent_<sistema>_<nome>_<modelo>.md`.
- **Recusas:** ficam registradas como tais. O *fallback* automático para outro modelo não é usado: quebraria a regra de mesmo modelo em todas as variantes.

### 7. Adaptadores (`llm/anthropic_provider.py`, `llm/openai_provider.py`)
- **Anthropic (Messages API):**
  - sistema no campo próprio;
  - resultados de ferramenta agrupados numa única mensagem;
  - ferramentas `strict`, `output_config.format` e `output_config.effort`;
  - cache de prompt automático.
- **OpenAI (Responses API):**
  - `instructions`, `function_call_output`, ferramentas `strict` e `text.format`;
  - `reasoning.effort`;
  - `store=False`, com reenvio dos itens de raciocínio;
  - tokens em cache separados da entrada.
- **Testes:** a tradução é feita por funções puras, testadas sem rede.
- **Configurações candidatas** em `configs/llm/`: Claude Opus 5.5, Claude Sonnet 5.5, gpt-5.6-sol (preço promocional até 2026-11-30) e gpt-6.1-sol.
- **Chaves:** ficam no `.env`, que fica fora do git e é carregado pelo `tcc-rca` sem sobrescrever variáveis já exportadas.

### 8. Conjuntos de casos
- **Teste:** as consultas com pacote da E6 — 45 no Telecom e 120 no Bank.
- **Piloto:** as consultas antes do primeiro dia de teste — 6 no Telecom (2020-04-11) e 16 no Bank (2021-03-04). Como não têm pacote, usam só a variante A1.
- **Escolhas no piloto:** a escolha do modelo e qualquer ajuste de prompt acontecem no piloto, nunca no teste.

### 9. Prompts (`configs/prompts/`)
- **Arquivos:**
  - `agent_system.md`: papel, como investigar, aviso de que um KPI anômalo pode ser sintoma de outra falha, e regras de citação e de não inventar nada;
  - um bloco por variante;
  - `agent_user.md`: o texto original da consulta, a janela, o número de falhas, os campos e os candidatos;
  - `agent_retry.md`.
- **Sem exemplos:** nenhum exemplo de poucas amostras (*few-shot*).
- **Versão:** o hash de cada template fica registrado em cada execução (`prompts.json`).

### 10. Execução e artefatos
- **Comando:** `tcc-rca agent`.
- **Artefatos:**
  - `metrics.json`, com contagens reprodutíveis;
  - o log completo de chamadas;
  - `cases.jsonl` e as transcrições por caso;
  - `predictions_s<k>.csv`, no formato oficial por repetição, que a E8 avalia com o script oficial;
  - `cost.json`.
- **Ensaio offline:** o provedor `fake_agent` é determinístico e sem valor científico. Ele roda o pipeline inteiro offline e é usado nos testes e nos ensaios. Nos dados reais, as 135 execuções do Telecom, as 120 do Bank e as 22 do piloto foram todas válidas.

## Pendências
- Escolha do provedor e do modelo pelo autor, com o piloto, e a chave de API no `.env` (`docs/BLOQUEIOS.md`).
- Execução do piloto e das variantes, iniciadas pelo autor.

## Consequências
- A E8 avalia `predictions_s<k>.csv` com o script oficial e compara as variantes por caso (McNemar, bootstrap). A fidelidade das explicações (RQ4) usa as citações, que já são validadas contra os ids que existiram.
- Trocar de provedor é trocar `configs/llm/<modelo>.yaml`, sem mudar código.
