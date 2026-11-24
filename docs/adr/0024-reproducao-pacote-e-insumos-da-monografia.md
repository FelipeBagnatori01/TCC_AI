# ADR 0024 — Reprodução de ponta a ponta, pacote do cache e insumos da monografia (E9)

- **Status:** aceito (2026-11-23)
- **Data:** 2026-11-23
- **Etapa:** E9

## Contexto

A E9 pede três coisas:

1. que um terceiro, com o OpenRCA baixado, reproduza os resultados da LSTM e do XAI, e os do agente a partir do cache;
2. que cada número da monografia tenha um artefato que o gerou;
3. que não haja chave de API nem telemetria no histórico do Git.

Até a E8, o pipeline exigia cerca de 20 comandos manuais. O cache do LLM (47 MB) e as execuções do agente ficavam fora do Git e não estavam disponíveis a terceiros.

## Decisões

1. **Um comando, três níveis** (`tcc-rca reproduce`, `src/tcc_rca/reproduce.py`; alvos `make reproduce`, `reproduce-agent` e `reproduce-full`):
   - **`principal`:** a cadeia que sustenta a tese, por sistema: verificação dos dados, conversão, séries, tarefas, baselines, LSTM e comparação pareada, XAI e pacotes, seleção de evidência, e a figura de RQ1.
   - **`agente-cache`:** refaz as 18 execuções de teste do agente (Opus, Haiku e Sonnet × A1–A3 × 2 sistemas) a partir do cache, com teto de US$ 0, e depois a avaliação da E8 e as comparações do ADR 0023. **Não precisa de chave de API**; há teste automatizado disso.
   - **`completo`:** o principal mais os relatórios da E1, todos os controles de sensibilidade e os modelos alternativos (ADR 0018).
   - **Como roda:** cada passo é um processo separado (o OpenMP do LightGBM não convive com o torch multithread, ADR 0010) e a primeira falha interrompe tudo. O registro fica em `experiments/reproduce/<data>-<nível>/`.
   - **Caminho dos dados:** vem da configuração. Um arquivo local opcional (`configs/local.yaml`, fora do Git) entra por último em cada passo.
2. **Verificação** (`tcc-rca reproduce-check`, `evaluation/reproduce_check.py`): compara as tabelas JSON/CSV regeneradas com as versionadas no Git (padrão: `HEAD`).
   - **Ignorados:** os nomes de pasta de execução (contêm data e hora) e os campos de tempo de relógio.
   - **Comparação exata:** textos, inteiros e booleanos.
   - **Números reais:**
     - **Mesma plataforma** (macOS arm64, CPU, Python 3.11, torch 2.15): igualdade exata (`rtol` = `atol` = 0) é o critério.
     - **Outro hardware:** o torch determinístico não garante bits iguais entre arquiteturas. O relatório lista os desvios, e a leitura usa `--atol 0.01` sobre valores já arredondados a 4 casas. Uma divergência maior que isso é reportada, não escondida.
3. **Pacote do cache** (`tcc-rca cache-bundle`/`cache-restore`, `src/tcc_rca/bundle.py`), publicado como anexo de uma Release do GitHub e **nunca commitado**. Ele contém:
   - o cache do LLM;
   - as pastas das 18 execuções originais do agente, com as quais o `evaluate-agent` compara os replays;
   - os pacotes de evidência que o agente viu;
   - um `MANIFEST.json` com SHA-256 de cada arquivo;
   - um `LEIA-ME.txt` com o conteúdo e a licença: trechos de telemetria do OpenRCA, CC BY-NC 4.0, uso não comercial com citação.

   **Garantias:**
   - A geração recusa qualquer arquivo com padrão de chave de API.
   - A restauração confere os checksums e não sobrescreve arquivo diferente.
   - **Se os pacotes de evidência regenerados diferirem dos originais** (o que faria o cache errar, porque as saídas das ferramentas entram na chave), a diferença é relatada. A opção `--use-reference-evidence` usa os originais e guarda os regenerados em `<nome>_regenerado`.
4. **Repositório privado:** o repositório está privado em 2026-11-23, e uma Release de repositório privado só é acessível a quem tem acesso a ele. Para o critério "um terceiro reproduz", o repositório precisa ficar público (ou o terceiro ser convidado) antes da defesa. **Decisão do autor**, registrada em `docs/BLOQUEIOS.md`.
5. **Insumos da monografia** (`tcc-rca thesis-assets`, `src/tcc_rca/thesis_assets.py`), gerados só a partir de `reports/tables`. **Formato: Word (`.docx`) e `.csv`**, por decisão do autor.
   - **Arquivos em `reports/monografia/`:**
     - `tabelas.docx` e `figuras.docx`, com título acima ("Tabela N – Título") e fonte abaixo, conforme a ABNT NBR 14724; tabelas só com fios horizontais, conforme as normas de apresentação tabular do IBGE;
     - um `.csv` por tabela (`;`, vírgula decimal, UTF-8 com BOM);
     - figuras PNG a 300 dpi;
     - `MANIFEST.csv`, com a origem, o comando e o commit de cada item;
     - `numeros.csv`, com os números citados no texto, cada um com o arquivo e a chave JSON de onde sai.
   - **Critério 2:** a monografia só cita números desses arquivos ou das tabelas de `reports/tables`.
   - **Estrutura usada no mapa dos ADRs:** a padrão de metodologia (dados, tarefas, modelos, explicabilidade, agente, avaliação), já que o autor não tem um sumário definido.
6. **Varredura do histórico** (`tcc-rca repo-audit`, `make audit-repo`): todos os blobs alcançáveis por qualquer ref. Procura:
   - conteúdo com padrão de chave de API;
   - caminhos proibidos: telemetria, `record.csv`/`query.csv`, `.env`, pesos, parquet, cache e execuções;
   - arquivos acima de 5 MB.

   O relatório fica em `reports/auditoria_git.md`.
7. **Recomendações para a indústria:** texto genérico para um ambiente de telecomunicações. O trabalho foi feito com recursos próprios e os dados públicos do OpenRCA, sem acesso a ambiente de nenhuma empresa (decisão do autor).

## Fora do escopo

- **Piloto da E7:** as execuções do piloto (escolha do modelo, ADR 0017) não entram no `agente-cache`. Foram a etapa de escolha, não resultado, e as tabelas delas continuam versionadas.
- **Bank A3 do Sonnet:** a execução interrompida (ADR 0023) não entra no pacote. Só a retomada entra, e ela contém todas as 120 consultas.

## Consequências

- Um terceiro com o OpenRCA e o anexo da Release roda `make reproduce` e `make reproduce-agent` e confere com `make reproduce-check`, sem chave de API.
- A geração das tabelas e figuras da monografia é repetível e rastreável até o JSON de origem.

## Resultado (2026-11-23)

- **Replay sem chave:** teste automatizado (`tests/test_llm_client.py::test_cache_replay_needs_no_api_key`).
- **Nível `agente-cache`** (`experiments/reproduce/20261123T200226-agente-cache`): os 24 passos terminaram sem erro.
  - As 18 execuções saíram inteiras do cache, com 0 chamadas pagas.
  - O `evaluate-agent` e o `compare-models` confirmam previsões e `metrics.json` idênticos aos originais nas 18.
  - **Verificação contra o `HEAD`:**
    - 8 tabelas regeneradas iguais;
    - 12 novas (as tabelas de replay do Haiku e do Sonnet, que antes não existiam);
    - 4 com diferença esperada: as tabelas `model_compare` ganharam a seção `replay`, ausente na versão anterior porque esses replays não tinham sido feitos.
  - Commitadas as tabelas novas, a verificação passa a ter estas como referência.
- **Pacote:** `tcc_rca_cache_llm.tar.gz`, com 92 MB, 11.983 arquivos e as 18 execuções; SHA-256 `028962e5e38e8cc217a9a3ccdbdae3cea21ca8c6e92268649947126dfaff502b`.
  - **Teste de restauração** numa pasta vazia: checksums conferidos, cache idêntico ao original e pacotes de evidência iguais nos dois sistemas.
  - **Publicação:** Release [`cache-llm-2026-11-23`](https://github.com/FelipeBagnatori01/TCC_AI/releases/tag/cache-llm-2026-11-23); o SHA-256 calculado pelo GitHub confere.
- **Histórico do Git:** 24 commits e 666 arquivos verificados, nenhum achado (`reports/auditoria_git.md`).
- **Insumos da monografia:** 7 tabelas, 4 figuras e 26 números rastreados (`reports/monografia/`). Os valores conferem com `docs/RESULTADOS.md`.
- **Reprodução num clone limpo** (2026-11-23/24, mesma máquina, dados ligados por `configs/local.yaml`):
  - `make reproduce`: os 18 passos sem erro.
  - `cache-restore` do anexo da Release: SHA-256 conferido, e os pacotes de evidência regenerados no clone são idênticos aos que o agente viu (45 no Telecom, 120 no Bank).
  - `make reproduce-agent`: os 24 passos sem erro; as 18 execuções saíram do cache com 0 chamadas pagas, previsões e `metrics.json` iguais aos originais.
  - `make reproduce-check` com tolerância zero: **49 tabelas iguais e nenhuma diferente**. As 56 não regeneradas são as execuções originais do agente e do piloto (os replays gravam tabelas `_replay` à parte) e as do nível `completo` (relatórios da E1, controles e modelos alternativos).
  - **Correções feitas no caminho:**
    - checagem prévia dos dados, do submódulo e do pacote restaurado, porque sem o cache os replays gravavam tabelas vazias;
    - a verificação considera a última execução de cada nível;
    - os tempos de relógio (`seconds`, `train_seconds`, `inference_ms_per_window` e agregados) são ignorados, porque eram as únicas diferenças.
- **Pendente com o autor:**
  - a reprodução em outra máquina (critério com `--atol 0.01`), se houver uma disponível;
  - **tornar o repositório público** (bloqueio registrado).
