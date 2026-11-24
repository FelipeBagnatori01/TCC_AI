# Detecção precoce e explicação automática de causas raiz de falhas em sistemas de telecomunicações a partir de dados multimodais de observabilidade

Trabalho de conclusão de curso (UNIFESP). O sistema tem três partes:

1. **Detecção:** uma rede **LSTM** detecta falhas a partir de séries de KPIs de observabilidade (métricas e agregados de traces).
2. **Explicabilidade:** **SHAP** e **LIME** explicam cada detecção por componente e por KPI.
3. **Diagnóstico:** um **agente de LLM** (Claude Opus 5.5), com ferramentas de leitura da telemetria e essas evidências, aponta a causa raiz: instante, componente e motivo.

**Por que "detecção precoce" e não "predição":** não há sinal precursor das falhas nos KPIs (E3, ADR 0007). A tarefa é detectar o início da falha nos primeiros minutos, não antecipá-la.

A avaliação usa o benchmark público **OpenRCA** (XU et al., ICLR 2025). O subconjunto Telecom é o cenário principal e o Bank, o segundo sistema.

## Perguntas de pesquisa e onde estão as respostas

| Pergunta | Resposta curta | Detalhes |
|---|---|---|
| RQ1: a LSTM detecta melhor que baselines? | A LSTM previsora tem a maior AUC-PR nos dois sistemas, mas empata com o melhor candidato em cada um (logística L1 e classificadora no Telecom, z-score local no Bank) | `docs/RESULTADOS.md` (E4, E5 e comparação de candidatos) |
| RQ2: SHAP/LIME localizam o componente? | A LSTM previsora + SHAP é a melhor nos dois sistemas (HR@1 de 0,42 no Telecom e 0,48 no Bank) | `docs/RESULTADOS.md` (E6) |
| RQ3: as evidências melhoram o agente? | No Opus, não: A3 × A1 empata nos dois sistemas. Nos modelos mais baratos (Haiku e Sonnet, exploratório), sim, mas eles não alcançam o Opus | `docs/RESULTADOS.md` (E7/E8 e ADR 0023) |
| RQ4: fidelidade e custo | Nenhum ID, componente ou KPI inventado; cerca de 1% de números sem correspondência; US$ 0,16 a 0,23 por consulta | `docs/RESULTADOS.md`, `docs/FICHA_AGENTE.md` |

## Requisitos

- **Software:** [uv](https://docs.astral.sh/uv/), que instala o Python 3.11 fixado e as dependências travadas em `uv.lock`, e `git`.
- **Hardware usado** (resultados de referência): Apple M4 Pro, 12 núcleos, 24 GB de RAM, só CPU.
- **Disco:**
  - OpenRCA Telecom: 16 GB;
  - OpenRCA Bank: 25 GB;
  - `data/interim`: 6 GB;
  - `experiments/`: cerca de 4 GB;
  - cache do LLM: 47 MB.
- **API de LLM:** não é necessária para reproduzir. Só para novas execuções pagas do agente, com a chave em `.env`.

## Instalação

```bash
git clone --recurse-submodules https://github.com/FelipeBagnatori01/TCC_AI.git
cd TCC_AI
make setup   # uv sync, cria .env a partir de .env.example e instala o pre-commit
make test    # testes (dados sintéticos; não precisam do OpenRCA)
```

O OpenRCA entra como submódulo em `external/OpenRCA`, no commit fixado no ADR 0003, porque o script oficial de avaliação vem de lá. **Se o clone foi feito sem `--recurse-submodules`,** rode `git submodule update --init` antes de reproduzir; sem isso, o script oficial não existe e a avaliação falha.

## Dados

1. **Baixe** do [Google Drive do OpenRCA](https://drive.google.com/drive/folders/1wGiEnu4OkWrjPxfx5ZTROnU37-5UDoPM?usp=drive_link), o link do README do repositório oficial:
   - as pastas `Telecom/` e `Bank/`, cada uma com `record.csv`, `query.csv` e `telemetry/`;
   - o arquivo `AppTelecomDeploymentList.xlsx`.
2. **Aponte o projeto para eles,** por um destes caminhos:
   - **links simbólicos** em `data/raw/openrca/`, que é o padrão. A pasta não existe num clone novo, então crie-a antes:
     ```bash
     mkdir -p data/raw/openrca
     ln -s /caminho/Telecom data/raw/openrca/Telecom
     ln -s /caminho/Bank data/raw/openrca/Bank
     ln -s /caminho/AppTelecomDeploymentList.xlsx data/raw/openrca/
     ```
   - **ou,** se `Telecom/`, `Bank/` e o `.xlsx` já estão juntos numa pasta: copie `configs/local.example.yaml` para `configs/local.yaml` (fora do Git), aponte `paths.data_root` para essa pasta e rode os comandos com `CONFIG`. Por exemplo:
     ```bash
     make reproduce CONFIG="configs/base.yaml configs/local.yaml"
     ```
3. **Confira:** o `make reproduce` checa primeiro se os dados e o submódulo estão no lugar e, se faltar algo, diz o que fazer. A conferência completa é com `uv run tcc-rca verify` (tamanhos e checksums contra `data/MANIFEST.csv`).

A telemetria do OpenRCA é licenciada sob **CC BY-NC 4.0** (uso não comercial, com citação) e **nunca é versionada** neste repositório.

## Reprodução

| Comando | O que faz | Duração estimada (máquina acima, somando os logs das execuções originais) |
|---|---|---|
| `make reproduce` | Nível `principal`, para os dois sistemas: conversão, séries, tarefas, baselines (E4), LSTM (E5), XAI e pacotes de evidência (E6), seleção de evidência (ADR 0020) e a figura de RQ1 | cerca de 1 a 1,5 h (o treino da LSTM leva 10 min no Telecom e 24 min no Bank) |
| `make reproduce-agent` | Nível `agente-cache`: as 18 execuções de teste do agente (Opus, Haiku e Sonnet × A1–A3 × 2 sistemas) a partir do cache, sem chave e com teto de US$ 0, depois a avaliação da E8 e a comparação de modelos | cerca de 25 min |
| `make reproduce-full` | Nível `completo`: o principal mais os relatórios da E1, os controles de sensibilidade e os modelos alternativos (ADR 0018) | cerca de 5 a 6 h |
| `make reproduce-check` | Compara as tabelas regeneradas com as versionadas e escreve `experiments/reproduce/<execução>/verificacao.md` | segundos |

**Passo a passo para o agente a partir do cache:**

1. Rode `make reproduce` primeiro: as ferramentas do agente leem as séries e os pacotes regenerados.
2. Baixe o anexo `tcc_rca_cache_llm.tar.gz` da Release [`cache-llm-2026-11-23`](https://github.com/FelipeBagnatori01/TCC_AI/releases/tag/cache-llm-2026-11-23). SHA-256: `028962e5e38e8cc217a9a3ccdbdae3cea21ca8c6e92268649947126dfaff502b`.
3. Restaure o pacote, com o caminho completo do arquivo baixado (e as mesmas configurações do `make reproduce`, se usou `configs/local.yaml`):
   ```bash
   uv run tcc-rca cache-restore --config configs/base.yaml configs/local.yaml --bundle ~/Downloads/tcc_rca_cache_llm.tar.gz
   make reproduce-agent CONFIG="configs/base.yaml configs/local.yaml"
   make reproduce-check CONFIG="configs/base.yaml configs/local.yaml"
   ```
   O `make reproduce-agent` não começa se o pacote não estiver restaurado. O `reproduce-check` considera a última execução de cada nível (`principal` e `agente-cache`).

**Como funciona:**
- **Restauração:** o `cache-restore` confere o SHA-256 de cada arquivo e informa se os pacotes de evidência regenerados são idênticos aos que o agente viu.
- **Se não forem:** a chave do cache inclui as saídas das ferramentas, então o replay pararia na primeira consulta divergente (sem pagar nada). Nesse caso, `--use-reference-evidence` usa os pacotes originais, e a diferença deve ser relatada.
- **Critério de aprovação:**
  - na mesma plataforma, igualdade exata (`make reproduce-check`);
  - em outro hardware, os desvios aparecem no relatório (`uv run tcc-rca reproduce-check --atol 0.01`), porque o torch determinístico não garante os mesmos bits entre arquiteturas (ADR 0024).

Para ver os comandos de cada nível sem executar: `uv run tcc-rca reproduce --level principal --dry-run`.

## Onde está cada coisa

| Caminho | Conteúdo |
|---|---|
| `docs/RESULTADOS.md` | Todos os resultados, por etapa, com IC e vereditos pareados |
| `docs/adr/` e `docs/MAPA_ADRS.md` | As 24 decisões de projeto, mapeadas para a metodologia |
| `docs/AMEACAS_VALIDADE.md` | Ameaças à validade (30) e o que foi mitigado |
| `docs/MODEL_CARD_LSTM.md`, `docs/FICHA_AGENTE.md` | Model card da LSTM e ficha do agente |
| `docs/RECOMENDACOES_INDUSTRIA.md` | Recomendações para aplicação em ambiente de telecomunicações |
| `docs/RASTREABILIDADE.md` | Cada resultado ligado ao artefato que o gerou |
| `reports/tables/` | Tabelas de resultado (JSON, CSV e Markdown), geradas pelo código |
| `reports/monografia/` | Tabelas (`.docx` e `.csv`) e figuras no padrão ABNT, com `MANIFEST.csv` e `numeros.csv` (`uv run tcc-rca thesis-assets`) |
| `notebooks/` | Análises que leem só `reports/tables` (`make notebooks`) |
| `src/tcc_rca/` | Código: ingestão, séries, tarefas, modelos, explicabilidade, agente e avaliação |
| `configs/` | Toda configuração de experimento (YAML validado; campo desconhecido é erro) |

## Limitações (resumo)

- **Tamanho amostral:** 45 consultas testadas no Telecom e 120 no Bank. Efeitos de até cerca de 0,13 na nota passam por empate.
- **LLM:** uma repetição por consulta; um único modelo principal, com Haiku e Sonnet só de forma exploratória; possível contaminação do LLM pelo benchmark público.
- **Fidelidade:** a auditoria foi feita por LLM, o mesmo modelo do agente, sem validação humana.
- **Ambiente:** o benchmark difere de um ambiente de produção (falhas injetadas, telemetria limpa, poucos dias).
- **Lista completa:** `docs/AMEACAS_VALIDADE.md`.

## Citação dos dados

XU, J. et al. OpenRCA: Can Large Language Models Locate the Root Cause of Software Failures? In: International Conference on Learning Representations (ICLR), 2025. Repositório: https://github.com/microsoft/OpenRCA (código MIT; telemetria CC BY-NC 4.0).

## Licença

| Conteúdo | Licença |
|---|---|
| Código (`src/`, `tests/`, `configs/`, `Makefile`, `notebooks/`) | [MIT](LICENSE) |
| Textos (`README.md`, `docs/`, incluindo os ADRs) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Resultados derivados da telemetria do OpenRCA (`reports/`) e o anexo da Release `cache-llm-2026-11-23` | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), a licença dos dados de origem |

Os trechos copiados dos prompts do agente do OpenRCA (`configs/prompts/candidates_*.yaml` e `systems.yaml`) seguem a licença MIT do código do OpenRCA (Microsoft). Os avisos estão em [`NOTICE`](NOTICE).
