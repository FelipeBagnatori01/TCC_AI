# ADR 0001 — Stack técnica

- **Status:** aceito
- **Data:** 2026-11-10
- **Etapa:** E0

## Contexto

O projeto precisa de execução reprodutível por terceiros com um comando (R5), processamento de dezenas de GB de telemetria sem carregar tudo em memória (R1/E1), uma LSTM em PyTorch com XAI (SHAP, LIME, captum) e um agente de LLM trocável por configuração (R6).

## Decisões

| Camada | Escolha | Justificativa |
|---|---|---|
| Ambiente | Python 3.11 com **uv** e `uv.lock` versionado | Instala o Python fixado, resolve rápido e trava dependências. Poetry era a alternativa; uv foi escolhido por não exigir Python pré-instalado. |
| Configuração | YAML + **pydantic v2** (`extra="forbid"`, imutável) | Campo desconhecido gera erro explícito; tipos validados no carregamento. |
| Rastreamento | Diretório por execução (`experiments/<timestamp>-<nome>/`) com JSON/YAML, sem MLflow | A alternativa JSONL/parquet prevista no plano basta e não adiciona servidor nem dependência; MLflow pode ser adotado depois sem mudar o contrato. |
| Dados (E1+) | polars lazy, pyarrow, parquet | Leitura por dia, sem carregar tudo. Dependências adicionadas na E1. |
| Modelo | PyTorch | Exigido pela LSTM e pelo GradientExplainer. |
| LLM | Interface `LLMProvider` + registro por nome; cliente próprio com cache em disco, log JSONL e teto de custo | Trocar de modelo/provedor é mudança de configuração. Na E0 só existe o provedor `fake`; adaptadores reais entram quando o provedor for escolhido (antes da E7). |
| Qualidade | pytest, ruff, mypy estrito em `src/`, pre-commit | Conforme o plano. |

## Detalhes de comportamento do cliente de LLM

- Chave de cache = SHA-256 do JSON canônico de (provedor, modelo, mensagens, temperatura, max_tokens).
- Acerto de cache custa 0, não chama o provedor e continua sendo registrado no log.
- Teto de orçamento: a execução para **antes** da próxima chamada paga quando o gasto acumulado já atingiu o teto. Uma única chamada pode ultrapassar o teto por seu próprio custo; o custo real é reportado.
- Seed global fixa Python, NumPy e PyTorch e ativa `torch.use_deterministic_algorithms(True)`.

## Consequências

- Dependências de dados e XAI não estão instaladas na E0; entram junto das etapas que as usam, mantendo o lockfile coerente com o código existente.
- Sem MLflow, comparar execuções é feito por scripts sobre os JSON (E8).
