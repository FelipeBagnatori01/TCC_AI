# Mapa dos ADRs para a metodologia da monografia (E9)

**Estrutura usada:** a padrão, por decisão do autor (ADR 0024):

- **3 Metodologia:**
  - 3.1 Dados;
  - 3.2 Tarefas e divisão temporal;
  - 3.3 Baselines e protocolo de avaliação;
  - 3.4 Modelos de detecção;
  - 3.5 Explicabilidade;
  - 3.6 Agente de LLM;
  - 3.7 Avaliação integrada e análise estatística;
  - 3.8 Reprodutibilidade.
- **4 Resultados.**
- **5 Discussão e ameaças à validade.**

**Status:**
- **vigente:** vale como escrito;
- **alterado:** parte foi substituída por um ADR posterior, indicado na coluna "Relação";
- **exploratório:** decidido depois de ver resultados de teste, relatado à parte.

| ADR | Título | Seção | Status | Relação e observação |
|---|---|---|---|---|
| 0001 | Stack técnica | 3.8 | vigente | Python 3.11, uv, polars, PyTorch, shap/lime/captum |
| 0002 | Fuso horário: UTC em todo o pipeline | 3.1 | vigente | OpenRCA vem em UTC+8; conversão num único módulo |
| 0003 | Licença CC BY-NC 4.0 e OpenRCA como dependência | 3.1, 3.8 | vigente | Submódulo fixado; script oficial de avaliação |
| 0004 | Ingestão: parquet por dia, esquemas explícitos, leitura lazy | 3.1 | vigente | — |
| 0005 | Séries: passo, agregação, preenchimento, seleção e escala | 3.1 | vigente | A isenção de `disponibilidade` foi decidida depois de ver os rótulos: controle relatado em toda a tese (ameaça nº 1) |
| 0006 | Arestas de dependência (traces e lista de implantação) | 3.1 | vigente | Usadas pelo agente e por localizadores |
| 0007 | Tarefas, janelas e rótulos | 3.2 | vigente | Detecção (H = 0), W = 20, D = 5; sem precursor |
| 0008 | Divisão temporal em origem móvel | 3.2 | vigente | Dias duplicados excluídos; 10 dobras no Telecom |
| 0009 | Marco go/no-go da E3 (+ Plano B: Bank) | 3.1, 3.2 | vigente | Inclusão do Bank como segundo sistema |
| 0010 | Baselines e protocolo de avaliação | 3.3, 3.7 | alterado | Pontuação de janela e janelas-oráculo revistas no 0019 |
| 0011 | Ingestão de mais de um sistema | 3.1 | vigente | — |
| 0012 | Séries do Bank | 3.1 | vigente | Logs em templates drain3; preenchimento adaptativo |
| 0013 | Tarefas e divisão do Bank | 3.2 | vigente | 8 dobras, 120 falhas testadas |
| 0014 | z-score local e referência de frequência | 3.3 | vigente | Melhor baseline de detecção |
| 0015 | LSTM: arquitetura, normalização, busca e avaliação | 3.4 | alterado | Pontuação nos últimos D + 1 minutos (0019) |
| 0016 | Explicabilidade e pacotes de evidência | 3.5, 3.6 | alterado | A janela das evidências passou da classificadora para a previsora (0021) |
| 0017 | Agente de LLM: ferramentas, isolamento, variantes | 3.6 | vigente | Modelo escolhido no piloto, nunca no teste |
| 0018 | Modelos alternativos explicáveis | 3.4, 4 | vigente | Complemento: confirma o papel da previsora; o agente não muda (ameaça nº 19) |
| 0019 | Pontuação nos últimos minutos e janelas-oráculo | 3.3, 3.4, 3.5 | vigente | Decidido depois de ver o teste; controles de janela inteira relatados (ameaça nº 21) |
| 0020 | Seleção prequencial da evidência | 3.6, 4 | vigente | Resultado: os pacotes não mudam; a cadeia do pacote sem LLM é a referência do agente |
| 0021 | A previsora escolhe a janela das evidências | 3.5, 3.6 | vigente | Fixado antes de qualquer execução de teste do agente |
| 0022 | Avaliação integrada do agente (E8) | 3.7, 4 | vigente | A3 × A1 é a comparação principal; adendo: auditoria de fidelidade por LLM (ameaça nº 28) |
| 0023 | Modelo mais barato com as mesmas ferramentas | 4, 5 | exploratório | Haiku e Sonnet; decidido depois de ver a E8 (ameaça nº 27) |
| 0024 | Reprodução, pacote do cache e insumos da monografia | 3.8 | vigente | `make reproduce`, Release com o cache, `thesis-assets`, `repo-audit` |

## Por seção

| Seção | ADRs |
|---|---|
| 3.1 Dados | 0002, 0003, 0004, 0005, 0006, 0009, 0011, 0012 |
| 3.2 Tarefas e divisão temporal | 0007, 0008, 0009, 0013 |
| 3.3 Baselines e protocolo | 0010, 0014, 0019 |
| 3.4 Modelos de detecção | 0015, 0018, 0019 |
| 3.5 Explicabilidade | 0016, 0019, 0021 |
| 3.6 Agente de LLM | 0016, 0017, 0020, 0021 |
| 3.7 Avaliação integrada | 0010, 0022 |
| 3.8 Reprodutibilidade | 0001, 0003, 0024 |
| 4 Resultados (complementos) | 0018, 0020, 0022, 0023 |
| 5 Discussão e ameaças | 0005, 0019, 0023 e `docs/AMEACAS_VALIDADE.md` |
