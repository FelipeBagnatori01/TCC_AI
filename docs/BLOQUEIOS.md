# Bloqueios

Registro de dados, acessos ou definições ausentes (R10). Nenhum dado é inventado para contorná-los.

| Data | Etapa | Bloqueio | Status |
|---|---|---|---|
| — | — | Nenhum bloqueio registrado até o fim da E0. | — |
| 2026-11-12 | E7 | Provedor e modelo de LLM a escolher pelo autor (configurações candidatas em `configs/llm/`) e chave de API no `.env`. O piloto e as execuções das variantes dependem disso; a implementação e os testes não. | Resolvido em 2026-11-16: chave no `.env`; Claude Opus 5.5 escolhido no piloto (ADR 0017) |
| 2026-11-23 | E9 | O repositório `FelipeBagnatori01/TCC_AI` é privado, e o anexo da Release com o cache do LLM só é acessível a quem tem acesso ao repositório. O critério "um terceiro reproduz" depende de torná-lo público (ou convidar o terceiro) antes da defesa (ADR 0024). | Resolvido em 2026-11-24: repositório público; o anexo da Release baixa sem autenticação. Licenças: MIT no código, CC BY 4.0 nos textos, CC BY-NC 4.0 nos resultados derivados dos dados (`LICENSE`, `NOTICE`, README) |
