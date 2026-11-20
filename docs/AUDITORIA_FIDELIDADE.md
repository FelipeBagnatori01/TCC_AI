# Auditoria manual da fidelidade das explicações (E8, RQ4)

A auditoria mede quantas afirmações das explicações do agente são **sustentadas pelos dados que o próprio agente consultou**. Ela complementa a verificação automática do `evaluate-agent` (IDs, nomes de componentes e KPIs, números), que só confere se cada item existe nas saídas das ferramentas. A auditoria confere se a afirmação inteira está correta.

## Amostra (fixada antes de auditar; ADR 0022)

- **Tamanho:** 30 explicações, 5 por sistema × variante (Telecom/Bank × A1/A2/A3).
- **Sorteio:** semente 0, sem reposição.
- **Afirmações:** 4 frases por explicação, sorteadas entre as frases com algo verificável (número, nome de componente ou KPI, ID de evidência). São 120 afirmações.
- **Cegamento:** a planilha mostra só o código da explicação (E01 a E30) e esconde a variante. A chave fica em `reports/audit/auditoria_chave.json`; não a abra antes de terminar. Os IDs `C#`/`K#`/`N#` citados ainda podem revelar a variante, e isso está registrado como limitação.

## Arquivos

- **Planilha:** `reports/audit/auditoria_fidelidade.csv`, uma linha por afirmação. Preencha só as colunas `classificacao` e `observacao`.
- **Evidência de cada explicação:** `experiments/auditoria_fidelidade/E##.md` (fora do git). Contém a explicação completa, as afirmações numeradas e a saída de cada chamada de ferramenta, com os argumentos.
- **Ambiente:** os horários estão em UTC+8, como o agente os viu. Não use o gabarito: a auditoria mede fidelidade aos dados, não acerto da causa.

## Rótulos (coluna `classificacao`)

| Rótulo | Quando usar |
|---|---|
| `sustentada` | Tudo o que a frase afirma pode ser conferido nas saídas das ferramentas: valores (com o arredondamento escrito), instantes, nomes, relações de dependência e o ID citado. |
| `parcial` | Parte da frase é sustentada e parte não: por exemplo, o valor está certo e o instante não, ou o ID citado não contém o dado. |
| `nao_sustentada` | A frase contradiz as saídas, ou afirma um dado que não aparece em nenhuma delas (valor inventado, componente ou KPI que não foi consultado, ID trocado). |
| `nao_verificavel` | A frase é uma inferência ou opinião ("a causa mais provável é...", "compatível com falha de rede") e não um dado. Também quando a frase é um dado que as ferramentas não teriam como mostrar. |

**Regras de desempate:**
- **Conclusão com dado:** "a causa é o Tomcat02 por volta de 07:32" é `nao_verificavel` quanto à causa. Se a frase também traz um dado conferível, como um valor ou um instante de pico, classifique o dado.
- **Arredondamento:** "cerca de 68 ms" para 67,9 é `sustentada`. "Valores entre 4.000 e 9.300" é `sustentada` se o intervalo cobre os valores mostrados.
- **Contas sobre os dados:** "caiu 95%" ou "triplicou" são `sustentada` se a conta bate com os valores mostrados.
- **Na dúvida** entre `parcial` e `nao_sustentada`, use `parcial` e explique em `observacao`.

## Depois de preencher

Rode `uv run tcc-rca evaluate-agent --config configs/base.yaml [configs/system/bank.yaml] configs/experiment/agent_eval.yaml`. A seção `audit` de `reports/tables/agent_eval_<sistema>.json` passa a trazer a fração de cada rótulo por variante, com IC por bootstrap sobre explicações. A planilha nunca é reescrita.
