# ADR 0016 — Explicabilidade (SHAP, LIME, IG), medição das explicações e pacotes de evidência (E6)

- **Status:** aceito (E6 aprovada em 2026-11-12)
- **Data:** 2026-11-12
- **Etapa:** E6 (Telecom e Bank)

## Contexto

O plano pede explicar as janelas sinalizadas pela LSTM com SHAP e LIME, conferir com Integrated Gradients, agregar as atribuições por atributo, componente e categoria de KPI, medir a localização via XAI (HR@k, MRR) contra a heurística |z| e o ranqueamento nativo, medir fidelidade, estabilidade e concordância, e gerar o pacote de evidências que alimenta o agente (E7), sem nenhuma informação de rótulo (R3).

A E5 deixou dois modelos por dobra e seed: a previsora (formulação principal, cuja pontuação é `ForecastScorer.score`) e a classificadora (melhor detector no Telecom). Os dois são explicados.

## Decisões

### 1. Funções explicadas (`explain/functions.py`)
- **Previsora:** `ForecastScoreModule` reimplementa em PyTorch a pontuação da E5:
  - erro normalizado;
  - média dos 10 maiores por minuto;
  - máximo dos minutos.

  Fica diferenciável para SHAP e IG. Os alvos ausentes entram por uma máscara fixa da janela explicada. Um teste garante que o resultado é igual ao de `ForecastScorer`, inclusive com ausentes.
- **Classificadora:** `ClassifierProbModule`, a probabilidade sigmoide (igual a `classifier_scores`, testado).
- **Seed explicada:** 0. As seeds 1–4 entram só na medição de estabilidade.

### 2. Explicadores (`explain/attributions.py`)
- **SHAP:** `GradientExplainer` (expected gradients), fundo de 100 janelas normais de treino da dobra sorteadas com seed fixa, 200 amostras. Saída: tempo × atributo.
  - O KernelExplainer não foi usado: com gradiente disponível ele é mais caro e não acrescenta nada.
  - O DeepExplainer não foi usado, por ter suporte limitado a LSTM.
- **LIME:** representação binária por série `componente::kpi`. Um atributo "desligado" troca a série inteira da janela pela **referência normal**: a mediana, por passo e atributo, das janelas normais de treino da dobra. Parâmetros:
  - 3.000 amostras; a primeira mantém tudo, e as demais desligam um número uniforme de séries;
  - distância de cosseno × 100 com núcleo exponencial de largura 25;
  - regressão ridge ponderada do `LimeBase`, sem seleção de atributos.
- **Integrated Gradients (captum):** 32 passos, partindo da mesma referência normal, nas 3 primeiras falhas testadas de cada dobra (verificação cruzada do SHAP).

### 3. Agregação (`explain/aggregate.py`)
- **Atributo:** soma no tempo da atribuição com sinal.
- **Componente:**
  - **soma** das importâncias positivas dos seus KPIs, ou seja, o que empurrou a pontuação para cima;
  - **média das 3 maiores**, a mesma variante da E4/E5.
- **Categoria de KPI:** soma das importâncias positivas por categoria do catálogo da E2.
  - A **categoria de motivo sugerida** é a de maior peso.
  - O acerto é medido contra o mapa motivo → categorias fixado em `configs/base.yaml` (`explain.reason_categories`) **antes** de medir.

### 4. Duas famílias de janelas
- **Janelas-oráculo, só para medir (RQ2):** terminam D = 5 min após o início de cada falha testada, como na E4/E5. Usam rótulos e por isso ficam em `evaluation/xai.py`, código de avaliação.
- **Janelas do detector, para o agente:** em cada consulta do OpenRCA, a janela de 20 min com maior pontuação do detector dentro da janela da consulta (`candidate_end_rows`). Nunca usam o início rotulado.

### 5. Medição
- **Localização via XAI:**
  - HR@1/3/5 e MRR com IC por bootstrap sobre falhas;
  - comparação **pareada** (mesmas falhas) contra o ranqueamento nativo da E5 (mesma seed) e as heurísticas z global e z local da E4.
- **Categoria sugerida:** taxa de acerto com duas referências:
  - `prior_treino`: a categoria mais frequente nas falhas de treino e validação da dobra;
  - `uniforme`: o acerto esperado de uma categoria sorteada entre as presentes.

  Há também a comparação pareada com `prior_treino`.
- **Fidelidade:**
  - **deleção:** troca os k atributos mais importantes pela referência normal, para k ∈ {1, 2, 5, 10, 20, 50};
  - **inserção:** insere esses atributos na referência.

  Média da curva relativa à pontuação da **janela completa**, contra 5 ordens aleatórias.
- **Estabilidade:**
  - Jaccard e Kendall dos 10 atributos mais importantes do SHAP entre a seed 0 e as seeds 1–4;
  - Jaccard entre janelas vizinhas (±1 min, mesmo dia).
- **Concordância:**
  - SHAP × LIME (Jaccard e Kendall nos 10 mais importantes; mesmo componente no topo);
  - IG × SHAP.
- **Saídas por execução:** `xai_ranks.csv`, `xai_categories.csv` e `xai_quality.csv` (por janela), em `experiments/<run>/artifacts/`.

### 6. Pacote de evidências (`explain/evidence.py`, esquema pydantic versão 1.0)
- **Conteúdo:**
  - a consulta e a janela explicada, em UTC+8;
  - detector, pontuação, limiar da validação e alarme;
  - modelo explicado;
  - componentes `C#` com pesos SHAP e LIME;
  - KPIs `K#` com:
    - categoria, unidade e descrição;
    - atribuição SHAP e peso LIME;
    - instante de pico do desvio;
    - **valor bruto no pico**;
    - mediana normal de treino;
    - valor normalizado;
  - categoria sugerida com os pesos por categoria;
  - ranqueamento nativo `N#` (soma dos erros da previsora).
- **Garantias (testadas):**
  - campos extras são rejeitados;
  - não há campo de causa, motivo ou rótulo;
  - o módulo não importa nenhum leitor de rótulo.
- **Fora do git:** os pacotes ficam em `data/processed/<sistema>/evidence/<nome>/query_NNN.json`.
- **Detector × modelo explicado (decisão do autor em 2026-11-12, tomada depois de ver os resultados):**
  - a **classificadora** escolhe a janela e dá pontuação e alarme, por ser o melhor detector da E5;
  - as atribuições citáveis vêm da **previsora**.
  - **Motivo:** as explicações da classificadora mudam quase todas entre seeds (Jaccard 0,20–0,26, Kendall negativo), enquanto as da previsora são estáveis (0,81–0,82) e concordam com LIME e IG.
  - **Custo:** no Bank, o SHAP da classificadora localiza melhor (HR@1 0,44 contra 0,36).
  - É um desvio pós-teste, registrado como ameaça nº 11.

### 7. Execução
- **Dispositivo:** CPU. Atribuições iguais às do MPS (correlação 1,0000; mesmo KPI no topo em 100% das janelas), com velocidade parecida, porque o modelo é pequeno. O MPS fica disponível por `explain.device: mps`.
- **Custo:** cerca de 75 s (Telecom) e 200 s (Bank) por execução completa, incluindo o controle de estabilidade e os pacotes.
- **Defeito corrigido durante a medição:** a inserção estava normalizada pelo último passo (top-50), não pela janela completa, como a definição pretendia. O defeito foi identificado porque a inserção aleatória da classificadora saiu maior que a do SHAP. Há teste calculado à mão.

## Resultados

Os números estão em `docs/RESULTADOS.md` (seção E6) e em `reports/tables/xai_<sistema>_<nome>.md`. Em resumo:

- **Localização:**
  - no Telecom, a XAI fica entre a heurística e o nativo: SHAP sobre a previsora tem HR@1 0,33, inferior ao nativo (0,44) e superior às duas heurísticas;
  - no Bank, SHAP sobre a classificadora é a melhor localização até aqui (HR@1 0,44, MRR 0,61).
- **Fidelidade:** retirando os atributos apontados, a curva de deleção fica em média em 0,58–0,75 da pontuação original; em ordem aleatória, em cerca de 1,0.
- **Estabilidade e concordância:** altas na previsora e baixas na classificadora.
- **Categoria sugerida:** não supera a frequência de treino em nenhum caso. A previsora concentra peso em "disco".

## Consequências

- **E7:** o agente recebe os pacotes com as evidências identificadas (`C#`, `K#`, `N#`) e deve citá-las. A categoria sugerida **não** é confiável como motivo: o agente deve raciocinar sobre os KPIs, e o texto deve dizer isso.
- **Atribuição não é causalidade:** os pacotes dizem a que o modelo reagiu, não o que causou a falha. Descoberta causal (Ikram et al., 2022) fica como abordagem complementar, citada na monografia.
- **Controle sem isenção:** toda tabela da E6 tem a versão `xai_sem_isencao`.
