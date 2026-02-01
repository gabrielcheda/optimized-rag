## 1. Perguntas de Recuperação Direta (Fatos Específicos)

*Estas perguntas testam se o seu RAG consegue localizar informações explícitas.*

* **O que é o DW-GRPO e quais são os três objetivos principais que ele equilibra?** 
R: 

---

* **Quais são as quatro fases do processo Deep GraphRAG descritas no Algoritmo System 1?** 
R:

---

* **Diferencie os conceitos de "Internal Evaluation" e "External Evaluation" de acordo com o levantamento sobre avaliação de RAG.** 
cls

---

* **O que caracteriza o problema "lost-in-the-middle" em modelos de contexto longo?** 

---

* **Quais são os componentes da "RAG Triad" mencionada no framework TruEra?** 

---

## 2. Perguntas de Raciocínio e Comparação (Complexas)

*Estas perguntas testam a capacidade do sistema de sintetizar conceitos distribuídos no texto.*

* **Explique a analogia entre "System 1" e "System 2" aplicada aos paradigmas de RAG Predefinido e Agêntico.** 


---

* **Como o Deep GraphRAG utiliza a busca em feixe (beam search) para navegar na hierarquia de comunidades do grafo?** 


* 
**Compare as abordagens baseadas em "Prompt" vs. "Training" para o desenvolvimento de RAG Agêntico.** 


* 
**De que forma o método "ReZero" (Retry-Zero) incentiva a persistência em agentes de busca?** 



---

## 3. Perguntas Técnicas e Matemáticas

*Úteis para verificar se o RAG recupera fórmulas e definições técnicas com precisão.*

* 
**Como é definida a recompensa de concisão () no framework Deep GraphRAG?** 


* 
**Qual é a fórmula para o cálculo do *Thrust metric* utilizado para avaliar o conhecimento de um LLM?** 


* 
**Descreva o funcionamento do mecanismo de pesos dinâmicos  no DW-GRPO, utilizando a função softmax.** 

---

## 4. Perguntas de Avaliação e Métricas

*Focadas no documento de revisão de avaliação (RAG-PAPER.pdf).*

* 
**Quais são as métricas sugeridas para avaliar a segurança (Safety) de um sistema RAG contra ataques adversários?** 

{'agent_response': 'Para avaliar a segurança de um sistema RAG contra ataques adversários, são sugeridas algumas métricas, incluindo o uso da abordagem SafeRAG, que classifica tarefas de ataque em quatro categorias com conjuntos de dados específicos. Além disso, o framework VERA utiliza amostragem bootstrap para calcular limites de confiança sobre as métricas de segurança, enquanto a abordagem de red teaming do DeepTeam identifica vulnerabilidades por meio de testes sistemáticos [2].', 'intent': <QueryIntent.QUESTION_ANSWERING: 'question_answering'>, 'retrieved_docs': 5, 'refinement_count': 0, 'quality_score': 0.85}

* 
**O que é a métrica "Semantic Perplexity" (SePer) e qual sua utilidade?** 


* 
**Quais métricas compõem o grupo de "Rank-Based Metrics" para avaliação de recuperação (IR)?** 

# Nada a ver com o assunto:

Qual é o impacto específico na latência de busca ao utilizar o motor de indexação "DiskANN" em vez de "HNSW" para conjuntos de dados vetoriais que excedem a memória RAM disponível?

Como a técnica de "Ring Attention" permite que modelos de contexto infinito processem sequências de milhões de tokens em clusters de GPUs, e qual a diferença dessa abordagem para o "Long-Context" mencionado nos artigos?

De que maneira a biblioteca "AutoGPT" implementa a "memória de curto prazo" em loops de RAG agêntico para evitar a repetição de consultas de busca que já falharam?