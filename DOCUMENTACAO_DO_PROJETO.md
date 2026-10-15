# Documentação Técnica e Arquitetural do Projeto

**Título do Trabalho:** Avaliação da Resiliência de Redes Neurais de Grafos a Concept Drift na Detecção de Botnets em IoT  
**Autora:** Letícia Alves de Pontes  
**Orientador:** Prof. Me. Paulo Rogério de Mello Cardoso  
**Data:** Outubro de 2026  

---

## 1. Visão Geral e Fundamentação do Projeto

### 1.1 Contexto e Questão de Pesquisa

Redes de Internet das Coisas (IoT) são alvos recorrentes de infecções por *botnets* (como Mirai, Hajime, Gafgyt e variantes), devido à vulnerabilidade de firmware, senhas padrão e ausência de mecanismos tradicionais de defesa perimetral.

A detecção clássica de intrusão baseia-se em classificadores tabulares operando sobre atributos isolados de fluxos de rede (ex.: duração, portas, bytes e pacotes). Contudo, sistemas em produção enfrentam uma limitação crítica: **a degradação acelerada de desempenho causada por mudanças na distribuição dos dados ao longo do tempo (*Concept Drift*) e entre ambientes distintos (*Domain Shift*)**.

Este projeto investiga se **Redes Neurais de Grafos (GNNs)** — capazes de modelar as interações topológicas entre dispositivos comunicantes (endereços IP) além dos atributos numéricos de fluxo — mantêm maior resiliência estatística sob alterações temporais e estruturais de rede.

### 1.2 Hipótese Científica e Critério Operacional de Resiliência

* **Hipótese:** Modelos que capturam a topologia de comunicação (estrutural e temporal) degradam significativamente menos o seu poder preditivo frente a drift do que modelos tabulares pontuais.
* **Critério Operacional:** Uma arquitetura baseada em grafos é considerada resiliente se a queda no $F_1\text{-Score}$ macro sob drift for inferior em pelo menos **$10\text{ pontos percentuais}$** em relação ao baseline tabular (*Random Forest*).
* **Validação Estatística:** Diferenças de acerto entre os modelos serão avaliadas pelo **Teste de McNemar** ($\alpha = 0,05$), com 5 repetições com sementes aleatórias fixadas (`42, 43, 44, 45, 46`).

---

## 2. Metodologia Experimental e Cenários de Avaliação

Com base nas deliberações da banca de qualificação, o projeto estabeleceu quatro pilares metodológicos:

### 2.1 Cenário T (Temporal Drift — IoT-23)

* **Objetivo:** Avaliar a degradação sob drift temporal gradual na mesma infraestrutura.
* **Dataset:** **IoT-23** (capturas reais de tráfego de dispositivos IoT infectados geradas pelo Stratosphere Lab em formato Zeek `conn.log.labeled`).
* **Protocolo de Separação:** As capturas são ordenadas cronologicamente a partir do primeiro timestamp de comunicação real ($\min(ts)$). O conjunto de treinamento é formado pelas capturas mais antigas e o conjunto de teste pelas mais recentes (divisão 80/20 ou 70/30). Não há embaralhamento temporal (*zero data leakage*).

### 2.2 Cenário D (Domain Shift / Heterogeneidade entre Bases)

* **Objetivo:** Avaliar a capacidade de transferência cruzada entre diferentes topologias, equipamentos físicos e vetores de ataque.
* **Datasets:** Treinamento conduzido no **IoT-23** e teste cego no **TON-IoT** (Network Dataset).
* **Isolamento de Efeitos:** O projeto separa formalmente a perda de desempenho decorrente da harmonização de esquemas (*feature mapping*) da degradação real do classificador.

### 2.3 Protocolo de Isonomia (Fairness Protocol)

Para assegurar comparabilidade científica:

* Se o modelo proposto (**PPT-GNN**) utilizar adaptação com poucos exemplos (*few-shot*) ou ajuste de prompt em um subconjunto de validação reservado, **todos os baselines** (*Random Forest*, *E-GraphSAGE* e *EvolveGCN-H*) receberão rigorosamente o mesmo orçamento e conjunto de amostras para recalibração.

### 2.4 Prevenção de Distorção em Dados Desbalanceados

* O *undersampling* aleatório estratificado é aplicado **exclusivamente no conjunto de treinamento**.
* O conjunto de teste retém a distribuição severa e realista de tráfego legítimo versus malicioso.

---

## 3. Estrutura Completa de Diretórios do Repositório

```text
gnn-drift-botnet-detection/
├── pyproject.toml                     # Configuração de empacotamento, dependências e pytest
├── README.md                          # Guia de execução, instalação e visão geral
├── PLANO_DESENVOLVIMENTO_CRONOGRAMA.md # Cronograma de semanas, entregas e mitigação de riscos
├── DOCUMENTACAO_DO_PROJETO.md         # Documentação técnica e arquitetural detalhada
├── .gitignore                         # Regras de exclusão (dados brutos, parquets, logs, .env)
├── .env.example                       # Modelo de variáveis de ambiente
├── .agents/                           # Customizações do agente
│   └── skills/
│       └── gnn-drift-botnet/          # Skill de instruções para agentes autônomos
│           └── SKILL.md
├── src/                               # Módulos Python reutilizáveis
│   ├── __init__.py
│   ├── config.py                      # Resolução de caminhos, reprodutibilidade e detecção Colab
│   ├── data.py                        # Ingestão Zeek, streaming, Parquet, splits e balanceamento
│   ├── harmonization.py               # Mapeamento para esquema canônico de 14 atributos
│   ├── graphs.py                      # Heurística de pesos de arestas, snapshots e PyG Data
│   ├── shift.py                       # Quantificação estatística de drift (MMD, Wasserstein)
│   └── models/                        # Implementações das arquiteturas avaliadas
│       └── __init__.py
├── notebooks/                         # Cadernos interativos de exploração e execução
│   ├── 00_setup.ipynb                 # Setup de runtime, checagem de GPU e Google Drive
│   └── 01_iot23_exploracao.ipynb      # Download do IoT-23, conversão Parquet e análise
├── tests/                             # Suíte de testes automatizados com pytest
│   ├── __init__.py
│   └── test_data.py                   # Testes unitários do parser Zeek e integridade
├── data/                              # Dados brutos e processados (ignorado no git)
│   └── parquet/                       # Arquivos comprimidos em Parquet
└── results/                           # Logs de treinamento, predições e pesos (ignorado no git)
```

---

## 4. Detalhamento dos Módulos do Sistema (`src/`)

### 4.1 `src/config.py` — Configurações e Reprodutibilidade

* **Caminhos Adaptativos:** Detecta automaticamente se o código está rodando no Google Colab (com ou sem Google Drive montado) ou em ambiente local, garantindo que diretórios `data/` e `results/` sejam criados e resolvidos de forma consistente.
* **Fixação de Sementes:** A função `set_seed(seed=42)` fixa sementes no `random`, `numpy` e `torch` (habilitando `torch.backends.cudnn.deterministic = True`).
* **Segurança:** As configurações são desacopladas de credenciais e arquivos sensíveis.

### 4.2 `src/data.py` — Pipeline de Ingestão e Processamento

* **Download Automatizado:** Funções `download_iot23_lightweight()` (URL oficial do Stratosphere Lab) e `download_ton_iot_kaggle()` (via Kaggle API com tratamento para credenciais ausentes).
* **Parser Robusto de Logs Zeek:** `parse_zeek_header()` lê dinamicamente a linha `#fields` do cabeçalho do Zeek, eliminando dependência de posições estáticas de colunas.
* **Desacoplamento de Rótulos:** `parse_label_columns()` decompõe a última coluna concatenada do IoT-23 em:
  * Rótulo binário normalizado (`Benign` ou `Malicious`).
  * Subclasse/Família (`detailed_label`, ex.: *Mirai*, *PartOfAHorizontalPortScan*, *C&C*).
* **Streaming de Baixo Consumo de RAM:** `read_conn_log_batches()` utiliza leitura por *chunks* (padrão de 200.000 linhas) com tipagem otimizada (`float32`, `category`, `string`), viabilizando o processamento de dezenas de milhões de fluxos sem estourar a memória.
* **Armazenamento Colunar:** `convert_capture_to_parquet()` converte logs brutos em Parquet com compressão Snappy.
* **Divisão Cronológica:** `list_iot23_captures()` extrai o menor timestamp epoch de cada captura para ordenar a geração das partições de treino e teste em `generate_temporal_splits()`.
* **Balanceamento de Treino:** `stratified_train_undersampling()` ajusta a proporção entre maioria e minoria no treino, preservando a ordenação temporal.

### 4.3 `src/harmonization.py` — Esquema Canônico Unificado

Para viabilizar o Cenário D (Domain Shift), os fluxos de diferentes ferramentas e datasets são convertidos para uma representação canônica comum composta por 14 atributos:

| Atributo | Tipo | Descrição |
| :--- | :--- | :--- |
| `ts` | float64 | Timestamp do início do fluxo (epoch) |
| `src_ip` | string | Endereço IP do host de origem |
| `src_port` | float32 | Porta de origem do fluxo |
| `dst_ip` | string | Endereço IP do host de destino |
| `dst_port` | float32 | Porta de destino do fluxo |
| `proto` | string | Protocolo de transporte (tcp, udp, icmp) |
| `duration` | float32 | Duração do fluxo em segundos |
| `orig_bytes` | float32 | Total de bytes transmitidos pela origem |
| `resp_bytes` | float32 | Total de bytes transmitidos pelo destino |
| `orig_pkts` | float32 | Total de pacotes transmitidos pela origem |
| `resp_pkts` | float32 | Total de pacotes transmitidos pelo destino |
| `label` | int64 | 0 = Benign, 1 = Malicious |
| `detailed_label` | string | Família de ataque ou benigno |
| `dataset_source` | string | Identificador da base de origem (`iot23`, `ton_iot`, `ctu13`) |

Funções implementadas: `harmonize_iot23`, `harmonize_ton_iot` e `harmonize_ctu13`.

### 4.4 `src/graphs.py` — Extração de Topologia e Snapshots Temporais

* **Heurística de Pesos das Arestas:**
  Para representar a intensidade relativa de comunicação de um host $u$ com um host $v$:
  $$w(u, v) = \frac{f(u, v)}{\sum_i f(u, i)}$$
  Onde $f(u, v)$ é a contagem de fluxos direcionados de $u$ para $v$. Calculada pela função `compute_edge_weights()`.
* **Snapshots Temporais:** `create_temporal_snapshots()` segmenta fluxos ordenados no tempo em janelas configuráveis (`window_seconds`, padrão 3.600s = 1h), gerando a sequência de grafos para o modelo temporal (*EvolveGCN-H*).
* **Estrutura PyG:** `build_pyg_graph_data()` mapeia endereços IP únicos para índices de nós, monta a matriz `edge_index`, vetor de pesos `edge_weight`, matriz de atributos de arestas `edge_attr` e constrói o objeto `Data` do PyTorch Geometric.

### 4.5 `src/shift.py` — Quantificação Estatística de Drift

Módulo reservado para o cálculo quantitativo de divergência de distribuição entre as janelas temporais e entre as bases, empregando métricas como:

* *Maximum Mean Discrepancy* (MMD).
* Distância de Wasserstein em distribuições de graus de entrada/saída.
* Teste Kolmogorov-Smirnov (KS) em distribuições de *embeddings*.

---

## 5. Modelos em Desenvolvimento e Comparação

| Modelo | Paradigma | Representação de Entrada | Papel no TCC |
| :--- | :--- | :--- | :--- |
| **Random Forest** | Tabular | 5 atributos numéricos contínuos de fluxo | Baseline clássico (sem informação relacional de grafo) |
| **E-GraphSAGE** | Grafo Estático | Grafo com atributos nas arestas e pesos $w(u,v)$ | Baseline de grafo estático com agregação de vizinhança |
| **EvolveGCN-H** | Grafo Dinâmico | Sequência de snapshots temporais com GRU | Baseline dinâmico para modelagem de evolução temporal |
| **PPT-GNN** | Pré-treino + Prompt | Grafo pré-treinado auto-supervisionado com adaptação | Modelo principal proposto (resiliência via pre-training) |

---

## 6. Fluxos de Trabalho e Execução

### 6.1 Execução Local

1. Criação e ativação da venv: `python3 -m venv .venv && source .venv/bin/activate`
2. Instalação em modo editável: `pip install -e ".[test]"` (ou `.[dev]`)
3. Execução dos testes unitários: `python3 -m pytest`

### 6.2 Execução no Google Colab via Extensão do VS Code

1. Conexão com o kernel do Google Colab (GPU T4) pelo VS Code.
2. Execução de `notebooks/00_setup.ipynb` (instalação de `.[colab]` com `torch-geometric`).
3. Execução de `notebooks/01_iot23_exploracao.ipynb` (download rápido do dataset, conversão Parquet e divisão temporal).
4. Download dos resultados para o ambiente local via `files.download()` ou pelo painel do VS Code.

---

## 7. Skill de Suporte Multiagente Criada

Para permitir que agentes adicionais compreendam as regras, arquitetura, APIs e restrições sem perda de contexto, foi criada a skill:

* **Arquivo:** [`.agents/skills/gnn-drift-botnet/SKILL.md`](file:///home/ubuntu22/projects/personal/gnn-drift-botnet-detection/.agents/skills/gnn-drift-botnet/SKILL.md)
* **Conteúdo:** Modelos mentais do projeto, regras de isonomia, mapeamento de APIs de dados e grafos, fórmulas matemáticas de pesos de arestas e instruções para a fase de modelagem e testes estatísticos.
