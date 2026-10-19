# Avaliação da Resiliência de Redes Neurais de Grafos a Concept Drift na Detecção de Botnets em IoT

Repositório do Trabalho de Conclusão de Curso (TCC) para avaliação da robustez e resiliência de modelos de aprendizado de máquina e Redes Neurais de Grafos (*Random Forest*, *E-GraphSAGE*, *EvolveGCN-H* e *PPT-GNN*) sob cenários de **Concept Drift** (mudança temporal gradual) e **Domain Shift** (mudança entre bases de dados heterogêneas) no contexto de redes IoT (*Internet of Things*).

---

## 📌 Cenários de Avaliação

- **Cenário T (Temporal Drift — IoT-23):** Treinamento nas capturas mais antigas e avaliação nas capturas mais recentes, seguindo estritamente a ordem cronológica do tráfego de rede (`min(ts)`).
- **Cenário D (Domain Shift / Heterogeneidade entre Bases):** Avaliação de transferência cruzada entre bases com diferentes infraestruturas e vetores de ataque (treino em **IoT-23** e teste em **TON-IoT**), após harmonização prévia de esquemas de atributos de fluxo de rede (14 colunas canônicas).

---

## 📂 Estrutura do Repositório

Todo o código do pipeline foi consolidado em um **único notebook autocontido** para facilitar a execução direta no Google Colab (via VS Code ou navegador) ou localmente, sem dependência de arquivos `.py` externos:

```text
gnn-drift-botnet-detection/
├── notebooks/
│   └── gnn_drift_botnet_detection.ipynb              # Notebook único autocontido (Setup, Ingestão, Harmonização, Grafos, Testes e Execução)
├── pyproject.toml                      # Lista de dependências do projeto
├── README.md                           # Guia de execução e documentação do projeto
├── PLANO_DESENVOLVIMENTO_CRONOGRAMA.md # Cronograma, decisões metodológicas e checklist
├── DOCUMENTACAO_DO_PROJETO.md          # Documentação detalhada da arquitetura e metodologia
├── data/                               # Diretório local para dados brutos e Parquet (ignorado no git)
└── results/                            # Logs de treinamento, splits e métricas (ignorado no git)
```

---

## 🚀 Como Executar (`notebooks/gnn_drift_botnet_detection.ipynb`)

O notebook [`notebooks/gnn_drift_botnet_detection.ipynb`](notebooks/gnn_drift_botnet_detection.ipynb) é **100% autocontido** e está organizado em 6 seções sequenciais:

1. **Seção 1 — Instalação de Dependências, Configuração de Ambiente e Sementes:**
   - Instala automaticamente `torch-geometric`, `pyarrow`, `scikit-learn`, `kaggle`, `tqdm`, `matplotlib` e `seaborn`.
   - Detecta se está no Google Colab (com ou sem Google Drive montado) ou no computador local.
   - Fixa as sementes aleatórias (`SEED = 42`).
2. **Seção 2 — Ingestão, Parsing Zeek, Parquet e Divisão Temporal:**
   - Funções para download do IoT-23 (`conn.log.labeled`) e TON-IoT (Kaggle), leitura em *streaming chunks*, conversão para Parquet com PyArrow e divisão temporal estrita por `min(ts)`.
3. **Seção 3 — Harmonização para o Esquema Canônico (14 Colunas):**
   - Padroniza fluxos do IoT-23, TON-IoT e CTU-13 no esquema canônico unificado.
4. **Seção 4 — Construção de Grafos, Pesos de Arestas $w(u,v)$ e Snapshots Temporais:**
   - Calcula a heurística de pesos normalizados $w(u,v)$, gera janelas temporais (`create_temporal_snapshots`) e monta objetos `Data` do *PyTorch Geometric*.
5. **Seção 5 — Validação Rápida de Integridade (Auto-Teste Sintético):**
   - Valida automaticamente todo o pipeline com uma amostra sintética antes de rodar nos dados reais.
6. **Seção 6 — Execução do Pipeline de Dados (Cenário T — IoT-23):**
   - Baixa os logs do IoT-23, converte para Parquet, plota a distribuição de tráfego malicioso por captura e exporta `splits.md`.
