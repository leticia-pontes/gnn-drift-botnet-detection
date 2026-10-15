# Avaliação da Resiliência de Redes Neurais de Grafos a Concept Drift na Detecção de Botnets em IoT

Repositório do Trabalho de Conclusão de Curso (TCC) para avaliação da robustez e resiliência de modelos de aprendizado de máquina e Redes Neurais de Grafos (*Random Forest*, *E-GraphSAGE*, *EvolveGCN-H* e *PPT-GNN*) sob cenários de **Concept Drift** (mudança temporal gradual) e **Domain Shift** (mudança entre bases de dados heterogêneas) no contexto de redes IoT (*Internet of Things*).

---

## 📌 Cenários de Avaliação

- **Cenário T (Temporal Drift — IoT-23):** Treinamento nas capturas mais antigas e avaliação nas capturas mais recentes, seguindo estritamente a ordem cronológica do tráfego de rede (partição 80/20 ou 70/30).
- **Cenário D (Domain Shift / Heterogeneidade entre Bases):** Avaliação de transferência cruzada entre bases com diferentes infraestruturas e vetores de ataque (treino em **IoT-23** e teste em **TON-IoT** / **Bot-IoT**), após harmonização prévia de esquemas de atributos de fluxo de rede.

---

## 📂 Estrutura do Repositório

```text
gnn-drift-botnet-detection/
├── pyproject.toml                     # Configuração de empacotamento moderno e dependências
├── README.md                          # Guia de execução e documentação do projeto
├── PLANO_DESENVOLVIMENTO_CRONOGRAMA.md # Cronograma, decisões metodológicas e checklist
├── src/                               # Código-fonte modular do projeto
│   ├── __init__.py
│   ├── config.py                      # Configurações de caminhos, sementes e detecção de ambiente
│   ├── data.py                        # Ingestão, parsing Zeek, conversão Parquet e splits temporais
│   ├── harmonization.py               # Harmonização de atributos para esquema canônico comum
│   ├── graphs.py                      # Extração de grafos, pesos de arestas e snapshots temporais
│   ├── shift.py                       # Quantificação e métricas de drift estatístico
│   └── models/                        # Implementação dos modelos baseline e GNNs
│       └── __init__.py
├── notebooks/                         # Cadernos de automação e exploração
│   ├── 00_setup.ipynb                 # Setup de runtime, montagem do Drive e checagem de GPU
│   └── 01_iot23_exploracao.ipynb      # Download do IoT-23, conversão Parquet e análise
├── tests/                             # Suíte de testes automatizados com pytest
│   ├── __init__.py
│   └── test_data.py                   # Testes unitários do parser Zeek e integridade de dados
├── data/                              # Diretório local para dados brutos e processados (ignorado no git)
│   └── parquet/                       # Capturas convertidas em formato Parquet
└── results/                           # Logs de treinamento, checkpoints e métricas (ignorado no git)
```

---

## ⚙️ Pré-requisitos

- **Python:** Versão `>= 3.10` instalada.
- **Git:** Para versionamento do repositório.
- **Espaço em Disco:** Recomenda-se ao menos **10 GB a 15 GB** livres para armazenamento dos arquivos brutos e arquivos Parquet gerados.
- **Aceleração por Hardware (Opcional para exploração / Recomendado para treino):** Placa de vídeo NVIDIA compatível com CUDA para treinamento das GNNs (*PyTorch Geometric*). No ambiente local com CPU, é possível executar todo o pipeline de dados, testes e modelos tabulares (*Random Forest*).

---

## 🚀 1. Executando o Projeto do Zero no Ambiente Local

Siga os passos abaixo para configurar o ambiente e executar os módulos a partir do zero no seu computador.

### Passo 1.1: Clonar e acessar o repositório

```bash
git clone https://github.com/leticia-pontes/gnn-drift-botnet-detection.git
cd gnn-drift-botnet-detection
```

### Passo 1.2: Criar e ativar o ambiente virtual (venv)

```bash
# Criação do ambiente virtual
python3 -m venv .venv

# Ativação no Linux/macOS:
source .venv/bin/activate

# Ativação no Windows (PowerShell):
# .venv\Scripts\Activate.ps1
# Ativação no Windows (Prompt de Comando CMD):
# .venv\Scripts\activate.bat
```

### Passo 1.3: Atualizar o gerenciador de pacotes pip

```bash
pip install --upgrade pip
```

### Passo 1.4: Instalar as dependências do projeto

O projeto utiliza o padrão moderno de empacotamento `pyproject.toml`. Escolha a opção de acordo com o seu objetivo:

- **Opção A — Desenvolvimento Local e Testes (Recomendado para início):**
  Instala o projeto em modo editável (`-e`) com dependências de manipulação de dados (`pandas`, `numpy`, `pyarrow`, `matplotlib`, `seaborn`, `requests`, `tqdm`) e o pacote de testes (`pytest`):
  ```bash
  pip install -e ".[test]"
  ```

- **Opção B — Ambiente Completo com Suporte a Modelos e Kaggle:**
  Instala todas as dependências adicionais (`torch-geometric`, `scikit-learn`, `kaggle`):
  ```bash
  pip install -e ".[dev]"
  ```

### Passo 1.5: Validar a instalação com os testes unitários

Execute a suíte de testes com o `pytest` para confirmar que os parsers e a integridade de dados estão operacionais:

```bash
python3 -m pytest
```

*Saída esperada: todos os testes em `tests/test_data.py` devem passar com sucesso (`PASSED`).*

---

## 🔄 2. Executando o Pipeline de Dados Passo a Passo

Você pode executar o pipeline completo tanto através dos notebooks interativos quanto diretamente via scripts ou console interativo Python (`python3` / `ipython`).

### Passo 2.1: Download e Conversão do Dataset IoT-23 (Cenário T)

Execute no terminal Python para baixar o dataset leve oficial do IoT-23 (arquivos `conn.log.labeled`), extrair e converter para Parquet otimizado:

```python
import sys
from pathlib import Path
from src.config import DATA_DIR, PARQUET_DIR, set_seed
from src.data import (
    download_iot23_lightweight,
    list_iot23_captures,
    convert_capture_to_parquet,
    generate_temporal_splits,
)

set_seed(42)

# 1. Download e extração automática dos logs do IoT-23
download_iot23_lightweight(dest_dir=DATA_DIR)

# 2. Varredura das capturas e derivação cronológica baseada em min(ts)
df_captures = list_iot23_captures(base_dir=DATA_DIR)
print(f"Total de capturas encontradas: {len(df_captures)}")
print(df_captures[["capture_name", "size_mb", "derived_start_date"]].head())

# 3. Conversão de cada captura para Parquet (otimizado para baixo uso de RAM)
for idx, row in df_captures.iterrows():
    raw_path = Path(row["filepath"])
    out_parquet = PARQUET_DIR / f"{row['capture_name']}.parquet"
    if not out_parquet.exists():
        print(f"Convertendo [{idx+1}/{len(df_captures)}]: {row['capture_name']}...")
        convert_capture_to_parquet(raw_path, out_parquet)

print(f"✓ Capturas convertidas com sucesso em: {PARQUET_DIR}")

# 4. Geração da partição temporal estrita (80% treino cronológico / 20% teste)
train_caps, test_caps = generate_temporal_splits(df_captures, train_ratio=0.8)
print(f"Capturas de Treino ({len(train_caps)}): {train_caps}")
print(f"Capturas de Teste ({len(test_caps)}): {test_caps}")
```

### Passo 2.2: Harmonização de Esquemas de Rede

Para unificar dados de diferentes fontes em um formato canônico de 14 colunas:

```python
import pandas as pd
from src.config import PARQUET_DIR
from src.harmonization import harmonize_iot23

# Exemplo de leitura de uma captura convertida e aplicação de harmonização
sample_parquet = next(PARQUET_DIR.glob("*.parquet"))
df_raw = pd.read_parquet(sample_parquet)
df_canonical = harmonize_iot23(df_raw)

print("Colunas do esquema canônico:")
print(df_canonical.columns.tolist())
print(df_canonical.head())
```

### Passo 2.3: Construção de Grafos e Snapshots Temporais

Para modelos de grafos temporais como *EvolveGCN-H*, utilize as funções de `src/graphs.py`:

```python
from src.graphs import compute_edge_weights, create_temporal_snapshots

# 1. Cálculo da heurística de pesos das arestas: w(u, v) = f(u, v) / sum_i f(u, i)
edge_weights_df = compute_edge_weights(df_canonical)
print(edge_weights_df.head())

# 2. Segmentação em janelas temporais contíguas (ex: 3600 segundos = 1 hora)
snapshots = create_temporal_snapshots(df_canonical, window_seconds=3600.0)
print(f"Total de snapshots temporais gerados: {len(snapshots)}")
```

### Passo 2.4: Download do TON-IoT via Kaggle API (Cenário D - Opcional)

Para o Cenário de *Domain Shift* (transferência para o TON-IoT):

1. Obtenha sua chave de API do Kaggle em: `https://www.kaggle.com/settings` -> **Create New Token** (arquivo `kaggle.json`).
2. Posicione o arquivo no seu diretório de usuário:
   - Linux/macOS: `~/.kaggle/kaggle.json` e ajuste a permissão: `chmod 600 ~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<SeuUsuario>\.kaggle\kaggle.json`
   - Ou defina as variáveis de ambiente:
     ```bash
     export KAGGLE_USERNAME="seu_usuario"
     export KAGGLE_KEY="sua_chave"
     ```
3. Execute o download através da função:
   ```python
   from src.data import download_ton_iot_kaggle
   download_ton_iot_kaggle()
   ```

---

## ☁️ 3. Como Rodar no Google Colab (com GPU)

Para treinamento dos modelos de grafos pesados (*E-GraphSAGE*, *EvolveGCN-H*, *PPT-GNN*), recomenda-se a utilização de uma GPU T4 ou superior no Google Colab.

### Via Extensão do VS Code (Google Colab for VS Code)
1. Abra a pasta do projeto no **VS Code**.
2. Conecte ao runtime do Colab com GPU selecionada via extensão.
3. Abra e execute em sequência:
   - [notebooks/00_setup.ipynb](file:///home/ubuntu22/projects/personal/gnn-drift-botnet-detection/notebooks/00_setup.ipynb): monta o Google Drive (`/content/drive/MyDrive/tcc/data/`), verifica a GPU com `torch.cuda.is_available()` e instala o projeto com `pip install -e ".[colab]"`.
   - [notebooks/01_iot23_exploracao.ipynb](file:///home/ubuntu22/projects/personal/gnn-drift-botnet-detection/notebooks/01_iot23_exploracao.ipynb): baixa o dataset leve, converte para Parquet e gera o relatório estatístico.

### Via Interface Web Tradicional do Google Colab
1. No Colab, clone o seu repositório ou faça upload dos arquivos:
   ```bash
   !git clone https://github.com/leticia-pontes/gnn-drift-botnet-detection.git
   %cd gnn-drift-botnet-detection
   ```
2. Abra `notebooks/00_setup.ipynb` e rode as células de configuração e instalação.

---

## 🛠️ 4. Solução de Problemas Comuns (Troubleshooting)

- **`ModuleNotFoundError: No module named 'src'`**
  - **Causa:** O pacote não foi instalado em modo editável ou o comando foi executado fora da raiz do projeto.
  - **Solução:** Certifique-se de estar na raiz do repositório com o ambiente virtual ativado e execute `pip install -e ".[test]"`. Ao rodar comandos Python, utilize a raiz ou execute os testes com `python3 -m pytest`.

- **`command not found: pytest`**
  - **Causa:** O pacote `pytest` não foi instalado ou o PATH do ambiente virtual não está ativo.
  - **Solução:** Ative a venv com `source .venv/bin/activate` e rode `pip install -e ".[test]"`. Como alternativa direta, utilize `python3 -m pytest`.

- **Falha de memória ao processar capturas grandes:**
  - O módulo [src/data.py](file:///home/ubuntu22/projects/personal/gnn-drift-botnet-detection/src/data.py) já implementa leitura em lotes (*chunks* de 200.000 linhas) através da função `read_conn_log_batches`. Evite carregar múltiplos logs brutos inteiros em memória simultaneamente; converta-os para Parquet e leia apenas as colunas necessárias com `pd.read_parquet(caminho, columns=[...])`.

- **Erro de autenticação no Kaggle (`401 - Unauthorized`):**
  - Verifique se o arquivo `kaggle.json` possui permissões restritas (`chmod 600 ~/.kaggle/kaggle.json`) e se a sua conta aceitou os termos de uso do dataset no site do Kaggle.

---

## 📅 5. Próximos Passos de Desenvolvimento

Consulte o documento completo [PLANO_DESENVOLVIMENTO_CRONOGRAMA.md](file:///home/ubuntu22/projects/personal/gnn-drift-botnet-detection/PLANO_DESENVOLVIMENTO_CRONOGRAMA.md) para acompanhar as fases de:
1. **Fase 2:** Implementação dos classificadores em `src/models/` (*Random Forest*, *E-GraphSAGE*, *EvolveGCN-H* e *PPT-GNN*).
2. **Fase 3:** Avaliação estatística comparativa (Teste de McNemar, degradação de F1-Score macro e ROC-AUC).
3. **Fase 4:** Redação final dos resultados e conclusão da monografia de TCC.

## ☁️ 3. Como Rodar no Google Colab (com GPU)

Para treinamento dos modelos de grafos pesados (*E-GraphSAGE*, *EvolveGCN-H*, *PPT-GNN*), recomenda-se a utilização de uma GPU T4 ou superior no Google Colab.

### Via Extensão do VS Code (Google Colab for VS Code)
1. Abra a pasta do projeto no **VS Code**.
2. Conecte ao runtime do Colab com GPU selecionada via extensão.
3. Abra e execute em sequência:
   - [notebooks/00_setup.ipynb](file:///home/ubuntu22/projects/personal/gnn-drift-botnet-detection/notebooks/00_setup.ipynb): monta o Google Drive (`/content/drive/MyDrive/tcc/data/`), verifica a GPU com `torch.cuda.is_available()` e instala o projeto com `pip install -e ".[colab]"`.
   - [notebooks/01_iot23_exploracao.ipynb](file:///home/ubuntu22/projects/personal/gnn-drift-botnet-detection/notebooks/01_iot23_exploracao.ipynb): baixa o dataset leve, converte para Parquet e gera o relatório estatístico.

### Via Interface Web Tradicional do Google Colab
1. No Colab, clone o seu repositório ou faça upload dos arquivos:
   ```bash
   !git clone https://github.com/leticia-pontes/gnn-drift-botnet-detection.git
   %cd gnn-drift-botnet-detection
   ```
2. Abra `notebooks/00_setup.ipynb` e rode as células de configuração e instalação.

---

## 🛠️ 4. Solução de Problemas Comuns (Troubleshooting)

- **`ModuleNotFoundError: No module named 'src'`**
  - **Causa:** O pacote não foi instalado em modo editável ou o comando foi executado fora da raiz do projeto.
  - **Solução:** Certifique-se de estar na raiz do repositório com o ambiente virtual ativado e execute `pip install -e ".[test]"`. Ao rodar comandos Python, utilize a raiz ou execute os testes com `python3 -m pytest`.

- **`command not found: pytest`**
  - **Causa:** O pacote `pytest` não foi instalado ou o PATH do ambiente virtual não está ativo.
  - **Solução:** Ative a venv com `source .venv/bin/activate` e rode `pip install -e ".[test]"`. Como alternativa direta, utilize `python3 -m pytest`.

- **Falha de memória ao processar capturas grandes:**
  - O módulo [src/data.py](file:///home/ubuntu22/projects/personal/gnn-drift-botnet-detection/src/data.py) já implementa leitura em lotes (*chunks* de 200.000 linhas) através da função `read_conn_log_batches`. Evite carregar múltiplos logs brutos inteiros em memória simultaneamente; converta-os para Parquet e leia apenas as colunas necessárias com `pd.read_parquet(caminho, columns=[...])`.

- **Erro de autenticação no Kaggle (`401 - Unauthorized`):**
  - Verifique se o arquivo `kaggle.json` possui permissões restritas (`chmod 600 ~/.kaggle/kaggle.json`) e se a sua conta aceitou os termos de uso do dataset no site do Kaggle.

---

## 📅 5. Próximos Passos de Desenvolvimento

Consulte o documento completo [PLANO_DESENVOLVIMENTO_CRONOGRAMA.md](file:///home/ubuntu22/projects/personal/gnn-drift-botnet-detection/PLANO_DESENVOLVIMENTO_CRONOGRAMA.md) para acompanhar as fases de:
1. **Fase 2:** Implementação dos classificadores em `src/models/` (*Random Forest*, *E-GraphSAGE*, *EvolveGCN-H* e *PPT-GNN*).
2. **Fase 3:** Avaliação estatística comparativa (Teste de McNemar, degradação de F1-Score macro e ROC-AUC).
3. **Fase 4:** Redação final dos resultados e conclusão da monografia de TCC.

