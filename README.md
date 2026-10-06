# Avaliação da Resiliência de Redes Neurais de Grafos a Concept Drift na Detecção de Botnets em IoT

Repositório de TCC para avaliação da robustez de Redes Neurais de Grafos (Random Forest, E-GraphSAGE, EvolveGCN-H e PPT-GNN) sob Concept Drift e Domain Drift em IoT.

## Cenários de Avaliação

- **Cenário T (Temporal - IoT-23):** Treino nas capturas mais antigas e teste nas mais recentes em ordem cronológica estrita.
- **Cenário D (Entre Domínios):** Treino no IoT-23 e teste no Bot-IoT.

---

## 1. Como Rodar no Ambiente Local (Desenvolvimento e Testes)

Ideal para máquinas com recursos modestos, executando testes unitários e verificações de código. O projeto utiliza o padrão moderno `pyproject.toml`.

```bash
# 1. Clonar o repositório
git clone https://github.com/leticia-pontes/gnn-drift-botnet-detection.git
cd gnn-drift-botnet-detection

# 2. Criar e ativar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate

# 3. Instalar dependências leves + pacote de testes em modo editável (-e)
pip install -e ".[test]"

# 4. Executar a suíte de testes unitários
pytest
```

---

## 2. Como Rodar no Google Colab via Extensão do VS Code

1. Abra a pasta do projeto no **VS Code**.
2. Conecte ao runtime do Colab usando a extensão oficial (**Google Colab for VS Code**).
3. Abra `notebooks/00_setup.ipynb` e execute as células em ordem:
   - O setup detectará se o Google Drive foi montado.
   - Caso o Drive esteja montado, os dados e Parquets ficarão em `/content/drive/MyDrive/tcc/data/`.
   - Se o Drive não estiver montado, os dados ficarão em `/content/tcc/data/` (atenção: volátil).
   - Instala as dependências pesadas e o projeto em modo editável com:

     ```bash
     pip install -e ".[colab]"
     ```

4. Abra `notebooks/01_iot23_exploracao.ipynb` para baixar a versão leve do IoT-23, converter para Parquet e gerar a partição temporal `splits.md`.
