"""Módulo de configuração de caminhos, parâmetros de execução e reprodutibilidade."""

import os
import sys
import random
import logging
from pathlib import Path
from typing import Dict, Any

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Semente padrão para reprodutibilidade
SEED: int = int(os.environ.get("EXPERIMENT_SEED", 42))


def set_seed(seed: int = SEED) -> None:
    """Fixa as sementes em bibliotecas padrão, numpy e torch (se disponível)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def is_colab() -> bool:
    """Detecta se o código está sendo executado em um ambiente Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def is_drive_mounted() -> bool:
    """Verifica se o Google Drive está montado no caminho padrão do Colab."""
    return Path("/content/drive/MyDrive").exists()


def get_base_dirs() -> Dict[str, Path]:
    """
    Determina os diretórios de dados e resultados de forma adaptativa.
    
    Prioridade:
    1. Variáveis de ambiente DATA_DIR e RESULTS_DIR (se definidas).
    2. Se no Colab e Drive montado: /content/drive/MyDrive/tcc/...
    3. Se no Colab sem Drive: /content/tcc/... (com alerta de volatilidade).
    4. Local: pastas 'data/' e 'results/' relativas à raiz do projeto.
    """
    env_data = os.environ.get("DATA_DIR")
    env_results = os.environ.get("RESULTS_DIR")
    
    if env_data and env_results:
        data_dir = Path(env_data)
        results_dir = Path(env_results)
        logger.info(f"Usando caminhos definidos via variáveis de ambiente: {data_dir}")
    elif is_colab():
        if is_drive_mounted():
            base_drive = Path("/content/drive/MyDrive/tcc")
            data_dir = base_drive / "data"
            results_dir = base_drive / "results"
            logger.info("Google Colab detectado com Google Drive montado. Armazenamento persistente ativo.")
        else:
            base_ephemeral = Path("/content/tcc")
            data_dir = base_ephemeral / "data"
            results_dir = base_ephemeral / "results"
            logger.warning(
                "ATENÇÃO: Google Drive NÃO está montado no Colab! "
                "Os dados serão salvos em disco volátil (/content/tcc) e serão "
                "PERDIDOS permanentemente ao encerrar a sessão de runtime."
            )
    else:
        # Caminho local relativo à raiz do repositório
        repo_root = Path(__file__).resolve().parent.parent
        data_dir = repo_root / "data"
        results_dir = repo_root / "results"
        logger.info(f"Ambiente local detectado. Diretório base: {repo_root}")

    # Criação das pastas caso não existam
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "parquet").mkdir(parents=True, exist_ok=True)

    return {
        "DATA_DIR": data_dir,
        "PARQUET_DIR": data_dir / "parquet",
        "RESULTS_DIR": results_dir,
    }


# Inicialização dos caminhos expostos
_DIRS = get_base_dirs()
DATA_DIR: Path = _DIRS["DATA_DIR"]
PARQUET_DIR: Path = _DIRS["PARQUET_DIR"]
RESULTS_DIR: Path = _DIRS["RESULTS_DIR"]

