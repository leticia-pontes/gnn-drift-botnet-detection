"""Módulo de ingestão, parsing Zeek, processamento de lotes e divisão temporal."""

import re
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator
import pandas as pd
from tqdm import tqdm

from src.config import DATA_DIR, PARQUET_DIR, logger

# URL oficial documentada pelo Stratosphere Lab para o arquivo leve do IoT-23
# (contendo apenas conn.log.labeled de cada captura)
# Referência: https://www.stratosphereips.org/datasets-iot23
IOT23_LEIGHTWEIGHT_URL: str = (
    "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/iot_23_datasets_small.tar.gz"
)

# Tipagem compacta para colunas conhecidas do Zeek conn.log
ZEEK_DITYPES: Dict[str, str] = {
    "ts": "float64",
    "uid": "string",
    "id.orig_h": "string",
    "id.orig_p": "float32",  # float suporta NaN antes de casting
    "id.resp_h": "string",
    "id.resp_p": "float32",
    "proto": "category",
    "service": "category",
    "duration": "float32",
    "orig_bytes": "float32",
    "resp_bytes": "float32",
    "conn_state": "category",
    "local_orig": "category",
    "local_resp": "category",
    "missed_bytes": "float32",
    "history": "string",
    "orig_pkts": "float32",
    "orig_ip_bytes": "float32",
    "resp_pkts": "float32",
    "resp_ip_bytes": "float32",
}


def download_iot23_lightweight(
    dest_dir: Path = DATA_DIR,
    url: str = IOT23_LEIGHTWEIGHT_URL,
    force_download: bool = False,
) -> Path:
    """
    Baixa e descompacta a versão leve do dataset IoT-23 contendo conn.log.labeled.
    
    Verifica se os arquivos já foram baixados antes de realizar nova requisição.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_name = "iot_23_datasets_small.tar.gz"
    archive_path = dest_dir / archive_name
    extracted_flag = dest_dir / ".iot23_extracted"

    if extracted_flag.exists() and not force_download:
        logger.info("Dataset IoT-23 já descompactado em %s", dest_dir)
        return dest_dir

    if not archive_path.exists() or force_download:
        logger.info("Iniciando download do IoT-23 (versão leve) de %s", url)
        try:
            with tqdm(unit="B", unit_scale=True, miniters=1, desc=archive_name) as t:
                def reporthook(block_num: int, block_size: int, total_size: int):
                    if total_size > 0:
                        t.total = total_size
                    t.update(block_size)

                urllib.request.urlretrieve(url, archive_path, reporthook=reporthook)
        except Exception as e:
            logger.error(
                "Falha ao baixar o arquivo da URL oficial (%s): %s\n"
                "Caso a conexão caia ou a URL mude, baixe manualmente o arquivo "
                "'iot_23_datasets_small.tar.gz' do site https://www.stratosphereips.org/datasets-iot23 "
                "e coloque-o em %s", url, e, dest_dir
            )
            raise e

    logger.info("Descompactando arquivo %s...", archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)

    extracted_flag.touch()
    logger.info("Extração concluída com sucesso.")
    return dest_dir


def parse_zeek_header(filepath: Path) -> List[str]:
    """
    Lê as primeiras linhas de um arquivo Zeek conn.log para extrair a lista de colunas.
    
    Localiza a linha iniciada por '#fields'.
    """
    fields: List[str] = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#fields"):
                fields = line.split("\t")[1:]
                break
            if not line.startswith("#"):
                # Saiu do cabeçalho sem encontrar #fields
                break

    if not fields:
        raise ValueError(f"Não foi possível encontrar '#fields' no arquivo {filepath}")
    return fields


def parse_label_columns(raw_series: pd.Series) -> pd.DataFrame:
    """
    Efetua o parsing da coluna de rótulo composto do IoT-23.
    
    No IoT-23, as colunas finais (tunnel_parents, label, detailed-label) frequentemente
    aparecem juntas separadas por espaços múltiplos ou tabs.
    Normaliza a classe binária em: Benign ou Malicious, e extrai o detailed_label.
    """
    def _parse_val(val: str) -> Tuple[str, str]:
        if not isinstance(val, str) or val.strip() == "-" or not val.strip():
            return "Benign", "Benign"
        
        val_clean = val.strip()
        # Normalização de rótulo binário
        if "benign" in val_clean.lower():
            binary_label = "Benign"
        else:
            binary_label = "Malicious"

        # Tentativa de extração da família/ataque do detailed-label
        # Padrões comuns: "Malicious   PartOfAHorizontalPortScan" ou "Attack   C&C"
        parts = [p.strip() for p in re.split(r"\s+", val_clean) if p.strip()]
        if len(parts) >= 2:
            detailed = parts[-1]
            if detailed.lower() == "benign":
                detailed = "Benign"
        else:
            detailed = "Unknown" if binary_label == "Malicious" else "Benign"

        return binary_label, detailed

    parsed = raw_series.apply(_parse_val)
    return pd.DataFrame(parsed.tolist(), index=raw_series.index, columns=["label", "detailed_label"])


def read_conn_log_batches(
    filepath: Path,
    chunksize: int = 200_000,
    max_rows: Optional[int] = None,
) -> Generator[pd.DataFrame, None, None]:
    """
    Lê um arquivo conn.log.labeled em chunks com baixo consumo de memória.
    
    Trata campos '-' como NaN e ajusta colunas de rótulo mescladas.
    """
    raw_fields = parse_zeek_header(filepath)
    rows_yielded = 0

    chunks = pd.read_csv(
        filepath,
        sep="\t",
        comment="#",
        names=raw_fields,
        na_values=["-", "(empty)"],
        dtype=str,
        chunksize=chunksize,
        low_memory=True,
    )

    for chunk in chunks:
        if max_rows and rows_yielded >= max_rows:
            break

        if max_rows and (rows_yielded + len(chunk)) > max_rows:
            chunk = chunk.iloc[: max_rows - rows_yielded]

        # Trata o campo de rótulo (que costuma ser a última coluna)
        last_col = raw_fields[-1]
        labels_df = parse_label_columns(chunk[last_col])
        chunk["label"] = labels_df["label"].astype("category")
        chunk["detailed_label"] = labels_df["detailed_label"].astype("category")

        # Conversão de tipos seguros
        for col, dtype in ZEEK_DITYPES.items():
            if col in chunk.columns:
                try:
                    if dtype == "category":
                        chunk[col] = chunk[col].astype("category")
                    else:
                        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
                except Exception:
                    pass

        rows_yielded += len(chunk)
        yield chunk


def convert_capture_to_parquet(
    capture_log_path: Path,
    output_parquet_path: Path,
    chunksize: int = 200_000,
    max_rows: Optional[int] = None,
) -> None:
    """Converte um conn.log.labeled para formato Parquet com compressão Snappy."""
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    first_chunk = True

    for chunk in read_conn_log_batches(capture_log_path, chunksize=chunksize, max_rows=max_rows):
        if first_chunk:
            chunk.to_parquet(output_parquet_path, engine="pyarrow", index=False)
            first_chunk = False
        else:
            chunk.to_parquet(
                output_parquet_path,
                engine="pyarrow",
                index=False,
                partition_cols=None,
            )


def list_iot23_captures(base_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Varre o diretório em busca de capturas do IoT-23 e calcula metadados.
    
    A data de coleta é derivada rigorosamente do menor valor do campo 'ts' (epoch).
    """
    records = []
    log_files = list(base_dir.rglob("conn.log.labeled"))
    if not log_files:
        log_files = list(base_dir.rglob("conn.log"))

    logger.info("Encontrados %d arquivos de log em %s", len(log_files), base_dir)

    for path in log_files:
        capture_name = path.parent.name
        file_size_mb = path.stat().st_size / (1024 * 1024)
        
        min_ts = None
        line_count = 0
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.startswith("#"):
                        line_count += 1
                        if min_ts is None:
                            parts = line.split("\t")
                            try:
                                min_ts = float(parts[0])
                            except (ValueError, IndexError):
                                pass
        except Exception as e:
            logger.warning("Erro ao ler metadados de %s: %s", path, e)

        start_date = pd.to_datetime(min_ts, unit="s") if min_ts else pd.NaT

        records.append({
            "capture_name": capture_name,
            "filepath": str(path),
            "size_mb": round(file_size_mb, 2),
            "total_lines_approx": line_count,
            "first_timestamp_epoch": min_ts,
            "derived_start_date": start_date,
        })

    df = pd.DataFrame(records)
    if not df.empty and "derived_start_date" in df.columns:
        df = df.sort_values(by="derived_start_date", ascending=True).reset_index(drop=True)
    return df


def generate_temporal_splits(
    captures_summary_df: pd.DataFrame,
    train_ratio: float = 0.7,
) -> Tuple[List[str], List[str]]:
    """
    Gera uma partição temporal em ordem cronológica estrita (sem embaralhamento).
    
    Retorna (lista_treino, lista_teste) contendo os nomes das capturas.
    """
    sorted_df = captures_summary_df.sort_values("derived_start_date").reset_index(drop=True)
    n_total = len(sorted_df)
    n_train = max(1, int(n_total * train_ratio))

    train_captures = sorted_df.iloc[:n_train]["capture_name"].tolist()
    test_captures = sorted_df.iloc[n_train:]["capture_name"].tolist()

    return train_captures, test_captures

