"""Testes unitários para o módulo de leitura e parsing de dados Zeek."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.data import (
    parse_zeek_header,
    parse_label_columns,
    read_conn_log_batches,
    list_iot23_captures,
)

# Amostra sintética de conn.log.labeled contendo casos de borda do Zeek:
# - Cabeçalho padrão com #fields
# - Valores '-' que devem virar nulos
# - Rótulo concatenado com espaços e variações de Benign e Malicious
ZEEK_SYNTHETIC_DATA = (
    "#separator \\x09\n"
    "#set_separator ,\n"
    "#empty_field (empty)\n"
    "#unset_field -\n"
    "#path conn\n"
    "#open 2026-10-06-00-00-00\n"
    "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice\tduration\torig_bytes\tresp_bytes\tconn_state\tlocal_orig\tlocal_resp\tmissed_bytes\thistory\torig_pkts\torig_ip_bytes\tresp_pkts\tresp_ip_bytes\ttunnel_parents\tlabel\n"
    "1532165000.000000\tC1\t192.168.1.1\t1000\t192.168.1.2\t80\ttcp\thttp\t1.5\t100\t200\tSF\t-\t-\t0\tShADadFf\t10\t500\t12\t600\t-\t-   benign   -\n"
    "1532165010.000000\tC2\t192.168.1.1\t1001\t192.168.1.3\t443\ttcp\t-\t-\t-\t-\tS0\t-\t-\t0\tS\t1\t40\t0\t0\t-\t-   Malicious   PartOfAHorizontalPortScan\n"
    "1532165020.000000\tC3\t192.168.1.5\t5353\t224.0.0.251\t5353\tudp\tdns\t0.05\t50\t0\tSF\t-\t-\t0\tD\t1\t78\t0\t0\t-\tBenign\n"
    "1532165030.000000\tC4\t192.168.1.1\t1002\t192.168.1.4\t23\ttcp\t-\t0.1\t20\t30\tSF\t-\t-\t0\tShA\t2\t80\t2\t80\t-\t(empty)   Malicious   Mirai\n"
)


@pytest.fixture
def synthetic_zeek_file(tmp_path: Path) -> Path:
    """Cria um arquivo temporário com o log sintético do Zeek."""
    log_file = tmp_path / "conn.log.labeled"
    log_file.write_text(ZEEK_SYNTHETIC_DATA, encoding="utf-8")
    return log_file


def test_parse_zeek_header(synthetic_zeek_file: Path):
    fields = parse_zeek_header(synthetic_zeek_file)
    assert len(fields) == 22
    assert fields[0] == "ts"
    assert fields[2] == "id.orig_h"
    assert fields[-1] == "label"


def test_parse_label_columns():
    raw_series = pd.Series([
        "-   benign   -",
        "-   Malicious   PartOfAHorizontalPortScan",
        "Benign",
        "(empty)   Malicious   Mirai",
        "-",
    ])
    df_labels = parse_label_columns(raw_series)
    
    assert df_labels.loc[0, "label"] == "Benign"
    assert df_labels.loc[0, "detailed_label"] == "Benign"
    
    assert df_labels.loc[1, "label"] == "Malicious"
    assert df_labels.loc[1, "detailed_label"] == "PartOfAHorizontalPortScan"
    
    assert df_labels.loc[2, "label"] == "Benign"
    assert df_labels.loc[3, "label"] == "Malicious"
    assert df_labels.loc[3, "detailed_label"] == "Mirai"
    assert df_labels.loc[4, "label"] == "Benign"


def test_read_conn_log_batches_nulls_and_types(synthetic_zeek_file: Path):
    chunks = list(read_conn_log_batches(synthetic_zeek_file, chunksize=2))
    assert len(chunks) == 2  # 4 linhas totais / chunksize 2 = 2 chunks
    
    df_all = pd.concat(chunks, ignore_index=True)
    assert len(df_all) == 4
    
    # Validação de valor nulo '-' convertido para NaN
    assert pd.isna(df_all.loc[1, "duration"])
    assert pd.isna(df_all.loc[1, "orig_bytes"])
    
    # Validação de tipagem numérica
    assert df_all["ts"].dtype == "float64"
    assert df_all.loc[0, "ts"] == 1532165000.0


def test_list_iot23_captures_derived_date(tmp_path: Path):
    capture_dir = tmp_path / "CTU-IoT-Malicious-1-1"
    capture_dir.mkdir(parents=True)
    log_file = capture_dir / "conn.log.labeled"
    log_file.write_text(ZEEK_SYNTHETIC_DATA, encoding="utf-8")

    df_captures = list_iot23_captures(base_dir=tmp_path)
    assert len(df_captures) == 1
    assert df_captures.loc[0, "capture_name"] == "CTU-IoT-Malicious-1-1"
    # 1532165000 epoch corresponde a 2018-07-21
    assert df_captures.loc[0, "derived_start_date"].year == 2018

