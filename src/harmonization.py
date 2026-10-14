"""Módulo de harmonização de esquemas de fluxos de rede para o pipeline de detecção."""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Atributos canônicos unificados para fluxos de rede
CANONICAL_COLUMNS: List[str] = [
    "ts",           # Timestamp do início do fluxo (epoch float)
    "src_ip",       # IP de origem (string)
    "src_port",     # Porta de origem (int ou float se nulo)
    "dst_ip",       # IP de destino (string)
    "dst_port",     # Porta de destino (int ou float se nulo)
    "proto",        # Protocolo (tcp, udp, icmp, etc.)
    "duration",     # Duração em segundos
    "orig_bytes",   # Bytes enviados pela origem
    "resp_bytes",   # Bytes enviados pelo destino
    "orig_pkts",    # Pacotes enviados pela origem
    "resp_pkts",    # Pacotes enviados pelo destino
    "label",        # 0 = Benign, 1 = Malicious
    "detailed_label",  # Subclasse/Família (Mirai, PortScan, etc.)
    "dataset_source",  # Identificador da base ('iot23', 'ton_iot', 'ctu13', 'bot_iot')
]

NUMERICAL_FEATURE_COLS: List[str] = [
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
]


def harmonize_iot23(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmoniza um DataFrame de conn.log (Zeek) do IoT-23 para o esquema canônico.
    """
    col_map = {
        "id.orig_h": "src_ip",
        "id.orig_p": "src_port",
        "id.resp_h": "dst_ip",
        "id.resp_p": "dst_port",
    }
    res = df.rename(columns=col_map).copy()

    # Preenchimento de colunas ausentes
    if "ts" not in res.columns:
        raise ValueError("DataFrame do IoT-23 precisa conter a coluna 'ts'.")

    # Mapeamento do label textual para binário (0 = Benign, 1 = Malicious)
    if "label" in res.columns:
        if res["label"].dtype == "object" or res["label"].dtype.name == "category":
            res["label"] = res["label"].apply(
                lambda x: 0 if str(x).strip().lower() == "benign" else 1
            ).astype(np.int64)
    else:
        res["label"] = 0

    if "detailed_label" not in res.columns:
        res["detailed_label"] = "Benign"

    # Conversão de numéricos e preenchimento de nulos com zero
    for col in NUMERICAL_FEATURE_COLS:
        if col in res.columns:
            res[col] = pd.to_numeric(res[col], errors="coerce").fillna(0.0).astype(np.float32)
        else:
            res[col] = np.float32(0.0)

    res["dataset_source"] = "iot23"

    for col in ["src_ip", "dst_ip", "proto"]:
        if col in res.columns:
            res[col] = res[col].astype(str)
        else:
            res[col] = "unknown"

    return res[CANONICAL_COLUMNS]


def harmonize_ton_iot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmoniza o CSV de tráfego de rede do TON-IoT para o esquema canônico.
    
    O TON-IoT Network Dataset (Zeek CSV) normalmente contém colunas como:
    ts, src_ip, src_port, dst_ip, dst_port, proto, duration,
    src_bytes, dst_bytes, src_pkts, dst_pkts, type (ou label).
    """
    col_map = {
        "src_bytes": "orig_bytes",
        "dst_bytes": "resp_bytes",
        "src_pkts": "orig_pkts",
        "dst_pkts": "resp_pkts",
        "type": "detailed_label",
    }
    res = df.rename(columns=col_map).copy()

    # Tratamento de rótulos do TON-IoT
    if "label" in res.columns:
        # Se for numérico (0 ou 1) ou texto
        res["label"] = pd.to_numeric(res["label"], errors="coerce").fillna(0).astype(np.int64)
    elif "detailed_label" in res.columns:
        res["label"] = res["detailed_label"].apply(
            lambda x: 0 if str(x).strip().lower() in ("normal", "benign") else 1
        ).astype(np.int64)
    else:
        res["label"] = 0

    if "detailed_label" not in res.columns:
        res["detailed_label"] = res["label"].apply(lambda x: "Benign" if x == 0 else "Malicious")

    for col in NUMERICAL_FEATURE_COLS:
        if col in res.columns:
            res[col] = pd.to_numeric(res[col], errors="coerce").fillna(0.0).astype(np.float32)
        else:
            res[col] = np.float32(0.0)

    if "ts" in res.columns:
        res["ts"] = pd.to_numeric(res["ts"], errors="coerce").fillna(0.0).astype(np.float64)
    else:
        res["ts"] = 0.0

    res["dataset_source"] = "ton_iot"

    for col in ["src_ip", "dst_ip", "proto"]:
        if col in res.columns:
            res[col] = res[col].astype(str)
        else:
            res[col] = "unknown"

    return res[CANONICAL_COLUMNS]


def harmonize_ctu13(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmoniza o formato NetFlow (.binetflow) do CTU-13 para o esquema canônico.
    
    Colunas típicas do NetFlow / Argus:
    StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr, Dport, State, sTos, dTos,
    TotPkts, TotBytes, SrcBytes, Label
    """
    col_map = {
        "SrcAddr": "src_ip",
        "Sport": "src_port",
        "DstAddr": "dst_ip",
        "Dport": "dst_port",
        "Proto": "proto",
        "Dur": "duration",
        "SrcBytes": "orig_bytes",
        "TotBytes": "total_bytes",
        "TotPkts": "orig_pkts",
        "Label": "raw_label",
    }
    res = df.rename(columns=col_map).copy()

    # Se resp_bytes não vier explícito, deriva: TotBytes - SrcBytes
    if "resp_bytes" not in res.columns:
        if "total_bytes" in res.columns and "orig_bytes" in res.columns:
            tot = pd.to_numeric(res["total_bytes"], errors="coerce").fillna(0.0)
            src = pd.to_numeric(res["orig_bytes"], errors="coerce").fillna(0.0)
            res["resp_bytes"] = np.maximum(0.0, tot - src).astype(np.float32)
        else:
            res["resp_bytes"] = np.float32(0.0)

    if "resp_pkts" not in res.columns:
        res["resp_pkts"] = np.float32(0.0)

    # Conversão de StartTime (ex: '2011/08/10 09:46:53.047277') para timestamp epoch
    if "StartTime" in res.columns:
        res["ts"] = pd.to_datetime(res["StartTime"], errors="coerce").astype("int64") // 10**9
    elif "ts" in res.columns:
        res["ts"] = pd.to_numeric(res["ts"], errors="coerce").fillna(0.0).astype(np.float64)
    else:
        res["ts"] = 0.0

    # Rótulo do CTU-13: normalmente contém 'Botnet', 'Background', 'Normal'
    if "raw_label" in res.columns:
        res["label"] = res["raw_label"].apply(
            lambda x: 1 if "botnet" in str(x).lower() else 0
        ).astype(np.int64)
        res["detailed_label"] = res["raw_label"].astype(str)
    else:
        res["label"] = 0
        res["detailed_label"] = "Unknown"

    for col in NUMERICAL_FEATURE_COLS:
        if col in res.columns:
            res[col] = pd.to_numeric(res[col], errors="coerce").fillna(0.0).astype(np.float32)
        else:
            res[col] = np.float32(0.0)

    res["dataset_source"] = "ctu13"

    for col in ["src_ip", "dst_ip", "proto"]:
        if col in res.columns:
            res[col] = res[col].astype(str)
        else:
            res[col] = "unknown"

    return res[CANONICAL_COLUMNS]

