"""Módulo para construção de grafos e snapshots temporais a partir de fluxos de rede."""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd


def compute_edge_weights(df_flows: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a heurística de pesos normalizados das arestas para tráfego entre IPs:
    
        w(u, v) = f(u, v) / sum_{i} f(u, i)
        
    Onde:
      f(u, v) é a contagem de fluxos direcionados de u para v.
      O denominador é a soma total de fluxos originados por u.
      
    Retorna um DataFrame com as colunas: ['src_ip', 'dst_ip', 'flow_count', 'weight']
    """
    if "src_ip" not in df_flows.columns or "dst_ip" not in df_flows.columns:
        raise ValueError("O DataFrame de fluxos deve conter as colunas 'src_ip' e 'dst_ip'.")

    # Contagem de fluxos f(u, v)
    edge_counts = (
        df_flows.groupby(["src_ip", "dst_ip"])
        .size()
        .reset_index(name="flow_count")
    )

    # Total de fluxos originados por cada u: sum_i f(u, i)
    out_degree_sum = (
        edge_counts.groupby("src_ip")["flow_count"]
        .transform("sum")
    )

    # Heurística de peso normalizado w(u, v)
    edge_counts["weight"] = (edge_counts["flow_count"] / out_degree_sum).astype(np.float32)

    return edge_counts


def create_temporal_snapshots(
    df_flows: pd.DataFrame,
    window_seconds: float = 3600.0,
    stride_seconds: Optional[float] = None,
) -> List[pd.DataFrame]:
    """
    Segmenta os fluxos ordenados temporalmente em uma sequência de janelas (snapshots).
    
    Parâmetros:
      df_flows: DataFrame contendo a coluna 'ts' (epoch float/segundos).
      window_seconds: Duração temporal da janela em segundos (ex: 3600 = 1 hora).
      stride_seconds: Passo de deslocamento da janela. Se None, assume janelas contíguas sem sobreposição (stride = window_seconds).
      
    Retorna:
      Lista de DataFrames correspondentes a cada snapshot temporal.
    """
    if "ts" not in df_flows.columns:
        raise ValueError("O DataFrame precisa conter a coluna 'ts' para gerar snapshots.")

    df_sorted = df_flows.sort_values("ts").reset_index(drop=True)
    if df_sorted.empty:
        return []

    min_ts = df_sorted["ts"].iloc[0]
    max_ts = df_sorted["ts"].iloc[-1]
    stride = stride_seconds if stride_seconds is not None else window_seconds

    snapshots = []
    current_start = min_ts

    while current_start <= max_ts:
        current_end = current_start + window_seconds
        mask = (df_sorted["ts"] >= current_start) & (df_sorted["ts"] < current_end)
        snap = df_sorted.loc[mask].copy()
        if not snap.empty:
            snapshots.append(snap)
        current_start += stride

    return snapshots


def build_pyg_graph_data(
    df_flows: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    node_id_map: Optional[Dict[str, int]] = None,
):
    """
    Converte um DataFrame de fluxos harmonizados em estrutura de grafo compatível com PyG.
    
    Constrói a matriz de arestas (edge_index), vetor de pesos (edge_attr / edge_weight),
    e mapeamento de nós (IPs).
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError(
            "PyTorch e PyTorch Geometric são necessários para construir objetos Data. "
            "Execute 'pip install torch torch-geometric'."
        )

    if feature_cols is None:
        feature_cols = ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]

    # Identificação única de todos os nós (IPs de origem e destino)
    all_ips = pd.concat([df_flows["src_ip"], df_flows["dst_ip"]]).unique()
    if node_id_map is None:
        node_id_map = {ip: idx for idx, ip in enumerate(all_ips)}

    # Mapeamento dos índices numéricos das arestas
    src_indices = df_flows["src_ip"].map(node_id_map).values
    dst_indices = df_flows["dst_ip"].map(node_id_map).values

    edge_index = torch.tensor(np.array([src_indices, dst_indices]), dtype=torch.long)

    # Heurística de pesos
    weights_df = compute_edge_weights(df_flows)
    merged_weights = df_flows.merge(
        weights_df[["src_ip", "dst_ip", "weight"]],
        on=["src_ip", "dst_ip"],
        how="left"
    )
    edge_weight = torch.tensor(merged_weights["weight"].values, dtype=torch.float32)

    # Atributos numéricos das arestas (fluxos)
    present_cols = [c for c in feature_cols if c in df_flows.columns]
    if present_cols:
        edge_attr = torch.tensor(df_flows[present_cols].values, dtype=torch.float32)
    else:
        edge_attr = edge_weight.unsqueeze(1)

    # Rótulos (binário por fluxo / aresta ou nó)
    if "label" in df_flows.columns:
        y = torch.tensor(df_flows["label"].values, dtype=torch.long)
    else:
        y = None

    num_nodes = len(node_id_map)
    # Inicialização dos atributos dos nós: matriz identidade ou grau
    x = torch.ones((num_nodes, 1), dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_weight=edge_weight,
        y=y,
        num_nodes=num_nodes
    )

    return data, node_id_map
