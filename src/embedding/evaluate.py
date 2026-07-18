from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sentence_transformers import util

from src.config import get_project_root


def _load_data(config: dict):
    root = get_project_root()
    paths = config["paths"]
    text_col = config["preprocessing"]["text_column"]
    df = pd.read_csv(root / paths["bertopic_ready"])
    embeddings = np.load(root / paths["embeddings"])
    return df, embeddings, text_col


def compare_complaints(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    text_col: str,
    idx1: int,
    idx2: int,
) -> dict:
    sim = util.cos_sim(embeddings[idx1], embeddings[idx2]).item()
    return {
        "similarity": round(sim, 4),
        "complaint_1": df.iloc[idx1][text_col][:500],
        "complaint_2": df.iloc[idx2][text_col][:500],
    }


def find_extreme_pairs(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    text_col: str,
    sample_size: int,
    seed: int,
) -> dict:
    random.seed(seed)
    sample_idx = random.sample(range(len(df)), min(sample_size, len(df)))
    sample_emb = embeddings[sample_idx]
    sim_matrix = util.cos_sim(sample_emb, sample_emb).numpy()
    np.fill_diagonal(sim_matrix, -1)

    max_pos = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
    min_pos_mat = sim_matrix.copy()
    np.fill_diagonal(min_pos_mat, 2)
    min_pos = np.unravel_index(np.argmin(min_pos_mat), min_pos_mat.shape)

    return {
        "most_similar": {
            "similarity": float(sim_matrix[max_pos]),
            "idx1": sample_idx[max_pos[0]],
            "idx2": sample_idx[max_pos[1]],
        },
        "least_similar": {
            "similarity": float(min_pos_mat[min_pos]),
            "idx1": sample_idx[min_pos[0]],
            "idx2": sample_idx[min_pos[1]],
        },
    }


def top_k_accuracy(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    sample_size: int,
    top_k: int,
    seed: int,
) -> dict:
    random.seed(seed)
    sample_idx = random.sample(range(len(df)), min(sample_size, len(df)))
    sample_emb = embeddings[sample_idx]
    sim = util.cos_sim(sample_emb, sample_emb).numpy()
    np.fill_diagonal(sim, -1)

    upper = np.triu_indices(len(sample_idx), k=1)
    scores = sim[upper]
    top_indices = np.argsort(scores)[-top_k:][::-1]

    same_product = 0
    same_issue = 0
    same_both = 0

    for t in top_indices:
        i, j = upper[0][t], upper[1][t]
        idx1, idx2 = sample_idx[i], sample_idx[j]
        p1, p2 = df.iloc[idx1]["Product"], df.iloc[idx2]["Product"]
        iss1, iss2 = df.iloc[idx1]["Issue"], df.iloc[idx2]["Issue"]
        if p1 == p2:
            same_product += 1
        if iss1 == iss2:
            same_issue += 1
        if p1 == p2 and iss1 == iss2:
            same_both += 1

    return {
        "top_k": top_k,
        "same_product_pct": round(same_product / top_k * 100, 2),
        "same_issue_pct": round(same_issue / top_k * 100, 2),
        "same_both_pct": round(same_both / top_k * 100, 2),
    }


def nearest_neighbor_accuracy(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    sample_size: int,
    seed: int,
) -> dict:
    random.seed(seed)
    sample_idx = random.sample(range(len(df)), min(sample_size, len(df)))
    sample_emb = embeddings[sample_idx]
    sim = util.cos_sim(sample_emb, sample_emb).numpy()
    np.fill_diagonal(sim, -1)

    same_product = 0
    same_issue = 0
    same_both = 0

    for i in range(len(sample_idx)):
        nn = np.argmax(sim[i])
        idx1, idx2 = sample_idx[i], sample_idx[nn]
        p1, p2 = df.iloc[idx1]["Product"], df.iloc[idx2]["Product"]
        iss1, iss2 = df.iloc[idx1]["Issue"], df.iloc[idx2]["Issue"]
        if p1 == p2:
            same_product += 1
        if iss1 == iss2:
            same_issue += 1
        if p1 == p2 and iss1 == iss2:
            same_both += 1

    n = len(sample_idx)
    return {
        "sample_size": n,
        "product_accuracy_pct": round(same_product / n * 100, 2),
        "issue_accuracy_pct": round(same_issue / n * 100, 2),
        "both_accuracy_pct": round(same_both / n * 100, 2),
    }


def run(config: dict) -> dict:
    df, embeddings, text_col = _load_data(config)
    seed = config["project"]["random_seed"]
    eval_cfg = config["evaluation"]

    print(f"Data shape: {df.shape}, Embeddings shape: {embeddings.shape}")

    pairs = find_extreme_pairs(df, embeddings, text_col, eval_cfg["embedding_test_sample_size"], seed)
    print(f"Most similar pair similarity: {pairs['most_similar']['similarity']:.4f}")
    print(f"Least similar pair similarity: {pairs['least_similar']['similarity']:.4f}")

    ak = top_k_accuracy(df, embeddings, eval_cfg["embedding_pair_sample_size"], eval_cfg["top_k_similar_pairs"], seed)
    print(f"Top-{ak['top_k']} similar pairs: Same Product={ak['same_product_pct']}%, Same Issue={ak['same_issue_pct']}%")

    nn = nearest_neighbor_accuracy(df, embeddings, eval_cfg["embedding_pair_sample_size"], seed)
    print(f"Nearest Neighbor: Product={nn['product_accuracy_pct']}%, Issue={nn['issue_accuracy_pct']}%")

    return {"extreme_pairs": pairs, "top_k_accuracy": ak, "nearest_neighbor": nn}
