from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.cluster import BaseCluster
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.feature_extraction.text import CountVectorizer

from src.config import get_project_root
from src.validators import ensure_file_exists


def compute_topic_centroids(embeddings: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    centroids = {}
    for tid in np.unique(labels):
        if tid == -1:
            continue
        mask = labels == tid
        centroids[int(tid)] = embeddings[mask].mean(axis=0)
    print(f"Computed centroids for {len(centroids)} topics (embedding dim={list(centroids.values())[0].shape[0]})")
    return centroids


def save_topic_centroids(centroids: dict[int, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, centroids)
    print(f"Topic centroids saved: {path} ({len(centroids)} topics)")


def load_data(config: dict) -> tuple:
    root = get_project_root()
    paths = config["paths"]
    pre = config["preprocessing"]
    text_col_candidates = config["bertopic"]["text_column_candidates"]

    df = pd.read_csv(root / paths["bertopic_ready"])
    embeddings = np.load(root / paths["embeddings"])
    labels = np.load(root / paths["cluster_labels"])

    text_column = None
    for col in text_col_candidates:
        if col in df.columns:
            text_column = col
            break
    if text_column is None:
        text_column = df.select_dtypes(include=[object]).columns[0]

    documents = df[text_column].astype(str).tolist()
    print(f"Using text column: '{text_column}'")
    print(f"Loaded {len(documents):,} documents, embeddings {embeddings.shape}, "
          f"labels {labels.shape}")
    return df, documents, embeddings, labels, text_column


def fit_bertopic(
    documents: list[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    config: dict,
) -> BERTopic:
    bertopic_cfg = config["bertopic"]

    vectorizer_model = CountVectorizer(
        stop_words=bertopic_cfg.get("vectorizer_stop_words", [])
    )

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=BaseDimensionalityReduction(),
        hdbscan_model=BaseCluster(),
        vectorizer_model=vectorizer_model,
        calculate_probabilities=bertopic_cfg.get("calculate_probabilities", False),
    )

    print("Fitting BERTopic with precomputed labels and embeddings...")
    topics, _ = topic_model.fit_transform(documents, embeddings=embeddings, y=labels)
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info) - 1
    print(f"Discovered {n_topics} semantic clusters (excluding outliers)")
    return topic_model


def save_model(topic_model: BERTopic, path: Path, serialization: str = "safetensors") -> None:
    path.mkdir(parents=True, exist_ok=True)
    topic_model.save(path, serialization=serialization)
    print(f"Model saved: {path}")


def run(config: dict) -> BERTopic:
    root = get_project_root()
    paths = config["paths"]

    df, documents, embeddings, labels, text_column = load_data(config)

    assert len(df) == len(embeddings) == len(labels), (
        f"Length mismatch: df={len(df)}, embeddings={len(embeddings)}, labels={len(labels)}"
    )

    topic_model = fit_bertopic(documents, embeddings, labels, config)
    save_model(topic_model, root / paths["bertopic_model_dir"], config["bertopic"]["serialization"])

    centroids = compute_topic_centroids(embeddings, labels)
    save_topic_centroids(centroids, root / paths["topic_centroids"])

    return topic_model
