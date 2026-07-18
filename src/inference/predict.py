from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.config import get_project_root
from src.inference.schemas import TopicPrediction
from src.validators import ensure_file_exists


def load_lookup(root: Path, lookup_path: str) -> dict[int, str]:
    full_path = root / lookup_path
    ensure_file_exists(full_path, "Topic lookup CSV")
    lookup_frame = pd.read_csv(full_path)
    return lookup_frame.set_index("Topic")["Topic_Label"].to_dict()


def load_topic_centroids(root: Path, centroids_path: str) -> dict[int, np.ndarray]:
    full_path = root / centroids_path
    ensure_file_exists(full_path, "Topic centroids")
    centroids = np.load(full_path, allow_pickle=True).item()
    return centroids


def predict_with_centroids(
    text: str,
    embedding_model: SentenceTransformer,
    centroids: dict[int, np.ndarray],
    lookup: dict[int, str],
    top_n: int = 3,
) -> list[TopicPrediction]:
    ordered_ids = sorted(centroids.keys())
    centroid_matrix = np.array([centroids[tid] for tid in ordered_ids])
    query_vector = embedding_model.encode([text], convert_to_numpy=True)
    similarities = cosine_similarity(query_vector, centroid_matrix)[0]
    ranking = np.argsort(similarities)[::-1][:top_n]
    return [
        TopicPrediction(
            topic_id=int(ordered_ids[idx]),
            label=lookup.get(int(ordered_ids[idx]), "Unknown Topic"),
            score=float(similarities[idx]),
        )
        for idx in ranking
    ]


def predict_with_bertopic(
    text: str,
    topic_model,
    lookup: dict[int, str],
    top_n: int = 1,
) -> list[TopicPrediction]:
    topics, probs = topic_model.transform([text])
    topic_id = int(topics[0])
    try:
        if probs is not None and hasattr(probs, "__len__") and len(probs) > 0:
            if hasattr(probs[0], "__getitem__") and topic_id < len(probs[0]):
                score = float(probs[0][topic_id])
            else:
                score = 0.0
        else:
            score = 0.0
    except (TypeError, IndexError):
        score = 0.0
    return [
        TopicPrediction(
            topic_id=topic_id,
            label=lookup.get(topic_id, "Unknown Topic"),
            score=score,
        )
    ][:top_n]


def predict(
    text: str,
    config: dict,
    method: str = "centroid_similarity",
) -> list[TopicPrediction]:
    root = get_project_root()
    inf_cfg = config["inference"]
    emb_cfg = config["embedding"]

    lookup = load_lookup(root, inf_cfg["lookup_path"])

    if method == "centroid_similarity":
        centroids = load_topic_centroids(root, config["paths"]["topic_centroids"])
        embedding_model = SentenceTransformer(
            emb_cfg["model_name"],
            trust_remote_code=True,
        )
        return predict_with_centroids(
            text, embedding_model, centroids, lookup,
            top_n=inf_cfg.get("top_n", 3),
        )

    elif method == "bertopic_transform":
        from bertopic import BERTopic
        model_dir = root / inf_cfg["model_dir"]
        ensure_file_exists(model_dir, "BERTopic model directory")
        embedding_model = SentenceTransformer(
            emb_cfg["model_name"],
            trust_remote_code=True,
        )
        topic_model = BERTopic.load(model_dir, embedding_model=embedding_model)
        return predict_with_bertopic(
            text, topic_model, lookup,
            top_n=inf_cfg.get("top_n", 1),
        )

    else:
        raise ValueError(f"Unknown inference method: {method}")


def run(config: dict) -> None:
    print("Inference module loaded successfully.")
    print(f"Default method: {config['inference']['method']}")
    example_text = "Someone stole my credit card and made unauthorized purchases."
    predictions = predict(example_text, config)
    print(f"\nExample prediction for: '{example_text}'")
    for pred in predictions:
        print(f"  Topic {pred.topic_id}: {pred.label} (score={pred.score:.4f})")
