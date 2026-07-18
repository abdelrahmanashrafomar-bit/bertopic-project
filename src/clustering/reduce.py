from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from umap import UMAP

from src.config import get_project_root
from src.validators import ensure_file_exists


def fit_umap(embeddings: np.ndarray, params: dict) -> np.ndarray:
    print(f"Fitting UMAP with params: {params}")
    reducer = UMAP(**params)
    embeddings_umap = reducer.fit_transform(embeddings)
    print(f"UMAP output shape: {embeddings_umap.shape}")
    return embeddings_umap


def evaluate_trustworthiness(
    original: np.ndarray,
    reduced: np.ndarray,
    n_neighbors: int,
    sample_size: int | None = None,
    seed: int = 42,
) -> float:
    if sample_size and sample_size < len(original):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(original), size=sample_size, replace=False)
        original = original[idx]
        reduced = reduced[idx]
    score = sklearn_trustworthiness(original, reduced, n_neighbors=n_neighbors)
    return float(score)


def tune_umap(
    embeddings: np.ndarray,
    umap_config: dict,
    hdbscan_config: dict,
    seed: int = 42,
) -> pd.DataFrame:
    import hdbscan

    sample_size = umap_config["tuning"]["sample_size"]
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(embeddings), size=min(sample_size, len(embeddings)), replace=False)
    emb_sample = embeddings[sample_idx]

    results = []
    for n_neighbors in umap_config["tuning"]["n_neighbors"]:
        reducer = UMAP(
            n_neighbors=n_neighbors,
            n_components=umap_config["tuning"]["n_components"],
            min_dist=umap_config["tuning"]["min_dist"],
            metric=umap_config["tuning"]["metric"],
            random_state=umap_config["tuning"]["random_state"],
        )
        umap_emb = reducer.fit_transform(emb_sample)

        trust = sklearn_trustworthiness(
            emb_sample,
            umap_emb,
            n_neighbors=umap_config["tuning"]["trustworthiness_neighbors"],
        )

        clusterer = hdbscan.HDBSCAN(**hdbscan_config["umap_tuning_eval"])
        labels = clusterer.fit_predict(umap_emb)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = float(np.mean(labels == -1))
        dbcv = getattr(clusterer, "relative_validity_", float("nan"))

        results.append({
            "n_neighbors": n_neighbors,
            "trustworthiness": round(trust, 6),
            "dbcv": round(float(dbcv), 6),
            "n_clusters": n_clusters,
            "noise_ratio": round(noise_ratio, 4),
        })

    results_df = pd.DataFrame(results).sort_values("dbcv", ascending=False).reset_index(drop=True)
    print("\nUMAP tuning results:")
    print(results_df.to_string(index=False))
    return results_df


def fit_visualization_umap(embeddings: np.ndarray, params: dict) -> np.ndarray:
    print("Fitting 2D UMAP for visualization...")
    reducer = UMAP(**params)
    return reducer.fit_transform(embeddings)


def run(config: dict) -> np.ndarray:
    root = get_project_root()
    umap_cfg = config["umap"]
    hdbscan_cfg = config["hdbscan"]
    seed = config["project"]["random_seed"]
    paths = config["paths"]

    emb_path = root / paths["embeddings"]
    ensure_file_exists(emb_path, "Embeddings")
    embeddings = np.load(emb_path)

    print(f"Loaded embeddings: {embeddings.shape}")

    results_df = tune_umap(embeddings, umap_cfg, hdbscan_cfg, seed)
    best = results_df.iloc[0]
    final_params = dict(umap_cfg["final"])
    final_params["n_neighbors"] = int(best["n_neighbors"])
    print(f"\nBest config: n_neighbors={final_params['n_neighbors']}, "
          f"n_components={final_params['n_components']}, min_dist={final_params['min_dist']}")

    embeddings_umap = fit_umap(embeddings, final_params)
    np.save(root / paths["embeddings_umap"], embeddings_umap)
    print(f"UMAP embeddings saved: {root / paths['embeddings_umap']}")

    trust = evaluate_trustworthiness(
        embeddings, embeddings_umap,
        n_neighbors=umap_cfg["trustworthiness"]["n_neighbors"],
        sample_size=umap_cfg["trustworthiness"]["sample_size"],
        seed=seed,
    )
    print(f"Trustworthiness score: {trust:.4f}")

    return embeddings_umap
