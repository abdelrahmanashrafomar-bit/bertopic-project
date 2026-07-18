from __future__ import annotations

import numpy as np
import pandas as pd
import hdbscan

from src.config import get_project_root
from src.validators import ensure_file_exists


def tune_hdbscan(embeddings_umap: np.ndarray, config: dict) -> pd.DataFrame:
    hdbscan_cfg = config["hdbscan"]
    results = []

    for min_cluster_size in hdbscan_cfg["tuning"]["min_cluster_size"]:
        for min_samples in hdbscan_cfg["tuning"]["min_samples"]:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=hdbscan_cfg["tuning"]["metric"],
                cluster_selection_method=hdbscan_cfg["tuning"]["cluster_selection_method"],
                gen_min_span_tree=hdbscan_cfg["tuning"]["gen_min_span_tree"],
            )
            labels = clusterer.fit_predict(embeddings_umap)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = float(np.mean(labels == -1))
            dbcv = getattr(clusterer, "relative_validity_", float("nan"))

            results.append({
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "noise_ratio": round(noise_ratio, 4),
                "dbcv": round(float(dbcv), 6),
            })
            print(f"mcs={min_cluster_size}, ms={min_samples} -> "
                  f"clusters={n_clusters}, noise={noise_ratio:.3f}, dbcv={dbcv:.3f}")

    results_df = pd.DataFrame(results).sort_values("dbcv", ascending=False).reset_index(drop=True)
    print("\nHDBSCAN tuning results (top 5):")
    print(results_df.head().to_string(index=False))
    return results_df


def fit_hdbscan(embeddings_umap: np.ndarray, params: dict) -> np.ndarray:
    print(f"Fitting HDBSCAN with params: min_cluster_size={params.get('min_cluster_size')}, "
          f"min_samples={params.get('min_samples')}")
    clusterer = hdbscan.HDBSCAN(**params)
    labels = clusterer.fit_predict(embeddings_umap)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = float(np.mean(labels == -1))
    print(f"Result: {n_clusters} clusters, {noise_ratio:.1%} noise")
    return labels


def analyze_clusters(labels: np.ndarray) -> dict:
    series = pd.Series(labels)
    sizes = series.value_counts()
    noise_count = sizes.get(-1, 0)
    cluster_sizes = sizes[sizes.index != -1]

    analysis = {
        "n_clusters": len(cluster_sizes),
        "n_noise": int(noise_count),
        "n_total": len(labels),
        "noise_pct": round(noise_count / len(labels) * 100, 2),
        "min_cluster_size": int(cluster_sizes.min()),
        "max_cluster_size": int(cluster_sizes.max()),
        "median_cluster_size": int(cluster_sizes.median()),
    }
    return analysis


def run(config: dict) -> np.ndarray:
    root = get_project_root()
    hdbscan_cfg = config["hdbscan"]
    paths = config["paths"]

    umap_path = root / paths["embeddings_umap"]
    ensure_file_exists(umap_path, "UMAP embeddings")
    embeddings_umap = np.load(umap_path)
    print(f"Loaded UMAP embeddings: {embeddings_umap.shape}")

    results_df = tune_hdbscan(embeddings_umap, config)
    best = results_df.iloc[0]

    final_params = dict(hdbscan_cfg["final"])
    final_params["min_cluster_size"] = int(best["min_cluster_size"])
    final_params["min_samples"] = int(best["min_samples"])

    labels = fit_hdbscan(embeddings_umap, final_params)
    analysis = analyze_clusters(labels)

    print(f"\nFinal clustering:")
    print(f"  Clusters: {analysis['n_clusters']}")
    print(f"  Noise: {analysis['n_noise']:,} ({analysis['noise_pct']}%)")
    print(f"  Size range: {analysis['min_cluster_size']} - {analysis['max_cluster_size']} "
          f"(median: {analysis['median_cluster_size']})")

    np.save(root / paths["cluster_labels"], labels)
    print(f"Cluster labels saved: {root / paths['cluster_labels']}")

    return labels
