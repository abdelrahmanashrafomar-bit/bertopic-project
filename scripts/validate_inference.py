from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, get_project_root
from src.inference.predict import predict, load_lookup, load_topic_centroids
from src.inference.schemas import TopicPrediction
from src.preprocessing.clean import clean_cfpb_text
from src.validators import ensure_file_exists


def run_inference_validation(config: dict, sample_size: int = 100, seed: int = 42) -> dict:
    root = get_project_root()
    paths = config["paths"]
    inf_cfg = config["inference"]
    text_column = config["preprocessing"]["text_column"]

    labeled_path = root / paths["final_labeled_topics"]
    if not labeled_path.exists():
        bertopic_ready = root / paths["bertopic_ready"]
        ensure_file_exists(bertopic_ready, "BERTopic-ready CSV")
        df = pd.read_csv(bertopic_ready)
        labels_path = root / paths["cluster_labels"]
        ensure_file_exists(labels_path, "Cluster labels")
        df["Topic"] = np.load(labels_path)
    else:
        df = pd.read_csv(labeled_path)

    print(f"Loaded dataset: {len(df):,} rows")
    print(f"Using text column: '{text_column}'")

    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(df), size=min(sample_size, len(df)), replace=False)
    sample_df = df.iloc[sample_indices]

    correct = 0
    total = len(sample_df)
    results = []

    for idx, row in sample_df.iterrows():
        raw_text = str(row[text_column])
        true_topic = int(row["Topic"])

        predictions = predict(raw_text, config, method=inf_cfg.get("method", "centroid_similarity"))
        predicted_topic = predictions[0].topic_id
        is_match = predicted_topic == true_topic

        if is_match:
            correct += 1

        results.append({
            "index": int(idx),
            "text_preview": clean_cfpb_text(raw_text)[:80],
            "true_topic": true_topic,
            "predicted_topic": predicted_topic,
            "score": predictions[0].score,
            "label": predictions[0].label,
            "match": is_match,
        })

    accuracy = correct / total * 100
    print(f"\n{'=' * 80}")
    print(f"INFERENCE VALIDATION RESULTS")
    print(f"{'=' * 80}")
    print(f"Samples tested: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'=' * 80}")

    results_df = pd.DataFrame(results)
    mismatches = results_df[~results_df["match"]]
    if len(mismatches) > 0:
        print(f"\nMismatches ({len(mismatches)}):")
        for _, row in mismatches.iterrows():
            print(f"  [{row['text_preview']}]")
            print(f"    True: Topic {row['true_topic']} | Predicted: Topic {row['predicted_topic']} ({row['label']}, score={row['score']:.4f})")

    summary = {
        "sample_size": total,
        "correct": correct,
        "accuracy_pct": round(accuracy, 2),
        "results": results_df,
    }
    return summary


def main():
    config = load_config()
    summary = run_inference_validation(config, sample_size=100, seed=42)
    return summary


if __name__ == "__main__":
    main()
