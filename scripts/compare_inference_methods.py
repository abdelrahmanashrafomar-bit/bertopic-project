from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, get_project_root
from src.inference.schemas import TopicPrediction


def load_lookup(root: Path, lookup_path: str) -> dict[int, str]:
    import pandas as pd
    full_path = root / lookup_path
    lookup_frame = pd.read_csv(full_path)
    return lookup_frame.set_index("Topic")["Topic_Label"].to_dict()


def load_topic_centroids(root: Path, centroids_path: str) -> dict[int, np.ndarray]:
    import numpy as np
    full_path = root / centroids_path
    centroids = np.load(full_path, allow_pickle=True).item()
    return centroids


def predict_bertopic(text: str, topic_model, lookup: dict[int, str]):
    topics, probs = topic_model.transform([text])
    topic_id = int(topics[0])
    score = float(probs[0][topic_id]) if probs is not None and len(probs) > 0 else 0.0
    label = lookup.get(topic_id, "Unknown Topic")
    return topic_id, label, score


def predict_centroid(
    text: str,
    embedding_model,
    centroids: dict[int, np.ndarray],
    lookup: dict[int, str],
    top_n: int = 3,
):
    ordered_ids = sorted(centroids.keys())
    centroid_matrix = np.array([centroids[tid] for tid in ordered_ids])
    query_vector = embedding_model.encode([text], convert_to_numpy=True)
    similarities = cosine_similarity(query_vector, centroid_matrix)[0]
    ranking = np.argsort(similarities)[::-1][:top_n]
    return [
        (int(ordered_ids[idx]), lookup.get(int(ordered_ids[idx]), "Unknown"), float(similarities[idx]))
        for idx in ranking
    ]


def main():
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from sklearn.metrics.pairwise import cosine_similarity

    root = Path(__file__).resolve().parent.parent
    config_path = root / "config.yaml"

    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    emb_cfg = config["embedding"]
    inf_cfg = config["inference"]
    paths = config["paths"]

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(emb_cfg["model_name"], trust_remote_code=True)

    print("Loading BERTopic model...")
    topic_model = BERTopic.load(root / inf_cfg["model_dir"], embedding_model=embedding_model)

    print("Loading topic centroids...")
    centroids = np.load(root / paths["topic_centroids"], allow_pickle=True).item()

    print("Loading topic lookup...")
    lookup_frame = pd.read_csv(root / inf_cfg["lookup_path"])
    lookup = lookup_frame.set_index("Topic")["Topic_Label"].to_dict()

    test_inputs = [
        "Someone stole my identity and opened a credit card in my name",
        "I demand validation of this debt under the FDCPA",
        "My mortgage escrow payment was not applied correctly",
        "There are errors on my credit report that need to be fixed",
        "I was charged an annual fee I did not agree to",
        "A debt collector keeps calling me at work after I told them to stop",
        "My student loan servicer won't process my forbearance request",
        "I found unauthorized hard inquiries on my credit report",
        "My bank account was charged without my authorization",
        "The credit bureau won't investigate my dispute",
        "I was a victim of identity theft and need accounts blocked",
        "My car loan was reported as repossessed but I'm current on payments",
    ]

    results = []
    for text in test_inputs:
        bt_id, bt_label, bt_prob = predict_bertopic(text, topic_model, lookup)
        centroid_results = predict_centroid(text, embedding_model, centroids, lookup, top_n=3)
        c_id = centroid_results[0][0]
        c_label = centroid_results[0][1]
        c_sim = centroid_results[0][2]
        match = "YES" if bt_id == c_id else "NO"
        results.append({
            "Input": text[:60],
            "BERTopic Topic ID": bt_id,
            "BERTopic Label": bt_label,
            "Probability": round(bt_prob, 4),
            "Centroid Topic ID": c_id,
            "Centroid Label": c_label,
            "Similarity": round(c_sim, 4),
            "Match": match,
        })

    df = pd.DataFrame(results)
    print("\n" + "=" * 140)
    print("INFERENCE METHOD COMPARISON: BERTopic.transform() vs Centroid Similarity")
    print("=" * 140)
    print(df.to_string(index=False))
    print("=" * 140)

    matches = df["Match"].value_counts()
    print(f"\nSummary: {matches.get('YES', 0)}/{len(df)} exact matches (same Topic ID)")

    match_details = df[df["Match"] == "NO"]
    if len(match_details) > 0:
        print("\nMismatch details:")
        for _, row in match_details.iterrows():
            print(f"  Input: {row['Input'][:50]}...")
            print(f"    BERTopic:  Topic {row['BERTopic Topic ID']} - {row['BERTopic Label']} (p={row['Probability']:.4f})")
            print(f"    Centroid:  Topic {row['Centroid Topic ID']} - {row['Centroid Label']} (sim={row['Similarity']:.4f})")

    print("\n" + "=" * 140)
    print("RECOMMENDATION")
    print("=" * 140)

    match_rate = matches.get("YES", 0) / len(df) * 100
    if match_rate >= 80:
        print(f"High agreement ({match_rate:.0f}% match). BERTopic.transform() should be primary, centroid similarity as fallback.")
    elif match_rate >= 50:
        print(f"Moderate agreement ({match_rate:.0f}% match). Centroid similarity should remain primary. Investigate mismatches.")
    else:
        print(f"Low agreement ({match_rate:.0f}% match). Centroid similarity should remain primary. BERTopic.transform() may need investigation.")


if __name__ == "__main__":
    main()
