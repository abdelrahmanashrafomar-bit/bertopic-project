from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_project_root
from src.validators import ensure_file_exists


def check_artifacts(config: dict) -> dict:
    root = get_project_root()
    paths = config["paths"]
    results = {"passed": [], "failed": [], "warnings": []}

    required = [
        ("embeddings.npy", paths["embeddings"]),
        ("cluster_labels.npy", paths["cluster_labels"]),
        ("topic_centroids.npy", paths["topic_centroids"]),
        ("topic_lookup.csv", paths["topic_lookup"]),
        ("labels.csv", paths["labels"]),
        ("BERTopic model", paths["bertopic_model_dir"]),
    ]

    for label, rel_path in required:
        full = root / rel_path
        if full.exists():
            results["passed"].append(f"{label} exists: {full}")
        else:
            results["failed"].append(f"{label} MISSING: {full}")

    return results


def verify_topic_mapping(config: dict) -> dict:
    root = get_project_root()
    paths = config["paths"]
    results = {"passed": [], "failed": [], "warnings": []}

    lookup_path = root / paths["topic_lookup"]
    labels_path = root / paths["labels"]

    if not lookup_path.exists():
        results["failed"].append(f"topic_lookup.csv not found: {lookup_path}")
        return results
    if not labels_path.exists():
        results["failed"].append(f"labels.csv not found: {labels_path}")
        return results

    lookup = pd.read_csv(lookup_path)
    labels = pd.read_csv(labels_path)

    if "Topic" not in lookup.columns or "Topic_Label" not in lookup.columns:
        results["failed"].append("topic_lookup.csv missing required columns: Topic, Topic_Label")
    else:
        results["passed"].append(f"topic_lookup.csv: {len(lookup)} topics, columns OK")

    if "Topic" not in labels.columns or "Topic_Label" not in labels.columns:
        results["failed"].append("labels.csv missing required columns: Topic, Topic_Label")
    else:
        results["passed"].append(f"labels.csv: {len(labels)} topics, columns OK")

    lookup_set = set(lookup["Topic"])
    labels_set = set(labels["Topic"])
    only_in_lookup = lookup_set - labels_set
    only_in_labels = labels_set - lookup_set

    if only_in_lookup:
        results["warnings"].append(f"Topics in lookup but not in labels: {sorted(only_in_lookup)}")
    if only_in_labels:
        results["warnings"].append(f"Topics in labels but not in lookup: {sorted(only_in_labels)}")
    if not only_in_lookup and not only_in_labels:
        results["passed"].append("topic_lookup.csv and labels.csv have identical Topic sets")

    merged = lookup.merge(labels, on="Topic", suffixes=("_lookup", "_labels"))
    mismatches = merged[merged["Topic_Label_lookup"] != merged["Topic_Label_labels"]]
    if len(mismatches) > 0:
        results["failed"].append(f"{len(mismatches)} label mismatches between lookup and labels CSVs")
        for _, row in mismatches.iterrows():
            results["warnings"].append(
                f"  Topic {row['Topic']}: lookup='{row['Topic_Label_lookup']}' vs labels='{row['Topic_Label_labels']}'"
            )
    else:
        results["passed"].append("topic_lookup.csv and labels.csv labels are identical")

    return results


def verify_topic_centroids(config: dict) -> dict:
    root = get_project_root()
    paths = config["paths"]
    results = {"passed": [], "failed": [], "warnings": []}

    centroids_path = root / paths["topic_centroids"]
    if not centroids_path.exists():
        results["failed"].append(f"topic_centroids.npy not found: {centroids_path}")
        return results

    try:
        centroids = np.load(centroids_path, allow_pickle=True).item()
        if not isinstance(centroids, dict):
            results["failed"].append("topic_centroids.npy is not a dict")
            return results

        n_topics = len(centroids)
        topic_ids = sorted(centroids.keys())
        dims = {v.shape for v in centroids.values()}
        results["passed"].append(f"topic_centroids.npy: {n_topics} topics, dimensions: {dims}")

        if len(dims) > 1:
            results["warnings"].append(f"Centroid dimensions vary: {dims}")

        lookup_path = root / paths["topic_lookup"]
        if lookup_path.exists():
            lookup = pd.read_csv(lookup_path)
            lookup_ids = set(lookup["Topic"])
            centroid_ids = set(centroids.keys())
            missing_in_lookup = centroid_ids - lookup_ids
            missing_in_centroids = lookup_ids - centroid_ids
            if missing_in_lookup:
                results["warnings"].append(f"Topics in centroids but not in lookup: {sorted(missing_in_lookup)}")
            if missing_in_centroids:
                results["warnings"].append(f"Topics in lookup but not in centroids: {sorted(missing_in_centroids)}")
            if not missing_in_lookup and not missing_in_centroids:
                results["passed"].append("All centroid topic IDs have corresponding lookup entries")

    except Exception as e:
        results["failed"].append(f"Error loading topic_centroids.npy: {e}")

    return results


def verify_cluster_labels(config: dict) -> dict:
    root = get_project_root()
    paths = config["paths"]
    results = {"passed": [], "failed": [], "warnings": []}

    labels_path = root / paths["cluster_labels"]
    if not labels_path.exists():
        results["failed"].append(f"cluster_labels.npy not found: {labels_path}")
        return results

    try:
        labels = np.load(labels_path)
        unique, counts = np.unique(labels, return_counts=True)
        n_clusters = len(unique) - (1 if -1 in unique else 0)
        noise_count = int(counts[unique == -1].item()) if -1 in unique else 0
        results["passed"].append(
            f"cluster_labels.npy: {len(labels)} labels, {n_clusters} clusters, "
            f"{noise_count} noise ({noise_count/len(labels)*100:.1f}%)"
        )

        centroids_path = root / paths["topic_centroids"]
        if centroids_path.exists():
            centroids = np.load(centroids_path, allow_pickle=True).item()
            centroid_ids = set(centroids.keys())
            label_ids = set(unique) - {-1}
            missing_in_centroids = label_ids - centroid_ids
            missing_in_labels = centroid_ids - label_ids
            if missing_in_centroids:
                results["warnings"].append(f"Cluster labels have topics not in centroids: {sorted(missing_in_centroids)}")
            if missing_in_labels:
                results["warnings"].append(f"Centroids have topics not in cluster labels: {sorted(missing_in_labels)}")
            if not missing_in_centroids and not missing_in_labels:
                results["passed"].append("Cluster labels and centroids topic IDs match")

    except Exception as e:
        results["failed"].append(f"Error loading cluster_labels.npy: {e}")

    return results


def verify_embeddings(config: dict) -> dict:
    root = get_project_root()
    paths = config["paths"]
    results = {"passed": [], "failed": [], "warnings": []}

    emb_path = root / paths["embeddings"]
    if not emb_path.exists():
        results["failed"].append(f"embeddings.npy not found: {emb_path}")
        return results

    try:
        embeddings = np.load(emb_path, mmap_mode="r")
        results["passed"].append(f"embeddings.npy: shape={embeddings.shape}, dtype={embeddings.dtype}")
    except Exception as e:
        results["failed"].append(f"Error loading embeddings.npy: {e}")

    return results


def verify_bertopic_model(config: dict) -> dict:
    root = get_project_root()
    paths = config["paths"]
    results = {"passed": [], "failed": [], "warnings": []}

    model_dir = root / paths["bertopic_model_dir"]
    if not model_dir.exists():
        results["failed"].append(f"BERTopic model directory not found: {model_dir}")
        return results

    required_files = ["config.json", "topics.json", "topic_embeddings.safetensors"]
    for fname in required_files:
        fpath = model_dir / fname
        if fpath.exists():
            results["passed"].append(f"BERTopic model file exists: {fname}")
        else:
            results["failed"].append(f"BERTopic model file MISSING: {fname}")

    return results


def verify_data_consistency(config: dict) -> dict:
    root = get_project_root()
    paths = config["paths"]
    results = {"passed": [], "failed": [], "warnings": []}

    emb_path = root / paths["embeddings"]
    labels_path = root / paths["cluster_labels"]

    if emb_path.exists() and labels_path.exists():
        try:
            emb = np.load(emb_path, mmap_mode="r")
            labels = np.load(labels_path)
            if len(emb) == len(labels):
                results["passed"].append(f"Data consistent: embeddings ({len(emb)}) == cluster labels ({len(labels)})")
            else:
                results["failed"].append(
                    f"Data MISMATCH: embeddings ({len(emb)}) != cluster labels ({len(labels)})"
                )
        except Exception as e:
            results["failed"].append(f"Error checking data consistency: {e}")

    return results


def run_inference_comparison(config: dict, output_path: Path | None = None) -> dict:
    results = {"passed": [], "failed": [], "warnings": []}

    if output_path is None:
        output_path = get_project_root() / "outputs" / "inference_comparison_results.csv"

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

    try:
        from sentence_transformers import SentenceTransformer
        from bertopic import BERTopic
        from sklearn.metrics.pairwise import cosine_similarity

        root = get_project_root()
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

        rows = []
        for text in test_inputs:
            bt_topics, bt_probs = topic_model.transform([text])
            bt_id = int(bt_topics[0])
            bt_prob = float(bt_probs[0][bt_id]) if bt_probs is not None and len(bt_probs) > 0 else 0.0
            bt_label = lookup.get(bt_id, "Unknown")

            ordered_ids = sorted(centroids.keys())
            centroid_matrix = np.array([centroids[tid] for tid in ordered_ids])
            query_vec = embedding_model.encode([text], convert_to_numpy=True)
            sims = cosine_similarity(query_vec, centroid_matrix)[0]
            top_idx = np.argsort(sims)[::-1][0]
            c_id = int(ordered_ids[top_idx])
            c_sim = float(sims[top_idx])
            c_label = lookup.get(c_id, "Unknown")

            match = "YES" if bt_id == c_id else "NO"

            rows.append({
                "Input": text[:80],
                "BERTopic_Topic_ID": bt_id,
                "BERTopic_Label": bt_label,
                "BERTopic_Probability": round(bt_prob, 4),
                "Centroid_Topic_ID": c_id,
                "Centroid_Label": c_label,
                "Centroid_Similarity": round(c_sim, 4),
                "Match": match,
            })

        df = pd.DataFrame(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        results["passed"].append(f"Inference comparison saved: {output_path}")

        match_rate = (df["Match"] == "YES").mean() * 100
        results["passed"].append(f"Match rate: {match_rate:.0f}% ({int(df['Match'].value_counts().get('YES', 0))}/{len(df)})")

        print("\n" + "=" * 140)
        print(df.to_string(index=False))
        print("=" * 140)

    except ImportError as e:
        results["warnings"].append(f"Cannot run inference comparison (missing deps on this machine): {e}")
        results["warnings"].append("Run on Lightning AI / cloud GPU with full dependencies installed")
    except Exception as e:
        results["failed"].append(f"Inference comparison error: {e}")

    return results


def run(config: dict) -> dict:
    print("=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    all_results = {}

    print("\n[1/6] Checking artifacts...")
    all_results["artifacts"] = check_artifacts(config)

    print("\n[2/6] Verifying topic mapping...")
    all_results["topic_mapping"] = verify_topic_mapping(config)

    print("\n[3/6] Verifying topic centroids...")
    all_results["topic_centroids"] = verify_topic_centroids(config)

    print("\n[4/6] Verifying cluster labels...")
    all_results["cluster_labels"] = verify_cluster_labels(config)

    print("\n[5/6] Verifying data consistency...")
    all_results["data_consistency"] = verify_data_consistency(config)

    print("\n[6/6] Inference comparison (requires GPU) — skipped. Run scripts/compare_inference_methods.py on Lightning AI.")
    all_results["inference_comparison"] = {
        "passed": [],
        "failed": [],
        "warnings": ["Inference comparison requires GPU. Run scripts/compare_inference_methods.py on Lightning AI."],
    }

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    total_passed = 0
    total_failed = 0
    total_warnings = 0

    for section, section_results in all_results.items():
        n_p = len(section_results.get("passed", []))
        n_f = len(section_results.get("failed", []))
        n_w = len(section_results.get("warnings", []))
        total_passed += n_p
        total_failed += n_f
        total_warnings += n_w
        status = "PASS" if n_f == 0 else "FAIL"
        print(f"  [{status}] {section}: {n_p} passed, {n_f} failed, {n_w} warnings")
        for msg in section_results.get("failed", []):
            print(f"    FAIL: {msg}")
        for msg in section_results.get("warnings", []):
            print(f"    WARN: {msg}")

    print(f"\nTotal: {total_passed} passed, {total_failed} failed, {total_warnings} warnings")

    if total_failed > 0:
        print("\nBUGS FOUND — review failed checks above before proceeding.")

    return all_results
