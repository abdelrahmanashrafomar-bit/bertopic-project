from __future__ import annotations

from pathlib import Path

import pandas as pd
from bertopic import BERTopic

from src.config import get_project_root
from src.validators import ensure_file_exists


def visualize_topics(
    topic_model: BERTopic,
    save_path: Path,
    custom_labels: bool = True,
) -> None:
    print("Generating intertopic distance map...")
    fig = topic_model.visualize_topics(custom_labels=custom_labels)
    fig.write_html(save_path)
    print(f"Saved: {save_path}")


def visualize_hierarchy(
    topic_model: BERTopic,
    save_path: Path,
    custom_labels: bool = True,
) -> None:
    print("Generating topic hierarchy dendrogram...")
    fig = topic_model.visualize_hierarchy(custom_labels=custom_labels)
    fig.write_html(save_path)
    print(f"Saved: {save_path}")


def visualize_topics_over_time(
    topic_model: BERTopic,
    documents: list[str],
    timestamps: list,
    save_path: Path,
    nr_bins: int = 20,
    target_topics: list[int] | None = None,
    custom_labels: bool = True,
) -> None:
    if nr_bins > 100 and len(set(timestamps)) > 100:
        print(f"WARNING: {len(set(timestamps))} unique timestamps. "
              f"Using nr_bins={nr_bins} may be slow.")
        print("Skipping topics_over_time visualization. "
              "Set nr_bins <= 100 or reduce timestamp cardinality.")

    print("Calculating topic distributions over time...")
    topics_over_time = topic_model.topics_over_time(
        docs=documents,
        timestamps=timestamps,
        nr_bins=nr_bins,
    )

    target = target_topics if target_topics else None
    fig = topic_model.visualize_topics_over_time(
        topics_over_time,
        topics=target,
        custom_labels=custom_labels,
    )
    fig.write_html(save_path)
    print(f"Saved: {save_path}")


def run(config: dict) -> dict[str, Path]:
    root = get_project_root()
    paths = config["paths"]
    tot_config = config["topics_over_time"]
    bertopic_cfg = config["bertopic"]

    model_path = root / paths["bertopic_model_dir"]
    ensure_file_exists(model_path / "config.json", "BERTopic model")

    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(
        config["embedding"]["model_name"],
        trust_remote_code=True,
    )
    topic_model = BERTopic.load(model_path, embedding_model=embedding_model)

    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = {}

    dist_path = output_dir / paths["intertopic_distance_map_html"].name
    visualize_topics(topic_model, dist_path)
    saved["intertopic_distance_map"] = dist_path

    hier_path = output_dir / paths["topic_hierarchy_html"].name
    visualize_hierarchy(topic_model, hier_path)
    saved["topic_hierarchy"] = hier_path

    df = pd.read_csv(root / paths["final_labeled_topics"])

    for col in bertopic_cfg["date_column_candidates"]:
        if col in df.columns:
            date_column = col
            break
    else:
        print("No date column found, skipping topics_over_time.")
        return saved

    documents = df[bertopic_cfg["text_column_candidates"][0]].astype(str).tolist()
    df[date_column] = pd.to_datetime(df[date_column])
    timestamps = df[date_column].tolist()

    tot_path = output_dir / paths["topics_evolution_over_time_html"].name
    visualize_topics_over_time(
        topic_model, documents, timestamps, tot_path,
        nr_bins=tot_config["nr_bins"],
        target_topics=tot_config["target_topics"],
    )
    saved["topics_over_time"] = tot_path

    return saved
