from __future__ import annotations

import argparse

from src.config import RunMetadata, load_config


def run_preprocessing(config: dict) -> None:
    print("Running preprocessing...")
    from src.preprocessing.preprocess import run
    run(config)


def run_embedding(config: dict) -> None:
    print("Generating embeddings...")
    from src.embedding.generate import run
    run(config)


def run_evaluate_embedding(config: dict) -> None:
    print("Evaluating embeddings...")
    from src.embedding.evaluate import run
    run(config)


def run_umap(config: dict) -> None:
    print("Running UMAP reduction...")
    from src.clustering.reduce import run
    run(config)


def run_clustering(config: dict) -> None:
    print("Running HDBSCAN clustering...")
    from src.clustering.cluster import run
    run(config)


def run_compute_centroids(config: dict) -> None:
    print("Computing topic centroids...")
    from src.topic_modeling.fit import compute_topic_centroids, save_topic_centroids
    from src.topic_modeling.fit import load_data
    from src.config import get_project_root

    root = get_project_root()
    df, documents, embeddings, labels, text_column = load_data(config)
    centroids = compute_topic_centroids(embeddings, labels)
    save_topic_centroids(centroids, root / config["paths"]["topic_centroids"])


def run_topic_modeling(config: dict) -> None:
    print("Running BERTopic fitting...")
    from src.topic_modeling.fit import run
    run(config)


def run_labeling(config: dict) -> None:
    print("Running Gemini topic labeling...")
    from src.topic_modeling.label import run
    run(config)


def run_visualize(config: dict) -> None:
    print("Running visualizations...")
    from src.topic_modeling.visualize import run
    run(config)


def run_validation(config: dict) -> None:
    print("Running validation...")
    from src.validation.validate import run
    run(config)


def run_inference(config: dict) -> None:
    print("Running inference...")
    from src.inference.predict import run
    run(config)


STEP_MAP = {
    "pring": run_preprocessing,
    "embedding": run_embedding,
    "evaluate": run_evaluate_embedding,
    "umap": run_umap,
    "clustering": run_clustering,
    "compute_centroids": run_compute_centroids,
    "topic_modeling": run_topic_modeling,
    "labeling": run_labeling,
    "visualize": run_visualize,
    "inference": run_inference,
    "validate": run_validation,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BERTopic CFPB Complaint Analysis Pipeline"
    )
    parser.add_argument(
        "--step",
        choices=list(STEP_MAP.keys()) + ["all"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: config.yaml in project root)",
    )
    return parser.parse_args(argv)


def load_config_from_args(custom_path: str | None = None) -> dict:
    if custom_path:
        import yaml
        from pathlib import Path
        with open(Path(custom_path), "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return load_config()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config_from_args(args.config)

    metadata = RunMetadata(config)

    if args.step == "all":
        steps = list(STEP_MAP.keys())
    else:
        steps = [args.step]

    for step_name in steps:
        func = STEP_MAP[step_name]
        func(config)
        metadata.steps_completed.append(step_name)

    metadata.save()
    print(f"\nPipeline complete. Steps run: {', '.join(metadata.steps_completed)}")


if __name__ == "__main__":
    main()
