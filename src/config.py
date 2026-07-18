from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    return PROJECT_ROOT


class RunMetadata:
    def __init__(self, config: dict):
        self.config = config
        self.timestamp: str = ""
        self.steps_completed: list[str] = []
        self.n_documents: int = 0
        self.n_topics: int = 0
        self.outlier_pct: float = 0.0
        self.embedding_model: str = ""
        self.umap_params: dict = {}
        self.hdbscan_params: dict = {}

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
            "steps_completed": self.steps_completed,
            "n_documents": self.n_documents,
            "n_topics": self.n_topics,
            "outlier_pct": round(self.outlier_pct, 2),
            "embedding_model": self.embedding_model,
            "umap_params": self.umap_params,
            "hdbscan_params": self.hdbscan_params,
            "config_snapshot": self.config,
        }

    def save(self, path: Path | str | None = None) -> None:
        if path is None:
            path = get_project_root() / "outputs" / "run_metadata.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
