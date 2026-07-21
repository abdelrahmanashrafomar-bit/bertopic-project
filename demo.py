"""Terminal demo script for BERTopic CFPB Complaint Analyzer.

Records beautifully for LinkedIn screen capture.
Usage:  python demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from bertopic import BERTopic
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.preprocessing.clean import clean_cfpb_text


# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
CENTROIDS_PATH = PROJECT_ROOT / "artifacts" / "topic_centroids.npy"
LOOKUP_PATH = PROJECT_ROOT / "artifacts" / "topic_lookup.csv"
TOPICS_JSON_PATH = PROJECT_ROOT / "artifacts" / "model" / "topics.json"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"

console = Console()


# ---------------------------------------------------------------------------
# Data loading (lazy — happens once at first prediction)
# ---------------------------------------------------------------------------
_embedding_model: SentenceTransformer | None = None
_topic_model: BERTopic | None = None
_centroids: dict[int, np.ndarray] | None = None
_lookup: dict[int, str] | None = None
_topic_keywords: dict[str, list[str]] | None = None
_topic_sizes: dict[str, int] | None = None
_emb_cfg: dict | None = None


def _load_artifacts() -> None:
    """Load all artifacts once and cache them in module globals."""
    global _embedding_model, _topic_model, _centroids, _lookup, _topic_keywords, _topic_sizes, _emb_cfg

    if _embedding_model is not None:
        return  # already loaded

    with console.status("[bold yellow]Loading artifacts ...", spinner="dots"):
        # Config
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        _emb_cfg = config["embedding"]

        # Embedding model (same dtype/device_map as training)
        dtype = getattr(torch, _emb_cfg.get("torch_dtype", "float16"), torch.float16)
        _embedding_model = SentenceTransformer(
            _emb_cfg["model_name"],
            model_kwargs={"torch_dtype": dtype, "device_map": _emb_cfg.get("device_map", "auto")},
            trust_remote_code=True,
        )

        # Centroids
        _centroids = np.load(CENTROIDS_PATH, allow_pickle=True).item()

        # Topic lookup
        import pandas as pd
        lookup_df = pd.read_csv(LOOKUP_PATH)
        _lookup = lookup_df.set_index("Topic")["Topic_Label"].to_dict()

        # BERTopic model (for c-TF-IDF matching)
        _topic_model = BERTopic.load(MODEL_DIR, embedding_model=_embedding_model)

        # Topic keywords + sizes from BERTopic saved JSON
        with open(TOPICS_JSON_PATH, encoding="utf-8") as f:
            data = json.load(f)
        reps = data["topic_representations"]
        _topic_keywords = {
            tid: [kw for kw, _ in kws[:8]]
            for tid, kws in reps.items()
        }
        _topic_sizes = {str(k): v for k, v in data.get("topic_sizes", {}).items()}


def _predict_centroid(text: str) -> dict:
    """Predict using centroid cosine similarity."""
    vec = _embedding_model.encode([text], convert_to_numpy=True)
    ordered_ids = sorted(_centroids.keys())
    centroid_mat = np.array([_centroids[tid] for tid in ordered_ids])
    sims = cosine_similarity(vec, centroid_mat)[0]
    best_idx = int(np.argmax(sims))
    topic_id = int(ordered_ids[best_idx])
    score = float(sims[best_idx])
    str_id = str(topic_id)
    return {
        "method": "Centroid Similarity",
        "topic_id": topic_id,
        "label": _lookup.get(topic_id, "Unknown Topic"),
        "score": score,
        "keywords": _topic_keywords.get(str_id, []),
        "size": _topic_sizes.get(str_id, 0),
    }


def _predict_bertopic(text: str) -> dict:
    """Predict using BERTopic c-TF-IDF transform."""
    topics, probs = _topic_model.transform([text])
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
    str_id = str(topic_id)
    return {
        "method": "BERTopic c-TF-IDF",
        "topic_id": topic_id,
        "label": _lookup.get(topic_id, "Unknown Topic"),
        "score": score,
        "keywords": _topic_keywords.get(str_id, []),
        "size": _topic_sizes.get(str_id, 0),
    }


def predict(text: str) -> list[dict]:
    """Predict using both methods and return results."""
    _load_artifacts()
    text = clean_cfpb_text(text)
    return [_predict_centroid(text), _predict_bertopic(text)]


# ---------------------------------------------------------------------------
# Welcome banner
# ---------------------------------------------------------------------------
def show_banner() -> None:
    banner = Text()
    banner.append("╔══════════════════════════════════════════════════════════════════╗\n")
    banner.append("║", style="bold cyan")
    banner.append("             CFPB Complaint Topic Analyzer            ", style="bold white")
    banner.append("║", style="bold cyan")
    banner.append("\n║", style="bold cyan")
    banner.append("    Centroid Similarity  vs  BERTopic c-TF-IDF  •  F2LLM-1.7B    ", style="dim white")
    banner.append("║", style="bold cyan")
    banner.append("\n╚══════════════════════════════════════════════════════════════════╝", style="bold cyan")
    console.print(Panel(banner, style="cyan", padding=(1, 2)))
    console.print()


# ---------------------------------------------------------------------------
# Results panel
# ---------------------------------------------------------------------------
def show_results(text: str, results: list[dict]) -> None:
    # Complaint panel
    complaint_panel = Panel(
        Text(text, style="white", no_wrap=False),
        title="[bold]Complaint[/]",
        border_style="dim white",
        padding=(1, 2),
    )

    # Build a side-by-side grid for both methods
    compare = Table.grid(padding=(1, 3))
    compare.add_column(justify="center")
    compare.add_column(justify="center")

    panels = []
    for result in results:
        topic_id = result["topic_id"]
        label = result["label"]
        score = result["score"]
        keywords = result["keywords"]
        size = result["size"]

        pct = score * 100
        bar_len = 20
        filled = int(bar_len * score)
        bar = "█" * filled + "░" * (bar_len - filled)
        if score >= 0.7:
            bar_color = "green"
        elif score >= 0.4:
            bar_color = "yellow"
        else:
            bar_color = "red"

        info = Table.grid(padding=(0, 2))
        info.add_column(style="bold white", justify="right")
        info.add_column(style="white")

        info.add_row("Topic ID", f"[bold cyan]{topic_id}[/]")
        info.add_row("Label", f"[bold green]{label}[/]")
        info.add_row("Confidence", f"[{bar_color}]{bar}  {pct:.1f}%[/]")
        info.add_row(
            "Keywords",
            "  ".join(f"[magenta]{kw}[/]" for kw in keywords[:5]),
        )
        info.add_row("Topic size", f"[bold]{size:,}[/] docs" if size else "[dim]N/A[/]")

        panels.append(Panel(info, title=f"[bold]{result['method']}[/]", border_style="green", padding=(1, 2)))

    compare.add_row(*panels)

    layout = Layout()
    layout.split_column(
        Layout(complaint_panel, size=5),
        Layout(compare, size=10),
    )

    console.print(layout)
    console.print()

    # Agreement indicator
    if results[0]["topic_id"] == results[1]["topic_id"]:
        console.print("  [bold green]✓ Both methods agree[/]")
    else:
        console.print("  [bold yellow]⚠ Methods disagree — centroid may need review[/]")
    console.print()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    show_banner()

    # Pre-load in background the first time
    with console.status("[bold yellow]Loading artifacts ...", spinner="dots"):
        _load_artifacts()
    console.print("[dim]✓ Model and artifacts loaded[/]\n")

    while True:
        text = Prompt.ask("[bold cyan]Enter a consumer complaint[/]").strip()

        if not text:
            console.print("[yellow]Empty input. Try again or press Ctrl+C to exit.[/]\n")
            continue

        # Spinner during prediction
        with console.status("[bold yellow]Analyzing complaint...", spinner="dots"):
            results = predict(text)

        show_results(text, results)

        again = Prompt.ask(
            "[dim]Analyze another?[/]",
            choices=["y", "n"],
            default="y",
        )
        if again.lower() != "y":
            break

    console.print("\n[bold green]✓ Demo complete.[/] Thanks for watching!\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[bold yellow]Interrupted. Exiting.[/]")
        sys.exit(0)
