"""Terminal demo script for BERTopic CFPB Complaint Analyzer.

Records beautifully for LinkedIn screen capture.
Usage:  python demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
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


# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
CENTROIDS_PATH = PROJECT_ROOT / "artifacts" / "topic_centroids.npy"
LOOKUP_PATH = PROJECT_ROOT / "artifacts" / "topic_lookup.csv"
TOPICS_JSON_PATH = PROJECT_ROOT / "artifacts" / "model" / "topics.json"

console = Console()


# ---------------------------------------------------------------------------
# Data loading (lazy — happens once at first prediction)
# ---------------------------------------------------------------------------
_embedding_model: SentenceTransformer | None = None
_centroids: dict[int, np.ndarray] | None = None
_lookup: dict[int, str] | None = None
_topic_keywords: dict[str, list[str]] | None = None
_topic_sizes: dict[str, int] | None = None


def _load_artifacts() -> None:
    """Load all artifacts once and cache them in module globals."""
    global _embedding_model, _centroids, _lookup, _topic_keywords, _topic_sizes

    if _embedding_model is not None:
        return  # already loaded

    with console.status("[bold yellow]Loading artifacts ...", spinner="dots"):
        # Config
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        model_name = config["embedding"]["model_name"]

        # Embedding model
        _embedding_model = SentenceTransformer(model_name, trust_remote_code=True)

        # Centroids
        _centroids = np.load(CENTROIDS_PATH, allow_pickle=True).item()

        # Topic lookup
        import pandas as pd
        lookup_df = pd.read_csv(LOOKUP_PATH)
        _lookup = lookup_df.set_index("Topic")["Topic_Label"].to_dict()

        # Topic keywords + sizes from BERTopic saved JSON
        with open(TOPICS_JSON_PATH, encoding="utf-8") as f:
            data = json.load(f)
        reps = data["topic_representations"]
        _topic_keywords = {
            tid: [kw for kw, _ in kws[:8]]
            for tid, kws in reps.items()
        }
        _topic_sizes = {str(k): v for k, v in data.get("topic_sizes", {}).items()}


def predict(text: str) -> dict:
    """Predict the best topic for a complaint text.

    Returns a dict with topic_id, label, score, keywords, and size.
    """
    _load_artifacts()

    # Encode query
    vec = _embedding_model.encode([text], convert_to_numpy=True)

    # Cosine similarity against all centroids
    ordered_ids = sorted(_centroids.keys())
    centroid_mat = np.array([_centroids[tid] for tid in ordered_ids])
    sims = cosine_similarity(vec, centroid_mat)[0]
    best_idx = int(np.argmax(sims))
    topic_id = ordered_ids[best_idx]
    score = float(sims[best_idx])

    str_id = str(topic_id)
    return {
        "topic_id": topic_id,
        "label": _lookup.get(topic_id, "Unknown Topic"),
        "score": score,
        "keywords": _topic_keywords.get(str_id, []),
        "size": _topic_sizes.get(str_id, 0),
    }


# ---------------------------------------------------------------------------
# Welcome banner
# ---------------------------------------------------------------------------
def show_banner() -> None:
    banner = Text()
    banner.append("╔══════════════════════════════════════════════════════╗\n")
    banner.append("║", style="bold cyan")
    banner.append("        CFPB Complaint Topic Analyzer       ", style="bold white")
    banner.append("║", style="bold cyan")
    banner.append("\n║", style="bold cyan")
    banner.append("    BERTopic • centroid similarity • F2LLM-1.7B    ", style="dim white")
    banner.append("║", style="bold cyan")
    banner.append("\n╚══════════════════════════════════════════════════════╝", style="bold cyan")
    console.print(Panel(banner, style="cyan", padding=(1, 2)))
    console.print()


# ---------------------------------------------------------------------------
# Results panel
# ---------------------------------------------------------------------------
def show_results(text: str, result: dict) -> None:
    topic_id = result["topic_id"]
    label = result["label"]
    score = result["score"]
    keywords = result["keywords"]
    size = result["size"]

    # Colour confidence bar
    pct = score * 100
    bar_len = 30
    filled = int(bar_len * score)
    bar = "█" * filled + "░" * (bar_len - filled)
    if score >= 0.7:
        bar_color = "green"
    elif score >= 0.4:
        bar_color = "yellow"
    else:
        bar_color = "red"

    # Complaint panel
    complaint_panel = Panel(
        Text(text, style="white", no_wrap=False),
        title="[bold]Complaint[/]",
        border_style="dim white",
        padding=(1, 2),
    )

    # Topic info table
    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold white", justify="right")
    info.add_column(style="white")

    info.add_row("Topic ID", f"[bold cyan]{topic_id}[/]")
    info.add_row("Label", f"[bold green]{label}[/]")
    info.add_row("Confidence", f"[{bar_color}]{bar}  {pct:.1f}%[/]")
    info.add_row(
        "Keywords",
        "  ".join(f"[magenta]{kw}[/]" for kw in keywords[:6]),
    )
    info.add_row("Topic size", f"[bold]{size:,}[/] documents" if size else "[dim]N/A[/]")

    results_panel = Panel(
        info,
        title="[bold]Prediction[/]",
        border_style="green",
        padding=(1, 2),
    )

    layout = Layout()
    layout.split_column(
        Layout(complaint_panel, size=5),
        Layout(results_panel, size=8),
    )

    console.print(layout)
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
            result = predict(text)

        show_results(text, result)

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
