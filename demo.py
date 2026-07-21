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
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.preprocessing.clean import clean_cfpb_text


# ---------------------------------------------------------------------------
# Project root & config path — artifact paths are read from config.yaml
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH  = PROJECT_ROOT / "config.yaml"

console = Console()


# ---------------------------------------------------------------------------
# Module-level artifact cache (lazy — loaded exactly once)
# ---------------------------------------------------------------------------
_loaded: bool = False                                   # explicit load flag
_embedding_model: SentenceTransformer | None = None
_topic_model: BERTopic | None = None
_centroids: dict[int, np.ndarray] | None = None
_lookup: dict[int, str] | None = None
_topic_keywords: dict[str, list[str]] | None = None
_topic_sizes: dict[str, int] | None = None
_emb_cfg: dict | None = None
_n_training_docs: int = 0                               # for startup stats


# ---------------------------------------------------------------------------
# Similarity thresholds — calibrated for cosine similarity in this embedding
# space (F2LLM embeddings typically cluster between 0.60 – 0.95)
# ---------------------------------------------------------------------------
_SIM_HIGH      = 0.80   # green
_SIM_MED       = 0.65   # yellow
_SIM_LOW_WARN  = 0.55   # below this → show "Low confidence" warning


def _load_artifacts() -> None:
    """Load all artifacts once and cache in module globals.

    All artifact paths are resolved from config.yaml — the single source
    of truth — instead of being hardcoded in this file.
    Raises a styled Rich error and exits if any file is missing or corrupt.
    """
    global _loaded, _embedding_model, _topic_model, _centroids, _lookup
    global _topic_keywords, _topic_sizes, _emb_cfg, _n_training_docs

    if _loaded:
        return  # guard uses a dedicated flag, not a None-check on one variable

    try:
        # ── Config ─────────────────────────────────────────────────────────
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        _emb_cfg = config["embedding"]
        paths    = config["paths"]

        # Resolve every artifact path from config — consistent with the rest
        # of the codebase; a single path change in config.yaml propagates here
        centroids_path = PROJECT_ROOT / paths["topic_centroids"]
        lookup_path    = PROJECT_ROOT / paths["topic_lookup"]
        model_dir      = PROJECT_ROOT / paths["bertopic_model_dir"]
        topics_json    = Path(model_dir) / "topics.json"

        # ── Embedding model ─────────────────────────────────────────────────
        # Loaded with the exact same settings as training (generate.py):
        # torch_dtype, device_map, trust_remote_code, and explicit CUDA move.
        dtype = getattr(torch, _emb_cfg.get("torch_dtype", "float16"), torch.float16)
        _embedding_model = SentenceTransformer(
            _emb_cfg["model_name"],
            model_kwargs={
                "torch_dtype": dtype,
                "device_map": _emb_cfg.get("device_map", "auto"),
            },
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            _embedding_model = _embedding_model.to("cuda")

        # ── Topic centroids ──────────────────────────────────────────────────
        raw = np.load(centroids_path, allow_pickle=True).item()
        if not isinstance(raw, dict):
            raise TypeError(
                f"topic_centroids.npy must contain a dict, got {type(raw).__name__}"
            )
        _centroids = raw

        # ── Topic label lookup ────────────────────────────────────────────────
        import pandas as pd
        lookup_df = pd.read_csv(lookup_path)
        _lookup = lookup_df.set_index("Topic")["Topic_Label"].to_dict()

        # ── BERTopic model (c-TF-IDF transform path) ──────────────────────────
        # The same _embedding_model object is passed so BERTopic.transform()
        # uses F2LLM-1.7B — consistent with training.
        _topic_model = BERTopic.load(str(model_dir), embedding_model=_embedding_model)

        # ── Topic keywords + sizes from topics.json ───────────────────────────
        with open(topics_json, encoding="utf-8") as f:
            data = json.load(f)
        reps = data["topic_representations"]
        _topic_keywords = {
            tid: [kw for kw, _ in kws[:8]]
            for tid, kws in reps.items()
        }
        sizes = data.get("topic_sizes", {})
        _topic_sizes = {str(k): v for k, v in sizes.items()}
        _n_training_docs = sum(sizes.values())

        _loaded = True  # only set True after everything succeeds

    except FileNotFoundError as exc:
        console.print(f"\n[bold red]✗ Missing artifact:[/] {exc}")
        console.print(
            "[dim]Run the full training pipeline first:\n"
            "  python main.py --step all[/]\n"
        )
        sys.exit(1)
    except Exception as exc:
        console.print(f"\n[bold red]✗ Failed to load artifacts:[/] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _predict_centroid(vec: np.ndarray) -> dict:
    """Predict topic by cosine similarity to stored topic centroids.

    vec — precomputed query embedding, shape (1, dim).
    """
    ordered_ids  = sorted(_centroids.keys())
    centroid_mat = np.array([_centroids[tid] for tid in ordered_ids])
    sims         = cosine_similarity(vec, centroid_mat)[0]
    best_idx     = int(np.argmax(sims))
    topic_id     = int(ordered_ids[best_idx])
    score        = float(sims[best_idx])
    str_id       = str(topic_id)
    return {
        "method":         "Centroid Similarity",
        "topic_id":       topic_id,
        "label":          _lookup.get(topic_id, "Unknown Topic"),
        "score":          score,
        "keywords":       _topic_keywords.get(str_id, []),
        "size":           _topic_sizes.get(str_id, 0),
        "low_confidence": score < _SIM_LOW_WARN,
    }


def _predict_bertopic(text: str, vec: np.ndarray) -> dict:
    """Predict topic using BERTopic c-TF-IDF transform.

    Score is cosine similarity between the query embedding and the centroid of
    the BERTopic-predicted topic.  We do NOT use BERTopic's native probability
    output because the model was trained with calculate_probabilities=False,
    which means transform() always returns None/empty for probs — so the raw
    probability value would always be 0.0, which is misleading.

    vec — precomputed query embedding (same vector used by centroid method),
    shape (1, dim).  Passing it in avoids a redundant encode() call.
    """
    topics, _ = _topic_model.transform([text])
    topic_id  = int(topics[0])

    # Score: cosine similarity to the predicted topic's centroid.
    # If topic_id == -1 (outlier) it has no centroid entry → score = 0.0.
    if topic_id in _centroids:
        centroid = _centroids[topic_id].reshape(1, -1)
        score    = float(cosine_similarity(vec, centroid)[0][0])
    else:
        score = 0.0

    str_id = str(topic_id)
    return {
        "method":         "BERTopic c-TF-IDF",
        "topic_id":       topic_id,
        "label":          _lookup.get(topic_id, "Unknown Topic"),
        "score":          score,
        "keywords":       _topic_keywords.get(str_id, []),
        "size":           _topic_sizes.get(str_id, 0),
        "low_confidence": score < _SIM_LOW_WARN,
    }


def predict(text: str) -> list[dict]:
    """Clean the input, embed it once, then run both inference methods.

    Returns an empty list if the input is empty after cleaning.
    """
    _load_artifacts()
    text = clean_cfpb_text(text)

    if not text.strip():
        return []

    # Single encode call — the same embedding vector is reused by both methods,
    # keeping both scores comparable and avoiding a redundant forward pass.
    vec = _embedding_model.encode([text], convert_to_numpy=True)
    return [_predict_centroid(vec), _predict_bertopic(text, vec)]


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


def show_stats() -> None:
    """Print a one-line summary of loaded artifacts after startup."""
    n_topics = len(_centroids)
    device   = f"GPU · {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
    console.print(
        f"  [dim]✓ Loaded:[/] [bold]{n_topics}[/] topics · "
        f"[bold]{_n_training_docs:,}[/] training documents · "
        f"[bold]{_emb_cfg['model_name']}[/] · "
        f"[bold]{device}[/]\n"
    )


# ---------------------------------------------------------------------------
# Results panel
# ---------------------------------------------------------------------------
def show_results(text: str, results: list[dict]) -> None:
    # Guard: empty results mean the input was blank after cleaning
    if not results:
        console.print(
            "[yellow]⚠ Input was empty after cleaning. "
            "Please type a real complaint.[/]\n"
        )
        return

    # Complaint panel
    complaint_panel = Panel(
        Text(text, style="white", no_wrap=False),
        title="[bold]Complaint[/]",
        border_style="dim white",
        padding=(1, 2),
    )

    # Side-by-side method panels
    compare = Table.grid(padding=(1, 3))
    compare.add_column(justify="center")
    compare.add_column(justify="center")

    panels = []
    for result in results:
        topic_id   = result["topic_id"]
        label      = result["label"]
        score      = result["score"]
        keywords   = result["keywords"]
        size       = result["size"]
        low_conf   = result["low_confidence"]

        pct     = score * 100
        bar_len = 20
        filled  = int(bar_len * score)
        bar     = "█" * filled + "░" * (bar_len - filled)

        # Thresholds calibrated for F2LLM cosine similarity range (0.60 – 0.95)
        if score >= _SIM_HIGH:
            bar_color = "green"
        elif score >= _SIM_MED:
            bar_color = "yellow"
        else:
            bar_color = "red"

        info = Table.grid(padding=(0, 2))
        info.add_column(style="bold white", justify="right")
        info.add_column(style="white")

        info.add_row("Topic ID",   f"[bold cyan]{topic_id}[/]")
        info.add_row("Label",      f"[bold green]{label}[/]")
        info.add_row(
            "Similarity",
            f"[{bar_color}]{bar}  {pct:.1f}%[/]  [dim](cosine)[/]",
        )
        info.add_row(
            "Keywords",
            "  ".join(f"[magenta]{kw}[/]" for kw in keywords[:5]),
        )
        info.add_row(
            "Topic size",
            f"[bold]{size:,}[/] docs" if size else "[dim]N/A[/]",
        )
        if low_conf:
            info.add_row(
                "[yellow]⚠ Warning[/]",
                "[yellow]Low similarity — may not match any known topic[/]",
            )

        panels.append(
            Panel(
                info,
                title=f"[bold]{result['method']}[/]",
                border_style="green",
                padding=(1, 2),
            )
        )

    compare.add_row(*panels)

    layout = Layout()
    layout.split_column(
        Layout(complaint_panel, size=5),
        Layout(compare, size=12),   # slightly taller to fit optional warning row
    )

    console.print(layout)
    console.print()

    # Agreement indicator — shows topic labels when methods disagree
    if results[0]["topic_id"] == results[1]["topic_id"]:
        console.print("  [bold green]✓ Both methods agree[/]")
    else:
        c_label = results[0]["label"]
        b_label = results[1]["label"]
        console.print(
            f"  [bold yellow]⚠ Methods disagree[/] — "
            f"Centroid: [cyan]{c_label}[/]  ·  BERTopic: [cyan]{b_label}[/]"
        )
    console.print()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    show_banner()

    with console.status("[bold yellow]Loading artifacts ...", spinner="dots"):
        _load_artifacts()

    show_stats()

    while True:
        text = Prompt.ask("[bold cyan]Enter a consumer complaint[/]").strip()

        if not text:
            console.print("[yellow]Empty input. Try again or press Ctrl+C to exit.[/]\n")
            continue

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
