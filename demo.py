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
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.preprocessing.clean import clean_cfpb_text


# ---------------------------------------------------------------------------
# Project root & config path — all artifact paths read from config.yaml
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH  = PROJECT_ROOT / "config.yaml"

console = Console()


# ---------------------------------------------------------------------------
# Module-level artifact cache (lazy — loaded exactly once)
# ---------------------------------------------------------------------------
_loaded: bool = False
_embedding_model: SentenceTransformer | None = None
_topic_model: BERTopic | None = None
_centroids: dict[int, np.ndarray] | None = None
_lookup: dict[int, str] | None = None
_topic_keywords: dict[str, list[str]] | None = None
_topic_sizes: dict[str, int] | None = None
_emb_cfg: dict | None = None
_n_training_docs: int = 0


# ---------------------------------------------------------------------------
# Similarity thresholds — calibrated for F2LLM cosine range (0.60 – 0.95)
# ---------------------------------------------------------------------------
_SIM_HIGH     = 0.80   # green
_SIM_MED      = 0.65   # yellow
_SIM_LOW_WARN = 0.55   # below this → warn


def _load_artifacts() -> None:
    """Load all artifacts once and cache in module globals.

    All paths are resolved from config.yaml — the single source of truth.
    A styled Rich error + sys.exit(1) is raised for any missing file.
    """
    global _loaded, _embedding_model, _topic_model, _centroids, _lookup
    global _topic_keywords, _topic_sizes, _emb_cfg, _n_training_docs

    if _loaded:
        return

    try:
        # ── Config ─────────────────────────────────────────────────────────
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        _emb_cfg = config["embedding"]
        paths    = config["paths"]

        centroids_path = PROJECT_ROOT / paths["topic_centroids"]
        lookup_path    = PROJECT_ROOT / paths["topic_lookup"]
        model_dir      = PROJECT_ROOT / paths["bertopic_model_dir"]
        topics_json    = Path(model_dir) / "topics.json"

        # ── Embedding model — identical settings to training (generate.py) ──
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

        # ── BERTopic model — primary inference engine ─────────────────────────
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

        _loaded = True  # only set after all artifacts load successfully

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
# Inference
# ---------------------------------------------------------------------------

def predict(text: str) -> dict | None:
    """Clean, embed, and return a single BERTopic prediction.

    Returns None if the input is empty after cleaning.

    The result dict contains two confidence signals:
    - ``bertopic_score``:  cosine similarity of the query to the BERTopic-
                           predicted topic's centroid.  Measures how well the
                           complaint fits the specific topic BERTopic chose.
    - ``embedding_score``: maximum cosine similarity across *all* topic
                           centroids.  Measures the best geometric match
                           regardless of which topic was picked.  If this is
                           much higher than ``bertopic_score``, BERTopic
                           assigned a topic that is not the nearest centroid.
    """
    _load_artifacts()
    text = clean_cfpb_text(text)
    if not text.strip():
        return None

    # Single encode call — vector reused for both signals
    vec = _embedding_model.encode([text], convert_to_numpy=True)

    # ── BERTopic topic assignment ────────────────────────────────────────────
    topics, _ = _topic_model.transform([text])
    topic_id  = int(topics[0])

    # Signal 1 — BERTopic similarity:
    #   cosine(query, centroid of the BERTopic-chosen topic)
    #   topic_id == -1 (outlier) has no centroid → score = 0.0
    if topic_id in _centroids:
        centroid       = _centroids[topic_id].reshape(1, -1)
        bertopic_score = float(cosine_similarity(vec, centroid)[0][0])
    else:
        bertopic_score = 0.0

    # Signal 2 — Embedding similarity:
    #   max cosine across ALL topic centroids (best geometric fit)
    ordered_ids    = sorted(_centroids.keys())
    centroid_mat   = np.array([_centroids[tid] for tid in ordered_ids])
    sims           = cosine_similarity(vec, centroid_mat)[0]
    embedding_score = float(np.max(sims))

    str_id = str(topic_id)
    return {
        "topic_id":        topic_id,
        "label":           _lookup.get(topic_id, "Unknown Topic"),
        "bertopic_score":  bertopic_score,
        "embedding_score": embedding_score,
        "keywords":        _topic_keywords.get(str_id, []),
        "size":            _topic_sizes.get(str_id, 0),
        "low_confidence":  bertopic_score < _SIM_LOW_WARN,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(score: float, length: int = 20) -> tuple[str, str, str]:
    """Return (bar_string, pct_string, color) for a cosine similarity score."""
    filled = int(length * min(max(score, 0.0), 1.0))
    bar    = "█" * filled + "░" * (length - filled)
    pct    = f"{score * 100:.1f}%"
    if score >= _SIM_HIGH:
        color = "green"
    elif score >= _SIM_MED:
        color = "yellow"
    else:
        color = "red"
    return bar, pct, color


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
    banner.append("          BERTopic · F2LLM-1.7B · Cosine Similarity          ", style="dim white")
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
# Results display — clean vertical layout
# ---------------------------------------------------------------------------

def show_results(text: str, result: dict | None) -> None:
    if result is None:
        console.print(
            "[yellow]⚠ Input was empty after cleaning. "
            "Please type a real complaint.[/]\n"
        )
        return

    label          = result["label"]
    topic_id       = result["topic_id"]
    b_score        = result["bertopic_score"]
    e_score        = result["embedding_score"]
    keywords       = result["keywords"]
    size           = result["size"]
    low_conf       = result["low_confidence"]

    # ── 1. Complaint ─────────────────────────────────────────────────────────
    console.print(Panel(
        Text(f'"{text}"', style="white", no_wrap=False),
        title="[bold]Complaint[/]",
        border_style="dim white",
        padding=(0, 2),
    ))
    console.print()

    # ── 2. Predicted Topic ───────────────────────────────────────────────────
    topic_display = Text()
    topic_display.append(f"\n  {label}\n", style="bold bright_white")
    topic_display.append(f"  Topic ID {topic_id}", style="dim cyan")
    if topic_id == -1:
        topic_display.append("  [Outlier — does not match known topics]", style="dim yellow")

    console.print(Panel(
        topic_display,
        title="[bold cyan]Predicted Topic[/]",
        border_style="cyan",
        padding=(0, 2),
    ))
    console.print()

    # ── 3. Confidence Signals ────────────────────────────────────────────────
    b_bar, b_pct, b_color = _make_bar(b_score)
    e_bar, e_pct, e_color = _make_bar(e_score)

    signals = Table.grid(padding=(0, 2))
    signals.add_column(style="dim white", justify="right", min_width=22)
    signals.add_column()

    signals.add_row(
        "BERTopic similarity",
        f"[{b_color}]{b_bar}[/]  [{b_color}]{b_pct}[/]",
    )
    signals.add_row(
        "Embedding similarity",
        f"[{e_color}]{e_bar}[/]  [{e_color}]{e_pct}[/]",
    )
    if low_conf:
        signals.add_row(
            "",
            "[yellow]⚠  Low similarity — topic may not be a strong match[/]",
        )

    console.print(Panel(
        signals,
        title="[bold]Confidence Signals[/]",
        border_style="dim green",
        padding=(1, 2),
    ))
    console.print()

    # ── 4. Top Keywords ──────────────────────────────────────────────────────
    kw_text = Text()
    for kw in keywords[:6]:
        kw_text.append("  ✓ ", style="bold green")
        kw_text.append(kw, style="magenta")
        kw_text.append("  ")

    size_line = Text()
    size_line.append("\n\n  Topic Documents: ", style="dim white")
    size_line.append(f"{size:,}" if size else "N/A", style="bold white")

    kw_block = Table.grid()
    kw_block.add_column()
    kw_block.add_row(kw_text)
    kw_block.add_row(size_line)

    console.print(Panel(
        kw_block,
        title="[bold]Top Keywords[/]",
        border_style="dim magenta",
        padding=(1, 2),
    ))
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
