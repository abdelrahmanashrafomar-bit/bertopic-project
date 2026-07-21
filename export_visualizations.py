"""Export all BERTopic visualizations as HTML + PNG for portfolio."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"
FINAL_CSV = PROJECT_ROOT / "outputs" / "cfpb_final_with_labeled_topics.csv"
TOPICS_JSON = MODEL_DIR / "topics.json"
HTML_DIR = PROJECT_ROOT / "outputs" / "visualizations" / "html"
PNG_DIR = PROJECT_ROOT / "outputs" / "visualizations" / "png"

HTML_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str, status: str = "INFO") -> None:
    icon = {"OK": "✓", "WARN": "⚠", "ERR": "✗", "INFO": "→"}.get(status, "→")
    print(f"  {icon} {msg}")


def save_both(fig, name: str) -> bool:
    html_path = HTML_DIR / f"{name}.html"
    png_path = PNG_DIR / f"{name}.png"

    fig.write_html(str(html_path))
    log(f"Exported {html_path.name}", "OK")

    try:
        fig.write_image(str(png_path), width=1200, height=800, scale=2)
        log(f"Exported {png_path.name}", "OK")
        return True
    except Exception as e:
        log(f"PNG export failed for {name}: {e}", "WARN")
        return False


def main() -> None:
    print("\n  Exporting BERTopic visualizations\n")
    print(f"  {'─' * 50}")

    # Load config
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config["embedding"]["model_name"]

    # Load embedding model
    log("Loading embedding model...")
    try:
        embedding_model = SentenceTransformer(model_name, trust_remote_code=True)
        log("Embedding model loaded", "OK")
    except Exception as e:
        log(f"Failed to load embedding model: {e}", "ERR")
        sys.exit(1)

    # Load BERTopic model
    log("Loading BERTopic model...")
    try:
        topic_model = BERTopic.load(MODEL_DIR, embedding_model=embedding_model)
        log("BERTopic model loaded", "OK")
    except Exception as e:
        log(f"Failed to load BERTopic model: {e}", "ERR")
        sys.exit(1)

    print(f"\n  {'─' * 50}")
    print()

    # ------------------------------------------------------------------
    # 1. Intertopic Distance Map
    # ------------------------------------------------------------------
    log("Intertopic Distance Map...")
    try:
        fig = topic_model.visualize_topics(custom_labels=True)
        save_both(fig, "intertopic_distance_map")
    except Exception as e:
        log(f"Intertopic distance map failed: {e}", "ERR")

    # ------------------------------------------------------------------
    # 2. Topic Hierarchy Dendrogram
    # ------------------------------------------------------------------
    log("Topic Hierarchy...")
    try:
        fig = topic_model.visualize_hierarchy(custom_labels=True)
        save_both(fig, "topic_hierarchy")
    except Exception as e:
        log(f"Topic hierarchy failed: {e}", "ERR")

    # ------------------------------------------------------------------
    # 3. Topic Similarity Heatmap
    # ------------------------------------------------------------------
    log("Topic Similarity Heatmap...")
    try:
        fig = topic_model.visualize_heatmap(custom_labels=True)
        save_both(fig, "topic_similarity_heatmap")
    except Exception as e:
        log(f"Topic similarity heatmap failed: {e}", "ERR")

    # ------------------------------------------------------------------
    # 4. Top Words per Topic (bar chart)
    # ------------------------------------------------------------------
    log("Top Words per Topic...")
    try:
        fig = topic_model.visualize_barchart(custom_labels=True, top_n_topics=20)
        save_both(fig, "top_words_per_topic")
    except Exception as e:
        log(f"Top words bar chart failed: {e}", "ERR")

    # ------------------------------------------------------------------
    # 5. Topic Frequency Bar Chart
    # ------------------------------------------------------------------
    log("Topic Frequency Bar Chart...")
    try:
        freq_df = topic_model.get_topic_freq()
        fig = topic_model.visualize_topics(
            custom_labels=True,
        )
        save_both(fig, "topic_frequencies")
    except Exception as e:
        log(f"Topic frequency chart failed: {e}", "ERR")

    # ------------------------------------------------------------------
    # 6. Topic Frequency (manual bar chart via Plotly)
    # ------------------------------------------------------------------
    log("Topic Frequency Bar Chart (detailed)...")
    try:
        import plotly.express as px

        freq = topic_model.get_topic_freq()
        freq = freq[freq["Topic"] != -1].head(30)
        id_to_label = dict(zip(freq["Topic"], freq["Name"]))
        freq["Label"] = freq["Topic"].map(id_to_label)
        fig = px.bar(
            freq,
            x="Count",
            y="Label",
            orientation="h",
            title="Top 30 Topics by Complaint Volume",
            labels={"Count": "Number of Complaints", "Label": "Topic"},
            color="Count",
            color_continuous_scale="Blues",
            height=800,
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            font=dict(size=12),
        )
        save_both(fig, "topic_frequency_bar")
    except Exception as e:
        log(f"Topic frequency bar chart failed: {e}", "ERR")

    # ------------------------------------------------------------------
    # 7. Document Visualization (2D UMAP projection)
    # ------------------------------------------------------------------
    log("Document Visualization...")
    try:
        df_labeled = pd.read_csv(FINAL_CSV)
        docs = df_labeled["clean_narrative"].fillna("").astype(str).tolist()
        # Use a sample if too large
        sample_size = min(5000, len(docs))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(docs), size=sample_size, replace=False)
        docs_sample = [docs[i] for i in idx]

        log(f"Rendering {sample_size} documents (may take a moment)...")
        fig = topic_model.visualize_documents(
            docs_sample,
            custom_labels=True,
            hide_document_hover=False,
        )
        save_both(fig, "document_projection")
    except Exception as e:
        log(f"Document visualization failed: {e}", "ERR")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    html_files = list(HTML_DIR.glob("*.html"))
    png_files = list(PNG_DIR.glob("*.png"))
    print(f"\n  {'─' * 50}")
    print(f"\n  All visualizations exported successfully.")
    print(f"  HTML: {len(html_files)} files in {HTML_DIR}")
    print(f"  PNG:  {len(png_files)} files in {PNG_DIR}")
    print()


if __name__ == "__main__":
    main()
