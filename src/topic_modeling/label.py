from __future__ import annotations

import json
import os
import time
from pathlib import Path
import pandas as pd
from bertopic import BERTopic
from google import genai
from google.genai import types
from tqdm import tqdm

from src.config import get_project_root
from src.validators import ensure_file_exists


def load_env_api_key(root: Path) -> str:
    env_path = root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip()
                os.environ["GEMINI_API_KEY"] = key
                return key
    return os.environ.get("GEMINI_API_KEY", "")


class GeminiLabeler:
    def __init__(self, config: dict):
        gemini_cfg = config["gemini"]
        root = get_project_root()
        load_env_api_key(root)
        self.client = genai.Client()
        self.model = gemini_cfg["model"]
        self.batch_size = gemini_cfg["topic_label_batch_size"]
        self.max_retries = gemini_cfg["max_retries"]
        self.retry_delay = gemini_cfg["retry_delay_seconds"]
        self.keywords_limit = gemini_cfg["label_keywords_limit"]
        self.doc_chars = gemini_cfg["representative_doc_chars"]

    def label_topics(self, topic_model: BERTopic) -> dict[int, str]:
        topic_info = topic_model.get_topic_info()
        labels: dict[int, str] = {-1: "Outliers / Unclassified"}
        valid_topics = topic_info[topic_info["Topic"] != -1]
        batches = [
            valid_topics[i:i + self.batch_size]
            for i in range(0, len(valid_topics), self.batch_size)
        ]
        print(f"Grouped {len(valid_topics)} topics into {len(batches)} batches")

        for batch_idx, batch in enumerate(tqdm(batches, desc="Labeling")):
            batch_data = []
            for _, row in batch.iterrows():
                t_id = int(row["Topic"])
                keywords = [w for w, _ in topic_model.get_topic(t_id)][:self.keywords_limit]
                rep_docs = row.get("Representative_Docs")
                sample = (rep_docs[0][:self.doc_chars] if isinstance(rep_docs, list) and rep_docs
                          else "No sample text")
                batch_data.append({
                    "topic_id": t_id,
                    "keywords": keywords,
                    "sample_text": sample,
                })

            prompt = self._build_prompt(batch_data)
            result = self._call_gemini(prompt, batch_idx)
            if result:
                for item in result:
                    labels[int(item["topic_id"])] = item["label"]

        return labels

    def _build_prompt(self, batch_data: list[dict]) -> str:
        prompt = """You are an expert financial consumer complaint analyst. Your job is to generate highly concise, professional, and descriptive 3-to-5 word labels for multiple clusters of customer complaints.

Below is a JSON array of complaint clusters. For each object, analyze the 'keywords' and 'sample_text', then return a JSON array with the exact same topic_id and a concise label.

Complaint clusters:
{data}

Return only JSON in this format:
[
  {"topic_id": 0, "label": "Example Topic Label"}
]"""
        return prompt.format(data=json.dumps(batch_data, indent=2))

    def _call_gemini(self, prompt: str, batch_idx: int) -> list[dict] | None:
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    ),
                )
                return json.loads(response.text.strip())
            except Exception as e:
                print(f"\n[Error on Batch {batch_idx + 1}] {e}. "
                      f"Retrying in {self.retry_delay}s...")
                time.sleep(self.retry_delay)
        return None


def save_labels(labels: dict[int, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([
        {"Topic": tid, "Topic_Label": label}
        for tid, label in labels.items()
    ]).sort_values("Topic").reset_index(drop=True)
    frame.to_csv(path, index=False)
    print(f"Labels saved: {path} (n={len(frame)})")


def run(config: dict) -> dict[int, str]:
    from src.topic_modeling.fit import load_data

    root = get_project_root()
    paths = config["paths"]

    model_path = root / paths["bertopic_model_dir"]
    ensure_file_exists(model_path / "config.json", "BERTopic model config")
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer(
        config["embedding"]["model_name"],
        trust_remote_code=True,
    )
    topic_model = BERTopic.load(model_path, embedding_model=embedding_model)

    labeler = GeminiLabeler(config)
    labels = labeler.label_topics(topic_model)

    save_labels(labels, root / paths["topic_lookup"])
    save_labels(labels, root / paths["labels"])

    df, documents, embeddings, cluster_labels, text_col = load_data(config)
    df["Topic"] = cluster_labels
    df["Topic_Label"] = df["Topic"].map(labels)
    df.to_csv(root / paths["final_labeled_topics"], index=False)
    print(f"Final labeled dataset saved: {root / paths['final_labeled_topics']}")

    return labels
