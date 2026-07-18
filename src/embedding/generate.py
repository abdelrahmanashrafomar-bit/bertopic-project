from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from src.config import get_project_root
from src.validators import ensure_file_exists


def load_model(model_name: str, torch_dtype: str, device_map: str) -> SentenceTransformer:
    print(f"Loading embedding model: {model_name}")
    dtype = getattr(torch, torch_dtype, torch.float16)
    model = SentenceTransformer(
        model_name,
        model_kwargs={"torch_dtype": dtype, "device_map": device_map},
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")
    return model


def generate_embeddings(
    model: SentenceTransformer,
    documents: list[str],
    batch_size: int,
) -> np.ndarray:
    print(f"Generating embeddings for {len(documents):,} documents...")
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def save_embeddings(embeddings: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    print(f"Embeddings saved: {path}")


def run(config: dict) -> np.ndarray:
    root = get_project_root()
    emb_cfg = config["embedding"]
    paths = config["paths"]

    data_path = root / paths["bertopic_ready"]
    ensure_file_exists(data_path, "BERTopic-ready dataset")
    df = pd.read_csv(data_path)
    documents = df[config["preprocessing"]["text_column"]].astype(str).tolist()
    print(f"Loaded {len(documents):,} documents")

    model = load_model(emb_cfg["model_name"], emb_cfg["torch_dtype"], emb_cfg["device_map"])
    embeddings = generate_embeddings(model, documents, emb_cfg["batch_size"])
    save_embeddings(embeddings, root / paths["embeddings"])
    return embeddings
