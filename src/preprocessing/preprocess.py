from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import get_project_root
from src.preprocessing.clean import (
    add_word_count,
    build_language_detector,
    clean_cfpb_text,
    detect_language,
)
from src.validators import ensure_file_exists


def load_raw_data(path: Path) -> pd.DataFrame:
    ensure_file_exists(path, "Raw complaints CSV")
    print(f"Loading raw data from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def sample_data(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    print(f"Sampling {n:,} rows (seed={seed})...")
    return df.sample(n=n, random_state=seed)


def clean_texts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    print("Cleaning complaint narratives...")
    df = df.copy()
    tqdm.pandas(desc="Cleaning")
    df[column] = df[column].fillna("")
    tqdm.pandas()
    df[column] = df[column].progress_apply(clean_cfpb_text)
    df[column] = df[column].str.strip()
    return df


def run_language_detection(
    df: pd.DataFrame, column: str, language_candidates: list[str]
) -> pd.DataFrame:
    print("Detecting languages...")
    detector = build_language_detector(language_candidates)
    df = df.copy()
    df["language"] = df[column].apply(lambda t: detect_language(detector, t))
    return df


def remove_duplicates(df: pd.DataFrame, column: str) -> pd.DataFrame:
    print(f"Removing duplicates by '{column}'...")
    before = len(df)
    df = df.drop_duplicates(subset=[column])
    after = len(df)
    print(f"Deduplicated: {before:,} -> {after:,} ({before - after:,} removed)")
    return df


def save_dataframe(df: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"{label} saved: {path}")


def run(config: dict) -> pd.DataFrame:
    root = get_project_root()
    pre = config["preprocessing"]
    paths = config["paths"]
    seed = config["project"]["random_seed"]

    df = load_raw_data(root / paths["raw_complaints"])
    df = sample_data(df, pre["sample_size"], seed)
    df = clean_texts(df, pre["text_column"])
    df = add_word_count(df, pre["text_column"])
    df = run_language_detection(df, pre["text_column"], pre["language_candidates"])
    save_dataframe(df, root / paths["sample_50k"], "Sample 50k")

    df_dedup = remove_duplicates(df, pre["deduplicate_column"])
    save_dataframe(df_dedup, root / paths["bertopic_ready"], "BERTopic-ready dataset")

    print(
        f"Preprocessing summary: {len(df_dedup):,} unique complaints, "
        f"word_count median={df_dedup['word_count'].median():.0f}"
    )
    return df_dedup
