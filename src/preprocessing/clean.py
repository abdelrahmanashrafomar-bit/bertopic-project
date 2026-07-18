from __future__ import annotations

import re

import pandas as pd
from lingua import Language, LanguageDetectorBuilder


def clean_cfpb_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\b[Xx]{1,2}[/\-][Xx]{1,2}[/\-]\d{4}\b", "[DATE]", text)
    text = re.sub(r"\b[Xx]{2,}\b", "[PROTECTED]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_language(detector, text: str) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    return detector.detect_language_of(text)


def build_language_detector(language_candidates: list[str] | None = None):
    if language_candidates is None:
        language_candidates = [
            Language.ENGLISH,
            Language.SPANISH,
            Language.FRENCH,
            Language.GERMAN,
            Language.CHINESE,
            Language.ARABIC,
        ]
    else:
        lang_map = {l.name: l for l in Language}
        language_candidates = [
            lang_map[name] for name in language_candidates if name in lang_map
        ]
    return LanguageDetectorBuilder.from_languages(*language_candidates).build()


def add_word_count(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df["word_count"] = df[column].fillna("").str.split().str.len()
    return df
