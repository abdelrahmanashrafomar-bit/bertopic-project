from __future__ import annotations

from pathlib import Path


def ensure_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def ensure_equal_length(*items: tuple[str, int]) -> None:
    if not items:
        return
    expected_name, expected_value = items[0]
    for name, value in items[1:]:
        if value != expected_value:
            raise ValueError(
                f"Length mismatch: {expected_name}={expected_value} but {name}={value}"
            )


def ensure_required_columns(columns: list[str], required: list[str]) -> None:
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
