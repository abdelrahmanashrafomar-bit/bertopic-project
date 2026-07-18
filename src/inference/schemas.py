from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TopicPrediction:
    topic_id: int
    label: str
    score: float
