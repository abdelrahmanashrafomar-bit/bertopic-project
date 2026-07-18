from __future__ import annotations

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.validation.validate import (
    check_artifacts,
    verify_topic_mapping,
    verify_topic_centroids,
    verify_cluster_labels,
    verify_embeddings,
    verify_bertopic_model,
    verify_data_consistency,
)

config = load_config()

checks = [
    ("check_artifacts", check_artifacts),
    ("verify_topic_mapping", verify_topic_mapping),
    ("verify_topic_centroids", verify_topic_centroids),
    ("verify_cluster_labels", verify_cluster_labels),
    ("verify_embeddings", verify_embeddings),
    ("verify_bertopic_model", verify_bertopic_model),
    ("verify_data_consistency", verify_data_consistency),
]

all_passed = 0
all_failed = 0
all_warnings = 0

for name, func in checks:
    t1 = time.time()
    r = func(config)
    elapsed = time.time() - t1
    n_p = len(r.get("passed", []))
    n_f = len(r.get("failed", []))
    n_w = len(r.get("warnings", []))
    all_passed += n_p
    all_failed += n_f
    all_warnings += n_w
    status = "PASS" if n_f == 0 else "FAIL"
    print(f"[{status}] {name}: {n_p} passed, {n_f} failed, {n_w} warnings ({elapsed:.1f}s)")
    for m in r.get("failed", []):
        print(f"  FAIL: {m}")
    for m in r.get("warnings", []):
        print(f"  WARN: {m}")

print(f"\nTotal: {all_passed} passed, {all_failed} failed, {all_warnings} warnings")