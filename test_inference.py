"""
test_inference.py

Predict the topic for a complaint using centroid similarity.

Usage:
    python test_inference.py

Requirements:
    - Run from the project root directory (bertopic-project/)
    - F2LLM-1.7B embedding model will be downloaded on first run (~6 GB)
    - A GPU is strongly recommended (will be slow on CPU)
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.inference.predict import predict
from src.config import load_config


def main():
    config = load_config()

    # ── Edit this sentence to test your own complaint ──
    my_complaint = "Someone stole my credit card and made unauthorized payments"
    # ──────────────────────────────────────────────────

    print("=" * 70)
    print("BERTopic CFPB Complaint Inference")
    print("=" * 70)
    print(f"\nInput complaint: \"{my_complaint}\"\n")

    # --- Method 1: Centroid Similarity (default) ---
    print("[Method 1] Centroid Similarity (CPU-friendly)")
    print("-" * 50)
    try:
        results = predict(my_complaint, config, method="centroid_similarity")
        for i, pred in enumerate(results, 1):
            print(f"  #{i}  Topic {pred.topic_id:>3}: {pred.label}")
            print(f"      Confidence: {pred.score:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  (Likely the embedding model failed to load. Needs GPU.)")

    print()

    # --- Method 2: BERTopic Transform (fallback) ---
    print("[Method 2] BERTopic Transform (requires GPU)")
    print("-" * 50)
    try:
        results = predict(my_complaint, config, method="bertopic_transform")
        for i, pred in enumerate(results, 1):
            print(f"  #{i}  Topic {pred.topic_id:>3}: {pred.label}")
            print(f"      Confidence: {pred.score:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  (Likely the embedding model or BERTopic model failed to load.)")

    print()
    print("=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
