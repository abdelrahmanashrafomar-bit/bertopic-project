# BERTopic CFPB Complaint Analysis Pipeline

A reproducible, modular pipeline for topic modeling on CFPB consumer complaint data using BERTopic with precomputed embeddings, UMAP reduction, and HDBSCAN clustering, plus LLM-powered topic labeling and CPU-friendly inference.

## Problem Statement

The CFPB (Consumer Financial Protection Bureau) receives millions of consumer complaints about financial products and services. Manually categorizing these complaints is time-consuming and inconsistent. This project uses topic modeling to automatically discover and label complaint themes — e.g., "Identity Theft Fraud Reporting," "Mortgage Escrow Payment Issues," "Debt Collection Verification Disputes" — enabling analysts to understand complaint patterns at scale.

A key requirement is that inference (classifying new complaints) must work **without retraining** the model, and must be runnable on CPU-only environments.

## Dataset

- **Source**: CFPB Consumer Complaint Database
- **Raw file**: `data/raw/complaints.csv`
- **Sample**: 50,000 complaints (randomly sampled, verified to match original product distribution)
- **Deduplicated**: 35,993 unique complaint narratives after removing exact duplicates
- **Text column**: `Consumer complaint narrative`
- **Language**: ~100% English (lingua-language-detector verified)

## Critical Design Note: Cluster ID Scheme

This pipeline uses **precomputed** clustering — HDBSCAN is fit once outside of BERTopic, and its labels are passed into BERTopic via `BaseCluster` rather than letting BERTopic re-cluster. This allows UMAP/HDBSCAN hyperparameters to be tuned and validated independently using DBCV before topic modeling runs.

BERTopic renumbers topics by descending cluster size after fitting (largest cluster → Topic `0`, etc.). The single `cluster_labels.npy` stores the **final renumbered topic IDs** (BERTopic's `topics_` output after `fit_transform`). Both `topic_centroids.npy` and `topic_lookup.csv` are keyed by these same final IDs, so no ID reconciliation is needed.

## Pipeline Architecture

```
                         ┌─────────────────────┐
                         │   config.yaml        │
                         │ (single source of    │
                         │  truth for paths,    │
                         │  params, methods)    │
                         └──────────┬──────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
    ▼                               ▼                               ▼
┌──────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  Preprocessing   │    │  Embedding Generation│    │  UMAP Reduction      │
│                  │    │                      │    │                      │
│ complaints.csv   │    │ SentenceTransformer  │    │  embeddings.npy      │
│       ▼          │    │   model.encode()     │    │       ▼              │
│ sample 50k rows  │    │       ▼              │    │ embeddings_umap.npy  │
│       ▼          │    │ embeddings.npy       │    │                      │
│ clean text       │    │                      │    │ tune_umap()          │
│ (redact PII)     │    │ Also: evaluate       │    │ (for future re-train)│
│       ▼          │    │ embedding quality    │    │                      │
│ detect language  │    │ via NN accuracy      │    │                      │
│       ▼          │    └──────────────────────┘    └──────────────────────┘
│ deduplicate      │
│       ▼          │
│ bertopic_ready   │
│ .csv             │
└──────────────────┘
         │
         ▼
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  HDBSCAN Clustering  │    │  BERTopic Fitting    │    │  Gemini Labeling     │
│                      │    │                      │    │                      │
│ embeddings_umap.npy  │    │ BaseDimensionality   │    │ BERTopic model  ────►│
│       ▼              │    │ Reduction +          │    │       ▼              │
│ tune_hdbscan()       │    │ BaseCluster          │    │ topic_lookup.csv     │
│ (grid search on DBCV)│    │ (precomputed, no     │    │ labels.csv           │
│       ▼              │───►│  re-clustering)      │    │       ▼              │
│ cluster_labels.npy   │    │       ▼              │    │ final labeled CSV    │
│ (final renumbered    │    │ cluster_labels.npy   │    └──────────────────────┘
│  topic IDs)          │    │ + topic_centroids    │
│                      │    │ + saved BERTopic     │
│                      │    │   model              │
└──────────────────────┘    └──────────────────────┘
         │                            │
         ▼                            ▼
                             ┌──────────────────────┐
                             │  Inference           │
                             │                      │
                             │ Two methods:         │
                             │                      │
                             │ 1. Centroid Similarity│
                             │    (default, CPU)    │
                             │    model.encode()    │
                             │                      │
                             │ 2. BERTopic.transform │
                             │    (GPU recommended) │
                             └──────────────────────┘
```

## Folder Structure

```
bertopic-project/
├── main.py                        # CLI entry point
├── config.yaml                    # Single source of truth
├── pyproject.toml                 # Dependencies & scripts
├── .gitignore
├── README.md
├── DEPLOYMENT.md                  # Deployment guide
│
├── src/                           # Python package
│   ├── __init__.py
│   ├── config.py                  # Config loader + RunMetadata
│   ├── validators.py              # File/column validation utilities
│   └── artifacts.py               # (reserved for future use)
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── clean.py               # Text cleaning, language detection
│   │   └── preprocess.py          # Preprocessing orchestration
│   │
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── generate.py            # Embedding generation via model.encode()
│   │   └── evaluate.py            # Embedding quality evaluation
│   │
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── reduce.py              # UMAP reduction + tuning
│   │   └── cluster.py             # HDBSCAN clustering + tuning
│   │
│   ├── topic_modeling/
│   │   ├── __init__.py
│   │   ├── fit.py                 # BERTopic fitting (precomputed y)
│   │   ├── label.py               # Gemini-powered topic labeling
│   │   └── visualize.py           # Intertopic distance, hierarchy
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── schemas.py             # TopicPrediction dataclass
│   │   └── predict.py             # Centroid + BERTopic.transform inference
│   │
│   └── validation/
│       ├── __init__.py
│       └── validate.py            # Artifact integrity checks
│
├── scripts/
│   ├── compare_inference_methods.py   # GPU-only comparison test
│   └── run_validation_checks.py       # Standalone validation runner
│
├── tests/
│   └── test_artifacts.py              # 19 unit tests
│
├── data/
│   ├── raw/complaints.csv             # Raw CFPB data
│   └── processed/                     # Preprocessed CSVs
│
├── artifacts/
│   ├── embeddings.npy                 # Precomputed corpus embeddings
│   ├── embeddings_umap.npy            # UMAP-reduced embeddings (5D)
│   ├── cluster_labels.npy             # Final topic IDs (BERTopic-renumbered)
│   ├── topic_centroids.npy            # Centroids keyed by final topic ID
│   ├── topic_lookup.csv               # Topic ID -> label
│   ├── labels.csv                     # Same mapping (redundant)
│   └── model/                         # Saved BERTopic model
│       ├── config.json
│       ├── topics.json
│       └── topic_embeddings.safetensors
│
├── outputs/                           # Generated reports & plots
│   ├── cfpb_final_with_labeled_topics.csv
│   ├── run_metadata.json
│   └── *.html                         # Visualization files
│
└── notebooks/                         # Original reference notebooks
    ├── 00_preprocessing.ipynb
    ├── 01_embeddings_generation.ipynb
    ├── 02_embeddings_evaluation.ipynb
    ├── 03_umap.ipynb
    ├── 04_umap_tuning.ipynb
    ├── 05_hdbscan.ipynb
    └── 06_bertopic_and_labels.ipynb
```

## Installation

### Prerequisites

- Python >= 3.12
- pip

### Setup

```bash
cd bertopic-project
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate    # Linux/Mac
pip install -e ".[dev]"
```

### Environment Variables

| Variable | Required For | Description |
|----------|-------------|-------------|
| `GEMINI_API_KEY` | Topic labeling | Google Gemini API key (set in `.env` file) |
| `HF_TOKEN` | Model download | Hugging Face token (optional, for rate limits) |

## Configuration

All configuration lives in `config.yaml`. Key sections:

```yaml
paths:
  embeddings: artifacts/embeddings.npy
  cluster_labels: artifacts/cluster_labels.npy
  topic_centroids: artifacts/topic_centroids.npy
  topic_lookup: artifacts/topic_lookup.csv
  bertopic_model_dir: artifacts/model

umap:
  final:
    n_neighbors: 15
    n_components: 5
    min_dist: 0.0
    metric: cosine
    random_state: 42
  tuning:
    n_neighbors: [10, 15, 30, 50]
    sample_size: 10000

hdbscan:
  final:
    metric: euclidean
    cluster_selection_method: eom
  tuning:
    min_cluster_size: [10, 15, 25, 40, 60]
    min_samples: [5, 10, 15]

bertopic:
  calculate_probabilities: false
  serialization: safetensors

inference:
  method: centroid_similarity    # default
  top_n: 3
  model_dir: artifacts/model
  lookup_path: artifacts/topic_lookup.csv

embedding:
  model_name: codefuse-ai/F2LLM-1.7B
```

## How to Run

### Validation (read-only, safe)

```bash
python -m main --step validate
python -m pytest tests/ -v
```

### Inference

```bash
# Run example inference (centroid similarity, CPU-safe)
python -m main --step inference

# Or use the API directly
python -c "
from src.inference.predict import predict
from src.config import load_config
config = load_config()
results = predict('I found unauthorized charges on my credit card.', config, method='centroid_similarity')
for pred in results:
    print(f'Topic {pred.topic_id}: {pred.label} (score={pred.score:.4f})')
"
```

### Full Training Pipeline

**WARNING**: Overwrites existing artifacts. Only run if you intend to retrain.

```bash
python -m main --step all
```

### Individual Training Steps

```bash
python -m main --step preprocessing
python -m main --step embedding
python -m main --step umap
python -m main --step clustering
python -m main --step topic_modeling
python -m main --step labeling          # requires GEMINI_API_KEY
python -m main --step visualize
```

## Example Prediction

```python
from src.inference.predict import predict
from src.config import load_config

config = load_config()

complaints = [
    "Someone opened a credit card in my name without permission",
    "I demand validation of this debt under the FDCPA",
    "My mortgage payment was not credited correctly",
]

for text in complaints:
    results = predict(text, config, method="centroid_similarity")
    best = results[0]
    print(f"Complaint: {text[:50]}...")
    print(f"  -> Topic {best.topic_id}: {best.label} (score={best.score:.4f})")
```

Expected output:
```
Complaint: Someone opened a credit card in my name without perm...
  -> Topic 0: Identity Theft Fraud Reporting (score=0.5231)
```

## Inference Methods

### Centroid Similarity (Default)

- **How it works**: Each topic has a centroid vector — mean of all complaint embeddings assigned to that topic. New complaints are embedded via `model.encode()` and compared to all centroids via cosine similarity.
- **Speed**: Fast (no BERTopic model loading, just embedding + matrix multiply)
- **GPU**: Not required
- **Returns**: Top N topics with similarity scores
- **Use when**: Fast, CPU-only inference

### BERTopic Transform

- **How it works**: Loads the saved BERTopic model and calls `topic_model.transform()` to assign a topic via the fitted c-TF-IDF representation. New complaints are embedded via `model.encode()` before transform.
- **Speed**: Slower (loads full BERTopic model + embedding model)
- **GPU**: Recommended for the embedding model
- **Returns**: Single topic with probability (if `calculate_probabilities: true` was set at fit time; otherwise returns placeholder score)
- **Use when**: You want BERTopic's native prediction on GPU

### Comparison

Run the GPU-only comparison script:

```bash
python scripts/compare_inference_methods.py
```

## Training Pipeline vs Inference Pipeline

| Aspect | Training Pipeline | Inference Pipeline |
|--------|------------------|-------------------|
| Steps | preprocessing → embedding → umap → clustering → topic_modeling → labeling | validate → inference |
| GPU required | Yes (embedding model) | No for centroid, yes for BERTopic transform |
| CPU-only | No | Yes |
| Modifies artifacts | Yes (overwrites) | No (read-only) |
| Required files | Raw CSV + config | artifacts/ + src/ + config |
| Safe to re-run | No (destructive) | Yes (idempotent) |

The project is designed so that once training is complete, inference can run independently with just:
- `topic_centroids.npy`
- `topic_lookup.csv` / `labels.csv`
- `artifacts/model/`
- `config.yaml`
- `src/`

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for:
- Minimal file set for inference-only deployment
- Lightning AI Studio setup
- Local inference server instructions
