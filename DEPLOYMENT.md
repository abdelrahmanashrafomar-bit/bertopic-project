# Deployment Instructions

## Required Files

```
bertopic-project/
├── config.yaml                  # Single source of truth
├── main.py                      # CLI entry point
├── pyproject.toml               # Dependencies
├── src/
│   ├── __init__.py
│   ├── config.py                # Config loader
│   ├── validators.py            # Validation utilities
│   ├── artifacts.py             # JSON helpers
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── schemas.py           # TopicPrediction dataclass
│   │   └── predict.py           # Both inference methods
│   └── validation/
│       ├── __init__.py
│       └── validate.py          # Artifact validation
├── artifacts/
│   ├── topic_lookup.csv         # Topic ID -> Label mapping
│   ├── labels.csv               # Same mapping (redundant)
│   ├── topic_centroids.npy      # Centroid vectors for similarity inference
│   ├── cluster_labels.npy       # Precomputed cluster labels
│   ├── embeddings.npy           # Precomputed embeddings
│   └── model/                   # Saved BERTopic model
│       ├── config.json
│       ├── topics.json
│       └── topic_embeddings.safetensors
├── .env                         # GEMINI_API_KEY (labeling only)
└── scripts/
    └── compare_inference_methods.py  # Inference comparison test
```

## Local Inference

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -e ".[dev]"
```

### Run inference

```bash
# Centroid similarity (default, no GPU needed for inference)
python -m main --step inference

# Or import in your code:
from src.inference.predict import predict
from src.config import load_config

config = load_config()
results = predict("Your complaint text here", config, method="centroid_similarity")
for pred in results:
    print(f"Topic {pred.topic_id}: {pred.label} (score={pred.score:.4f})")
```

### Run validation

```bash
python -m main --step validate
python -m pytest tests/ -v
```

## Lightning AI Deployment

### Setup

1. Create a new Studio or use an existing one with GPU (A10G or better).
2. Upload the required files (see list above).
3. Install dependencies:

```bash
pip install -e ".[dev]"
```

### Run inference comparison

```bash
python scripts/compare_inference_methods.py
```

### Run full pipeline (if retraining)

```bash
bertopic-pipeline --step all
```

### Run individual steps

```bash
bertopic-pipeline --step validate
bertopic-pipeline --step inference
```

## Inference API Usage

```python
from src.inference.predict import predict
from src.config import load_config

config = load_config()

# Centroid similarity (default, fast, no GPU needed)
results = predict("Your complaint text", config, method="centroid_similarity")
for pred in results:
    print(f"Topic {pred.topic_id}: {pred.label} (score={pred.score:.4f})")

# BERTopic transform (requires GPU for embedding model)
results = predict("Your complaint text", config, method="bertopic_transform")
for pred in results:
    print(f"Topic {pred.topic_id}: {pred.label} (score={pred.score:.4f})")
```

## Lightning AI Deployment

### Step 1: Create Studio
- Create a new Studio with GPU (A10G or better).
- Select Python 3.12 runtime.

### Step 2: Upload files
Upload the required files (see list above). The minimum set for inference only:
- `config.yaml`
- `main.py`
- `src/` (all files)
- `artifacts/` (all files)
- `pyproject.toml`

### Step 3: Install

```bash
pip install -e ".[dev]"
```

### Step 4: Run inference comparison

```bash
python scripts/compare_inference_methods.py
```

### Step 5: Use inference API

```python
from src.inference.predict import predict
from src.config import load_config

config = load_config()

# Centroid similarity (fast, no GPU)
results = predict("Your complaint text", config, method="centroid_similarity")

# BERTopic transform (requires GPU)
results = predict("Your complaint text", config, method="bertopic_transform")
```

## Inference Methods

| Method | Speed | GPU Required | Returns |
|--------|-------|-------------|---------|
| `centroid_similarity` | Fast | No | Top N topics with cosine similarity scores |
| `bertopic_transform` | Slow | Yes | Single topic with probability |

## Environment Variables

| Variable | Required For | Description |
|----------|-------------|-------------|
| `GEMINI_API_KEY` | Topic labeling | Google Gemini API key |
| `HF_TOKEN` | Model download | Hugging Face token (optional, for rate limits) |
