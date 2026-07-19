# BERTopic CFPB Complaint Analysis Pipeline

A modular, production-oriented topic modeling pipeline for the CFPB Consumer Complaint Database. Discovers complaint themes at scale using BERTopic with precomputed embeddings, UMAP reduction, HDBSCAN clustering, and LLM-powered labeling -- with CPU-friendly centroid similarity inference for deployment.

---

## Business Problem

The Consumer Financial Protection Bureau (CFPB) receives millions of consumer complaints about financial products and services. Each complaint is a free-text narrative describing issues with mortgages, debt collection, credit reporting, identity theft, and more.

**The challenge**: Manual categorization is slow, inconsistent, and does not scale. Predefined product/issue taxonomies miss emerging themes. Analysts need a way to automatically discover complaint patterns without predefined categories.

## Business Value

- **Theme discovery**: Automatically surfaces complaint topics (e.g., "Identity Theft Fraud Reporting," "Debt Collection Verification Disputes") without predefined categories.
- **Emerging pattern detection**: New fraud schemes or recurring consumer pain points appear as distinct clusters, enabling proactive investigation.
- **Analyst efficiency**: Reduces thousands of complaints into a handful of interpretable topics with human-readable labels.
- **Reporting support**: Labeled topics can feed dashboards, trend analysis, and regulatory reporting.
- **Reproducibility**: The pipeline is fully scripted and version-controlled -- no black-box notebook cells.

---

## Solution Architecture

```
+-----------------------------------------------------------+
|                     config.yaml                            |
|    (single source of truth for all paths, params, methods) |
+---------------------------+-------------------------------+
                            |
          +-----------------+-----------------+
          |                 |                 |
          v                 v                 v
+--------------------+ +--------------------+ +--------------------+
|  Preprocessing     | | Embedding Gen.     | | UMAP Reduction     |
|                    | |                    | |                    |
| complaints.csv     | | SentenceTrans.     | | embeddings.npy     |
|       v            | |   model.encode()   | |       v            |
| sample 50k rows    | |       v            | | embeddings_umap    |
|       v            | | embeddings.npy     | |       .npy         |
| clean text         | |  (2048-D vectors)  | |                    |
| (redact PII)       | |                    | | tune_umap()        |
|       v            | | Also: evaluate     | | (grid search on   |
| detect language    | | embedding quality  | | trustworthiness + |
|       v            | | via NN accuracy    | | DBCV)             |
| deduplicate        | +--------------------+ +--------------------+
|       v            |
| bertopic_ready     |
| .csv               |
+--------------------+
         |
         v
+--------------------+ +--------------------+ +--------------------+
| HDBSCAN Clustering | | BERTopic Fitting   | | Gemini Labeling    |
|                    | |                    | |                    |
| embeddings_umap    | | BaseDimensionality | | BERTopic model --->|
|       .npy         | | Reduction +        | |       v            |
|       v            | | BaseCluster        | | topic_lookup.csv   |
| tune_hdbscan()     | | (precomputed, no   | | labels.csv         |
| (grid search on    | |  re-clustering)    | |       v            |
| DBCV)              | |       v            | | final labeled CSV  |
|       v            | | topics (renumbered | +--------------------+
| cluster_labels     | |  by BERTopic in    |
|   .npy             | |  memory; not saved |
| (raw HDBSCAN       | |  to disk)          |
|  labels)           | | + topic_centroids  |
|                    | | + saved BERTopic   |
|                    | |   model            |
+--------------------+ +--------------------+
```

## Repository Structure

```
bertopic-project/
|
+-- main.py                        # CLI entry point: python -m main --step <name>
+-- config.yaml                    # Single source of truth (paths, params, methods)
+-- pyproject.toml                 # Dependencies, scripts, project metadata
+-- .gitignore
+-- README.md
+-- DEPLOYMENT.md                  # Deployment guide for Lightning AI / inference server
|
+-- src/                           # Python package -- all pipeline logic
|   +-- config.py                  # Config loader, project root resolver, RunMetadata
|   +-- validators.py              # Shared validation utilities (file existence, column checks)
|   |
|   +-- preprocessing/             # Data loading, cleaning, sampling, deduplication
|   |   +-- clean.py               # PII redaction, language detection, word count
|   |   +-- preprocess.py          # Orchestration: load -> sample -> clean -> dedup -> save
|   |
|   +-- embedding/                 # Text-to-vector conversion
|   |   +-- generate.py            # SentenceTransformer model loading + batch encoding
|   |   +-- evaluate.py            # NN accuracy, top-k similarity, extreme pair analysis
|   |
|   +-- clustering/                # Dimensionality reduction + clustering
|   |   +-- reduce.py              # UMAP fitting + grid search tuning (trustworthiness, DBCV)
|   |   +-- cluster.py             # HDBSCAN fitting + grid search tuning (DBCV)
|   |
|   +-- topic_modeling/            # BERTopic integration + labeling
|   |   +-- fit.py                 # BERTopic with precomputed labels (BaseCluster)
|   |   +-- label.py               # Gemini-powered topic labeling (batching, retries, JSON)
|   |   +-- visualize.py           # Intertopic distance, hierarchy, topics over time
|   |
|   +-- inference/                 # Production inference (CPU-friendly)
|   |   +-- schemas.py             # TopicPrediction dataclass
|   |   +-- predict.py             # Centroid similarity + BERTopic.transform
|   |
|   +-- validation/                # Artifact integrity checks
|       +-- validate.py            # 6 check suites, 18+ individual checks
|
+-- scripts/
|   +-- compare_inference_methods.py   # GPU-only: compares centroid vs BERTopic on 12 test cases
|   +-- run_validation_checks.py       # Standalone validation runner
|
+-- tests/
|   +-- test_artifacts.py              # 19 unit tests for artifact loading + validation
|
+-- data/
|   +-- raw/complaints.csv             # Raw CFPB data (large, gitignored)
|   +-- processed/                     # Preprocessed CSVs (gitignored)
|
+-- artifacts/
|   +-- embeddings.npy                 # 2048-D corpus embeddings
|   +-- embeddings_umap.npy            # UMAP-reduced embeddings (5D)
|   +-- cluster_labels.npy             # Raw HDBSCAN labels (pre-BERTopic renumbering)
|   +-- topic_centroids.npy            # Dict[topic_id -> centroid vector]
|   +-- topic_lookup.csv               # Topic ID -> human-readable label
|   +-- labels.csv                     # Same mapping (redundant, cross-validation)
|   +-- model/                         # Saved BERTopic model
|       +-- config.json
|       +-- topics.json
|       +-- topic_embeddings.safetensors
|
+-- outputs/                           # Generated reports, plots, metadata
|   +-- cfpb_final_with_labeled_topics.csv
|   +-- run_metadata.json
|   +-- *.html                         # Intertopic distance, hierarchy, topics over time
|
+-- notebooks/                         # Reference notebooks (original exploration)
    +-- 00_preprocessing.ipynb
    +-- 01_embeddings_generation.ipynb
    +-- 02_embeddings_evaluation.ipynb
    +-- 03_umap.ipynb
    +-- 04_umap_tuning.ipynb
    +-- 05_hdbscan.ipynb
    +-- 06_bertopic_and_labels.ipynb
```

## Training Pipeline

The training pipeline is designed to be run once (offline) on a GPU-capable machine. Each step produces artifacts consumed by the next.

```
Raw CSV (complaints.csv)
   |
   v
+---------------------------------------------------------------------+
| Step 1: Preprocessing                                                |
| ------------------------------------------------------------------- |
| load_raw_csv() -> sample 50k rows -> clean text (PII redaction)      |
| -> detect language -> deduplicate -> save bertopic_ready.csv         |
+---------------------------------------------------------------------+
   |
   v
+---------------------------------------------------------------------+
| Step 2: Embedding Generation                                        |
| ------------------------------------------------------------------- |
| SentenceTransformer (codefuse-ai/F2LLM-1.7B) -> model.encode()      |
| -> embeddings.npy (2048-D vectors, one per complaint)               |
+---------------------------------------------------------------------+
   |
   v
+---------------------------------------------------------------------+
| Step 3: UMAP Reduction                                               |
| ------------------------------------------------------------------- |
| tune_umap(): grid search n_neighbors [10, 15, 30, 50]               |
|   -> evaluate by trustworthiness + downstream DBCV                  |
| fit_umap(): reduce 2048-D -> 5-D with best params                    |
|   -> embeddings_umap.npy                                             |
+---------------------------------------------------------------------+
   |
   v
+---------------------------------------------------------------------+
| Step 4: HDBSCAN Clustering                                          |
| ------------------------------------------------------------------- |
| tune_hdbscan(): grid search min_cluster_size x min_samples          |
|   -> evaluate by DBCV (Density-Based Clustering Validation)         |
| fit_hdbscan(): final clustering with best params                     |
|   -> cluster_labels.npy (raw HDBSCAN labels)                        |
+---------------------------------------------------------------------+
   |
   v
+---------------------------------------------------------------------+
| Step 5: BERTopic Fitting                                            |
| ------------------------------------------------------------------- |
| BERTopic(umap_model=BaseDimensionalityReduction(),                  |
|          hdbscan_model=BaseCluster())                                |
| -> fit_transform(documents, embeddings, y=cluster_labels)            |
| -> renumbered topic IDs + topic_centroids.npy                       |
| -> saved model (artifacts/model/)                                   |
+---------------------------------------------------------------------+
   |
   v
+---------------------------------------------------------------------+
| Step 6: Gemini Labeling                                             |
| ------------------------------------------------------------------- |
| For each topic: c-TF-IDF keywords + representative doc -> Gemini API |
| -> topic_lookup.csv + labels.csv + final_labeled_topics.csv         |
+---------------------------------------------------------------------+
   |
   v
+---------------------------------------------------------------------+
| Step 7: Visualization                                               |
| ------------------------------------------------------------------- |
| visualize_topics() -> intertopic distance map (HTML)                |
| visualize_hierarchy() -> topic hierarchy dendrogram (HTML)          |
| visualize_topics_over_time() -> topic evolution (HTML)              |
+---------------------------------------------------------------------+
```

## Inference Pipeline

Inference is designed to be lightweight, CPU-compatible, and independent of the training pipeline.

```
User Complaint Text
        |
        v
+---------------------------+
|  SentenceTransformer      |
|  model.encode()           |
|  (uses encode() -- see    |
|   Known Issues for        |
|   encode_query() bug)     |
|  Output: 2048-D           |
|  query vector             |
+-------------+-------------+
              |
              v
+---------------------------+
|  Cosine Similarity        |
|  query vector x           |
|  topic_centroids          |
|  (matrix multiply)        |
+-------------+-------------+
              |
              v
+---------------------------+
|  Top-k Topics             |
|  (sorted by score)        |
+-------------+-------------+
              |
              v
+---------------------------+
|  topic_lookup.csv         |
|  (ID -> label)            |
+-------------+-------------+
              |
              v
+---------------------------+
|  TopicPrediction          |
|  {topic_id, label,        |
|   score}                  |
+---------------------------+
```

This architecture is production-friendly because:

- **No BERTopic dependency at inference time** -- centroid similarity only needs `numpy`, `pandas`, `scikit-learn`, and `sentence-transformers`.
- **No GPU required** -- the embedding model runs on CPU for single queries.
- **No Gemini API call** -- labels are precomputed and stored in `topic_lookup.csv`.
- **No retraining** -- inference is a simple matrix multiply against precomputed centroids.
- **Idempotent** -- running inference never modifies artifacts.

---

## Artifacts Explained

| Artifact | Produced By | Consumed By | Training | Inference | Deletable After Training |
|---|---|---|---|---|---|
| `embeddings.npy` | Step 2 (embedding) | Step 3 (UMAP) | Yes | No | Yes |
| `embeddings_umap.npy` | Step 3 (UMAP) | Step 4 (HDBSCAN) | Yes | No | Yes |
| `cluster_labels.npy` | Step 4 (HDBSCAN) | Step 5 (BERTopic) | Yes | No | Yes (raw HDBSCAN output; renumbered IDs only in centroids) |
| `topic_centroids.npy` | Step 5 (BERTopic) | Inference | Yes | **Yes** | No |
| `topic_lookup.csv` | Step 6 (Gemini) | Inference, Validation | Yes | **Yes** | No |
| `labels.csv` | Step 6 (Gemini) | Validation | Yes | No | Yes (redundant with lookup) |
| `model/` (BERTopic) | Step 5 (BERTopic) | Inference (fallback), Visualization | Yes | Optional | Yes (if not using BERTopic transform) |

---

## Project Lifecycle

```
TRAINING (offline, GPU required)
Run once on a GPU-capable machine. Produces all artifacts.
Steps: preprocessing -> embedding -> umap -> clustering ->
       topic_modeling -> labeling -> visualize
    |
    v
ARTIFACTS (version-controlled subset)
topic_centroids.npy | topic_lookup.csv | model/ (optional)
    |
    v
DEPLOYMENT (copy artifacts to target environment)
No retraining. No Gemini API. No clustering. No BERTopic fitting.
    |
    v
INFERENCE (runs many times, CPU-friendly)
embed query -> cosine similarity with centroids -> top-k topics
-> look up human-readable label -> return TopicPrediction
```

---

## Design Philosophy

This project intentionally avoids unnecessary complexity. Key principles:

- **No over-engineering**: The pipeline does exactly what it needs to and no more. Each module has a single responsibility.
- **Readable modular code**: Each file is under 150 lines. Functions are short and focused. Type hints everywhere.
- **Configuration-driven**: All paths, parameters, and methods live in `config.yaml`. No hardcoded values in source code.
- **Minimal dependencies**: Only what is actually used. No `mlflow`, `dvc`, `kubeflow`, or orchestration frameworks.
- **Clear separation**: Training and inference are independent code paths. Training is destructive; inference is read-only.
- **Inference does not require Gemini**: Labels are precomputed. No API calls at inference time.
- **Production inference only needs trained artifacts**: No BERTopic model, no clustering code, no training data.

---

## Engineering Decisions

### Why BERTopic?

BERTopic was chosen over LDA, NMF, or Top2Vec because:
- It produces interpretable topic representations via c-TF-IDF.
- It integrates with modern sentence embeddings (any SentenceTransformer model).
- It supports precomputed clustering via `BaseCluster`, enabling independent tuning of UMAP and HDBSCAN.
- It provides built-in visualization (intertopic distance, hierarchy, topics over time).

### Why SentenceTransformer embeddings?

SentenceTransformer models produce dense, semantically meaningful vectors that capture complaint similarity better than bag-of-words or TF-IDF representations. The chosen model (`codefuse-ai/F2LLM-1.7B`) produces 2048-D embeddings with strong semantic separation, validated via nearest-neighbor accuracy against CFPB product/issue labels.

### Why UMAP before clustering?

UMAP reduces 2048-D embeddings to 5 dimensions before clustering. This is necessary because HDBSCAN's density-based algorithm degrades in high-dimensional spaces (curse of dimensionality). UMAP preserves local neighborhood structure while removing noise dimensions, making clusters more separable.

### Why HDBSCAN instead of KMeans?

- HDBSCAN does not require specifying the number of clusters in advance -- it discovers the natural cluster structure.
- It handles noise: complaints that do not fit any cluster are labeled as outliers (topic -1) rather than being force-assigned to an arbitrary cluster.
- It can detect clusters of varying density, which is expected in real-world complaint data.

### Why precomputed clustering?

BERTopic normally runs UMAP + HDBSCAN internally. This project runs them independently and passes the labels into BERTopic via `BaseCluster`. This allows:

- **Independent tuning**: UMAP and HDBSCAN hyperparameters are optimized via grid search with DBCV before BERTopic is ever involved.
- **Validation**: Cluster quality can be assessed independently of topic modeling.
- **Reproducibility**: The clustering step is decoupled and can be re-run without re-fitting BERTopic.

### Why centroid similarity for inference?

Centroid similarity is the default inference method because:

- **Speed**: A single matrix multiply against precomputed centroids. No BERTopic model loading.
- **CPU-friendly**: The embedding model runs on CPU for single queries. No GPU required.
- **Deterministic**: Same input always produces the same output.
- **Top-k support**: Returns multiple topic candidates with similarity scores, enabling confidence-based decision making.

### Why keep BERTopic transform as a fallback?

BERTopic's native `transform()` method is retained as a validation/fallback method. It provides a second opinion on topic assignment and can be used when GPU is available. The two methods can be compared via `scripts/compare_inference_methods.py` to measure agreement.

---

## Trade-offs / Decisions Not Taken

### MMR (Maximal Marginal Relevance) for topic representation

BERTopic supports MMR to diversify c-TF-IDF keywords by reducing redundancy. This was evaluated but not adopted because:

- The current c-TF-IDF keywords already produce clean, interpretable topic representations (see Topic 0: `identity, theft, fraudulent, accounts, victim`).
- Gemini receives representative documents in addition to keywords, providing context beyond the keyword list.
- MMR adds complexity to the pipeline without measurable improvement in Gemini's labeling quality.
- Simpler pipeline is easier to maintain and debug.

### BERTopic's RepresentationModel abstraction

BERTopic provides a `RepresentationModel` interface for custom label generation. This was not used because a custom `GeminiLabeler` class provides:

- **Batching**: Topics are labeled in configurable batches (default 15) to stay within API limits.
- **Retries**: Automatic retry with exponential backoff on API failures.
- **JSON validation**: Gemini returns structured JSON that is validated before use.
- **Prompt customization**: The prompt can be tuned independently of BERTopic's internals.
- **Deterministic outputs**: Each topic is labeled independently; no shared state between calls.

### Topic reduction

BERTopic supports merging similar topics after fitting. This was intentionally not applied because:

- Preserving topic granularity allows analysts to see fine-grained complaint patterns.
- Similar topics can be merged downstream (in dashboards or reporting) without losing information.
- The current number of topics (~20) is already manageable for human review.

---

## Lessons Learned

### Modular ML pipelines matter

The original exploration was done in Jupyter notebooks. Refactoring into a modular `src/` package with a CLI entry point made the pipeline reproducible, testable, and deployable. Each module can be developed, tested, and debugged independently.

### Separating offline and online stages

Training (offline) and inference (online) have fundamentally different requirements. Training needs GPU, large memory, and all data. Inference needs low latency, CPU compatibility, and minimal dependencies. Designing for this separation from the start prevents architectural debt.

### Artifact management is infrastructure

Artifacts are the contract between training and inference. Versioning, validation, and documentation of artifacts is as important as the ML code itself. The validation module (`src/validation/validate.py`) checks artifact integrity before inference runs.

### Reproducibility requires discipline

- `config.yaml` captures every parameter -- no magic numbers in code.
- `RunMetadata` records the full config snapshot and execution context.
- Random seeds are set explicitly in config.
- The pipeline is fully scripted -- no manual steps.

### Deployment considerations shape architecture

Designing for deployment from the start (rather than as an afterthought) influenced every major decision: centroid similarity over BERTopic transform, config-driven paths, artifact validation, and separation of training and inference code paths.

### Balancing complexity and value

Every feature was evaluated against the question: "Does this meaningfully improve the output or the developer experience?" MMR, RepresentationModel, topic reduction, and orchestration frameworks were all considered and rejected because they added complexity without proportional benefit.

---

## Future Improvements

- **Online learning**: Incrementally update centroids as new complaints arrive without full retraining.
- **Confidence thresholds**: Reject low-confidence predictions and flag them for human review.
- **REST API**: Wrap the inference pipeline in a FastAPI endpoint for production serving.
- **CI/CD**: Add GitHub Actions for automated testing and artifact validation.
- **Monitoring**: Track prediction drift over time -- do new complaints map to existing topics or form new patterns?
- **Multi-language support**: Extend preprocessing to handle non-English complaints (lingua-language-detector is already integrated).

---

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
|---|---|---|
| `GEMINI_API_KEY` | Topic labeling | Google Gemini API key (set in `.env` file) |
| `HF_TOKEN` | Model download | Hugging Face token (optional, for rate limits) |

---

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

---

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

---

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

---

## Dataset

- **Source**: CFPB Consumer Complaint Database
- **Raw file**: `data/raw/complaints.csv`
- **Sample**: 50,000 complaints (randomly sampled, verified to match original product distribution)
- **Deduplicated**: 35,993 unique complaint narratives after removing exact duplicates
- **Text column**: `Consumer complaint narrative`
- **Language**: ~100% English (lingua-language-detector verified)

---

## Critical Design Note: Cluster ID Scheme

This pipeline uses **precomputed** clustering -- HDBSCAN is fit once outside of BERTopic, and its labels are passed into BERTopic via `BaseCluster` rather than letting BERTopic re-cluster. This allows UMAP/HDBSCAN hyperparameters to be tuned and validated independently using DBCV before topic modeling runs.

**Important**: `cluster_labels.npy` on disk stores the **raw HDBSCAN output** (not renumbered). BERTopic renumbers topics by descending cluster size during `fit_transform()` (largest cluster becomes Topic `0`, etc.), but the renumbered labels are only kept in memory and used to compute `topic_centroids.npy`. The centroids are keyed by these renumbered IDs, and `topic_lookup.csv` maps the same renumbered IDs to labels.

This means `cluster_labels.npy` (raw HDBSCAN IDs) and `topic_centroids.npy` (renumbered IDs) use **different ID schemes**. The centroids and lookup table are the authoritative source for inference; `cluster_labels.npy` is only needed if you want to reproduce or debug the raw clustering output.

---

## Known Issues

### Asymmetric embedding model: `encode()` vs `encode_query()`

The embedding model (`codefuse-ai/F2LLM-1.7B`) is an asymmetric model -- it uses different internal pooling for documents vs. queries. The current inference code uses `model.encode()` for both training corpus embeddings and inference-time query embeddings. Using `encode()` for inference queries instead of `encode_query()` can produce a distribution mismatch against the corpus embeddings, leading to confidently wrong topic assignments -- this was observed empirically during testing. The fix is a one-line change in `src/inference/predict.py`: replace `model.encode()` with `model.encode_query()` for the inference path.

### BERTopic transform comparison requires `calculate_probabilities: true`

The `scripts/compare_inference_methods.py` script compares centroid similarity against BERTopic's native transform. Since `calculate_probabilities` is set to `false` in `config.yaml`, BERTopic transform only returns a placeholder score (0.0) rather than a true probability. For a meaningful comparison, temporarily set `calculate_probabilities: true` in `config.yaml` and re-run the training pipeline (Step 5: topic_modeling).
