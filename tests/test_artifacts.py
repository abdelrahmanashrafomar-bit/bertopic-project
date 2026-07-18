from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture
def config():
    root = Path(__file__).resolve().parent.parent
    with open(root / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def project_root():
    return Path(__file__).resolve().parent.parent


class TestConfig:
    def test_load_config(self, config):
        assert isinstance(config, dict)
        assert "paths" in config
        assert "preprocessing" in config
        assert "embedding" in config
        assert "inference" in config

    def test_paths_exist(self, config, project_root):
        for key, rel_path in config["paths"].items():
            full = project_root / rel_path
            assert full.parent.exists(), f"Parent dir missing for {key}: {full.parent}"


class TestValidators:
    def test_ensure_file_exists_raises(self):
        from src.validators import ensure_file_exists
        from pathlib import Path
        import pytest
        with pytest.raises(FileNotFoundError):
            ensure_file_exists(Path("/nonexistent/path"), "test")

    def test_ensure_equal_length_passes(self):
        from src.validators import ensure_equal_length
        ensure_equal_length(("a", 5), ("b", 5), ("c", 5))

    def test_ensure_equal_length_raises(self):
        from src.validators import ensure_equal_length
        import pytest
        with pytest.raises(ValueError, match="Length mismatch"):
            ensure_equal_length(("a", 5), ("b", 3))

    def test_ensure_required_columns_passes(self):
        from src.validators import ensure_required_columns
        ensure_required_columns(["a", "b", "c"], ["a", "b"])

    def test_ensure_required_columns_raises(self):
        from src.validators import ensure_required_columns
        import pytest
        with pytest.raises(ValueError, match="Missing required columns"):
            ensure_required_columns(["a", "b"], ["a", "c"])


class TestConfig:
    def test_load_config(self, config):
        assert isinstance(config, dict)
        assert "paths" in config
        assert "preprocessing" in config
        assert "embedding" in config
        assert "inference" in config

    def test_paths_resolve(self, config):
        from src.config import get_project_root
        root = get_project_root()
        for key, rel_path in config["paths"].items():
            full = root / rel_path
            assert isinstance(full, type(root / "x"))


class TestTopicPrediction:
    def test_topic_prediction_creation(self):
        from src.inference.schemas import TopicPrediction
        tp = TopicPrediction(topic_id=5, label="Test Label", score=0.95)
        assert tp.topic_id == 5
        assert tp.label == "Test Label"
        assert tp.score == 0.95

    def test_topic_prediction_repr(self):
        from src.inference.schemas import TopicPrediction
        tp = TopicPrediction(topic_id=0, label="Outliers", score=0.5)
        assert "topic_id=0" in repr(tp)


class TestArtifactLoading:
    def test_topic_lookup_loads(self):
        from src.config import get_project_root
        import pandas as pd
        root = get_project_root()
        path = root / "artifacts" / "topic_lookup.csv"
        assert path.exists(), f"File not found: {path}"
        df = pd.read_csv(path)
        assert "Topic" in df.columns
        assert "Topic_Label" in df.columns
        assert len(df) > 0

    def test_labels_csv_loads(self):
        from src.config import get_project_root
        import pandas as pd
        root = get_project_root()
        path = root / "artifacts" / "labels.csv"
        assert path.exists(), f"File not found: {path}"
        df = pd.read_csv(path)
        assert "Topic" in df.columns
        assert "Topic_Label" in df.columns

    def test_topic_centroids_loads(self):
        from src.config import get_project_root
        import numpy as np
        root = get_project_root()
        path = root / "artifacts" / "topic_centroids.npy"
        assert path.exists(), f"File not found: {path}"
        centroids = np.load(path, allow_pickle=True).item()
        assert isinstance(centroids, dict)
        assert len(centroids) > 0
        for tid, vec in centroids.items():
            assert isinstance(tid, (int, np.integer))
            assert isinstance(vec, np.ndarray)

    def test_cluster_labels_loads(self):
        from src.config import get_project_root
        import numpy as np
        root = get_project_root()
        path = root / "artifacts" / "cluster_labels.npy"
        assert path.exists(), f"File not found: {path}"
        labels = np.load(path)
        assert labels.ndim == 1
        assert len(labels) > 0

    def test_embeddings_loads(self):
        from src.config import get_project_root
        import numpy as np
        root = get_project_root()
        path = root / "artifacts" / "embeddings.npy"
        assert path.exists(), f"File not found: {path}"
        emb = np.load(path, mmap_mode="r")
        assert emb.ndim == 2
        assert emb.shape[0] > 0


class TestTopicMapping:
    def test_topic_id_to_label(self):
        from src.config import get_project_root
        import pandas as pd
        root = get_project_root()
        lookup = pd.read_csv(root / "artifacts" / "topic_lookup.csv")
        mapping = lookup.set_index("Topic")["Topic_Label"].to_dict()
        assert -1 in mapping
        assert mapping[-1] == "Outliers / Unclassified"
        assert 0 in mapping
        assert isinstance(mapping[0], str)
        assert len(mapping[0]) > 0

    def test_labels_csv_matches_lookup(self):
        from src.config import get_project_root
        import pandas as pd
        root = get_project_root()
        lookup = pd.read_csv(root / "artifacts" / "topic_lookup.csv")
        labels = pd.read_csv(root / "artifacts" / "labels.csv")
        merged = lookup.merge(labels, on="Topic", suffixes=("_lookup", "_labels"))
        mismatches = merged[merged["Topic_Label_lookup"] != merged["Topic_Label_labels"]]
        assert len(mismatches) == 0, f"{len(mismatches)} label mismatches between CSVs"

    def test_all_topic_ids_have_labels(self):
        from src.config import get_project_root
        import pandas as pd
        import numpy as np
        root = get_project_root()
        lookup = pd.read_csv(root / "artifacts" / "topic_lookup.csv")
        centroids = np.load(root / "artifacts" / "topic_centroids.npy", allow_pickle=True).item()
        lookup_ids = set(lookup["Topic"])
        centroid_ids = set(centroids.keys())
        missing = centroid_ids - lookup_ids
        assert len(missing) == 0, f"Centroid topics missing from lookup: {missing}"


class TestCLI:
    def test_help(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "main", "--help"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0
        assert "--step" in result.stdout

    def test_validate_step(self):
        import subprocess
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [sys.executable, "-m", "main", "--step", "validate"],
            capture_output=True, text=True, cwd=root, timeout=60,
        )
        assert result.returncode == 0
        assert "VALIDATION REPORT" in result.stdout
        assert "Total:" in result.stdout
