from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "compare_umap_spans", Path(__file__).resolve().parents[2] / "scripts/compare_umap_spans.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_deadline_rejects_elapsed_budget() -> None:
    deadline = module.Deadline(1.0)
    deadline.ends_at = 0.0
    with pytest.raises(module.DeadlineExceeded, match="deadline expired"):
        deadline.remaining()


@pytest.mark.parametrize("seconds", [0, -1])
def test_deadline_requires_positive_budget(seconds: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        module.Deadline(seconds)


def test_worker_timeout_is_reported_as_incomplete_deadline() -> None:
    with pytest.raises(module.DeadlineExceeded, match="during worker execution") as error:
        module.run_with_deadline(
            [sys.executable, "-c", "import time; time.sleep(30)"], module.Deadline(0.1)
        )
    assert isinstance(error.value.__cause__, subprocess.TimeoutExpired)


def test_expiry_during_final_diagnostic_cannot_mark_suite_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sklearn.manifold

    clock = [0.0]
    calls = [0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module, "PRIMARY", (32, 2, 20))
    monkeypatch.setattr(module, "SMALL", ())

    def trustworthiness(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 4:
            clock[0] = 601.0
        return 0.95

    monkeypatch.setattr(sklearn.manifold, "trustworthiness", trustworthiness)
    embedding = np.zeros((32, 2), dtype="<f4")
    graph = (np.arange(32)[:, None] + np.arange(1, 15)[None, :]) % 32

    def worker(command, deadline):
        deadline.remaining()
        if "--_python-worker" in command:
            destination = Path(command[-1])
            np.save(destination / "embedding_0.npy", embedding)
            np.save(destination / "knn_0.npy", graph)
        else:
            destination = Path(command[5])
            destination.mkdir()
            embedding.tofile(destination / "embedding_0.f32le")
            graph.astype("<i4").tofile(destination / "graph_0.i32le")
        return subprocess.CompletedProcess(
            command, 0, json.dumps({"fit_seconds": [0.1], "fit_graph_width": 14}), ""
        )

    monkeypatch.setattr(module, "run_with_deadline", worker)
    binary = tmp_path / "mock-binary"
    binary.touch()
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["suite", "--binary", str(binary), "--runs", "1", "--output", str(output)],
    )
    assert module._cli() == 1
    result = json.loads(output.read_text())
    assert calls[0] == 4
    assert result["complete"] is False
    assert "deadline expired" in result["incomplete_reason"]


def test_nonself_filters_self_and_missing_neighbors() -> None:
    actual = np.array([[0, -1, 2, 1], [2, 1, -1, 0], [-1, 0, 2, 1]])
    np.testing.assert_array_equal(module.nonself(actual, 2), [[2, 1], [2, 0], [0, 1]])
    with pytest.raises(ValueError, match="fewer than 3"):
        module.nonself(actual, 3)


def test_recall_counts_distinct_in_radius_neighbors_only() -> None:
    matrix = np.array([[0.0], [1.0], [10.0], [2.0]])
    actual = np.array([[1, 1, 2], [0, 0, 2], [1, 1, 0], [1, 1, 2]])
    # Repeated near neighbors must not disguise the distinct far neighbors.
    assert module.tie_aware_recall(matrix, actual, np.ones(4)) == pytest.approx(3 / 12)


def test_recall_accepts_alternate_neighbors_at_exact_distance_ties() -> None:
    matrix = np.array([[0.0], [-1.0], [1.0]])
    actual = np.array([[2], [0], [0]])
    assert module.tie_aware_recall(matrix, actual, np.ones(3)) == 1.0


@pytest.fixture
def reference() -> dict:
    return {
        "complete": True,
        "configuration": {
            "primary": {"rows": 5000, "features": 32, "epochs": 20},
            "neighbors": 15,
            "components": 2,
            "init": "random",
            "metric": "euclidean",
            "input": "seeded clustered float32 (8 centers), little-endian row-major",
        },
        "quality": {
            "exact": {"cuda_trustworthiness": [0.941, 0.943]},
            "approximate": {"cuda_trustworthiness": [0.942, 0.943]},
        },
    }


def test_reference_accepts_matched_complete_workload(tmp_path: Path, reference: dict) -> None:
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))
    assert module.load_quality_reference(path) == reference


@pytest.mark.parametrize(
    ("field", "value"),
    [("epochs", 10), ("rows", 1024), ("features", 16)],
)
def test_reference_rejects_mismatched_workloads(
    tmp_path: Path, reference: dict, field: str, value: int
) -> None:
    reference["configuration"]["primary"][field] = value
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="primary workload"):
        module.load_quality_reference(path)


@pytest.mark.parametrize("bad_samples", [[0.94], [True, 0.94], [float("nan"), 0.94]])
def test_reference_rejects_unusable_samples(
    tmp_path: Path, reference: dict, bad_samples: list
) -> None:
    reference["quality"]["approximate"]["cuda_trustworthiness"] = bad_samples
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="samples"):
        module.load_quality_reference(path)


def test_reference_rejects_incomplete_or_missing_metadata(tmp_path: Path, reference: dict) -> None:
    for bad in [dict(reference, complete=False), {}, copy.deepcopy(reference)]:
        if "configuration" in bad and bad.get("complete") is True:
            del bad["configuration"]["metric"]
        path = tmp_path / "reference.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            module.load_quality_reference(path)
