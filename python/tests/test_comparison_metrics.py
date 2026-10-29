from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

evaluate_module = importlib.import_module("comparison.evaluate")
metrics_module = importlib.import_module("comparison.metrics")
evaluate = evaluate_module.evaluate
parse_args = evaluate_module.parse_args
EVALUATION_SEED = metrics_module.EVALUATION_SEED
embedding_metrics = metrics_module.embedding_metrics
representative_indices = metrics_module.representative_indices
unique_rows = metrics_module.unique_rows


def test_identity_embedding_has_perfect_metrics() -> None:
    x = np.random.default_rng(41).normal(size=(48, 5)).astype(np.float32)
    assert embedding_metrics(x, x) == {
        "trustworthiness_at_15": 1.0,
        "neighbor_overlap_at_15": 1.0,
    }


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_metrics_reject_nonfinite_values(bad: float) -> None:
    x = np.random.default_rng(7).normal(size=(40, 3))
    x[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        embedding_metrics(x, np.nan_to_num(x))


def test_representatives_are_seeded_without_replacement() -> None:
    first = representative_indices(100, count=20, seed=EVALUATION_SEED)
    assert np.array_equal(first, representative_indices(100, count=20, seed=EVALUATION_SEED))
    assert len(np.unique(first)) == 20
    assert np.all((first >= 0) & (first < 100))


def test_unique_rows_returns_first_and_inverse_mappings() -> None:
    x = np.asarray([[3, 4], [1, 2], [3, 4], [5, 6], [1, 2]], dtype=np.float32)
    values, first, inverse = unique_rows(x)
    assert np.array_equal(values, [[1, 2], [3, 4], [5, 6]])
    assert np.array_equal(first, [1, 0, 3])
    assert np.array_equal(values[inverse], x)
    assert np.array_equal(x[first], values)


def test_cli_configuration_and_gate_states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    matrix = np.random.default_rng(3).normal(size=(40, 4)).astype("<f4")
    input_path = tmp_path / "rows.f32"
    matrix.tofile(input_path)
    original_run = subprocess.run

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if len(command) < 5:
            return original_run(command, **_kwargs)
        target = Path(command[4])
        source = Path(command[3])
        x = np.load(source, allow_pickle=False)
        np.save(target, x[:, :2], allow_pickle=False)
        return subprocess.CompletedProcess(
            command, 0, json.dumps({"fit_seconds": [0.01], "warnings": []}), ""
        )

    monkeypatch.setattr(evaluate_module.subprocess, "run", fake_run)

    args = parse_args(
        [
            "--input",
            str(input_path),
            "--rows",
            "40",
            "--features",
            "4",
            "--device",
            "cpu",
            "--seeds",
            "5,6",
            "--representatives",
            "31",
            "--methods",
            "rust-unique,python-unique",
            "--output",
            str(tmp_path / "result.json"),
            "--min-trustworthiness",
            "0",
            "--max-runtime-seconds",
            "1000",
        ]
    )
    report = evaluate(args)
    assert report["configuration"]["seeds"] == [5, 6]
    assert report["configuration"]["representative_count"] == 31
    assert report["input"]["raw_rows"] == 40
    assert report["input"]["unique_rows"] == 40
    assert len(report["runs"]["rust-unique"]) == 2
    assert "deduplication_seconds" not in report["runs"]["rust-unique"][0]["stage_times"]
    assert report["runs"]["rust-unique"][0]["unique_strategy"] == "native unique fit"
    assert report["runs"]["rust-unique"][0]["embedding_rows"] == 40
    assert report["gates"]["rust-unique"]["trustworthiness_at_15"]["status"] == "pass"
    assert (
        report["gates"]["python-unique"]["neighbor_overlap_at_15"]["status"]
        == "pending_approval_not_evaluated"
    )
    assert (tmp_path / "result.json").is_file()
    assert (tmp_path / "result.html").is_file()


def test_native_stderr_spectral_fallback_is_failed_and_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = np.random.default_rng(4).normal(size=(40, 4)).astype("<f4")
    input_path = tmp_path / "rows.f32"
    matrix.tofile(input_path)

    def fallback_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        target = Path(command[4])
        x = np.load(command[3], allow_pickle=False)
        np.save(target, x[:, :2], allow_pickle=False)
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({"fit_seconds": [0.01], "warnings": []}),
            "[cuscan] WARNING: spectral initialization did not converge; "
            "using seeded uniform initialization",
        )

    monkeypatch.setattr(evaluate_module.subprocess, "run", fallback_run)
    args = parse_args(
        [
            "--input",
            str(input_path),
            "--rows",
            "40",
            "--features",
            "4",
            "--representatives",
            "31",
            "--methods",
            "rust-raw",
            "--output",
            str(tmp_path / "failed.json"),
        ]
    )
    with pytest.raises(RuntimeError, match="fallback"):
        evaluate(args)
    evidence = json.loads((tmp_path / "failed.json").read_text())
    assert evidence["status"] == "failed"
    assert "using seeded uniform" in evidence["failures"][0]["stderr"]


def test_child_timeout_raises_with_timeout_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout_run(_command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired("python", 1, output="partial", stderr="native partial")

    monkeypatch.setattr(evaluate_module.subprocess, "run", timeout_run)
    with pytest.raises(evaluate_module.ChildFitError, match="timeout") as error:
        evaluate_module._fit_in_child(
            np.ones((3, 2)), method="rust-raw", device="cpu", seed=1, n_epochs=2, timeout=1
        )
    assert error.value.evidence["status"] == "timeout"
    assert error.value.evidence["stderr"] == "native partial"


def test_cli_rejects_invalid_method_configuration() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input",
                "x",
                "--rows",
                "40",
                "--features",
                "4",
                "--output",
                "o",
                "--methods",
                "rust",
            ]
        )
