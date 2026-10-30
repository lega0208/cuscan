from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest

spec = importlib.util.spec_from_file_location(
    "compare_hdbscan_stages",
    Path(__file__).resolve().parents[2] / "comparison/compare_hdbscan_stages.py",
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_measured_summary_uses_per_run_residuals() -> None:
    result = {
        "warmup_seconds": 100,
        "fit_seconds": [10, 20, 30],
        "stage_seconds": {
            "core_distances": [1, 15, 2],
            "cluster_extraction": [8, 1, 3],
        },
    }
    summary = module.summarize(result, range(3))
    assert summary["total"] == 20000
    assert summary["core_distances"] == 2000
    assert summary["cluster_extraction"] == 3000
    assert summary["unattributed"] == 4000


@pytest.mark.parametrize("runs", [1, 3])
def test_summary_selects_exact_measured_fit_count(runs: int) -> None:
    result = {
        "warmup_seconds": 1000,
        "fit_seconds": [10, 20, 30],
        "stage_seconds": {},
    }
    summary = module.summarize(result, range(runs))
    assert summary["total"] == (10 if runs == 1 else 20) * 1000


def test_single_measured_sample_is_reported_as_its_sample() -> None:
    summary = module.summarize({"fit_seconds": [7.25], "stage_seconds": {}}, range(1))
    assert summary["total"] == 7250


def test_label_export_writes_integer_list_and_digest(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    metadata = module.write_labels(path, np.array([2, -1, 0], dtype=np.int64))
    assert json.loads(path.read_text()) == [2, -1, 0]
    assert metadata == {
        "labels_path": str(path),
        "labels_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_worker_parser_exposes_labels_output_attribute(monkeypatch, tmp_path: Path) -> None:
    worker_input = tmp_path / "input.f32le"
    labels_output = tmp_path / "labels.json"
    captured = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_hdbscan_stages.py",
            "--_python-worker",
            str(worker_input),
            "--_worker-rows",
            "4",
            "--_worker-features",
            "2",
            "--_worker-runs",
            "1",
            "--_worker-labels-output",
            str(labels_output),
        ],
    )
    monkeypatch.setattr(module, "_python_worker", captured.append)
    assert module.main() == 0
    args = captured[0]
    assert args.worker_labels_output == labels_output
    assert args.worker_input == worker_input
    assert (args.rows, args.features, args.runs) == (4, 2, 1)


@pytest.mark.parametrize("runs", [1, 3])
@pytest.mark.parametrize("fail", [False, True])
def test_python_worker_runs_warmup_and_measured_fits(
    monkeypatch, capsys, tmp_path: Path, runs: int, fail: bool
) -> None:
    worker_input = tmp_path / "input.f32le"
    np.arange(8, dtype="<f4").tofile(worker_input)
    labels_path = tmp_path / "labels.json"
    fit_calls = []
    labels_by_fit = []

    fast_module: Any = ModuleType("fast_hdbscan")
    fast_module.__path__ = []
    reference: Any = ModuleType("fast_hdbscan.hdbscan")
    approximate: Any = ModuleType("fast_hdbscan.nndescent")
    approximate.compute_mst_from_knn_graph = lambda: None

    def compute_mst():
        reference.build_kdtree()
        reference.parallel_boruvka()

    def original_fast_hdbscan():
        reference.compute_minimum_spanning_tree()
        reference.clusters_from_spanning_tree()

    reference.build_kdtree = lambda: None
    reference.parallel_boruvka = lambda: None
    reference.clusters_from_spanning_tree = lambda: None
    reference.compute_minimum_spanning_tree = compute_mst
    reference.fast_hdbscan = original_fast_hdbscan
    fast_module.hdbscan = reference
    fast_module.nndescent = approximate

    class Estimator:
        def __init__(self, **kwargs):
            self.labels_ = None

        def fit(self, matrix):
            fit_calls.append(matrix.copy())
            reference.fast_hdbscan()
            if fail and len(fit_calls) == 2:
                raise RuntimeError("expected fit failure")
            self.labels_ = np.array([len(fit_calls), -1, 0, 1])
            labels_by_fit.append(self.labels_.copy())

    fast_module.HDBSCAN = Estimator
    numba = SimpleNamespace(
        get_num_threads=lambda: 3,
        threading_layer=lambda: "workqueue",
        config=SimpleNamespace(THREADING_LAYER="default"),
    )
    monkeypatch.setitem(sys.modules, "fast_hdbscan", fast_module)
    monkeypatch.setitem(sys.modules, "fast_hdbscan.hdbscan", reference)
    monkeypatch.setitem(sys.modules, "fast_hdbscan.nndescent", approximate)
    monkeypatch.setitem(sys.modules, "numba", numba)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "0.3.2")
    ticks = iter(float(value) for value in range(100))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))
    original_hooks = vars(reference).copy()
    args = SimpleNamespace(
        worker_input=worker_input,
        rows=4,
        features=2,
        runs=runs,
        min_cluster_size=2,
        min_samples=2,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        worker_labels_output=labels_path,
    )

    if fail:
        with pytest.raises(RuntimeError, match="expected fit failure"):
            module._python_worker(args)
        assert vars(reference) == original_hooks
        assert len(fit_calls) == 2
        return

    module._python_worker(args)
    output = json.loads(capsys.readouterr().out)
    assert len(fit_calls) == runs + 1
    assert len(output["fit_seconds"]) == runs
    assert output["warmup_seconds"] > 0
    assert all(value > 0 for value in output["fit_seconds"])
    assert output["numba_threads"] == 3
    assert output["numba_threading_layer"] == "workqueue"
    assert json.loads(labels_path.read_text()) == labels_by_fit[-1].tolist()
    assert vars(reference) == original_hooks


def test_main_passes_label_outputs_and_records_label_metadata(monkeypatch, tmp_path: Path):
    binary = tmp_path / "native"
    binary.touch()
    json_path = tmp_path / "result.json"
    labels_dir = tmp_path / "labels"
    calls = []

    def fake_run(command, label):
        calls.append((command, label))
        if label == "native":
            path = Path(command[command.index("--labels-output") + 1])
            metadata = module.write_labels(path, np.array([1, -1]))
            return {"fit_seconds": [2.0], "stage_seconds": {}, **metadata}
        path = Path(command[command.index("--_worker-labels-output") + 1])
        metadata = module.write_labels(path, np.array([1, -1]))
        return {
            "fit_seconds": [3.0],
            "stage_seconds": {},
            "numba_threads": 2,
            "numba_threading_layer": "workqueue",
            **metadata,
        }

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_hdbscan_stages.py",
            "--rows",
            "4",
            "--features",
            "2",
            "--binary",
            str(binary),
            "--labels-dir",
            str(labels_dir),
            "--json",
            str(json_path),
        ],
    )
    monkeypatch.setattr(
        module,
        "make_input",
        lambda rows, features, *args: np.ones((rows, features), dtype="<f4").tofile(args[-1]),
    )
    monkeypatch.setattr(module, "run", fake_run)
    monkeypatch.setattr(module, "validate_native_stages", lambda _: None)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "0.3.2")
    monkeypatch.setattr(module.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda _: {0, 2, 4}, raising=False)
    assert module.main() == 0
    assert "--labels-output" in calls[0][0]
    assert "--_worker-labels-output" in calls[1][0]
    result = json.loads(json_path.read_text())
    labels = result["metadata"]["labels"]
    assert (
        labels["native_sha256"]
        == hashlib.sha256((labels_dir / "native-labels.json").read_bytes()).hexdigest()
    )
    assert (
        labels["python_sha256"]
        == hashlib.sha256((labels_dir / "python-labels.json").read_bytes()).hexdigest()
    )
    assert labels["scope"].startswith(
        "labels exported from each implementation's last measured fit"
    )
    threads = result["metadata"]["thread_metadata"]
    assert threads["numba_threads"] == 2
    assert threads["native_cpu_count"] == 8
    assert threads["available_affinity_count"] == 3


def test_unmeasured_stage_is_not_reported_as_zero() -> None:
    summary = module.summarize({"fit_seconds": [2], "stage_seconds": {}}, range(1))
    assert summary["core_distances"] is None
    assert summary["unattributed"] == 2000


def test_exclusive_timer_pauses_parent_and_switches_to_finalization() -> None:
    now = [0.0]
    timer = module.ExclusiveTimer(clock=lambda: now[0])
    with timer.measure("input_validation"):
        now[0] += 2
        with timer.measure("tree_build"):
            now[0] += 3
        now[0] += 2
        timer.switch("result_finalization")
        now[0] += 3
    assert timer.seconds == {
        "input_validation": 4,
        "tree_build": 3,
        "result_finalization": 3,
    }
    assert sum(timer.seconds.values()) == now[0]


@pytest.mark.parametrize("fail", [False, True])
def test_reference_hooks_restore_after_success_or_failure(fail: bool) -> None:
    now = [0.0]
    timer = module.ExclusiveTimer(clock=lambda: now[0])

    def tree():
        now[0] += 3

    def fused():
        now[0] += 5

    reference = SimpleNamespace(
        build_kdtree=tree,
        parallel_boruvka=fused,
        clusters_from_spanning_tree=fused,
    )
    approximate = SimpleNamespace(compute_mst_from_knn_graph=fused)

    def compute():
        now[0] += 2
        reference.build_kdtree()
        reference.parallel_boruvka()
        if fail:
            raise RuntimeError("fit failed")
        return "mst"

    reference.compute_minimum_spanning_tree = compute
    originals = vars(reference).copy()
    try:
        with module.measure_reference(reference, approximate, timer):
            with timer.measure("input_validation"):
                assert reference.compute_minimum_spanning_tree() == "mst"
    except RuntimeError as error:
        assert fail and str(error) == "fit failed"
    assert vars(reference) == originals
    assert approximate.compute_mst_from_knn_graph is fused
    assert timer.seconds == {
        "input_validation": 0,
        "mst_setup": 2,
        "tree_build": 3,
        "core_distances_and_mst": 5,
    }


def test_missing_reference_hook_restores_partially_installed_hooks() -> None:
    reference = SimpleNamespace(build_kdtree=lambda: None)
    original = reference.build_kdtree
    with pytest.raises(RuntimeError, match="timing hook missing"):
        with module.measure_reference(reference, SimpleNamespace(), module.ExclusiveTimer()):
            pytest.fail("missing required hooks must fail before measuring")
    assert reference.build_kdtree is original


@pytest.mark.parametrize(
    "native, reference, expected", [(5, 10, 50), (20, 10, -100), (0, 10, 100), (1, 0, None)]
)
def test_speedup_matches_original_time_reduction(native, reference, expected) -> None:
    left = dict.fromkeys((*module.STAGES, "unattributed", "total"), native)
    right = dict.fromkeys(left, reference)
    assert module.comparison_rows(left, right)[0]["speedup"] == expected


@pytest.mark.parametrize("duration", [-1.0, float("nan"), float("inf")])
def test_rejects_invalid_stage_durations(duration: float) -> None:
    with pytest.raises(ValueError):
        module.summarize({"fit_seconds": [2.0], "stage_seconds": {"mst": [duration]}}, range(1))


def test_rejects_overlapping_or_excess_stage_durations() -> None:
    with pytest.raises(ValueError):
        module.summarize(
            {"fit_seconds": [2.0], "stage_seconds": {"mst": [1.5], "tree_build": [1.0]}},
            range(1),
        )


@pytest.mark.parametrize(
    "result",
    [
        {"fit_seconds": [float("nan")], "stage_seconds": {}},
        {"fit_seconds": [-1.0], "stage_seconds": {}},
        {"fit_seconds": [2.0], "stage_seconds": {"mst": []}},
        {"fit_seconds": [2.0], "stage_seconds": {"mst": [1.0, 1.0]}},
    ],
)
def test_rejects_malformed_run_arrays(result) -> None:
    with pytest.raises(ValueError):
        module.summarize(result, range(1))


def test_obsolete_profiler_stage_is_not_silently_ignored() -> None:
    with pytest.raises(ValueError):
        module.summarize(
            {"fit_seconds": [2.0], "stage_seconds": {"minimum_spanning_tree": [1.0]}},
            range(1),
        )


def test_stale_native_companion_requires_rebuild() -> None:
    with pytest.raises(SystemExit, match="rebuild the native companion"):
        module.validate_native_stages(
            {"stage_seconds": {"minimum_spanning_tree": [1.0], "cluster_extraction": [0.5]}}
        )


def test_fused_reference_stage_has_no_misleading_speedup() -> None:
    native = module.summarize(
        {"fit_seconds": [3.0], "stage_seconds": {"core_distances": [1.0], "mst": [2.0]}},
        range(1),
    )
    reference = module.summarize(
        {"fit_seconds": [2.0], "stage_seconds": {"core_distances_and_mst": [2.0]}}, range(1)
    )
    rows = {row["stage"]: row for row in module.comparison_rows(native, reference)}
    assert rows["core_distances"]["python_ms"] is None
    assert rows["mst"]["python_ms"] is None
    assert rows["core_distances_and_mst"]["native_ms"] is None
    assert all(
        rows[stage]["speedup"] is None
        for stage in ("core_distances", "mst", "core_distances_and_mst")
    )
    assert "minimum_spanning_tree" not in rows
    assert "other" not in rows
    assert native["unattributed"] == 0
    assert reference["unattributed"] == 0


def test_parquet_selection_and_reproducible_sampling(tmp_path: Path) -> None:
    path = tmp_path / "input.parquet"
    pl.DataFrame(
        {
            "embedding_2d": [[i, -i] for i in range(20)],
            "embedding_16d": [[i, -i, i + 100] for i in range(20)],
            "embedding": [[i, -i, i + 100] for i in range(20)],
        }
    ).write_parquet(path)
    small = module.load_parquet(path, 8, "2d", 42)
    full = module.load_parquet(path, 8, "16d", 42)
    assert small.shape == (8, 2)
    assert full.shape == (8, 3)
    np.testing.assert_array_equal(small, full[:, :2])
    np.testing.assert_array_equal(small, module.load_parquet(path, 8, "2d", 42))


@pytest.mark.parametrize(
    "values",
    [[], [[1, 2]], [[1, 2, 3], [4, 5, 6]], [[1.0, float("nan")], [2.0, 3.0]], [[1, 2], [3]]],
)
def test_rejects_invalid_visualization_column(tmp_path: Path, values) -> None:
    path = tmp_path / "input.parquet"
    pl.DataFrame({"embedding_2d": values}).write_parquet(path)
    with pytest.raises(ValueError):
        module.load_parquet(path, 20, "2d", 42)
