from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

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


def test_warmed_summary_uses_per_run_residuals() -> None:
    result = {
        "fit_seconds": [100, 10, 20, 30],
        "stage_seconds": {
            "core_distances": [90, 1, 15, 2],
            "cluster_extraction": [5, 8, 1, 3],
        },
    }
    summary = module.summarize(result, range(1, 4))
    assert summary["total"] == 20000
    assert summary["core_distances"] == 2000
    assert summary["cluster_extraction"] == 3000
    assert summary["unattributed"] == 4000


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
            "embedding": [[i, -i, i + 100] for i in range(20)],
        }
    ).write_parquet(path)
    small = module.load_parquet(path, 8, "2d", 42)
    full = module.load_parquet(path, 8, "full", 42)
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
