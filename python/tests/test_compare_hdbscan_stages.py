from __future__ import annotations

import importlib.util
from pathlib import Path

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
            "minimum_spanning_tree": [90, 1, 15, 2],
            "cluster_extraction": [5, 8, 1, 3],
        },
    }
    summary = module.summarize(result, range(1, 4))
    assert summary["total"] == 20000
    assert summary["minimum_spanning_tree"] == 2000
    assert summary["cluster_extraction"] == 3000
    assert summary["other"] == 4000


def test_unmeasured_stage_is_not_reported_as_zero() -> None:
    summary = module.summarize({"fit_seconds": [2], "stage_seconds": {}}, range(1))
    assert summary["minimum_spanning_tree"] is None
    assert summary["other"] == 2000


@pytest.mark.parametrize(
    "native, reference, expected", [(5, 10, 50), (20, 10, -100), (0, 10, 100), (1, 0, None)]
)
def test_speedup_matches_original_time_reduction(native, reference, expected) -> None:
    left = dict.fromkeys((*module.STAGES, "other", "total"), native)
    right = dict.fromkeys(left, reference)
    assert module.comparison_rows(left, right)[0]["speedup"] == expected


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
