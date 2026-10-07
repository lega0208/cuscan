from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "compare_umap_spans", Path(__file__).resolve().parents[2] / "scripts/compare_umap_spans.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Deadline = module.Deadline
DeadlineExceeded = module.DeadlineExceeded
instrument_umap = module.instrument_umap
parse_chrome_trace = module.parse_chrome_trace


def test_chrome_trace_inclusive_durations_for_complete_and_nested_events() -> None:
    parsed = parse_chrome_trace(
        {
            "traceEvents": [
                {"name": "fit", "ph": "B", "ts": 1_000, "pid": 1, "tid": 2},
                {"name": "optimization", "ph": "B", "ts": 2_000, "pid": 1, "tid": 2},
                {"name": "layout_epoch", "ph": "X", "ts": 3_000, "dur": 2_000},
                {"name": "optimization", "ph": "E", "ts": 7_000, "pid": 1, "tid": 2},
                {"name": "fit", "ph": "E", "ts": 11_000, "pid": 1, "tid": 2},
            ]
        }
    )
    assert parsed["inclusive_seconds_by_span"] == {
        "fit": pytest.approx(0.010),
        "layout_epoch": pytest.approx(0.002),
        "optimization": pytest.approx(0.005),
    }
    inclusive = parsed["inclusive_seconds_by_span"]
    assert inclusive["optimization"] > inclusive["layout_epoch"]
    assert "must not be summed" in parsed["note"]


def test_instrumentation_wraps_actual_call_and_restores_on_exception() -> None:
    def pairwise(value: int) -> int:
        return value + 1

    original = pairwise
    module = SimpleNamespace(pairwise_distances=pairwise)
    samples: dict[str, list[float]] = defaultdict(list)
    with pytest.raises(RuntimeError, match="stop"):
        with instrument_umap(module, samples):
            assert module.pairwise_distances(4) == 5
            raise RuntimeError("stop")
    assert module.pairwise_distances is original
    assert len(samples["pairwise_distances"]) == 1
    assert samples["pairwise_distances"][0] >= 0


def test_instrumentation_restores_after_wrapped_function_raises() -> None:
    def fails() -> None:
        raise ValueError("original failure")

    module = SimpleNamespace(pairwise_distances=fails)
    original = module.pairwise_distances
    samples: dict[str, list[float]] = defaultdict(list)
    with instrument_umap(module, samples), pytest.raises(ValueError, match="original failure"):
        module.pairwise_distances()
    assert module.pairwise_distances is original
    assert len(samples["pairwise_distances"]) == 1


def test_deadline_rejects_elapsed_budget() -> None:
    deadline = Deadline(1.0)
    deadline.ends_at = 0.0
    with pytest.raises(DeadlineExceeded, match="deadline expired"):
        deadline.remaining()


def test_chrome_native_array_and_thread_local_nesting() -> None:
    parsed = parse_chrome_trace(
        [
            {"name": "fit", "ph": "B", "ts": 0, "pid": 1, "tid": 1},
            {"name": "other", "ph": "B", "ts": 500, "pid": 1, "tid": 2},
            {"ph": "E", "ts": 1500, "pid": 1, "tid": 2},
            {"ph": "E", "ts": 2000, "pid": 1, "tid": 1},
        ]
    )
    assert parsed["inclusive_seconds_by_span"] == {"fit": 0.002, "other": 0.001}
    assert parsed["unmatched_begin_events"] == 0
    assert parsed["unmatched_end_events"] == 0
