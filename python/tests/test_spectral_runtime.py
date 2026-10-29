from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location(
    "check_spectral_runtime", SCRIPTS / "check_spectral_runtime.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _runtime_response(**overrides: Any) -> dict[str, Any]:
    response: dict[str, Any] = {
        "runtime": True,
        "fallback": False,
        "warmups": 1,
        "runs": 5,
        "warmup_seconds": 0.5,
        "seconds": [1.0, 1.1, 1.2, 1.3, 1.4],
        "n": 4,
        "dimensions": 2,
        "unseeded": True,
    }
    response.update(overrides)
    return response


def _graph() -> dict[str, Any]:
    return {
        "n": 4,
        "heads": np.asarray([0, 0, 0, 1, 1, 2]),
        "tails": np.asarray([1, 2, 3, 2, 3, 3]),
        "weights": np.ones(6),
    }


def _mock_native_child(
    monkeypatch: pytest.MonkeyPatch,
    response: dict[str, Any],
    stderr: str = "",
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def completed(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        env = kwargs["env"]
        request_path = Path(env["CUSCAN_SPECTRAL_REQUEST"])
        response_path = Path(env["CUSCAN_SPECTRAL_RESPONSE"])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        response_path.write_text(json.dumps(response), encoding="utf-8")
        calls.append(
            {
                "command": command,
                "request": request,
                "timeout": kwargs["timeout"],
                "check": kwargs["check"],
                "capture_output": kwargs["capture_output"],
            }
        )
        return subprocess.CompletedProcess(command, 0, "probe complete", stderr)

    monkeypatch.setattr(module.subprocess, "run", completed)
    return calls


def test_native_probe_uses_runtime_contract_and_expected_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _mock_native_child(monkeypatch, _runtime_response())
    result = module._invoke_runtime(_graph(), 2, tmp_path / "fake-runner")

    assert result["seconds"] == [1.0, 1.1, 1.2, 1.3, 1.4]
    assert result["warmup_seconds"] == 0.5
    assert calls[0]["request"]["runtime"] is True
    assert calls[0]["request"]["seed"] == 42
    assert calls[0]["request"]["dimensions"] == 2
    assert calls[0]["command"] == [
        str(tmp_path / "fake-runner"),
        "--ignored",
        "--exact",
        module.PROBE_TEST,
        "--nocapture",
    ]
    assert calls[0]["timeout"] == 300
    assert calls[0]["check"] is False
    assert calls[0]["capture_output"] is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"runtime": False}, "runtime must be True", id="runtime-flag"),
        pytest.param({"fallback": True}, "fallback must be False", id="fallback-flag"),
        pytest.param({"unseeded": False}, "unseeded must be True", id="seeded-flag"),
        pytest.param({"warmups": 0}, "warmups must equal 1", id="warmup-count"),
        pytest.param({"runs": 4}, "runs must equal 5", id="run-count"),
        pytest.param({"n": 5}, "n must equal 4", id="vertex-count"),
        pytest.param({"dimensions": 1}, "dimensions must equal 2", id="dimension-count"),
        pytest.param({"seconds": [1.0] * 4}, "exactly 5 values", id="timing-count"),
        pytest.param(
            {"seconds": [1.0, 1.0, 1.0, float("nan"), 1.0]},
            "finite positive number",
            id="nonfinite-run-time",
        ),
        pytest.param(
            {"seconds": [1.0, 1.0, 1.0, -0.1, 1.0]},
            "finite positive number",
            id="negative-run-time",
        ),
        pytest.param({"warmup_seconds": 0}, "finite positive number", id="invalid-warmup-time"),
    ],
)
def test_native_probe_fails_closed_on_invalid_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    message: str,
) -> None:
    _mock_native_child(monkeypatch, _runtime_response(**overrides))

    with pytest.raises(module.HarnessError, match=message):
        module._invoke_runtime(_graph(), 2, tmp_path / "fake-runner")


def test_native_probe_fails_closed_when_stderr_reports_seeded_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_native_child(monkeypatch, _runtime_response(), "using seeded uniform fallback")

    with pytest.raises(module.HarnessError, match="seeded-uniform fallback"):
        module._invoke_runtime(_graph(), 2, tmp_path / "fake-runner")


def test_scipy_raw_validation_rejects_nonorthogonal_eigenvectors() -> None:
    operator, _ = module.symmetric_operator(_graph())
    values, vectors = np.linalg.eigh(operator.toarray())
    order = np.argsort(values)[::-1][:3]
    values = values[order]
    vectors = vectors[:, order]
    vectors[:, 2] = vectors[:, 1]

    with pytest.raises(module.HarnessError, match="validation exceeded limits"):
        module._validate_reference(values, vectors, operator)


def test_scipy_raw_validation_rejects_bad_residual() -> None:
    operator, _ = module.symmetric_operator(_graph())
    values, vectors = np.linalg.eigh(operator.toarray())
    order = np.argsort(values)[::-1][:3]
    values = values[order]
    vectors = vectors[:, order]
    values[1] += 0.01

    with pytest.raises(module.HarnessError, match="validation exceeded limits"):
        module._validate_reference(values, vectors, operator)


@pytest.mark.parametrize(
    ("scipy_component_seconds", "expected_ratio", "passes"),
    [
        pytest.param([25.25] * 5, 2.0, True, id="at-runtime-boundary"),
        pytest.param([25.0] * 5, 2.02, False, id="above-runtime-boundary"),
    ],
)
def test_graph_aggregation_sums_matched_runs_before_medians(
    scipy_component_seconds: list[float], expected_ratio: float, passes: bool
) -> None:
    rust_times = ([1.0, 1.0, 1.0, 100.0, 100.0], [100.0, 100.0, 3.0, 3.0, 3.0])
    components = [
        {
            "rust": {"seconds": values, "warmup_seconds": 0.1},
            "scipy": {"seconds": scipy_component_seconds, "warmup_seconds": 0.2},
        }
        for values in rust_times
    ]
    result = module._aggregate_graph(
        "graph",
        _graph(),
        2,
        Path("graph.npz"),
        "hash",
        [("component-1", {}), ("component-2", {})],
        [],
        components,
    )

    assert [run["rust_seconds_sum"] for run in result["matched_runs"]] == [
        101.0,
        101.0,
        4.0,
        103.0,
        103.0,
    ]
    assert result["rust_median_seconds"] == 101.0
    assert result["scipy_median_seconds"] == 2 * scipy_component_seconds[0]
    assert result["median_rust_over_scipy"] == pytest.approx(expected_ratio)
    assert result["passes_runtime_gate"] is passes


def test_component_split_remaps_vertices_and_skips_small_components() -> None:
    graph = {
        "n": 6,
        "heads": np.asarray([0, 2, 3, 1]),
        "tails": np.asarray([2, 3, 4, 5]),
        "weights": np.ones(4),
    }

    eligible, skipped = module._component_cases("graph", graph, dimensions=2)

    assert len(eligible) == 1
    name, component = eligible[0]
    assert name == "graph/component-4"
    assert component["n"] == 4
    assert np.array_equal(component["heads"], [0, 1, 2])
    assert np.array_equal(component["tails"], [1, 2, 3])
    assert np.array_equal(component["weights"], [1.0, 1.0, 1.0])
    assert skipped == [{"name": "graph/component-2", "n": 2, "reason": "n <= dimensions + 1"}]
