#!/usr/bin/env python3
"""Measure unseeded native spectral runtime against a SciPy reference.

Input loading, component splitting, normalized-operator construction, and the
SciPy worker's process startup are outside solver timings. The native probe
times Rust solve_spectral and its raw residual checks; SciPy timings cover only
eigsh/eigh acquisition, with raw checks outside those timings. Native timings
therefore conservatively include extra validation work. This is a runtime
harness, not a replacement for check_spectral.py's independent accuracy gates.
"""

import argparse
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import scipy
from check_spectral import graph_arrays, symmetric_operator
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

TIMEOUT_SECONDS = 300
WARMUPS = 1
RUNS = 5
DIMENSIONS = 2
SEED = 42
RESIDUAL_LIMIT = 1e-6
GRAM_LIMIT = 1e-8
RATIO_LIMIT = 2.0
PROBE_TEST = "standard_umap::init::solver_tests::external_spectral_probe"
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


class HarnessError(RuntimeError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}


def _threadpool_info():
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return {"available": False, "reason": "threadpoolctl is not installed"}
    try:
        return {"available": True, "pools": threadpool_info()}
    except Exception as exc:  # Information only; never alter the environment.
        return {"available": False, "error": repr(exc)}


def _sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _write_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _thread_environment():
    return {key: os.environ.get(key) for key in THREAD_ENV_KEYS}


def _component_cases(name, graph, dimensions):
    operator, _ = symmetric_operator(graph)
    count, labels = connected_components(operator, directed=False)
    eligible = []
    skipped = []
    if count == 1:
        if graph["n"] <= dimensions + 1:
            skipped.append({"name": name, "n": graph["n"], "reason": "n <= dimensions + 1"})
        else:
            eligible.append((name, graph))
        return eligible, skipped

    for label in range(count):
        vertices = np.flatnonzero(labels == label)
        component_name = f"{name}/component-{len(vertices)}"
        if len(vertices) <= dimensions + 1:
            skipped.append(
                {
                    "name": component_name,
                    "n": len(vertices),
                    "reason": "n <= dimensions + 1",
                }
            )
            continue
        remap = {int(vertex): index for index, vertex in enumerate(vertices)}
        mask = np.isin(graph["heads"], vertices) & np.isin(graph["tails"], vertices)
        child = {
            "n": len(vertices),
            "heads": np.asarray([remap[int(value)] for value in graph["heads"][mask]]),
            "tails": np.asarray([remap[int(value)] for value in graph["tails"][mask]]),
            "weights": graph["weights"][mask],
        }
        eligible.append((component_name, child))
    return eligible, skipped


def _positive_finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarnessError(f"runtime response {label} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise HarnessError(f"runtime response {label} must be a finite positive number")
    return result


def _exact_int(value, expected, label):
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise HarnessError(f"runtime response {label} must equal {expected}; got {value!r}")


def _invoke_runtime(graph, dimensions, test_runner):
    request = {
        "n": int(graph["n"]),
        "heads": np.asarray(graph["heads"], dtype=np.int64).tolist(),
        "tails": np.asarray(graph["tails"], dtype=np.int64).tolist(),
        "weights": [float(value) for value in np.asarray(graph["weights"]).reshape(-1)],
        "dimensions": int(dimensions),
        "seed": SEED,
        "runtime": True,
    }
    try:
        request_bytes = _json_bytes(request)
    except (TypeError, ValueError) as exc:
        raise HarnessError(f"cannot encode native request: {exc}") from exc

    with tempfile.TemporaryDirectory(prefix="fast-umap-spectral-runtime-") as directory:
        request_path = pathlib.Path(directory) / "graph.json"
        response_path = pathlib.Path(directory) / "result.json"
        request_path.write_bytes(request_bytes)
        command = [
            str(test_runner),
            "--ignored",
            "--exact",
            PROBE_TEST,
            "--nocapture",
        ]
        env = dict(os.environ)
        env["FAST_UMAP_SPECTRAL_REQUEST"] = str(request_path)
        env["FAST_UMAP_SPECTRAL_RESPONSE"] = str(response_path)
        try:
            child = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            evidence = {
                "command": command,
                "timeout_seconds": TIMEOUT_SECONDS,
                "stdout": _as_text(exc.stdout),
                "stderr": _as_text(exc.stderr),
            }
            raise HarnessError(
                f"native spectral runtime child exceeded {TIMEOUT_SECONDS}s", evidence
            ) from exc
        evidence = {
            "command": command,
            "returncode": child.returncode,
            "stdout": child.stdout,
            "stderr": child.stderr,
            "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        }
        if child.returncode != 0:
            raise HarnessError(
                f"native spectral runtime child exited with status {child.returncode}", evidence
            )
        if "using seeded uniform" in child.stderr.casefold():
            raise HarnessError("native stderr reports seeded-uniform fallback", evidence)
        if not response_path.is_file():
            raise HarnessError("native child passed without writing its response", evidence)
        try:
            result = json.loads(response_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HarnessError(f"cannot parse native runtime response: {exc}", evidence) from exc

        if not isinstance(result, dict):
            raise HarnessError("native runtime response must be a JSON object", evidence)
        expected_flags = {"runtime": True, "fallback": False, "unseeded": True}
        for field, expected in expected_flags.items():
            if result.get(field) is not expected:
                raise HarnessError(
                    f"native runtime response {field} must be {expected!r}; "
                    f"got {result.get(field)!r}",
                    evidence,
                )
        _exact_int(result.get("warmups"), WARMUPS, "warmups")
        _exact_int(result.get("runs"), RUNS, "runs")
        _exact_int(result.get("n"), int(graph["n"]), "n")
        _exact_int(result.get("dimensions"), int(dimensions), "dimensions")
        warmup_seconds = _positive_finite(result.get("warmup_seconds"), "warmup_seconds")
        seconds = result.get("seconds")
        if not isinstance(seconds, list) or len(seconds) != RUNS:
            raise HarnessError(
                f"native runtime response seconds must contain exactly {RUNS} values"
            )
        seconds = [
            _positive_finite(value, f"seconds[{index}]") for index, value in enumerate(seconds)
        ]
        return {
            "request_sha256": evidence["request_sha256"],
            "command": command,
            "returncode": child.returncode,
            "warmups": WARMUPS,
            "runs": RUNS,
            "warmup_seconds": warmup_seconds,
            "seconds": seconds,
            "stderr": child.stderr,
            "stdout": child.stdout,
            "response": result,
        }


def _as_text(value):
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _validate_reference(values, vectors, operator):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    vectors = np.asarray(vectors, dtype=np.float64)
    n = operator.shape[0]
    expected = DIMENSIONS + 1
    if vectors.shape != (n, expected) or values.shape != (expected,):
        raise HarnessError(
            f"SciPy raw eigenpair shape mismatch: values={values.shape}, vectors={vectors.shape}, "
            f"expected {(expected,)} and {(n, expected)}"
        )
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        raise HarnessError("SciPy raw eigenvalues or eigenvectors contain non-finite values")
    norms = np.linalg.norm(vectors, axis=0)
    norm_error = float(np.max(np.abs(norms - 1.0)))
    gram_error = vectors.T @ vectors - np.eye(expected)
    diagonal_error = float(np.max(np.abs(np.diag(gram_error))))
    off_diagonal = gram_error - np.diag(np.diag(gram_error))
    off_diagonal_error = float(np.max(np.abs(off_diagonal)))
    residuals = np.linalg.norm(operator @ vectors - vectors * values, axis=0) / np.maximum(
        1.0, np.abs(2.0 - values)
    )
    residual = float(np.max(residuals))
    metrics = {
        "max_raw_norm_error": norm_error,
        "max_gram_diagonal_error": diagonal_error,
        "max_gram_offdiagonal_error": off_diagonal_error,
        "max_normalized_residual": residual,
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise HarnessError(
            "SciPy raw eigenpair validation produced non-finite diagnostics", metrics
        )
    if (
        norm_error > GRAM_LIMIT
        or diagonal_error > GRAM_LIMIT
        or off_diagonal_error > GRAM_LIMIT
        or residual > RESIDUAL_LIMIT
    ):
        raise HarnessError(
            "SciPy raw eigenpair validation exceeded limits "
            f"(norm/Gram<={GRAM_LIMIT}, residual<={RESIDUAL_LIMIT})",
            metrics,
        )
    return metrics


def _solve_reference(operator, dense_operator, n, dimensions):
    k = dimensions + 1
    cutoff = max(64, 2 * k)
    if n <= cutoff:
        started = time.perf_counter()
        values, vectors = np.linalg.eigh(dense_operator)
        elapsed = time.perf_counter() - started
        indices = np.argsort(values)[::-1][:k]
        return values[indices], vectors[:, indices], "numpy.linalg.eigh", cutoff, elapsed
    ncv = min(n - 1, max(64, 2 * k + 1))
    started = time.perf_counter()
    values, vectors = eigsh(
        operator,
        k=k,
        which="LA",
        tol=1e-10,
        ncv=ncv,
        v0=None,
    )
    elapsed = time.perf_counter() - started
    indices = np.argsort(values)[::-1]
    return values[indices], vectors[:, indices], "scipy.sparse.linalg.eigsh", cutoff, elapsed


def _scipy_worker(request_path, response_path):
    # Loading graph arrays and constructing the normalized operator are not timed.
    with np.load(request_path) as data:
        graph = {
            "n": int(data["n"][0]),
            "heads": np.asarray(data["heads"], dtype=np.int64),
            "tails": np.asarray(data["tails"], dtype=np.int64),
            "weights": np.asarray(data["weights"], dtype=np.float64),
        }
        dimensions = int(data["dimensions"][0])
    operator, _ = symmetric_operator(graph)
    dense_operator = operator.toarray() if graph["n"] <= max(64, 2 * (dimensions + 1)) else None
    branch = "numpy.linalg.eigh" if dense_operator is not None else "scipy.sparse.linalg.eigsh"
    timings = []
    validation = []
    for index in range(WARMUPS + RUNS):
        values, vectors, selected_branch, cutoff, elapsed = _solve_reference(
            operator, dense_operator, graph["n"], dimensions
        )
        if selected_branch != branch:
            raise HarnessError("SciPy reference branch changed between runs")
        diagnostics = _validate_reference(values, vectors, operator)
        diagnostics["run"] = index
        diagnostics["warmup"] = index < WARMUPS
        validation.append(diagnostics)
        if index < WARMUPS:
            warmup_seconds = elapsed
        else:
            timings.append(elapsed)
    response = {
        "n": graph["n"],
        "dimensions": dimensions,
        "warmups": WARMUPS,
        "runs": RUNS,
        "warmup_seconds": float(warmup_seconds),
        "seconds": [float(value) for value in timings],
        "branch": branch,
        "dense_cutoff": cutoff,
        "eigsh": {
            "k": dimensions + 1,
            "which": "LA",
            "tol": 1e-10,
            "ncv": (
                min(graph["n"] - 1, max(64, 2 * (dimensions + 1) + 1))
                if branch == "scipy.sparse.linalg.eigsh"
                else None
            ),
            "v0": None,
        },
        "validation": validation,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "threadpool_info": _threadpool_info(),
        "thread_environment": _thread_environment(),
    }
    _write_json(pathlib.Path(response_path), response)


def _run_scipy_worker(graph, dimensions):
    with tempfile.TemporaryDirectory(prefix="fast-umap-scipy-runtime-") as directory:
        request_path = pathlib.Path(directory) / "request.npz"
        response_path = pathlib.Path(directory) / "response.json"
        np.savez(
            request_path,
            n=np.asarray([graph["n"]], dtype=np.int64),
            heads=np.asarray(graph["heads"], dtype=np.int64),
            tails=np.asarray(graph["tails"], dtype=np.int64),
            weights=np.asarray(graph["weights"], dtype=np.float64),
            dimensions=np.asarray([dimensions], dtype=np.int64),
        )
        command = [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "--scipy-worker",
            str(request_path),
            str(response_path),
        ]
        try:
            child = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                env=dict(os.environ),
            )
        except subprocess.TimeoutExpired as exc:
            evidence = {
                "command": command,
                "timeout_seconds": TIMEOUT_SECONDS,
                "stdout": _as_text(exc.stdout),
                "stderr": _as_text(exc.stderr),
            }
            raise HarnessError(
                f"SciPy reference worker exceeded {TIMEOUT_SECONDS}s", evidence
            ) from exc
        evidence = {
            "command": command,
            "returncode": child.returncode,
            "stdout": child.stdout,
            "stderr": child.stderr,
            "request_sha256": _sha256(request_path),
        }
        if child.returncode != 0:
            raise HarnessError(
                f"SciPy reference worker exited with status {child.returncode}", evidence
            )
        if not response_path.is_file():
            raise HarnessError(
                "SciPy reference worker passed without writing its response", evidence
            )
        try:
            result = json.loads(response_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HarnessError(f"cannot parse SciPy worker response: {exc}", evidence) from exc
        if not isinstance(result, dict):
            raise HarnessError("SciPy worker response must be a JSON object", evidence)
        _exact_int(result.get("n"), int(graph["n"]), "SciPy n")
        _exact_int(result.get("dimensions"), int(dimensions), "SciPy dimensions")
        _exact_int(result.get("warmups"), WARMUPS, "SciPy warmups")
        _exact_int(result.get("runs"), RUNS, "SciPy runs")
        result["warmup_seconds"] = _positive_finite(
            result.get("warmup_seconds"), "SciPy warmup_seconds"
        )
        seconds = result.get("seconds")
        if not isinstance(seconds, list) or len(seconds) != RUNS:
            raise HarnessError(f"SciPy seconds must contain exactly {RUNS} values", evidence)
        result["seconds"] = [
            _positive_finite(value, f"SciPy seconds[{index}]")
            for index, value in enumerate(seconds)
        ]
        if (
            not isinstance(result.get("validation"), list)
            or len(result["validation"]) != RUNS + WARMUPS
        ):
            raise HarnessError("SciPy worker did not report validation for every run", evidence)
        if result.get("branch") not in ("numpy.linalg.eigh", "scipy.sparse.linalg.eigsh"):
            raise HarnessError(
                f"unexpected SciPy reference branch: {result.get('branch')!r}", evidence
            )
        result["command"] = command
        result["request_sha256"] = evidence["request_sha256"]
        result["worker_stdout"] = child.stdout
        result["worker_stderr"] = child.stderr
        return result


def _run_component(name, graph, dimensions, test_runner):
    rust = _invoke_runtime(graph, dimensions, test_runner)
    scipy_result = _run_scipy_worker(graph, dimensions)
    return {
        "name": name,
        "n": int(graph["n"]),
        "dimensions": int(dimensions),
        "rust": rust,
        "scipy": scipy_result,
    }


def _aggregate_graph(
    name, graph, dimensions, source_path, input_sha256, eligible, skipped, components
):
    if len(components) != len(eligible):
        raise HarnessError(
            f"{name}: runtime/reference results incomplete for one or more components"
        )
    rust_runs = [sum(part["rust"]["seconds"][run] for part in components) for run in range(RUNS)]
    scipy_runs = [sum(part["scipy"]["seconds"][run] for part in components) for run in range(RUNS)]
    run_ratios = [rust_runs[index] / scipy_runs[index] for index in range(RUNS)]
    rust_median = float(np.median(rust_runs))
    scipy_median = float(np.median(scipy_runs))
    ratio = rust_median / scipy_median
    passes_runtime_gate = math.isfinite(ratio) and ratio <= RATIO_LIMIT
    return {
        "name": name,
        "source": str(source_path),
        "source_sha256": input_sha256,
        "n": int(graph["n"]),
        "dimensions": int(dimensions),
        "eligible_component_count": len(eligible),
        "skipped_components": skipped,
        "components": components,
        "matched_runs": [
            {
                "run": index + 1,
                "rust_seconds_sum": rust_runs[index],
                "scipy_seconds_sum": scipy_runs[index],
                "rust_over_scipy": run_ratios[index],
            }
            for index in range(RUNS)
        ],
        "rust_warmup_seconds_sum": sum(part["rust"]["warmup_seconds"] for part in components),
        "scipy_warmup_seconds_sum": sum(part["scipy"]["warmup_seconds"] for part in components),
        "rust_median_seconds": rust_median,
        "scipy_median_seconds": scipy_median,
        "median_rust_over_scipy": ratio,
        "ratio_limit": RATIO_LIMIT,
        "passes_runtime_gate": passes_runtime_gate,
    }


def _failure(case, exc):
    return {
        "case": case,
        "error": str(exc),
        "type": type(exc).__name__,
        "evidence": getattr(exc, "evidence", {}),
        "traceback": "".join(traceback.format_exception(exc)),
    }


def _run_harness(args):
    frozen_root = args.frozen_root
    test_runner = args.test_runner.resolve()
    script_path = pathlib.Path(__file__).resolve()
    repository = script_path.parent.parent
    source_paths = (
        "Cargo.toml",
        "Cargo.lock",
        "crates/fast-umap-core/Cargo.toml",
        "crates/fast-umap-core/src/init.rs",
        "crates/fast-umap-core/src/block_spectral.rs",
        "scripts/check_spectral.py",
    )
    report = {
        "harness": "check_spectral_runtime.py",
        "status": "running",
        "timeout_seconds": TIMEOUT_SECONDS,
        "configuration": {
            "dimensions": DIMENSIONS,
            "seed": SEED,
            "runtime": True,
            "warmups": WARMUPS,
            "runs": RUNS,
            "ratio_limit": RATIO_LIMIT,
            "component_skip_policy": "skip components with n <= dimensions + 1",
            "rust_command_arguments": ["--ignored", "--exact", PROBE_TEST, "--nocapture"],
            "rust_request_fields": [
                "n",
                "heads",
                "tails",
                "weights",
                "dimensions",
                "seed",
                "runtime",
            ],
            "rust_response_contract": {
                "runtime": True,
                "fallback": False,
                "warmups": WARMUPS,
                "runs": RUNS,
                "warmup_seconds": "finite positive number",
                "seconds": f"{RUNS} finite positive numbers",
                "n": "component vertex count",
                "dimensions": DIMENSIONS,
                "unseeded": True,
            },
            "reference": {
                "k": DIMENSIONS + 1,
                "which": "LA",
                "tol": 1e-10,
                "ncv": "min(n - 1, max(64, 2*k + 1))",
                "v0": None,
                "dense_branch": "numpy.linalg.eigh when n <= max(64, 2*k)",
                "sparse_branch": "scipy.sparse.linalg.eigsh otherwise",
                "timing": "Solver call only; excludes startup, graph/operator setup and validation",
            },
            "timing_comparison_disclosure": (
                "Native times solve_spectral including raw residual checks; "
                "SciPy times eigsh/eigh acquisition, excluding finite/norm/Gram/residual checks. "
                "Both exclude JSON work, startup and normalized-operator preparation. "
                "Native therefore includes extra validation work relative to SciPy."
            ),
            "accuracy_scope": (
                "Validates SciPy raw eigenpairs and native runtime protocol; "
                "does not replace check_spectral.py independent repeated-subspace acceptance."
            ),
        },
        "provenance": {
            "frozen_root": str(frozen_root),
            "test_runner": str(test_runner),
            "test_runner_sha256": _sha256(test_runner) if test_runner.is_file() else None,
            "script": str(script_path),
            "script_sha256": _sha256(script_path),
            "source_snapshot_sha256": {path: _sha256(repository / path) for path in source_paths},
            "source_snapshot_scope": (
                "Working source at measurement time; separate binary hash, "
                "not a reproducible-build attestation"
            ),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "threadpool_info": _threadpool_info(),
            "thread_environment": _thread_environment(),
            "thread_environment_mutated": False,
        },
        "graphs": [],
        "failures": [],
    }
    if not test_runner.is_file():
        report["failures"].append(
            _failure("startup", FileNotFoundError(f"test runner not found: {test_runner}"))
        )
        report["status"] = "failed"
        return report

    for kind in ("raw", "unique"):
        for size in (15000, 50000):
            source_path = frozen_root / f"rust-{kind}-{size}-42" / "spectral-graph.npz"
            name = str(source_path)
            try:
                if not source_path.is_file():
                    raise FileNotFoundError(f"required frozen graph missing: {source_path}")
                input_sha256 = _sha256(source_path)
                graph = graph_arrays(source_path)
                eligible, skipped = _component_cases(name, graph, DIMENSIONS)
                if not eligible:
                    raise HarnessError(
                        f"{name}: no component can supply {DIMENSIONS} spectral dimensions"
                    )
                components = []
                component_failures = []
                for component_name, component_graph in eligible:
                    try:
                        components.append(
                            _run_component(component_name, component_graph, DIMENSIONS, test_runner)
                        )
                    except Exception as exc:
                        component_failures.append(_failure(component_name, exc))
                if component_failures:
                    report["failures"].extend(component_failures)
                    graph_result = {
                        "name": name,
                        "source": str(source_path),
                        "source_sha256": input_sha256,
                        "n": int(graph["n"]),
                        "dimensions": DIMENSIONS,
                        "eligible_component_count": len(eligible),
                        "skipped_components": skipped,
                        "components": components,
                        "status": "failed",
                    }
                else:
                    graph_result = _aggregate_graph(
                        name,
                        graph,
                        DIMENSIONS,
                        source_path,
                        input_sha256,
                        eligible,
                        skipped,
                        components,
                    )
                    graph_result["status"] = (
                        "passed" if graph_result["passes_runtime_gate"] else "failed"
                    )
                    if not graph_result["passes_runtime_gate"]:
                        report["failures"].append(
                            _failure(
                                name,
                                HarnessError(
                                    f"{name}: median Rust/SciPy runtime ratio "
                                    f"{graph_result['median_rust_over_scipy']:.6g} exceeds "
                                    f"{RATIO_LIMIT:.3g}",
                                    {
                                        "ratio": graph_result["median_rust_over_scipy"],
                                        "limit": RATIO_LIMIT,
                                    },
                                ),
                            )
                        )
                report["graphs"].append(graph_result)
            except Exception as exc:
                report["failures"].append(_failure(name, exc))

    report["status"] = "passed" if not report["failures"] else "failed"
    return report


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compare frozen-graph unseeded Rust spectral runtime with SciPy.",
        epilog=(
            "Example: python scripts/check_spectral_runtime.py "
            "--test-runner artifacts/approved-plan-tooling/spectral-probe-test "
            "--output artifacts/approved-plan-baseline/runtime.json"
        ),
    )
    parser.add_argument(
        "--frozen-root",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/approved-plan-baseline"),
    )
    parser.add_argument(
        "--test-runner",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/approved-plan-tooling/spectral-probe-test"),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/approved-plan-baseline/spectral-runtime-results.json"),
    )
    parser.add_argument(
        "--failures",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/approved-plan-baseline/spectral-runtime-failures.json"),
    )
    parser.add_argument("--scipy-worker", nargs=2, metavar=("REQUEST", "RESPONSE"))
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.scipy_worker:
        _scipy_worker(*args.scipy_worker)
        return

    output_path = args.output
    failures_path = args.failures
    if output_path.resolve() == failures_path.resolve():
        print("--output and --failures must name different files", file=sys.stderr)
        raise SystemExit(2)
    try:
        report = _run_harness(args)
    except Exception as exc:
        report = {
            "harness": "check_spectral_runtime.py",
            "status": "failed",
            "timeout_seconds": TIMEOUT_SECONDS,
            "graphs": [],
            "failures": [_failure("harness", exc)],
        }
    failures_document = {
        "status": report.get("status", "failed"),
        "failures": report.get("failures", []),
        "completed_graphs": report.get("graphs", []),
    }
    write_errors = []
    try:
        _write_json(output_path, report)
    except Exception as exc:
        write_errors.append({"path": str(output_path), "error": repr(exc)})
    try:
        _write_json(failures_path, failures_document)
    except Exception as exc:
        write_errors.append({"path": str(failures_path), "error": repr(exc)})
    if write_errors:
        print(json.dumps({"persistence_errors": write_errors}, indent=2), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(report, indent=2, allow_nan=False))
    if report.get("status") != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
