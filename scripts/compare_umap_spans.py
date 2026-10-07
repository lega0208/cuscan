#!/usr/bin/env -S uv run --script
"""Compare umap-learn 0.5.12 with the CUDA-only standard UMAP profiler.

Run from the repository root with ``uv run --no-sync python scripts/compare_umap_spans.py``.
Build the companion first (not included in the runtime budget):
``cargo build --release --no-default-features --features cuda,tracing
--example profile_standard_cuda`` (one shell command).
Defaults are 512 rows, 16 features, 30 epochs, seed 42, 15 exact neighbors, and a
115-second global deadline covering all worker startup/import/JIT/CUDA warmups.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "target" / "release" / "examples" / "profile_standard_cuda"


class DeadlineExceeded(TimeoutError):
    """The global benchmark deadline expired."""


class Deadline:
    def __init__(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("deadline must be positive")
        self.ends_at = time.monotonic() + seconds

    def remaining(self) -> float:
        value = self.ends_at - time.monotonic()
        if value <= 0:
            raise DeadlineExceeded("global runtime deadline expired")
        return value


def run_with_deadline(command: list[str], deadline: Deadline) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=deadline.remaining(),
        )
    except subprocess.TimeoutExpired as error:
        raise DeadlineExceeded("global runtime deadline expired during worker execution") from error


def _timed_wrapper(function: Any, label: str, samples: dict[str, list[float]]) -> Any:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            samples[label].append(time.perf_counter() - start)

    return wrapped


@contextmanager
def instrument_umap(module: Any, samples: dict[str, list[float]]) -> Iterator[None]:
    """Measure real umap-learn callsites and restore all monkeypatches on exit."""
    targets = {
        "pairwise_distances": (module, "pairwise_distances"),
        "nearest_neighbors": (module, "nearest_neighbors"),
        "fuzzy_simplicial_set": (module, "fuzzy_simplicial_set"),
        "simplicial_set_embedding": (module, "simplicial_set_embedding"),
        "optimize_layout_euclidean": (module, "optimize_layout_euclidean"),
    }
    originals: list[tuple[Any, str, Any]] = []
    try:
        for label, (owner, attribute) in targets.items():
            original = getattr(owner, attribute, None)
            if original is None:
                continue
            originals.append((owner, attribute, original))
            setattr(owner, attribute, _timed_wrapper(original, label, samples))
        yield
    finally:
        for owner, attribute, original in originals:
            setattr(owner, attribute, original)


def _python_worker(input_path: Path, rows: int, features: int, epochs: int) -> dict[str, Any]:
    import numpy as np
    import umap
    import umap.umap_ as umap_module

    if umap.__version__ != "0.5.12":
        raise ValueError("This callsite mapping requires umap-learn 0.5.12")
    matrix = np.fromfile(input_path, dtype="<f4").reshape(rows, features)
    samples: dict[str, list[float]] = defaultdict(list)
    estimator = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        n_epochs=epochs,
        init="random",
        random_state=42,
        metric="euclidean",
        force_approximation_algorithm=False,
        low_memory=True,
        n_jobs=1,
        verbose=False,
    )
    with instrument_umap(umap_module, samples):
        estimator.fit(matrix)
        for durations in samples.values():
            durations.clear()
        start = time.perf_counter()
        estimator.fit(matrix)
        fit_seconds = time.perf_counter() - start

    stages = {name: sum(durations) for name, durations in samples.items()}
    simplicial = stages.get("simplicial_set_embedding", 0.0)
    optimizer = stages.get("optimize_layout_euclidean", 0.0)
    return {
        "fit_seconds": fit_seconds,
        "stages_seconds": stages,
        "embedding_pre_optimizer_seconds": max(0.0, simplicial - optimizer),
        "initialization_note": (
            "Random initialization is inside simplicial_set_embedding and is not exposed as a "
            "separate umap-learn call; embedding_pre_optimizer_seconds is its inclusive outer "
            "duration minus the nested optimizer duration, not an initialization-only measurement."
        ),
        "umap_learn_version": umap.__version__,
    }


def parse_chrome_trace(trace: dict[str, Any] | list[Any]) -> dict[str, Any]:
    """Aggregate inclusive durations from Chrome complete and begin/end events."""
    events = trace if isinstance(trace, list) else trace.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("Chrome trace must contain a traceEvents array")
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    stacks: dict[tuple[Any, Any], list[tuple[str, float]]] = defaultdict(list)
    unmatched_ends = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        phase = event.get("ph")
        name = str(event.get("name", "<unnamed>"))
        if phase == "X":
            duration = event.get("dur")
            if isinstance(duration, (int, float)) and duration >= 0:
                totals[name] += float(duration) / 1_000_000.0
                counts[name] += 1
        elif phase == "B":
            timestamp = event.get("ts")
            if isinstance(timestamp, (int, float)):
                stacks[(event.get("pid"), event.get("tid"))].append((name, float(timestamp)))
        elif phase == "E":
            key = (event.get("pid"), event.get("tid"))
            timestamp = event.get("ts")
            stack = stacks[key]
            if not isinstance(timestamp, (int, float)) or not stack:
                unmatched_ends += 1
                continue
            begin_name, begin = stack.pop()
            if timestamp < begin:
                raise ValueError(f"Chrome span {begin_name!r} has negative duration")
            totals[begin_name] += (float(timestamp) - begin) / 1_000_000.0
            counts[begin_name] += 1
    unmatched_begins = sum(len(stack) for stack in stacks.values())
    return {
        "inclusive_seconds_by_span": dict(sorted(totals.items())),
        "span_counts": dict(sorted(counts.items())),
        "unmatched_begin_events": unmatched_begins,
        "unmatched_end_events": unmatched_ends,
        "note": "Durations are inclusive; nested spans overlap and must not be summed.",
    }


def make_clustered_input(rows: int, features: int, destination: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(42)
    centers = rng.normal(0.0, 5.0, size=(8, features)).astype(np.float32)
    labels = np.arange(rows) % len(centers)
    rng.shuffle(labels)
    matrix = centers[labels] + rng.normal(0.0, 0.35, size=(rows, features)).astype(np.float32)
    matrix.astype("<f4", copy=False).tofile(destination)


def _python_output(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "fit_seconds" in parsed:
            return parsed
    raise ValueError("Python worker did not emit its result JSON")


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--deadline-seconds", type=float, default=115.0)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path, help="write the full comparison JSON")
    parser.add_argument("--trace", type=Path, help="copy the Rust Chrome trace here")
    parser.add_argument(
        "--_python-worker", nargs=4, metavar=("INPUT", "ROWS", "FEATURES", "EPOCHS")
    )
    args = parser.parse_args()

    if args._python_worker:
        result = _python_worker(Path(args._python_worker[0]), *map(int, args._python_worker[1:]))
        print(json.dumps(result))
        return 0
    if not 15 < args.rows < 4096 or args.features < 1 or args.epochs < 1:
        parser.error("rows must be 16..4095 (Python exact-search branch); features/epochs positive")
    if not 0 < args.deadline_seconds <= 115:
        parser.error("deadline must be positive and at most 115 seconds")
    binary = args.binary.resolve()
    if not binary.is_file():
        print(
            f"CUDA profiler executable not found: {binary}\n"
            "Build it first (build time is excluded):\n"
            "  cargo build --release --no-default-features --features cuda,tracing "
            "--example profile_standard_cuda",
            file=sys.stderr,
        )
        return 2

    deadline = Deadline(args.deadline_seconds)
    run_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="compare-umap-spans-") as temporary:
        temp = Path(temporary)
        input_path = temp / "input.f32le"
        trace_path = temp / "rust-trace.json"
        make_clustered_input(args.rows, args.features, input_path)
        python_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_python-worker",
            str(input_path),
            str(args.rows),
            str(args.features),
            str(args.epochs),
        ]
        try:
            python_process = run_with_deadline(python_command, deadline)
            python = _python_output(python_process.stdout)
            rust_process = run_with_deadline(
                [
                    str(binary),
                    str(input_path),
                    str(args.rows),
                    str(args.features),
                    str(args.epochs),
                    str(trace_path),
                ],
                deadline,
            )
        except (DeadlineExceeded, subprocess.CalledProcessError) as error:
            detail = getattr(error, "stderr", "")
            print(f"Comparison failed without fallback: {error}\n{detail}", file=sys.stderr)
            return 1
        rust = json.loads(rust_process.stdout.strip().splitlines()[-1])
        with trace_path.open(encoding="utf-8") as stream:
            rust_trace = parse_chrome_trace(json.load(stream))
        cuda_spans = rust_trace["inclusive_seconds_by_span"]
        python_stages = python["stages_seconds"]
        embedding_cuda = (
            cuda_spans.get("initialization", 0.0)
            + cuda_spans.get("optimization", 0.0)
            + cuda_spans.get("tensor_embedding", 0.0)
        )

        def stage_ratio(python_seconds: float, cuda_seconds: float) -> float | None:
            return python_seconds / cuda_seconds if cuda_seconds > 0.0 else None

        output = {
            "configuration": {
                "rows": args.rows,
                "features": args.features,
                "epochs": args.epochs,
                "neighbors": 15,
                "init": "random",
                "seed": 42,
                "neighbor_search": "exact",
                "input": "shared seeded clustered float32, little-endian row-major",
                "deadline_seconds": args.deadline_seconds,
            },
            "elapsed_seconds_including_warmups": time.monotonic() - run_start,
            "python": python,
            "rust_cuda": {
                **rust,
                "trace": rust_trace,
                "layout_note": (
                    "The CUDA layout kernel assigns one serial unit per epoch "
                    "(src/kernels/standard_umap/layout.rs); epoch spans measure host work, "
                    "not GPU execution timing."
                ),
            },
            "comparisons": {
                "fit_wall_ratio_python_over_rust": python["fit_seconds"] / rust["fit_seconds"],
                "host_span_ratios_python_over_cuda_not_gpu_speedups": {
                    "exact_neighbor_search": stage_ratio(
                        python_stages.get("pairwise_distances", 0.0)
                        + python_stages.get("nearest_neighbors", 0.0),
                        cuda_spans.get("neighbor_search", 0.0),
                    ),
                    "fuzzy_graph": stage_ratio(
                        max(
                            0.0,
                            python_stages.get("fuzzy_simplicial_set", 0.0)
                            - python_stages.get("nearest_neighbors", 0.0),
                        ),
                        cuda_spans.get("fuzzy_graph", 0.0),
                    ),
                    "embedding_setup_optimization_and_host_readback": stage_ratio(
                        python_stages.get("simplicial_set_embedding", 0.0), embedding_cuda
                    ),
                },
                "rust_unmapped_fit_seconds": max(
                    0.0,
                    rust["fit_seconds"]
                    - embedding_cuda
                    - cuda_spans.get("neighbor_search", 0.0)
                    - cuda_spans.get("fuzzy_graph", 0.0),
                ),
                "python_unmapped_fit_seconds": max(
                    0.0,
                    python["fit_seconds"]
                    - sum(
                        python_stages.get(stage, 0.0)
                        for stage in (
                            "pairwise_distances",
                            "fuzzy_simplicial_set",
                            "simplicial_set_embedding",
                        )
                    ),
                ),
                "note": (
                    "Python stage timings are inclusive at their actual callsites; fuzzy graph and "
                    "embedding subtimings nest inside fit, and embedding optimization nests inside "
                    "simplicial_set_embedding. Python residual is not a separately timed stage. "
                    "CUDA trace spans are inclusive and may overlap; host enqueue duration "
                    "is not GPU time. Embedding comparison includes CUDA tensor_embedding "
                    "to capture queued layout work at host readback, but may include earlier GPU "
                    "work too. Python neighbor time includes pairwise distances and nested exact "
                    "nearest_neighbors selection; graph time subtracts that nested selection. "
                    "No Flex fallback is attempted."
                ),
            },
        }
        if args.trace:
            args.trace.resolve().write_bytes(trace_path.read_bytes())
        rendered = json.dumps(output, indent=2, sort_keys=True)
        if args.output:
            args.output.resolve().write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_cli())
    except DeadlineExceeded as error:
        print(f"Comparison timed out without fallback: {error}", file=sys.stderr)
        raise SystemExit(1) from error
