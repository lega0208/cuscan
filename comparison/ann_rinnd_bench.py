#!/usr/bin/env -S uv run --no-sync
"""Measure ANN construction arms using the source-bound rinnd plan protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from comparison.umap06.datasets import generate as generate_umap06
    from comparison.umap06.datasets import load_fixture
else:
    from umap06.datasets import generate as generate_umap06
    from umap06.datasets import load_fixture

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "umap06"
QUERY_SEED = 20261119
TIMING_ORDER = ("A", "B", "B", "A")
PROFILE_FIELD = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
PRODUCT_PATHS = (
    "crates",
    "python",
    "Cargo.toml",
    "Cargo.lock",
    "pyproject.toml",
    "uv.lock",
)
SOURCE_PROVENANCE_FIELDS = (
    "base_sha",
    "tree_id",
    "2a_patch_sha256",
    "arm_patch_sha256",
    "extension_sha256",
)


class Worker:
    def __init__(
        self,
        binary: Path,
        input_path: Path,
        rows: int,
        features: int,
        backend: str,
        stderr_path: Path,
        *,
        profiled: bool = False,
    ) -> None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr_stream = stderr_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        if profiled:
            env["CUSCAN_ANN_PROFILE"] = "1"
        else:
            env.pop("CUSCAN_ANN_PROFILE", None)
        self.process = subprocess.Popen(
            [str(binary), str(input_path), str(rows), str(features), backend],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_stream,
            text=True,
            encoding="utf-8",
            env=env,
            bufsize=1,
        )
        self.binary = binary

    def command(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("worker pipes are unavailable")
        self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            raise RuntimeError(f"ANN worker exited before response (status={code})")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"ANN worker emitted invalid JSON: {line!r}") from error
        if not response.get("ok"):
            raise RuntimeError(f"ANN worker failed: {response.get('error', response)}")
        return response

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        status = self.process.wait()
        if self.process.stdout is not None:
            self.process.stdout.close()
        self.stderr_stream.close()
        if status:
            raise RuntimeError(f"ANN worker exited with status {status}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_at(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def source_identity(
    binary: Path,
    *,
    source_root: Path | None = None,
    provenance_path: Path | None = None,
    require_source_chain: bool = False,
) -> dict[str, Any]:
    """Resolve source identity from a checked-out root or binary-bound frozen receipt."""
    binary_hash = sha256(binary)
    if provenance_path is not None:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("binary_sha256") != binary_hash:
            raise ValueError(f"provenance binary_sha256 does not match {binary}: {provenance_path}")
        head = provenance.get("head")
        diff_hash = provenance.get("product_diff_sha256")
        if not isinstance(head, str) or not head or not isinstance(diff_hash, str) or not diff_hash:
            raise ValueError(
                f"provenance must contain non-empty head and product_diff_sha256: {provenance_path}"
            )
        identity = {
            "head": head,
            "product_diff_sha256": diff_hash,
            "identity_source": "binary-bound-provenance",
            "provenance_path": str(provenance_path.resolve()),
            "binary_sha256": binary_hash,
        }
        supplied_source_fields = [key for key in SOURCE_PROVENANCE_FIELDS if key in provenance]
        if require_source_chain and not supplied_source_fields:
            raise ValueError(f"provenance lacks the complete arm source chain: {provenance_path}")
        if supplied_source_fields:
            if len(supplied_source_fields) != len(SOURCE_PROVENANCE_FIELDS):
                missing = sorted(set(SOURCE_PROVENANCE_FIELDS) - set(supplied_source_fields))
                raise ValueError(
                    f"provenance must contain the complete arm source chain; missing {missing}: "
                    f"{provenance_path}"
                )
            for key in SOURCE_PROVENANCE_FIELDS:
                value = provenance[key]
                expected_length = 40 if key in ("base_sha", "tree_id") else 64
                if (
                    not isinstance(value, str)
                    or re.fullmatch(rf"[0-9a-f]{{{expected_length}}}", value) is None
                ):
                    raise ValueError(
                        f"provenance {key} must be a lowercase {expected_length}-digit hash"
                    )
                identity[key] = value
            if re.fullmatch(r"[0-9a-f]{40}", head) is None:
                raise ValueError("provenance head must be a lowercase Git object ID")
            if re.fullmatch(r"[0-9a-f]{64}", diff_hash) is None:
                raise ValueError("provenance product_diff_sha256 must be a lowercase SHA-256")
            arm = provenance.get("arm")
            if not isinstance(arm, str) or not arm:
                raise ValueError(f"provenance must contain a non-empty arm: {provenance_path}")
            if head != provenance["base_sha"]:
                raise ValueError(f"provenance head does not match base_sha: {provenance_path}")
            identity["arm"] = arm
        if source_root is not None:
            checked = source_identity(binary, source_root=source_root)
            if checked["head"] != head or checked["product_diff_sha256"] != diff_hash:
                raise ValueError("frozen provenance does not match the supplied source root")
        return identity
    root = (source_root or ROOT).resolve()
    head = git_at(root, "rev-parse", "HEAD")
    diff = subprocess.check_output(["git", "diff", "HEAD", "--", *PRODUCT_PATHS], cwd=root)
    return {
        "head": head,
        "product_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "identity_source": "worktree",
        "source_root": str(root),
        "binary_sha256": binary_hash,
    }


def write_input(path: Path, matrix: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    contiguous = np.asarray(matrix, dtype="<f4", order="C")
    contiguous.tofile(path)
    return {
        "path": str(path),
        "rows": int(contiguous.shape[0]),
        "features": int(contiguous.shape[1]),
        "dtype": "f32le",
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def fixture_inputs(
    out: Path, data_root: Path, fixtures: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """Materialize the five plan datasets as immutable, hashed f32le benchmark inputs."""
    inputs_dir = out / "inputs"
    generated: dict[str, dict[str, Any]] = {}
    if "S1" in fixtures:
        generated["S1"] = make_s1(inputs_dir / "S1.f32le")
    umap06_names = [
        name for name, fixture in (("S2", "mix_200k"), ("R3", "digits")) if name in fixtures
    ]
    if umap06_names:
        generate_umap06(
            data_root, ["mix_200k" if name == "S2" else "digits" for name in umap06_names]
        )
        if "S2" in fixtures:
            mix, _, _ = load_fixture(data_root, "mix_200k")
            generated["S2"] = write_input(inputs_dir / "S2.f32le", mix)
        if "R3" in fixtures:
            digits, _, _ = load_fixture(data_root, "digits")
            generated["R3"] = write_input(inputs_dir / "R3.f32le", digits)
    if "R1" in fixtures:
        generated["R1"] = parquet_input(
            ROOT / "comparison" / "embeddings.parquet", "embeddings", inputs_dir / "R1.f32le"
        )
    if "R2" in fixtures:
        generated["R2"] = parquet_sample(
            ROOT / "comparison" / "feedback_embedded.parquet",
            "embeddings",
            500_000,
            QUERY_SEED,
            inputs_dir / "R2.f32le",
        )
    expected_shapes = {
        "S1": (250_000, 256),
        "S2": (200_000, 32),
        "R1": (272_383, 128),
        "R2": (500_000, 256),
        "R3": (1_497, 64),
    }
    for name in fixtures:
        expected = expected_shapes[name]
        actual = (generated[name]["rows"], generated[name]["features"])
        if actual != expected:
            raise ValueError(f"{name} shape is {actual}, expected {expected}")
    return generated


def make_s1(destination: Path) -> dict[str, Any]:
    """Match compare_stages.py's clustered generator (eight centers, seed 42)."""
    rng = np.random.default_rng(42)
    rows, features = 250_000, 256
    centers = rng.normal(0.0, 1.0, size=(8, features)).astype(np.float32)
    labels = np.arange(rows) % 8
    rng.shuffle(labels)
    matrix = centers[labels] + rng.normal(0.0, 1.0, size=(rows, features)).astype(np.float32)
    return write_input(destination, matrix)


def parquet_array(path: Path, column: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"required local input is missing: {path}")
    import polars as pl

    frame = pl.read_parquet(path, columns=[column])
    values = frame[column].to_numpy()
    matrix = np.asarray(values.tolist() if values.dtype == object else values)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{path}:{column} must be a finite 2D matrix")
    return np.asarray(matrix, dtype=np.float32, order="C")


def parquet_input(path: Path, column: str, destination: Path) -> dict[str, Any]:
    matrix = parquet_array(path, column)
    result = write_input(destination, matrix)
    result["source"] = {"path": str(path), "sha256": sha256(path), "column": column}
    return result


def parquet_sample(
    path: Path, column: str, count: int, seed: int, destination: Path
) -> dict[str, Any]:
    matrix = parquet_array(path, column)
    if len(matrix) < count:
        raise ValueError(f"{path}:{column} has {len(matrix)} rows; {count} required")
    indices = np.sort(np.random.default_rng(seed).choice(len(matrix), count, replace=False))
    result = write_input(destination, matrix[indices])
    result["source"] = {
        "path": str(path),
        "sha256": sha256(path),
        "column": column,
        "sample_seed": seed,
        "sample_rows": count,
        "selection": "without replacement; selected source rows retained in original order",
        "indices_sha256": hashlib.sha256(np.asarray(indices, dtype="<u8").tobytes()).hexdigest(),
    }
    return result


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def exact_tie_aware_recall(
    matrix: np.ndarray,
    graph_path: Path,
    k: int,
    seed: int = QUERY_SEED,
    query_chunk_size: int = 8,
    reference_chunk_size: int = 4_096,
) -> dict[str, Any]:
    ids_path = Path(f"{graph_path}.ids.i32le")
    ids = np.fromfile(ids_path, dtype="<i4")
    rows = matrix.shape[0]
    if ids.size % rows:
        raise ValueError(f"invalid graph ID file length: {ids_path}")
    width = ids.size // rows
    if width < k:
        raise ValueError(f"graph width {width} is less than recall k={k}")
    ids = ids.reshape(rows, width)
    if np.any(ids < 0) or np.any(ids >= rows):
        raise ValueError("graph contains an out-of-range neighbor ID")
    queries = (
        np.arange(rows, dtype=np.int64)
        if rows <= 2_000
        else np.sort(np.random.default_rng(seed).choice(rows, 2_000, replace=False))
    )
    recalls = np.empty(len(queries), dtype=np.float64)
    for start in range(0, len(queries), query_chunk_size):
        selected = queries[start : start + query_chunk_size]
        query_values = np.asarray(matrix[selected], dtype=np.float64)
        distances = np.empty((len(selected), rows), dtype=np.float64)
        for reference_start in range(0, rows, reference_chunk_size):
            reference_end = min(reference_start + reference_chunk_size, rows)
            reference_values = np.asarray(matrix[reference_start:reference_end], dtype=np.float64)
            delta = query_values[:, None, :] - reference_values[None, :, :]
            distances[:, reference_start:reference_end] = np.einsum(
                "qnf,qnf->qn", delta, delta, optimize=True
            )
        distances[np.arange(len(selected)), selected] = np.inf
        boundary = np.partition(distances, k - 1, axis=1)[:, k - 1]
        candidate_ids = ids[selected, :k]
        candidate_distance = distances[np.arange(len(selected))[:, None], candidate_ids]
        recalls[start : start + len(selected)] = (
            np.count_nonzero(candidate_distance <= boundary[:, None], axis=1) / k
        )
    return {
        "mean": float(recalls.mean()),
        "minimum": float(recalls.min()),
        "queries": len(queries),
        "query_seed": seed,
        "query_rows_sha256": hashlib.sha256(np.asarray(queries, dtype="<u8").tobytes()).hexdigest(),
        "k_nonself": k,
        "tie_rule": "ANN neighbors at or below the exact k-th non-self distance count as hits",
    }


def run_worker(
    *,
    arm: str,
    binary: Path,
    fixture: str,
    input_info: dict[str, Any],
    backend: str,
    k: int,
    cap: int,
    seed: int,
    squared: bool,
    out: Path,
    protocol: str,
    profile: bool,
    runs: int,
    source: dict[str, Any],
) -> dict[str, Any]:
    input_path = Path(input_info["path"])
    rows, features = input_info["rows"], input_info["features"]
    prefix = out / "graphs" / protocol / fixture / backend / arm
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stderr_path = out / "logs" / protocol / fixture / backend / f"{arm}.stderr.log"
    worker = Worker(binary, input_path, rows, features, backend, stderr_path, profiled=profile)
    try:
        warmup = None
        if not profile:
            warmup = worker.command(
                {"op": "warmup", "k": k, "seed": seed, "cap": cap, "squared": squared}
            )
        responses = []
        for index in range(runs):
            graph_prefix = str(prefix) if index == runs - 1 else None
            response = worker.command(
                {
                    "op": "build",
                    "k": k,
                    "seed": seed,
                    "cap": cap,
                    "squared": squared,
                    "graph_prefix": graph_prefix,
                }
            )
            responses.append(response)
    finally:
        worker.close()
    result: dict[str, Any] = {
        "arm": arm,
        "source": source,
        "fixture": fixture,
        "backend": backend,
        "protocol": protocol,
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "input_sha256": input_info["sha256"],
        "rows": rows,
        "features": features,
        "k_nonself": k,
        "cap": cap,
        "seed": seed,
        "squared": squared,
        "warmup": warmup,
        "responses": responses,
        "stderr_log": str(stderr_path),
        "graph_prefix": str(prefix) if runs else None,
    }
    if profile:
        result["ann_profile_lines"] = []
        for line in stderr_path.read_text(encoding="utf-8").splitlines():
            if "ANN_PROFILE" in line:
                result["ann_profile_lines"].append(
                    {"raw": line, "fields": dict(PROFILE_FIELD.findall(line))}
                )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--binary", type=Path, help="single-arm baseline worker binary")
    mode.add_argument("--predecessor", type=Path, help="predecessor worker binary (arm A)")
    parser.add_argument("--candidate", type=Path, help="candidate worker binary (arm B)")
    parser.add_argument(
        "--predecessor-source-root", type=Path, help="predecessor checkout used to build arm A"
    )
    parser.add_argument(
        "--candidate-source-root", type=Path, help="candidate checkout used to build arm B"
    )
    parser.add_argument(
        "--predecessor-provenance",
        type=Path,
        help="frozen arm A JSON; R2 requires base/tree, patch and extension hashes",
    )
    parser.add_argument(
        "--candidate-provenance",
        type=Path,
        help="frozen arm B JSON; R2 requires base/tree, patch and extension hashes",
    )
    parser.add_argument("--arm-name", default="P0", help="single-arm label (default: P0)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--backends", nargs="+", default=["cpu", "cuda:0"])
    parser.add_argument(
        "--fixtures",
        nargs="+",
        choices=("S1", "S2", "R1", "R2", "R3"),
        default=["S1", "S2", "R1", "R2", "R3"],
    )
    parser.add_argument("--cap", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--squared", action="store_true")
    parser.add_argument("--timing-only", action="store_true", help="defer recall scoring")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.predecessor is not None and args.candidate is None:
        raise ValueError("--predecessor requires --candidate")
    if args.predecessor is None and args.candidate is not None:
        raise ValueError("--candidate requires --predecessor")
    if args.binary is not None and args.candidate is not None:
        raise ValueError("--candidate is only valid with --predecessor")
    arm_binaries = (
        {args.arm_name: args.binary.resolve()}
        if args.binary is not None
        else {"A": args.predecessor.resolve(), "B": args.candidate.resolve()}
    )
    for binary in arm_binaries.values():
        if not binary.is_file():
            raise FileNotFoundError(f"worker binary not found: {binary}")
    if len(arm_binaries) == 2:
        sources_supplied = (
            args.predecessor_source_root is not None or args.predecessor_provenance is not None,
            args.candidate_source_root is not None or args.candidate_provenance is not None,
        )
        if not all(sources_supplied):
            raise ValueError(
                "paired runs require source identity for both binaries via --*-source-root "
                "or --*-provenance"
            )
    if not args.fixtures or len(set(args.fixtures)) != len(args.fixtures):
        raise ValueError("--fixtures must contain one or more distinct fixture IDs")
    if "R2" in args.fixtures:
        if len(arm_binaries) != 2:
            raise ValueError("R2 performance replay requires paired arms with source provenance")
        if args.seed != 11:
            raise ValueError("R2 performance replay requires seed 11")
        if not args.timing_only:
            raise ValueError("R2 performance-first replay requires --timing-only")
        if args.predecessor_provenance is None or args.candidate_provenance is None:
            raise ValueError(
                "R2 performance replay requires --predecessor-provenance and --candidate-provenance"
            )
    input_info = fixture_inputs(out, args.data.resolve(), tuple(args.fixtures))
    if len(arm_binaries) == 1:
        arm = next(iter(arm_binaries))
        source_identities = {arm: source_identity(arm_binaries[arm], source_root=ROOT)}
    else:
        source_identities = {
            "A": source_identity(
                arm_binaries["A"],
                source_root=args.predecessor_source_root,
                provenance_path=args.predecessor_provenance,
                require_source_chain="R2" in args.fixtures,
            ),
            "B": source_identity(
                arm_binaries["B"],
                source_root=args.candidate_source_root,
                provenance_path=args.candidate_provenance,
                require_source_chain="R2" in args.fixtures,
            ),
        }
    k_by_fixture = {"S1": 29, "S2": 14, "R1": 29, "R2": 29, "R3": 29}
    manifest = {
        "schema": "cuscan-ann-rinnd-bench-v1",
        "source": {"arms": source_identities},
        "arms": {
            arm: {
                "binary": str(binary),
                "binary_sha256": sha256(binary),
                "source": source_identities[arm],
            }
            for arm, binary in arm_binaries.items()
        },
        "inputs": input_info,
        "settings": {
            "backends": args.backends,
            "cap": args.cap,
            "seed": args.seed,
            "squared": args.squared,
            "query_seed": QUERY_SEED,
            "timing_order": list(TIMING_ORDER),
            "timed_commands_per_arm": TIMING_ORDER.count("A"),
            "profiled_runs_per_arm": 1,
            "recall_deferred": args.timing_only,
        },
        "receipts": [],
    }
    dump(out / "manifest.json", manifest)
    for fixture in args.fixtures:
        info = input_info[fixture]
        k = k_by_fixture[fixture]
        for backend in args.backends:
            if backend != "cpu" and not backend.startswith("cuda:"):
                raise ValueError(f"backend must be cpu or cuda:<index>: {backend}")
            if len(arm_binaries) == 1:
                arm = next(iter(arm_binaries))
                receipt = run_worker(
                    arm=arm,
                    binary=arm_binaries[arm],
                    fixture=fixture,
                    input_info=info,
                    backend=backend,
                    k=k,
                    cap=args.cap,
                    seed=args.seed,
                    squared=args.squared,
                    out=out,
                    protocol="baseline",
                    profile=False,
                    runs=TIMING_ORDER.count("A"),
                    source=source_identities[arm],
                )
                times = [response["ann_ms"] for response in receipt["responses"]]
                receipt["ann_ms"] = times
                receipt["median_ann_ms"] = statistics.median(times)
                receipt["range_ann_ms"] = [min(times), max(times)]
                prefix = Path(receipt["graph_prefix"])
                if not args.timing_only:
                    receipt["recall"] = exact_tie_aware_recall(
                        np.memmap(
                            info["path"],
                            dtype="<f4",
                            mode="r",
                            shape=(info["rows"], info["features"]),
                        ),
                        prefix,
                        k,
                    )
                manifest["receipts"].append(receipt)
            else:
                workers = {
                    arm: Worker(
                        binary,
                        Path(info["path"]),
                        info["rows"],
                        info["features"],
                        backend,
                        out / "logs" / "timing" / fixture / backend / f"{arm}.stderr.log",
                    )
                    for arm, binary in arm_binaries.items()
                }
                try:
                    warmups = {
                        arm: worker.command(
                            {
                                "op": "warmup",
                                "k": k,
                                "seed": args.seed,
                                "cap": args.cap,
                                "squared": args.squared,
                            }
                        )
                        for arm, worker in workers.items()
                    }
                    timings = {arm: [] for arm in workers}
                    responses = {arm: [] for arm in workers}
                    for schedule_index, arm in enumerate(TIMING_ORDER):
                        remaining = TIMING_ORDER[schedule_index + 1 :]
                        graph_prefix = (
                            None
                            if arm in remaining
                            else str(out / "graphs" / "timing" / fixture / backend / arm)
                        )
                        if graph_prefix is not None:
                            Path(graph_prefix).parent.mkdir(parents=True, exist_ok=True)
                        response = workers[arm].command(
                            {
                                "op": "build",
                                "k": k,
                                "seed": args.seed,
                                "cap": args.cap,
                                "squared": args.squared,
                                "graph_prefix": graph_prefix,
                            }
                        )
                        responses[arm].append(response)
                        timings[arm].append(response["ann_ms"])
                finally:
                    for worker in workers.values():
                        worker.close()
                for arm, binary in arm_binaries.items():
                    prefix = out / "graphs" / "timing" / fixture / backend / arm
                    matrix = np.memmap(
                        info["path"], dtype="<f4", mode="r", shape=(info["rows"], info["features"])
                    )
                    receipt = {
                        "arm": arm,
                        "fixture": fixture,
                        "backend": backend,
                        "source": source_identities[arm],
                        "protocol": "abba",
                        "binary": str(binary),
                        "binary_sha256": sha256(binary),
                        "input_sha256": info["sha256"],
                        "rows": info["rows"],
                        "features": info["features"],
                        "k_nonself": k,
                        "cap": args.cap,
                        "seed": args.seed,
                        "squared": args.squared,
                        "warmup": warmups[arm],
                        "ann_ms": timings[arm],
                        "median_ann_ms": statistics.median(timings[arm]),
                        "range_ann_ms": [min(timings[arm]), max(timings[arm])],
                        "responses": responses[arm],
                        "recall_graph_prefix": str(prefix),
                        "stderr_log": str(
                            out / "logs" / "timing" / fixture / backend / f"{arm}.stderr.log"
                        ),
                    }
                    if not args.timing_only:
                        receipt["recall"] = exact_tie_aware_recall(matrix, prefix, k)
                    manifest["receipts"].append(receipt)
                for arm, binary in arm_binaries.items():
                    stage = run_worker(
                        arm=arm,
                        binary=binary,
                        fixture=fixture,
                        input_info=info,
                        backend=backend,
                        k=k,
                        cap=args.cap,
                        seed=args.seed,
                        squared=args.squared,
                        out=out,
                        protocol="profile",
                        profile=True,
                        runs=1,
                        source=source_identities[arm],
                    )
                    stage["protocol"] = "profiled_stage_only"
                    stage["stage_only"] = True
                    manifest["receipts"].append(stage)
            if len(arm_binaries) == 1:
                arm, binary = next(iter(arm_binaries.items()))
                stage = run_worker(
                    arm=arm,
                    binary=binary,
                    fixture=fixture,
                    input_info=info,
                    backend=backend,
                    k=k,
                    cap=args.cap,
                    seed=args.seed,
                    squared=args.squared,
                    out=out,
                    protocol="profile",
                    profile=True,
                    runs=1,
                    source=source_identities[arm],
                )
                stage["protocol"] = "profiled_stage_only"
                stage["stage_only"] = True
                manifest["receipts"].append(stage)
            dump(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
