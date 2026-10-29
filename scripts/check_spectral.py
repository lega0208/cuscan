#!/usr/bin/env python3
"""Timeout-isolated Rust/SciPy spectral regression harness."""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.sparse import coo_matrix, diags, load_npz
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

TIMEOUT_SECONDS = 300
EIGENGAP_CLUSTER = 1e-6
ANGLE_LIMIT = 1e-4


def graph_arrays(path):
    data = np.load(path)
    aliases = {
        "n": ("n", "n_vertices", "n_samples"),
        "heads": ("heads", "head", "row", "rows", "sources"),
        "tails": ("tails", "tail", "col", "cols", "targets"),
        "weights": ("weights", "weight", "data"),
    }
    result = {}
    for name, keys in aliases.items():
        key = next((key for key in keys if key in data), None)
        if key is None:
            if name == "n":
                continue
            try:
                matrix = load_npz(path).tocoo()
            except Exception as exc:
                raise ValueError(
                    f"{path}: cannot identify COO {name}; available={list(data.keys())}"
                ) from exc
            return {
                "n": int(matrix.shape[0]),
                "heads": matrix.row.astype(np.int64),
                "tails": matrix.col.astype(np.int64),
                "weights": matrix.data.astype(np.float64),
            }
        result[name] = data[key]
    if "n" in result:
        result["n"] = int(np.asarray(result["n"]).reshape(-1)[0])
    result["heads"] = np.asarray(result["heads"], dtype=np.int64).reshape(-1)
    result["tails"] = np.asarray(result["tails"], dtype=np.int64).reshape(-1)
    result["weights"] = (
        np.asarray(result["weights"], dtype=np.float32).astype(np.float64).reshape(-1)
    )
    if "n" not in result:
        result["n"] = int(max(result["heads"].max(initial=-1), result["tails"].max(initial=-1)) + 1)
    if not (len(result["heads"]) == len(result["tails"]) == len(result["weights"])):
        raise ValueError(f"{path}: COO arrays have different lengths")
    return result


def synthetic_cases(manifest):
    cases = []
    for entry in json.loads(manifest.read_text(encoding="utf-8")):
        path = manifest.parent / f"{entry['name']}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"synthetic manifest graph missing: {path}")
        cases.append((entry["name"], graph_arrays(path), int(entry["dims"])))
    two_cliques = (
        [(i, j, 1.0) for i in range(64) for j in range(i + 1, 64)]
        + [(i + 64, j + 64, 1.0) for i in range(64) for j in range(i + 1, 64)]
        + [(63, 64, 1.0)]
    )
    heads, tails, weights = zip(*two_cliques, strict=True)
    cases.append(
        (
            "near-repeated-clique-blocks",
            {
                "n": 128,
                "heads": np.asarray(heads),
                "tails": np.asarray(tails),
                "weights": np.asarray(weights),
            },
            2,
        )
    )
    return cases


def symmetric_operator(graph):
    n = graph["n"]
    directed = {}
    for i, j, value in zip(graph["heads"], graph["tails"], graph["weights"], strict=True):
        directed[(int(i), int(j))] = directed.get((int(i), int(j)), 0.0) + float(value)
    edges = {}
    for (i, j), value in directed.items():
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        edges[key] = max(edges.get(key, 0.0), value, directed.get((j, i), 0.0))
    rows, cols, vals = [], [], []
    for (i, j), value in edges.items():
        rows.extend((i, j))
        cols.extend((j, i))
        vals.extend((value, value))
    adjacency = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    if np.any(degree <= 0) or not np.isfinite(degree).all():
        raise ValueError("comparison graph must be one positive-degree connected component")
    inv = 1 / np.sqrt(degree)
    shifted = diags(inv) @ adjacency @ diags(inv) + diags(np.ones(n))
    return shifted, np.sqrt(degree)


def _reference_worker(graph, dimensions, count, dense=False):
    operator, trivial = symmetric_operator(graph)
    trivial = trivial / np.linalg.norm(trivial)
    if dense or count >= graph["n"] - 1:
        values, vectors = np.linalg.eigh(operator.toarray())
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        keep = np.abs(vectors.T @ trivial) < 1 - 1e-6
        return values[keep], vectors[:, keep]
    while True:
        # Default ncv=20 can converge one vector per repeated eigenvalue and
        # falsely report a gap. Use a broad deterministic reference subspace.
        values, vectors = eigsh(
            operator,
            k=count,
            which="LA",
            tol=1e-12,
            ncv=(
                graph["n"] - 1 if graph["n"] <= 512 else min(graph["n"] - 1, max(64, 2 * count + 1))
            ),
            v0=np.random.default_rng(42).normal(size=graph["n"]),
        )
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        keep = np.abs(vectors.T @ trivial) < 1 - 1e-6
        nontrivial_values, nontrivial_vectors = values[keep], vectors[:, keep]
        boundary = dimensions - 1
        if (
            len(nontrivial_values) > dimensions
            and abs(nontrivial_values[boundary] - nontrivial_values[boundary + 1])
            / max(1.0, abs(nontrivial_values[boundary]))
            > EIGENGAP_CLUSTER
        ):
            return nontrivial_values, nontrivial_vectors
        if count >= graph["n"] - 2:
            values, vectors = np.linalg.eigh(operator.toarray())
            order = np.argsort(values)[::-1]
            values, vectors = values[order], vectors[:, order]
            keep = np.abs(vectors.T @ trivial) < 1 - 1e-6
            return values[keep], vectors[:, keep]
        count = min(graph["n"] - 2, count + 8)


def scipy_reference(graph, dimensions, timeout):
    with tempfile.TemporaryDirectory(prefix="cuscan-scipy-") as directory:
        request = pathlib.Path(directory) / "request.npz"
        response = pathlib.Path(directory) / "response.npz"
        np.savez(
            request,
            n=np.asarray([graph["n"]]),
            heads=graph["heads"],
            tails=graph["tails"],
            weights=graph["weights"],
            dimensions=np.asarray([dimensions]),
            count=np.asarray([min(graph["n"] - 2, dimensions + 2)]),
        )
        started = time.monotonic()
        try:
            subprocess.run(
                [sys.executable, __file__, "--scipy-worker", str(request), str(response)],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"SciPy eigsh exceeded {timeout}s timeout") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"SciPy eigsh failed: {exc.stderr}\n{exc.stdout}") from exc
        with np.load(response) as result:
            return result["values"], result["vectors"], time.monotonic() - started


def invoke_probe(graph, dims, seed, test_runner, timeout):
    request = {
        "n": graph["n"],
        "heads": graph["heads"].tolist(),
        "tails": graph["tails"].tolist(),
        "weights": [
            float(value) if np.isfinite(value) else str(value) for value in graph["weights"]
        ],
        "dimensions": dims,
        "seed": seed,
    }
    with tempfile.TemporaryDirectory(prefix="cuscan-spectral-") as directory:
        path = pathlib.Path(directory) / "graph.json"
        response_path = pathlib.Path(directory) / "result.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        started = time.monotonic()
        env = dict(os.environ)
        env["CUSCAN_SPECTRAL_REQUEST"] = str(path)
        env["CUSCAN_SPECTRAL_RESPONSE"] = str(response_path)
        try:
            child = subprocess.run(
                [
                    str(test_runner),
                    "--ignored",
                    "--exact",
                    "standard_umap::init::solver_tests::external_spectral_probe",
                    "--nocapture",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"external spectral test exceeded {timeout}s timeout") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"external spectral test failed: {exc.stderr}\n{exc.stdout}"
            ) from exc
        if not response_path.is_file():
            raise RuntimeError(
                f"external spectral test passed without writing {response_path}: {child.stderr}"
            )
        elapsed = time.monotonic() - started
        result = json.loads(response_path.read_text(encoding="utf-8"))
        if result.get("fallback") is not False:
            raise RuntimeError("probe result must explicitly report fallback=false")
        result["child_wall_seconds"] = elapsed
        result["native_stderr"] = child.stderr
        return result


def run_case(name, graph, dims, seed, test_runner, timeout):
    started = time.monotonic()
    if not np.isfinite(graph["weights"]).all() or np.any(graph["weights"] <= 0):
        result = invoke_probe(graph, dims, seed, test_runner, timeout)
        if not result.get("expected_rejection") or "invalid weight" not in result.get("error", ""):
            raise AssertionError(
                f"{name}: expected invalid-weight precheck rejection, got {result}"
            )
        return {**result, "case": name, "wall_seconds": time.monotonic() - started}
    operator, _ = symmetric_operator(graph)
    component_count, labels = connected_components(operator, directed=False)
    if component_count > 1:
        parts = []
        for label in range(component_count):
            vertices = np.flatnonzero(labels == label)
            if len(vertices) <= dims + 1:
                # Production uses its existing small-component random layout.
                continue
            remap = {int(vertex): index for index, vertex in enumerate(vertices)}
            mask = np.isin(graph["heads"], vertices) & np.isin(graph["tails"], vertices)
            child = {
                "n": len(vertices),
                "heads": np.asarray([remap[int(x)] for x in graph["heads"][mask]]),
                "tails": np.asarray([remap[int(x)] for x in graph["tails"][mask]]),
                "weights": graph["weights"][mask],
            }
            parts.append(
                run_case(
                    f"{name}/component-{len(vertices)}", child, dims, seed, test_runner, timeout
                )
            )
        if not parts:
            raise AssertionError(f"{name}: no component can supply {dims} spectral dimensions")
        return {"case": name, "components": parts, "wall_seconds": time.monotonic() - started}
    rust = invoke_probe(graph, dims, seed, test_runner, timeout)
    requested = dims + 1
    max_dim = max(64, 2 * requested)
    if rust["requested"] != requested or rust["max_dim"] != max_dim:
        raise AssertionError(f"{name}: probe production request differs from k/M contract")
    if rust["production_dense"] != (graph["n"] <= max(64, 2 * requested, max_dim)):
        raise AssertionError(f"{name}: production solver used the wrong dense/sparse cutoff")
    production_values = np.asarray(rust["production_eigenvalues"])
    production = np.asarray(rust["production_eigenvectors"], dtype=float).T
    diagnostic_values = np.asarray(rust["eigenvalues"])
    diagnostic = np.asarray(rust["eigenvectors"], dtype=float).T
    if (
        production.shape != (graph["n"], len(production_values))
        or len(production_values) != rust["n_converged"]
    ):
        raise AssertionError(f"{name}: production raw eigenpair counts/shapes disagree")
    ref_values, ref, scipy_seconds = scipy_reference(graph, dims, timeout)
    operator, trivial = symmetric_operator(graph)
    # Verify every production pair including the trivial mode, independently of Rust.
    production_applied = operator @ production
    production_residuals = np.linalg.norm(
        production_applied - production * production_values, axis=0
    ) / np.maximum(1.0, np.abs(2.0 - production_values))
    if (
        not np.isfinite(production_values).all()
        or not np.isfinite(production).all()
        or not np.isfinite(production_residuals).all()
    ):
        raise AssertionError(f"{name}: non-finite production raw eigenpair evidence")
    production_gram = production.T @ production
    production_gram_error = production_gram - np.eye(production.shape[1])
    if not np.isfinite(production_gram_error).all():
        raise AssertionError(f"{name}: non-finite production raw Gram entries")
    production_diagonal = float(np.max(np.abs(np.diag(production_gram_error))))
    production_offdiagonal = float(
        np.max(np.abs(production_gram_error - np.diag(np.diag(production_gram_error))))
    )
    production_max_residual = float(np.max(production_residuals))
    if (
        production_max_residual > 1e-6
        or production_diagonal > 1e-8
        or production_offdiagonal > 1e-8
    ):
        raise AssertionError(
            f"{name}: production raw diagnostics residual={production_max_residual} "
            f"diag_gram={production_diagonal} offdiag_gram={production_offdiagonal}"
        )
    nontrivial_mask = np.abs(production.T @ (trivial / np.linalg.norm(trivial))) < 1 - 1e-6
    production_nontrivial = production[:, nontrivial_mask]
    if production_nontrivial.shape[1] < dims:
        raise AssertionError(f"{name}: production returned fewer than requested nontrivial pairs")
    # The SciPy reference is extended through the entire boundary eigenvalue cluster.
    boundary = ref_values[dims - 1]
    reference_cluster_end = dims
    while reference_cluster_end < len(ref_values) and abs(
        ref_values[reference_cluster_end] - boundary
    ) <= EIGENGAP_CLUSTER * max(1.0, abs(boundary)):
        reference_cluster_end += 1
    if reference_cluster_end == len(ref_values) and len(ref_values) < graph["n"] - 1:
        raise AssertionError(f"{name}: SciPy reference did not close full boundary cluster")
    full_reference = ref[:, :reference_cluster_end]
    production_overlap = production_nontrivial[:, :dims].T @ full_reference
    production_singular = np.linalg.svd(production_overlap, compute_uv=False)
    production_angle = float(np.arccos(np.clip(production_singular[-1], -1.0, 1.0)))
    diagnostic_count = diagnostic.shape[1]
    if (
        diagnostic.shape != (graph["n"], len(diagnostic_values))
        or diagnostic_count != rust["cluster_end"]
    ):
        raise AssertionError(f"{name}: supplemental diagnostic eigenpair counts/shapes disagree")
    if diagnostic_count != reference_cluster_end:
        raise AssertionError(
            f"{name}: supplemental/reference full-cluster counts differ: "
            f"{diagnostic_count} vs {reference_cluster_end}"
        )
    overlap = diagnostic.T @ ref[:, :diagnostic_count]
    smallest = np.linalg.svd(overlap, compute_uv=False)[-1]
    angle = float(np.arccos(np.clip(smallest, -1.0, 1.0)))
    applied = operator @ diagnostic
    residuals = np.linalg.norm(applied - diagnostic * diagnostic_values, axis=0) / np.maximum(
        1.0, np.abs(2.0 - diagnostic_values)
    )
    gram = diagnostic.T @ diagnostic
    orthogonality = float(np.max(np.abs(gram - np.eye(gram.shape[0]))))
    residual = float(np.max(residuals))
    eigenvalue_error = float(np.max(np.abs(diagnostic_values - ref_values[:diagnostic_count])))
    if (
        residual > 1e-6
        or orthogonality > 1e-8
        or angle > ANGLE_LIMIT
        or production_angle > ANGLE_LIMIT
    ):
        raise AssertionError(
            f"{name}: diagnostic residual={residual} orthogonality={orthogonality} "
            f"angle={angle}; production containment angle={production_angle}"
        )
    return {
        "case": name,
        "n": graph["n"],
        "dimensions": dims,
        "count": production.shape[1],
        "eigengap": rust["eigengap"],
        "cluster_size": rust["cluster_end"] - rust["cluster_start"],
        "principal_angle_radians": angle,
        "production_containment_angle_radians": production_angle,
        "compared_eigenvalue_count": diagnostic_count,
        "max_eigenvalue_error": eigenvalue_error,
        "max_residual": residual,
        "orthogonality": orthogonality,
        "production_max_residual": production_max_residual,
        "production_max_diagonal_gram_error": production_diagonal,
        "production_max_offdiagonal_gram_error": production_offdiagonal,
        "rust_seconds": rust["elapsed_seconds"],
        "child_wall_seconds": rust["child_wall_seconds"],
        "scipy_seconds": scipy_seconds,
        "production_dense": rust["production_dense"],
        "diagnostic_dense": rust["diagnostic_dense"],
        "production_requested": rust["requested"],
        "production_max_dim": rust["max_dim"],
        "diagnostic_requested": rust["diagnostic_requested"],
        "diagnostic_n_converged": rust["diagnostic_n_converged"],
        "wall_seconds": time.monotonic() - started,
        "fallback": rust["fallback"],
    }


def main():
    parser = argparse.ArgumentParser(
        epilog=(
            "Build the private test transport with `cargo test --release "
            "--no-default-features --features tracing --lib --no-run "
            "--message-format=json`; parse the compiler-artifact executable for "
            "the lib test target and copy it to artifacts/approved-plan-tooling/"
            "spectral-probe-test. No build runs inside this harness."
        )
    )
    parser.add_argument(
        "--frozen-root", type=pathlib.Path, default=pathlib.Path("artifacts/approved-plan-baseline")
    )
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument(
        "--synthetic-manifest",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/approved-plan-baseline/synthetic/manifest.json"),
    )
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--test-runner",
        type=pathlib.Path,
        default=pathlib.Path("artifacts/approved-plan-tooling/spectral-probe-test"),
    )
    parser.add_argument("--failures", type=pathlib.Path)
    parser.add_argument("--scipy-worker", nargs=2, metavar=("REQUEST", "RESPONSE"))
    args = parser.parse_args()
    if args.scipy_worker:
        request, response = args.scipy_worker
        with np.load(request) as data:
            graph = {key: data[key] for key in ("n", "heads", "tails", "weights")}
            graph["n"] = int(graph["n"][0])
            dimensions = int(data["dimensions"][0])
            count = int(data["count"][0])
        values, vectors = _reference_worker(graph, dimensions, count)
        np.savez(response, values=values, vectors=vectors)
        return
    if args.failures is None:
        args.failures = (
            args.output.parent / "spectral-check-failures.json"
            if args.output
            else pathlib.Path("spectral-check-failures.latest.json")
        )
    if not args.test_runner.is_file():
        error = (
            f"prebuilt library test runner not found: {args.test_runner}; "
            "see harness help for the cargo test --no-run build and artifact copy command"
        )
        args.failures.write_text(
            json.dumps({"failures": [{"case": "startup", "error": error}]}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise FileNotFoundError(error)
    try:
        cases = synthetic_cases(args.synthetic_manifest)
    except Exception as exc:
        args.failures.write_text(
            json.dumps({"failures": [{"case": "synthetic-manifest", "error": repr(exc)}]}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        raise
    if not args.synthetic_only:
        for kind in ("raw", "unique"):
            for size in (15000, 50000):
                path = args.frozen_root / f"rust-{kind}-{size}-42" / "spectral-graph.npz"
                if not path.exists():
                    error = f"required frozen graph missing: {path}"
                    args.failures.write_text(
                        json.dumps({"failures": [{"case": "startup", "error": error}]}, indent=2)
                        + "\n",
                        encoding="utf-8",
                    )
                    raise FileNotFoundError(error)
                try:
                    graph = graph_arrays(path)
                except Exception as exc:
                    args.failures.write_text(
                        json.dumps(
                            {"failures": [{"case": str(path), "error": repr(exc)}]}, indent=2
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    raise
                cases.append((str(path), graph, 2))
    results, failures = [], []
    for name, graph, dims in cases:
        try:
            results.append(run_case(name, graph, dims, args.seed, args.test_runner, args.timeout))
        except Exception as exc:
            failures.append({"case": name, "error": repr(exc)})
            args.failures.write_text(
                json.dumps({"failures": failures, "completed": results}, indent=2) + "\n",
                encoding="utf-8",
            )
    output = json.dumps(
        {"timeout_seconds": args.timeout, "results": results, "failures": failures}, indent=2
    )
    args.failures.write_text(
        json.dumps({"failures": failures, "completed": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(
            f"{len(failures)} spectral cases failed; details persisted to {args.failures}"
        )
    print(output)


if __name__ == "__main__":
    main()
