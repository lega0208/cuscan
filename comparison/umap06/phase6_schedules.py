"""Ordered item inventories for Phase 6 campaigns Q6, F6 and V6.

Pure construction: nothing here runs a command, builds a binary or touches the filesystem.
Each item is a dict with `kind`, `command` (argv list), `env` (overrides only) and
`binaries` (paths whose hashes the runner pins), plus descriptive metadata.
"""

import sys
from pathlib import Path

from comparison.umap06.spec import FIXTURES

BACKENDS = ("cuda", "cpu")
QUALITY_FIXTURES = ("mix_200k", "mix_50k", "curve_20k", "digits", "blobs_1k")
DEFAULTS = {
    "old": {"optimizer": "compatibility", "init": "spectral"},
    "new": {"optimizer": "adam", "init": "auto"},
}
STRICT_CUDA = {"CUSCAN_REQUIRE_CUDA": "1"}
FOCUSED_PYTHON = (
    "python/tests/test_standard_umap.py::test_defaults_are_auto_and_adam",
    "python/tests/test_standard_umap.py::test_default_fit_matches_explicit_adam_auto",
    "python/tests/test_standard_umap.py::test_native_default_arguments",
    "python/tests/test_standard_umap.py::test_default_surfaces_agree",
    "python/tests/test_standard_umap.py::test_phase4_settings_defaults",
    "python/tests/test_serialization.py::test_v5_library_defaults_round_trip",
    "python/tests/test_serialization.py::test_v5_explicit_previous_defaults_round_trip",
)


def item(kind, command, *, env=None, binaries=(), **metadata):
    return {
        "kind": kind,
        "command": [str(arg) for arg in command],
        "env": dict(env or {}),
        "binaries": [str(path) for path in binaries],
        **metadata,
    }


def run_module(*args):
    return [sys.executable, "-m", "comparison.umap06.run", *map(str, args)]


def quality(out, data, reference, upstream, upstream_hash):
    """Q6 in plan order; `--record-extension` only on the first cuscan item of each root."""
    roots = {name: Path(out) / name for name in ("fit", "transform", "seed47")}
    recorded = set()
    items = []

    def fit(root, fixture, backend, seeds, *, transform=False, halved=True):
        command = run_module(
            "cuscan",
            "--data", data,
            "--artifacts", roots[root],
            "--reference", reference,
            "--fixtures", fixture,
            "--backend", backend,
            "--seeds", *seeds,
            "--phase5",
            "--configs", "adam_recursive",
        )  # fmt: skip
        if halved:
            command += ["--scope", "halved-fit"]
        if not transform:
            command.append("--skip-transform")
        if root not in recorded:
            command.append("--record-extension")
            recorded.add(root)
        items.append(
            item(
                "fit",
                command,
                root=root,
                fixture=fixture,
                backend=backend,
                seeds=list(seeds),
                transform=transform,
            )
        )

    def score(root, fixture, *, fit_only=True):
        command = run_module(
            "score", "--data", data, "--artifacts", roots[root], "--fixtures", fixture
        )
        if fit_only:
            command.append("--fit-only")
        items.append(item("score", command, root=root, fixture=fixture, fit_only=fit_only))

    def receipt(label, evidence=None):
        command = run_module(
            "accept-phase5",
            "--data", data,
            "--artifacts", roots["fit"],
            "--upstream-reference", upstream,
            "--upstream-sha256", upstream_hash,
            "--scope", "halved-fit",
        )  # fmt: skip
        if evidence:
            command += ["--evidence-fixtures", *evidence]
        items.append(
            item(
                "receipt",
                command,
                root="fit",
                receipt=label,
                partial=evidence is not None,
                acceptance=str(roots["fit"] / "acceptance.json"),
            )
        )

    # 1. Receipt rows without mix_1m (19 rows), then partial receipt A.
    for fixture in QUALITY_FIXTURES:
        fit("fit", fixture, "cuda", (11, 29))
        fit("fit", fixture, "cpu", (11,) if fixture == "mix_200k" else (11, 29))
        score("fit", fixture)
    receipt("A", QUALITY_FIXTURES)
    # 2. G5: transforms, reporting only, scored with transform metrics.
    for fixture in ("curve_20k", "mix_50k"):
        for backend in BACKENDS:
            fit("transform", fixture, backend, (11,), transform=True)
        score("transform", fixture, fit_only=False)
    # 3. G3: seed 47 outside the halved scope, fit only, no receipt.
    for fixture in ("curve_20k", "mix_50k", "mix_200k"):
        for backend in BACKENDS:
            fit("seed47", fixture, backend, (47,), halved=False)
        score("seed47", fixture)
    # 4. mix_1m receipt rows, then receipt B over all 22 rows.
    for backend, seed in (("cuda", 11), ("cpu", 11), ("cuda", 29)):
        fit("fit", "mix_1m", backend, (seed,))
        score("fit", "mix_1m")
    receipt("B")
    return items


def epochs(fixture):
    return 200 if FIXTURES[fixture].rows > 10_000 else 500


def profiling(out, data, profile_binary, transform_binary, bench_binaries):
    """F6 in plan order; every item captures stderr and is reporting only."""
    out = Path(out)
    items = []

    def stages(fixture, backend, default, runs):
        spec = FIXTURES[fixture]
        trace = out / "stages" / f"{fixture}_{backend}_{default}.trace.json"
        settings = DEFAULTS[default]
        command = [
            profile_binary,
            Path(data) / fixture / "train.f32",
            spec.rows,
            spec.features,
            epochs(fixture),
            "--backend", backend,
            "--search", "approximate",
            "--optimizer", settings["optimizer"],
            "--init", settings["init"],
            "--seed", 11,
            "--runs", runs,
            "--trace", trace,
        ]  # fmt: skip
        items.append(
            item(
                "profile_stages",
                command,
                binaries=[profile_binary],
                fixture=fixture,
                backend=backend,
                default=default,
                pair=f"{fixture}_{backend}",
                runs=runs,
                trace=str(trace),
                capture_stderr=True,
            )
        )

    for backend, runs in (("cuda", 2), ("cpu", 1)):
        for fixture in ("mix_50k", "mix_200k"):
            for default in DEFAULTS:
                stages(fixture, backend, default, runs)
    for backend in BACKENDS:
        for optimizer in ("compatibility", "adam"):
            output = out / "transform" / f"{backend}_{optimizer}"
            items.append(
                item(
                    "profile_transform",
                    [
                        transform_binary,
                        "--backend",
                        backend,
                        "--optimizer",
                        optimizer,
                        "--output-dir",
                        output,
                    ],
                    binaries=[transform_binary],
                    backend=backend,
                    optimizer=optimizer,
                    pair=f"transform_{backend}",
                    output_dir=str(output),
                    capture_stderr=True,
                )
            )
    for backend in BACKENDS:
        binary = bench_binaries[backend]
        items.append(
            item(
                "bench",
                [binary, "--bench", "--quick"],
                binaries=[binary],
                backend=backend,
                capture_stderr=True,
            )
        )
    for backend, runs in (("cuda", 2), ("cpu", 1)):
        for default in DEFAULTS:
            stages("mix_1m", backend, default, runs)
    return items


def verification():
    """V6 in plan order, focused default tests before each full suite."""
    cargo = ["soldr", "cargo"]
    uv = ["uv", "run", "--no-sync"]
    return [
        item("test", [*cargo, "test", "-p", "umap-core", "default_"], focused=True),
        item("test", [*cargo, "test", "-p", "umap-core"]),
        item(
            "test",
            [*cargo, "test", "-p", "umap", "--test", "test_standard_estimator", "default_"],
            env=STRICT_CUDA,
            focused=True,
        ),
        item(
            "test",
            [*cargo, "test", "-p", "umap", "--test", "test_standard_serialization", "roundtrips"],
            env=STRICT_CUDA,
            focused=True,
        ),
        item(
            "test",
            [
                *cargo,
                "test",
                "-p",
                "umap",
                "--test",
                "test_standard_quality",
                "default_cluster_quality",
            ],
            env=STRICT_CUDA,
            focused=True,
        ),
        item("test", [*cargo, "test", "-p", "umap"], env=STRICT_CUDA),
        item("test", [*cargo, "test", "-p", "cuscan-common"]),
        item("test", [*uv, "pytest", *FOCUSED_PYTHON], env=STRICT_CUDA, focused=True),
        item("test", [*uv, "pytest", "python/tests"], env=STRICT_CUDA),
        item("lint", [*cargo, "fmt", "--all", "--", "--check"]),
        item("lint", [*cargo, "clippy", "--workspace", "--all-targets"]),
        item("lint", [*uv, "ruff", "check"]),
        item("lint", [*uv, "ruff", "format", "--check"]),
        item("lint", [*uv, "ty", "check"]),
    ]
