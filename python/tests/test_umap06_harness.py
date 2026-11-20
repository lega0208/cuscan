"""Regression tests for the pinned UMAP 0.6 reference and evaluation harness."""

import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from comparison import umap_reference as reference


def test_campaign_deadline_reaps_nested_worker(tmp_path):
    import json
    import os
    import subprocess

    ledger = tmp_path / "ledger.jsonl"
    destination = tmp_path / "child"
    pidfile = tmp_path / "worker.pid"
    worker = (
        "import os, pathlib, time; "
        f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    nested = (
        "import sys; from comparison.umap06.run import child; "
        f"child([sys.executable, '-c', {worker!r}], {str(destination)!r}, None)"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "comparison.umap06.campaign",
            "run",
            "--ledger",
            str(ledger),
            "--budget-seconds",
            "10",
            "--reserve-seconds",
            "2",
            "--",
            sys.executable,
            "-c",
            nested,
        ],
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert json.loads((destination / "child.json").read_text())["timed_out"]
    assert json.loads(ledger.read_text().splitlines()[-1])["outcome"] == "deadline"


def test_reference_environment():
    reference.require_reference()


@pytest.mark.parametrize("optimizer", reference.OPTIMIZERS)
def test_reference_options_are_explicit(optimizer):
    options = reference.reference_kwargs(optimizer=optimizer, init="spectral", n_neighbors=15)
    assert options == {
        "optimizer": optimizer,
        "init": "spectral",
        "n_neighbors": 15,
        "compatibility_layout": False,
        "nearest_neighbor_backend": "pynndescent",
    }


@pytest.mark.parametrize(
    "options",
    [
        {"optimizer": "unknown", "init": "spectral"},
        {"optimizer": "adam", "init": "auto"},
        {"optimizer": "adam", "init": "spectral", "compatibility_layout": False},
    ],
)
def test_reference_rejects_implicit_or_overridden_options(options):
    with pytest.raises(ValueError):
        reference.reference_kwargs(**options)


@pytest.mark.parametrize("package", ["umap-learn", "pynndescent"])
def test_reference_rejects_wrong_versions(monkeypatch, package):
    actual_version = metadata.version
    monkeypatch.setattr(
        reference.metadata,
        "version",
        lambda name: "0.0.0" if name == package else actual_version(name),
    )
    with pytest.raises(RuntimeError, match=package):
        reference.require_reference()


def test_reference_preserves_explicit_rinnd_backend():
    options = reference.reference_kwargs(
        optimizer="adam", init="spectral", nearest_neighbor_backend="rinnd"
    )
    assert options["nearest_neighbor_backend"] == "rinnd"


def test_exact_trustworthiness_matches_sklearn():
    import numpy as np
    from comparison.umap06.quality import score_fit
    from sklearn.manifold import trustworthiness

    rng = np.random.default_rng(71)
    x, y = rng.normal(size=(120, 8)), rng.normal(size=(120, 2))
    assert score_fit(x, y)["trustworthiness"] == pytest.approx(
        trustworthiness(x, y, n_neighbors=15), abs=1e-12
    )


def test_identity_metrics_and_index_ties():
    import numpy as np
    from comparison.umap06.quality import exact_neighbors, score_fit, score_transform

    x = np.arange(80, dtype=float).reshape(40, 2)
    fit = score_fit(x, x)
    assert fit["trustworthiness"] == 1
    assert fit["preservation"] == 1
    transformed = score_transform(x, x, x + 0.1, x + 0.1)
    assert transformed["trustworthiness"] == 1
    assert transformed["preservation_qr"] == 1
    ids, distances = exact_neighbors(np.array([[0.0], [1.0], [-1.0], [2.0]]), k=2)
    assert ids[0].tolist() == [1, 2]
    assert distances[0].tolist() == [1, 1]


def test_sampled_trustworthiness_uncertainty():
    import numpy as np
    from comparison.umap06.quality import trustworthiness

    rng = np.random.default_rng(47)
    x, y = rng.normal(size=(3000, 5)), rng.normal(size=(3000, 2))
    full = trustworthiness(x, y)
    sample = trustworthiness(x, y, query_ids=rng.choice(3000, 1000, replace=False))
    assert abs(full["value"] - sample["value"]) <= 4 * sample["standard_error"]


def test_spawned_quality_workers_after_native_cpu_fit(monkeypatch):
    import numpy as np
    from comparison.umap06 import quality
    from sklearn.manifold import trustworthiness as sklearn_trustworthiness

    from cuscan import UMAP

    rng = np.random.default_rng(71)
    x, y = rng.normal(size=(128, 8)), rng.normal(size=(128, 2))
    query_ids = rng.permutation(len(x))
    model = UMAP(
        n_neighbors=10,
        n_epochs=100,
        random_state=19,
        backend="cpu",
        neighbor_search="exact",
        init="random",
        optimizer="compatibility",
        learning_rate=0.0,
    ).fit(x[:32].astype(np.float32))
    assert np.isfinite(model.embedding_).all()

    monkeypatch.setattr(quality.os, "cpu_count", lambda: 1)
    serial = quality.trustworthiness(x, y, query_ids=query_ids)
    assert serial["value"] == pytest.approx(
        sklearn_trustworthiness(x, y, n_neighbors=15), abs=1e-12
    )

    executor = quality.ProcessPoolExecutor
    contexts = []

    def recording_executor(*args, **kwargs):
        contexts.append((args[0], kwargs["mp_context"].get_start_method()))
        return executor(*args, **kwargs)

    monkeypatch.setattr(quality.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(quality, "ProcessPoolExecutor", recording_executor)
    spawned = quality.trustworthiness(x, y, query_ids=query_ids)
    assert contexts == [(2, "spawn")]
    assert all(np.isfinite(value) for value in spawned.values())
    assert spawned == pytest.approx(serial, abs=1e-12)


@pytest.mark.parametrize("invalid", ["nonfinite", "rows", "dimensions"])
def test_quality_rejects_invalid_embeddings(invalid):
    import numpy as np
    from comparison.umap06.quality import score_fit

    x = np.ones((40, 3))
    y = np.ones((40, 2))
    if invalid == "nonfinite":
        y[0, 0] = np.nan
    elif invalid == "rows":
        y = y[:-1]
    else:
        y = y[:, 0]
    with pytest.raises(ValueError):
        score_fit(x, y)


def test_fixture_determinism_and_tamper_rejection(tmp_path):
    import numpy as np
    from comparison.umap06.datasets import generate, load_fixture

    first, second = tmp_path / "first", tmp_path / "second"
    assert generate(first, ["blobs_1k"]) == generate(second, ["blobs_1k"])
    train, queries, labels = load_fixture(first, "blobs_1k")
    assert train.shape == (1000, 32) and queries[200].shape == (200, 32)
    assert labels is not None and labels.shape == (1000,)
    assert np.array_equal(train, load_fixture(second, "blobs_1k")[0])
    path = first / "blobs_1k" / "train.f32"
    with path.open("r+b") as stream:
        stream.write(b"BAD!")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_fixture(first, "blobs_1k")
    with pytest.raises(ValueError, match="verification failed"):
        generate(first, ["blobs_1k"])


def test_untracked_fixture_is_not_overwritten(tmp_path):
    from comparison.umap06.datasets import generate

    (tmp_path / "blobs_1k").mkdir()
    with pytest.raises(FileExistsError):
        generate(tmp_path, ["blobs_1k"])


def test_matrix_configuration_and_warmup_coverage():
    from comparison.umap06.spec import CONFIGS, FIXTURES, expected_runs, warmup_plan

    assert set(CONFIGS) == {
        "classic_spectral",
        "classic_random",
        "adam_spectral",
        "adam_recursive",
        "momentum_spectral",
    }
    upstream = expected_runs("upstream")
    assert len(upstream) == 122
    identities = {tuple(sorted(item.items())) for item in upstream}
    assert len(identities) == 122
    for config in CONFIGS.values():
        reference.reference_kwargs(optimizer=config.optimizer, init=config.init)
    assert warmup_plan(FIXTURES["blobs_1k"]) == {"fit_rows": [1000], "transform_rows": [512]}
    assert warmup_plan(FIXTURES["mix_50k"]) == {
        "fit_rows": [2000, 1000],
        "transform_rows": [512, 2048],
    }
    for run in expected_runs("cuscan"):
        assert run["config"].startswith("classic_")
        assert run["backend"] in {"cpu", "cuda"}


def test_acceptance_pooled_variance_and_one_sided_rule():
    from comparison.umap06.quality import acceptance_margins, acceptance_pass

    records = [
        {"fixture": "f", "configuration": config, "trustworthiness": value, "preservation": 0.5}
        for config, values in (("a", (0.7, 0.9)), ("b", (0.8, 0.8)))
        for value in values
    ]
    margins = acceptance_margins(records)["f"]
    assert margins["trustworthiness"] == pytest.approx(0.2)
    assert margins["preservation"] == 0.02
    assert acceptance_pass(0.99, 0.8, 0.01)
    assert not acceptance_pass(0.78, 0.8, 0.01)


def test_lattice_ties_match_bounded_fixture_evaluator():
    import importlib.util

    import numpy as np
    from comparison.umap06.quality import trustworthiness

    path = (
        Path(__file__).resolve().parents[2] / "crates/umap/tests/fixtures/generate_standard_umap.py"
    )
    spec = importlib.util.spec_from_file_location("fixture_generator", path)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    grid = np.indices((8, 8)).reshape(2, -1).T.astype(float)
    embedded = np.random.default_rng(11).normal(size=(64, 2))
    assert trustworthiness(grid, embedded)["value"] == pytest.approx(
        generator.exact_trustworthiness_bounded(grid, embedded, 15), abs=1e-12
    )


def test_batch_scoring_shares_exact_rank_results():
    import numpy as np
    from comparison.umap06.quality import score_fit, score_fits

    rng = np.random.default_rng(101)
    x = rng.normal(size=(90, 5))
    embeddings = {"first": rng.normal(size=(90, 2)), "second": rng.normal(size=(90, 4))}
    scores = score_fits(x, embeddings)
    for name, y in embeddings.items():
        assert scores[name] == score_fit(x, y)


def test_batch_transform_scoring_matches_individual_runs():
    import numpy as np
    from comparison.umap06.quality import score_transform, score_transforms

    rng = np.random.default_rng(71)
    train, queries = rng.normal(size=(100, 5)), rng.normal(size=(50, 5))
    fits = {name: rng.normal(size=(100, 2)) for name in ("a", "b")}
    embedded = {name: rng.normal(size=(50, 2)) for name in fits}
    together = score_transforms(train, fits, queries, embedded)
    for name in fits:
        assert together[name] == score_transform(train, fits[name], queries, embedded[name])


def test_self_inclusive_upstream_recall_is_not_capped():
    import numpy as np
    from comparison.umap06.quality import exact_neighbors, knn_recall

    x = np.random.default_rng(47).normal(size=(60, 5))
    ids, distances = exact_neighbors(x, k=15)
    graph = np.column_stack((np.arange(len(x)), ids[:, :14]))
    exact = knn_recall(x, np.arange(len(x)), distances, graph)
    assert exact["recall"] == 1
    graph[0, 1] = -1
    missing = knn_recall(x, np.arange(len(x)), distances, graph)
    assert missing["missing"] == 1
    assert missing["recall"] == pytest.approx(1 - 1 / (14 * len(x)))


def test_hand_computed_rank_penalty():
    import numpy as np
    from comparison.umap06.quality import score_fit

    # Row zero's nearest embedded neighbor is row two: input rank 2, penalty 1.
    x = np.array([[0.0], [1.0], [3.0], [7.0], [15.0]])
    y = np.array([[0.0], [4.0], [1.0], [8.0], [16.0]])
    scores = score_fit(x, y, query_ids=np.array([0]), k=1)
    assert scores["trustworthiness"] == pytest.approx(1 - 2 / (2 * 5 - 3 - 1))
    assert scores["preservation"] == 0


def test_cache_tampering_and_overwrite_refusal(tmp_path):
    from comparison.umap06.datasets import generate
    from comparison.umap06.quality import build_cache, verify_cache

    generate(tmp_path, ["blobs_1k"])
    path = build_cache(tmp_path, "blobs_1k")
    assert verify_cache(tmp_path, "blobs_1k") == []
    assert build_cache(tmp_path, "blobs_1k") == path
    with path.open("r+b") as stream:
        stream.write(b"corrupt")
    assert verify_cache(tmp_path, "blobs_1k")
    with pytest.raises(ValueError):
        build_cache(tmp_path, "blobs_1k")


def test_check_reports_missing_runs_and_tampered_fixture(tmp_path, capsys):
    from argparse import Namespace

    from comparison.umap06 import run
    from comparison.umap06.datasets import generate
    from comparison.umap06.quality import build_cache

    args = Namespace(
        data=tmp_path / "data", artifacts=tmp_path / "runs", reference=tmp_path / "reference"
    )
    generate(args.data, ["blobs_1k"])
    build_cache(args.data, "blobs_1k")
    run.require_extension(args.artifacts, establish=True)
    with pytest.raises(SystemExit) as error:
        run.check(args)
    assert error.value.code == 1
    assert "missing run:" in capsys.readouterr().err
    with (args.data / "blobs_1k" / "train.f32").open("r+b") as stream:
        stream.write(b"BAD!")
    assert any("hash mismatch" in message for message in run.evidence_errors(args))


def test_decision_requires_observed_values_and_exact_scope(tmp_path):
    from comparison.umap06.run import decisions

    (tmp_path / "decisions.md").write_text(
        '- {"date":"2026-11-16","trigger":"timeout","option":"drop"}\n'
    )
    with pytest.raises(ValueError, match="observed"):
        decisions(tmp_path)


def test_reference_output_refuses_overwrite(tmp_path):
    from comparison.umap06.run import write_new

    path = tmp_path / "reference.json"
    write_new(path, {"real": "evidence"})
    with pytest.raises(FileExistsError):
        write_new(path, {"replacement": True})


def test_summary_indexes_exact_reference_configurations(tmp_path, monkeypatch):
    from argparse import Namespace

    from comparison.umap06 import run
    from comparison.umap06.spec import CONFIGS

    args = Namespace(
        data=tmp_path / "data",
        artifacts=tmp_path / "runs",
        reference=tmp_path / "reference",
        freeze=False,
    )
    run.require_extension(args.artifacts, establish=True)
    grouped = {
        "classic_spectral": [
            {
                "fixture": "blobs_1k",
                "configuration": "classic_spectral",
                "components": 2,
                "precomputed_knn": False,
                "metrics": {},
            }
        ]
    }
    monkeypatch.setattr(run, "aggregate", lambda records: grouped)
    upstream, _ = run.summary_data(args)
    assert set(upstream["configurations"]) == set(CONFIGS)
    for name, config in CONFIGS.items():
        assert upstream["configurations"][name]["optimizer"] == config.optimizer
        assert upstream["configurations"][name]["init"] == config.init
        assert upstream["configurations"][name]["aggregates"] == grouped.get(name, [])


def test_measurement_freshness_distinguishes_offline_sources(tmp_path):
    import json

    from comparison.umap06 import run
    from comparison.umap06.datasets import fixture_hash, generate

    data = tmp_path / "data"
    generate(data, ["blobs_1k"])
    path = tmp_path / "campaign/children/0/output/11/run.json"
    path.parent.mkdir(parents=True)
    hashes = {name: "frozen-measurement" for name in run.MEASUREMENT_SOURCES}
    recorded = {**hashes, "comparison/umap06/run.py": "older-offline-checker"}
    invocation = {
        "extension": {"sha256": "extension"},
        "provenance": {"source_hashes": recorded},
        "fixture_hashes": {"blobs_1k": fixture_hash(data, "blobs_1k")},
    }
    (path.parents[4] / "invocation.json").write_text(json.dumps(invocation))
    command = [
        "python",
        "-m",
        "comparison.umap06.upstream_worker",
        "--fixture",
        "blobs_1k",
        "--config",
        "classic_spectral",
        "--seeds",
        "11",
        "--components",
        "2",
        "--data",
        str(data),
        "--out",
        str(path.parents[1]),
    ]
    (path.parents[2] / "child.json").write_text(
        json.dumps({"command": command, "environment": {"NUMBA_NUM_THREADS": "15"}})
    )
    record = {
        "role": "upstream",
        "fixture": "blobs_1k",
        "config": "classic_spectral",
        "components": 2,
        "seed": 11,
        "precomputed_knn": False,
    }
    current = {**hashes, "comparison/umap06/run.py": "repaired-offline-checker"}
    run.validate_invocation(path, record, data, "extension", current, {})
    current["comparison/umap06/upstream_worker.py"] = "different-worker"
    with pytest.raises(ValueError, match="Source mismatch"):
        run.validate_invocation(path, record, data, "extension", current, {})


def test_stale_metric_scores_do_not_count_as_current(tmp_path):
    import json

    from comparison.umap06 import run

    directory = tmp_path / "synthetic-score-generation"
    directory.mkdir()
    path = directory / "scores.json"
    document = {"provenance": {"source_hashes": {}}, "records": [{"stale": True}]}
    path.write_text(json.dumps(document))
    assert run.scored_runs(tmp_path, {}) == {}
    document["provenance"]["source_hashes"] = {
        name: run.sha256(run.ROOT / name) for name in run.SCORE_SOURCES
    }
    path.write_text(json.dumps(document))
    with pytest.raises(KeyError, match="identity"):
        run.scored_runs(tmp_path, {})


def test_stage_receipt_accepts_cuda_device_debug_spelling(tmp_path, monkeypatch):
    import json

    from comparison.umap06 import run

    root = tmp_path / "artifacts"
    campaign = root / "synthetic-stages-generation"
    directory = campaign / "stages/mix_50k-cuda"
    directory.mkdir(parents=True)
    monkeypatch.setattr(run, "require_extension", lambda root: {"sha256": "extension"})
    monkeypatch.setattr(run.datasets, "fixture_hash", lambda root, name: "fixture")
    command = [
        "profile_standard_stages",
        str(tmp_path / "data/mix_50k/train.f32"),
        "50000",
        "32",
        "200",
        "--backend",
        "cuda",
        "--search",
        "approximate",
        "--init",
        "spectral",
        "--seed",
        "11",
        "--runs",
        "3",
        "--trace",
        str(directory / "trace.json"),
    ]
    (directory / "child.json").write_text(json.dumps({"returncode": 0, "command": command}))
    (campaign / "provenance.json").write_text(
        json.dumps({"extension": {"sha256": "extension"}, "fixture_hashes": {"mix_50k": "fixture"}})
    )
    (campaign / "build").mkdir()
    (campaign / "build/child.json").write_text(json.dumps({"returncode": 0}))
    output = {
        "backend": "Cuda { device_index: 0 }",
        "epochs": 200,
        "neighbors": 15,
        "fit_seconds": [1, 1, 1],
        "stage_seconds": {"optimization": [0.1, 0.1, 0.1]},
    }
    (directory / "stdout.txt").write_text(json.dumps(output))
    (directory / "stderr.txt").write_text("")
    (directory / "trace.json").write_text(json.dumps({"traceEvents": [{"name": "fit"}]}))
    assert run.stage_evidence(root)["mix_50k-cuda"]["outcome"] == "success"


def test_fixture_diff_preserves_exact_fields_and_float_tolerances():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "crates/umap/tests/fixtures/diff_standard_umap.py"
    spec = importlib.util.spec_from_file_location("fixture_diff", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = {"graph_cases": [{"rho": [1.0], "heads": [0]}]}
    after = {"graph_cases": [{"rho": [1.000001], "heads": [0]}]}
    records, violations = module.compare(before, after)
    assert violations == 0
    assert any(record["max_ulp_distance"] for record in records)
    after["graph_cases"][0]["heads"] = [1]
    assert module.compare(before, after)[1] == 1
    clocks = {"optimizer_probes": [{"epoch_of_next_sample": [1.0]}]}
    changed = {"optimizer_probes": [{"epoch_of_next_sample": [1.000001]}]}
    assert module.compare(clocks, changed)[1] == 1


@pytest.fixture
def phase2_campaign(tmp_path, monkeypatch):
    return build_campaign(tmp_path, monkeypatch)


def build_campaign(
    tmp_path,
    monkeypatch,
    names: tuple[str, ...] = ("tiny",),
    configs: tuple[str, ...] = ("classic_spectral",),
    settings=None,
):
    """Synthetic scored campaign; ``settings`` adds explicit worker flags and run records."""
    import json
    from argparse import Namespace

    from comparison.umap06 import run
    from comparison.umap06.spec import Fixture

    fixtures = {name: Fixture(name, 2, 2, (1,), (11, 29)) for name in names}
    monkeypatch.setattr(run, "FIXTURES", fixtures)
    rows: list[dict[str, Any]] = [
        dict(
            fixture=name,
            config=config,
            seed=seed,
            components=components,
            precomputed_knn=exact,
            backend=backend,
        )
        for name in names
        for config in configs
        for backend in ("cpu", "cuda")
        for components, exact in ((2, False), (16, False), (2, True))
        for seed in fixtures[name].seeds
    ]
    monkeypatch.setattr(run, "expected_runs", lambda role: rows)
    monkeypatch.setattr(run.datasets, "fixture_hash", lambda data, name: "fixture-hash")
    monkeypatch.setattr(run.datasets, "verify", lambda data: [])
    monkeypatch.setattr(run.quality, "verify_cache", lambda data, name: [])
    monkeypatch.setattr(run, "extension", lambda: {"path": "extension", "sha256": "ext"})
    monkeypatch.setattr(run, "validate_run", lambda path, record: None)
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "extension.json").write_text(json.dumps({"extension": {"sha256": "ext"}}))
    data = tmp_path / "data"
    data.mkdir()
    sources = {name: run.sha256(run.ROOT / name) for name in run.MEASUREMENT_SOURCES}
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        record = dict(row, role="cuscan")
        if settings is not None:
            record["settings"] = settings
        path = (
            root
            / f"campaign-{index}"
            / "children"
            / "child"
            / "output"
            / str(row["seed"])
            / "run.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(record))
        invocation = {
            "extension": {"sha256": "ext"},
            "provenance": {"source_hashes": sources},
            "fixture_hashes": {row["fixture"]: "fixture-hash"},
        }
        (path.parents[4] / "invocation.json").write_text(json.dumps(invocation))
        command = [
            "python",
            "-m",
            "comparison.umap06.cuscan_worker",
            "--fixture",
            row["fixture"],
            "--backend",
            row["backend"],
            "--init",
            run.CONFIGS[row["config"]].init,
            "--optimizer",
            run.CONFIGS[row["config"]].optimizer,
            "--components",
            str(row["components"]),
            "--seeds",
            str(row["seed"]),
            "--data",
            str(data),
            "--out",
            str(path.parents[1]),
            "--neighbor-search",
            "exact" if row["precomputed_knn"] else "approximate",
            *([] if settings is None else run.settings_arguments(settings)),
        ]
        (path.parents[2] / "child.json").write_text(
            json.dumps({"command": command, "returncode": 0})
        )
        records.append(
            {
                "identity": record,
                "run": str(path),
                "fit": {
                    "trustworthiness": 0.9,
                    "trustworthiness_se": 0.0,
                    "preservation": 0.9,
                    "diagnostics": {"pair_spearman": 0, "axis_std": [], "bounding_box": []},
                },
                "transforms": {}
                if row["precomputed_knn"]
                else {
                    "1": {"trustworthiness": 0.9, "trustworthiness_se": 0.0, "preservation_qr": 0.9}
                },
            }
        )
    score_path = root / "synthetic-score-1" / "scores.json"
    score_path.parent.mkdir()
    score_doc = {
        "provenance": {
            "source_hashes": {name: run.sha256(run.ROOT / name) for name in run.SCORE_SOURCES}
        },
        "records": records,
    }
    score_path.write_text(json.dumps(score_doc))
    upstream_records = [r for r in records if r["identity"]["backend"] == "cpu"]
    aggregates = run.aggregate(upstream_records)
    configurations = {
        name: {
            "optimizer": config.optimizer,
            "init": config.init,
            "aggregates": aggregates.get(name, []),
        }
        for name, config in run.CONFIGS.items()
    }
    provenance = dict(
        run.wheel_provenance(),
        fixture_hashes={name: "fixture-hash" for name in names},
        source_hashes=sources,
    )
    reference_runs = [
        {
            "identity": {
                "config": name,
                "resolved_optimizer": config.optimizer,
                "resolved_init": config.init,
                "params": {
                    "compatibility_layout": False,
                    "nearest_neighbor_backend": "pynndescent",
                },
            }
        }
        for name, config in run.CONFIGS.items()
    ]
    ref = {
        "schema": "cuscan-umap06-upstream-reference-v1",
        "provenance": provenance,
        "configurations": configurations,
        "aggregates": aggregates,
        "runs": reference_runs,
        "acceptance_rule": {
            "margins": {name: {"trustworthiness": 0.01, "preservation": 0.02} for name in names},
            "transform_margins_by_batch": {
                "1": {
                    name: {"transform_trustworthiness": 0.01, "preservation_qr": 0.02}
                    for name in names
                }
            },
        },
    }
    reference = tmp_path / "upstream.json"
    reference.write_text(json.dumps(ref))
    args = Namespace(
        artifacts=root,
        data=data,
        upstream_reference=reference,
        upstream_sha256=run.sha256(reference),
        unscored_fixtures=None,
        fit_only=False,
        exemption_reason=None,
    )
    return run, args, score_path, score_doc, ref


def test_phase2_acceptance_complete_and_fit_only(phase2_campaign):
    run, args, _, _, _ = phase2_campaign
    before = args.upstream_reference.read_bytes()
    result = run.phase2_acceptance(args)
    assert result["passed"], result["errors"]
    assert result["total_comparisons"] == 20
    assert all(
        not item["metric"].startswith("transform_")
        for item in result["comparisons"]
        if item["group"][3]
    )
    assert args.upstream_reference.read_bytes() == before
    assert result["unscored_fixtures"] == []


def test_phase2_acceptance_fit_only_ignores_transform_regression(phase2_campaign):
    import json

    run, args, score_path, score_doc, _ = phase2_campaign
    for record in score_doc["records"]:
        for metrics in record["transforms"].values():
            metrics["trustworthiness"] = 0.1
    score_path.write_text(json.dumps(score_doc))
    assert not run.phase2_acceptance(args)["passed"]
    args.fit_only = True
    missing_reason = run.phase2_acceptance(args)
    assert not missing_reason["passed"]
    assert any("--exemption-reason" in error for error in missing_reason["errors"])
    args.exemption_reason = "transform gap tracked as Phase 7"
    result = run.phase2_acceptance(args)
    assert result["passed"], result["errors"]
    assert result["fit_only"] is True
    assert result["exemption_reason"] == "transform gap tracked as Phase 7"
    transforms = [c for c in result["comparisons"] if c["metric"].startswith("transform_")]
    assert transforms and all(not c["gating"] for c in transforms)
    assert any(not c["passed"] for c in transforms)
    assert all(c["gating"] for c in result["comparisons"] if c not in transforms)


def test_phase2_acceptance_only_unscores_mix_1m(phase2_campaign):
    run, args, _, _, _ = phase2_campaign
    args.unscored_fixtures = ["tiny"]
    args.exemption_reason = "test"
    result = run.phase2_acceptance(args)
    assert not result["passed"]
    assert result["unscored_fixtures"] == ["tiny"]
    assert any("only mix_1m may be unscored" in error for error in result["errors"])


def test_phase2_known_exceptions_are_exactly_the_cuda_random_gap():
    from comparison.umap06 import run

    assert run.PHASE2_KNOWN_EXCEPTIONS == {
        ("cuda", "curve_20k", "classic_random", 2, False, "preservation")
    }


def test_rank_counting_matches_full_lexsort_with_ties():
    import numpy as np
    from comparison.umap06 import quality

    rng = np.random.default_rng(5)
    squared = rng.integers(0, 6, size=400).astype(np.float64)
    squared[17] = np.inf
    order = np.lexsort((np.arange(len(squared)), squared))
    expected = np.empty(len(squared), dtype=np.int64)
    expected[order] = np.arange(1, len(squared) + 1)
    ids = np.array([0, 3, 17, 399, 200, 201])
    actual = quality._ranks(squared, np.sort(squared), ids)
    np.testing.assert_array_equal(actual, expected[ids])


@pytest.mark.parametrize(
    "mutation",
    ["inequality", "missing_seed", "missing_backend", "missing_variant", "batch", "stale_score"],
)
def test_phase2_acceptance_rejects_incomplete_or_bad_scores(phase2_campaign, mutation):
    import json

    run, args, path, doc, _ = phase2_campaign
    if mutation == "inequality":
        for record in doc["records"]:
            record["fit"]["trustworthiness"] = 0.1
    elif mutation == "batch":
        doc["records"][0]["transforms"] = {}
    elif mutation == "stale_score":
        doc["provenance"]["source_hashes"] = {}
    else:
        for record in list(doc["records"]):
            identity = record["identity"]
            remove = (
                (mutation == "missing_seed" and identity["seed"] == 11)
                or (mutation == "missing_backend" and identity["backend"] == "cuda")
                or (mutation == "missing_variant" and identity["components"] == 16)
            )
            if remove:
                Path(record["run"]).unlink()
                doc["records"].remove(record)
    path.write_text(json.dumps(doc))
    assert not run.phase2_acceptance(args)["passed"]


@pytest.mark.parametrize(
    "mutation", ["hash", "extension", "source", "duplicate", "empty", "margin", "reference_source"]
)
def test_phase2_acceptance_integrity_failures(phase2_campaign, mutation):
    import json
    import shutil

    run, args, path, doc, ref = phase2_campaign
    before = args.upstream_reference.read_bytes()
    if mutation == "hash":
        args.upstream_sha256 = "0" * 64
    elif mutation == "extension":
        (args.artifacts / "extension.json").write_text(json.dumps({"extension": {"sha256": "old"}}))
    elif mutation == "source":
        run_path = Path(doc["records"][0]["run"])
        invocation = run_path.parents[4] / "invocation.json"
        value = json.loads(invocation.read_text())
        value["provenance"]["source_hashes"][run.MEASUREMENT_SOURCES[0]] = "stale"
        invocation.write_text(json.dumps(value))
    elif mutation == "duplicate":
        run_path = Path(doc["records"][0]["run"])
        shutil.copytree(run_path.parents[4], args.artifacts / "duplicate")
    elif mutation == "empty":
        path.unlink()
    elif mutation == "reference_source":
        ref["provenance"]["source_hashes"]["comparison/umap06/upstream_worker.py"] = "stale"
        args.upstream_reference.write_text(json.dumps(ref))
        args.upstream_sha256 = run.sha256(args.upstream_reference)
        before = args.upstream_reference.read_bytes()
    else:
        del ref["acceptance_rule"]["margins"]["tiny"]["trustworthiness"]
        args.upstream_reference.write_text(json.dumps(ref))
        args.upstream_sha256 = run.sha256(args.upstream_reference)
        before = args.upstream_reference.read_bytes()
    assert not run.phase2_acceptance(args)["passed"]
    assert args.upstream_reference.read_bytes() == before


def test_phase2_command_writes_failure_receipt(phase2_campaign):
    run, args, _, _, _ = phase2_campaign
    args.upstream_sha256 = "wrong"
    with pytest.raises(SystemExit) as exc:
        run.accept_phase2(args)
    assert exc.value.code == 1
    receipts = list(args.artifacts.glob("*-accept-phase2-*/acceptance.json"))
    assert len(receipts) == 1
    assert not run.read_json(receipts[0])["passed"]


def test_phase3_matrix_uses_frozen_upstream_variants():
    from comparison.umap06 import run

    rows = run.phase3_runs()
    modern = [row for row in rows if row["config"] in ("adam_spectral", "momentum_spectral")]
    upstream = [
        row
        for row in run.expected_runs("upstream")
        if row["config"] in ("adam_spectral", "momentum_spectral") and row["fixture"] != "mix_1m"
    ]
    assert {run.identity(dict(row, role="cuscan")) for row in modern} == {
        run.identity(dict(row, role="cuscan", backend=backend))
        for row in upstream
        for backend in ("cpu", "cuda")
    }
    assert {row["config"] for row in modern} == {"adam_spectral", "momentum_spectral"}
    assert any(row["fixture"] == "mix_200k" for row in modern)
    assert not any(row["fixture"] == "mix_1m" for row in modern)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "provenance"])
def test_phase3_acceptance_rejects_bad_evidence(phase2_campaign, monkeypatch, mutation):
    import json
    import shutil

    run, args, _, doc, _ = phase2_campaign
    rows = run.expected_runs("cuscan")
    monkeypatch.setattr(run, "phase3_runs", lambda: rows)
    assert run.phase2_acceptance(args, phase3=True)["passed"]
    run_path = Path(doc["records"][0]["run"])
    if mutation == "missing":
        run_path.unlink()
    elif mutation == "duplicate":
        shutil.copytree(run_path.parents[4], args.artifacts / "duplicate")
    else:
        invocation = run_path.parents[4] / "invocation.json"
        value = json.loads(invocation.read_text())
        value["provenance"]["source_hashes"][run.MEASUREMENT_SOURCES[0]] = "wrong"
        invocation.write_text(json.dumps(value))
    assert not run.phase2_acceptance(args, phase3=True)["passed"]


def test_phase3_transform_is_reported_but_not_gating(phase2_campaign, monkeypatch):
    import json

    run, args, path, doc, _ = phase2_campaign
    monkeypatch.setattr(run, "phase3_runs", lambda: run.expected_runs("cuscan"))
    for record in doc["records"]:
        for metrics in record["transforms"].values():
            metrics["trustworthiness"] = 0.1
    path.write_text(json.dumps(doc))
    result = run.phase2_acceptance(args, phase3=True)
    assert result["passed"], result["errors"]
    transforms = [item for item in result["comparisons"] if item["metric"].startswith("transform_")]
    assert transforms and any(not item["passed"] for item in transforms)
    assert all(not item["gating"] for item in transforms)


PHASE4_PERMITTED = ("digits", "blobs_1k", "curve_20k")
PHASE4_SETTINGS = {
    "negative_selection_range": 4096,
    "negative_sample_scale": 0.5,
    "exclude_graph_neighbors": True,
    "negative_sample_scale_adaptation_samples": 64,
}


def phase4_args(args, settings=PHASE4_SETTINGS, evidence=None):
    args.negative_selection_range = settings["negative_selection_range"]
    args.negative_sample_scale = settings["negative_sample_scale"]
    args.exclude_graph_neighbors = "true" if settings["exclude_graph_neighbors"] else "false"
    args.negative_sample_scale_adaptation_samples = settings[
        "negative_sample_scale_adaptation_samples"
    ]
    args.evidence_fixtures = evidence
    return args


def test_phase4_matrix_adds_frozen_modern_spectral_variants_only():
    from comparison.umap06 import run

    rows = run.phase4_runs()
    keys = [run.identity(dict(row, role="cuscan")) for row in rows]
    assert len(keys) == len(set(keys))
    classic = {run.identity(dict(row, role="cuscan")) for row in run.expected_runs("cuscan")}
    upstream_modern = [
        row for row in run.expected_runs("upstream") if row["config"] in run.MODERN_SPECTRAL
    ]
    modern = {
        run.identity(dict(row, role="cuscan", backend=backend))
        for row in upstream_modern
        for backend in ("cpu", "cuda")
    }
    assert set(keys) == classic | modern
    assert {row["config"] for row in rows} == {
        "classic_spectral",
        "classic_random",
        "adam_spectral",
        "momentum_spectral",
    }
    phase3 = {run.identity(dict(row, role="cuscan")) for row in run.phase3_runs()}
    assert phase3 < set(keys)
    # Every upstream modern fixture is required; Phase 3's large-fixture omission is not inherited.
    assert {row["fixture"] for row in upstream_modern} == {
        row["fixture"] for row in rows if row["config"] in run.MODERN_SPECTRAL
    }
    selected = run.selected_matrix(
        "cuscan", fixtures=list(PHASE4_PERMITTED), backend="cuda", phase4=True
    )
    assert selected and {row["fixture"] for row in selected} <= set(PHASE4_PERMITTED)
    assert {row["backend"] for row in selected} == {"cuda"}
    assert {"adam_spectral", "momentum_spectral", "classic_random"} <= {
        row["config"] for row in selected
    }
    with pytest.raises(ValueError):
        run.selected_matrix("cuscan", phase3=True, phase4=True)


def test_phase4_settings_arguments_round_trip_through_worker_parser():
    from argparse import Namespace

    from comparison.umap06 import run

    flags = run.settings_arguments(PHASE4_SETTINGS)
    assert flags == [
        "--negative-selection-range",
        "4096",
        "--exclude-graph-neighbors",
        "true",
        "--negative-sample-scale-adaptation-samples",
        "64",
        "--negative-sample-scale",
        "0.5",
    ]
    assert run.requested_settings(phase4_args(Namespace())) == PHASE4_SETTINGS
    assert run.requested_settings(Namespace()) == run.DEFAULT_SETTINGS
    assert "--negative-sample-scale" not in run.settings_arguments(run.DEFAULT_SETTINGS)
    with pytest.raises(ValueError):
        run.normalize_settings(dict(PHASE4_SETTINGS, exclude_graph_neighbors=1))


def test_phase4_worker_forwards_settings_to_fit_and_replay(tmp_path, monkeypatch):
    import json
    import sys

    import numpy as np
    from comparison.umap06 import cuscan_worker, run

    import cuscan

    constructed = []

    class FakeUMAP:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.n_components = kwargs["n_components"]
            self._model = self
            self.init = kwargs["init"]
            self.optimizer = kwargs["optimizer"]
            for name, value in kwargs.items():
                setattr(self, name, value)

        def export_state(self):
            return {}, {"init": self.init, "optimizer": self.optimizer}

        def fit_transform(self, x):
            return np.zeros((len(x), self.n_components), dtype=np.float32)

    train = np.zeros((5, 3), dtype=np.float32)
    monkeypatch.setattr(cuscan, "UMAP", FakeUMAP)
    monkeypatch.setattr(cuscan_worker, "load_fixture", lambda data, name: (train, {}, None))
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cuscan_worker",
            "--fixture",
            "digits",
            "--backend",
            "cpu",
            "--init",
            "spectral",
            "--optimizer",
            "adam",
            "--seeds",
            "11",
            "--out",
            str(out),
            "--neighbor-search",
            "exact",
            *run.settings_arguments(PHASE4_SETTINGS),
        ],
    )
    cuscan_worker.main()
    assert len(constructed) == 2
    for kwargs in constructed:
        assert {name: kwargs[name] for name in PHASE4_SETTINGS} == PHASE4_SETTINGS
    record = json.loads((out / "11" / "run.json").read_text())
    assert record["settings"] == PHASE4_SETTINGS
    assert record["config"] == "adam_spectral"


@pytest.fixture
def phase4_campaign(tmp_path, monkeypatch):
    run, args, path, doc, ref = build_campaign(
        tmp_path,
        monkeypatch,
        names=PHASE4_PERMITTED,
        configs=("classic_random", "adam_spectral", "momentum_spectral"),
        settings=PHASE4_SETTINGS,
    )
    rows = [dict(record["identity"]) for record in doc["records"]]
    for row in rows:
        row.pop("role")
        row.pop("settings")
    monkeypatch.setattr(run, "phase4_runs", lambda: rows)
    return run, phase4_args(args), path, doc, ref


def test_phase4_acceptance_complete_with_explicit_settings(phase4_campaign):
    run, args, _, _, _ = phase4_campaign
    result = run.phase2_acceptance(args, phase4=True)
    assert result["passed"], (result["errors"], result["incomplete"])
    assert result["schema"] == "cuscan-umap06-phase4-acceptance-v1"
    assert result["complete"] and result["excluded_fixtures"] == []
    assert result["settings"] == PHASE4_SETTINGS
    assert result["fit_only"] and "unscored_fixtures" not in result
    assert result["known_exceptions"] == [
        ["cuda", "curve_20k", "classic_random", 2, False, "preservation"]
    ]


def test_phase4_evidence_subset_is_reported_but_never_accepted(phase4_campaign):
    run, args, _, _, _ = phase4_campaign
    args.evidence_fixtures = ["digits", "curve_20k"]
    result = run.phase2_acceptance(args, phase4=True)
    assert result["scoped_quality_passed"], result["errors"]
    assert not result["complete"] and not result["passed"]
    assert result["excluded_fixtures"] == ["blobs_1k"]
    assert {item["group"][0] for item in result["comparisons"]} == {"digits", "curve_20k"}


@pytest.mark.parametrize(
    "mutation",
    ["missing_run", "missing_settings", "other_settings", "tampered_settings", "fit_regression"],
)
def test_phase4_acceptance_fails_closed_on_omissions(phase4_campaign, mutation):
    import json

    run, args, path, doc, _ = phase4_campaign
    target = next(
        r
        for r in doc["records"]
        if r["identity"]["fixture"] == "blobs_1k" and r["identity"]["config"] == "adam_spectral"
    )
    run_path = Path(target["run"])
    if mutation == "missing_run":
        run_path.unlink()
        doc["records"].remove(target)
        path.write_text(json.dumps(doc))
    elif mutation == "missing_settings":
        record = json.loads(run_path.read_text())
        del record["settings"]
        run_path.write_text(json.dumps(record))
    elif mutation == "other_settings":
        phase4_args(args, dict(PHASE4_SETTINGS, exclude_graph_neighbors=False))
    elif mutation == "tampered_settings":
        record = json.loads(run_path.read_text())
        record["settings"]["negative_sample_scale_adaptation_samples"] = 128
        run_path.write_text(json.dumps(record))
    else:
        for record in doc["records"]:
            if record["identity"]["config"] == "momentum_spectral":
                record["fit"]["trustworthiness"] = 0.1
        path.write_text(json.dumps(doc))
    result = run.phase2_acceptance(args, phase4=True)
    assert not result["passed"] and result["errors"]


def test_phase4_exception_is_not_extended(phase4_campaign):
    import json

    run, args, path, doc, _ = phase4_campaign
    for record in doc["records"]:
        identity = record["identity"]
        if identity["fixture"] == "curve_20k" and identity["config"] == "classic_random":
            if (identity["backend"], identity["components"], identity["precomputed_knn"]) == (
                "cuda",
                2,
                False,
            ):
                record["fit"]["preservation"] = 0.1
    path.write_text(json.dumps(doc))
    result = run.phase2_acceptance(args, phase4=True)
    assert result["passed"], result["errors"]
    known = [item for item in result["comparisons"] if item["known_exception"]]
    assert known and all(
        item["group"] == ["curve_20k", "classic_random", 2, False] for item in known
    )
    for record in doc["records"]:
        identity = record["identity"]
        if identity["fixture"] == "curve_20k" and identity["config"] == "adam_spectral":
            if identity["backend"] == "cuda":
                record["fit"]["preservation"] = 0.1
    path.write_text(json.dumps(doc))
    assert not run.phase2_acceptance(args, phase4=True)["passed"]


def test_phase4_command_writes_incomplete_receipt(phase4_campaign):
    run, args, _, _, _ = phase4_campaign
    args.evidence_fixtures = ["digits"]
    with pytest.raises(SystemExit) as exc:
        run.accept_phase4(args)
    assert exc.value.code == 1
    receipts = list(args.artifacts.glob("*-accept-phase4-*/acceptance.json"))
    assert len(receipts) == 1
    receipt = run.read_json(receipts[0])
    assert not receipt["passed"] and not receipt["complete"]
    assert receipt["excluded_fixtures"] == ["blobs_1k", "curve_20k"]


def run_fake_worker(tmp_path, monkeypatch, algorithm_args):
    """Run cuscan_worker.main with a fake UMAP; return (constructed kwargs, record)."""
    import json
    import sys

    import numpy as np
    from comparison.umap06 import cuscan_worker

    import cuscan

    constructed = []

    class FakeUMAP:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self.n_components = kwargs["n_components"]
            self.init = kwargs.get("init", "auto")
            self.optimizer = kwargs.get("optimizer", "adam")
            self._model = self
            for name, value in kwargs.items():
                setattr(self, name, value)

        def export_state(self):
            return {}, {"init": self.init, "optimizer": self.optimizer}

        def fit_transform(self, x):
            return np.zeros((len(x), self.n_components), dtype=np.float32)

    train = np.zeros((5, 3), dtype=np.float32)
    monkeypatch.setattr(cuscan, "UMAP", FakeUMAP)
    monkeypatch.setattr(cuscan_worker, "load_fixture", lambda data, name: (train, {}, None))
    out = tmp_path / "out"
    argv = ["cuscan_worker", "--fixture", "digits", "--backend", "cpu", *algorithm_args]
    argv += ["--seeds", "11", "--out", str(out), "--neighbor-search", "exact"]
    monkeypatch.setattr(sys, "argv", argv)
    cuscan_worker.main()
    return constructed, json.loads((out / "11" / "run.json").read_text())


def test_worker_library_defaults_omits_algorithm_keywords(tmp_path, monkeypatch):
    constructed, record = run_fake_worker(tmp_path, monkeypatch, ["--library-defaults"])
    assert len(constructed) == 2  # main fit and in-process repeat fit
    for kwargs in constructed:
        assert "optimizer" not in kwargs and "init" not in kwargs
    assert record["config"] == "default"
    assert record["library_defaults"] is True
    assert (record["init"], record["optimizer"]) == ("auto", "adam")


def test_worker_explicit_passes_algorithm_keywords(tmp_path, monkeypatch):
    constructed, record = run_fake_worker(
        tmp_path, monkeypatch, ["--init", "recursive", "--optimizer", "adam"]
    )
    assert len(constructed) == 2
    for kwargs in constructed:
        assert (kwargs["optimizer"], kwargs["init"]) == ("adam", "recursive")
    assert record["config"] == "adam_recursive"
    assert record["library_defaults"] is False


@pytest.mark.parametrize(
    "algorithm_args",
    [
        ["--library-defaults", "--init", "auto"],
        ["--library-defaults", "--optimizer", "adam"],
        ["--optimizer", "adam"],
    ],
)
def test_worker_rejects_conflicting_or_missing_algorithm_flags(
    tmp_path, monkeypatch, algorithm_args
):
    with pytest.raises(SystemExit) as exc:
        run_fake_worker(tmp_path, monkeypatch, algorithm_args)
    assert exc.value.code == 2


def test_probe_worker_command_library_defaults():
    from comparison.umap06 import phase5_probes

    explicit = phase5_probes.worker_command(
        {"fixture": "digits", "init": "recursive", "optimizer": "adam"},
        "cpu",
        Path("out"),
        Path("data"),
    )
    assert "--library-defaults" not in explicit
    assert explicit[explicit.index("--init") + 1] == "recursive"
    library = phase5_probes.worker_command(
        {"fixture": "digits", "library_defaults": True}, "cpu", Path("out"), Path("data")
    )
    assert "--library-defaults" in library
    assert "--init" not in library and "--optimizer" not in library


def write_side(root, items):
    import json

    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"items": items}))
    return root


def probe_item(identity, outcome="success", hashes=None, replay=0):
    item = {
        "kind": "probe",
        "backend": "cpu",
        "identity": identity,
        "replay": replay,
        "outcome": outcome,
    }
    if outcome == "success":
        item["hashes"] = hashes
    return item


def compare_expected_report(tmp_path, capsys, expected, baseline, candidate):
    import json
    from argparse import Namespace

    from comparison.umap06 import run

    path = tmp_path / "expected.json"
    path.write_text(json.dumps({"items": expected}))
    args = Namespace(
        expected=path,
        baseline=write_side(tmp_path / "baseline", baseline),
        candidate=write_side(tmp_path / "candidate", candidate),
    )
    try:
        run.compare_hashes(args)
        code = 0
    except SystemExit as exc:
        code = exc.code
    report = json.loads(capsys.readouterr().out)
    rows = {json.dumps(row["identity"], sort_keys=True): row for row in report["results"]}
    return code, rows, report


def test_compare_hashes_expected_statuses(tmp_path, capsys):
    import json

    adam = {"fixture": "digits", "optimizer": "adam", "init": "recursive"}
    absent = {"fixture": "curve_20k", "optimizer": "adam", "init": "recursive"}
    lone = {"fixture": "digits", "optimizer": "compatibility", "init": "spectral"}
    keyed = {"fixture": "blobs_1k", "optimizer": "adam", "init": "auto"}
    full = {"fit": "a", "transform_1": "b"}
    expected = [
        {"backend": "cpu", "identity": probe, "hash_keys": ["fit", "transform_1"]}
        for probe in (adam, absent, lone, keyed)
    ]
    baseline = [
        probe_item(adam, hashes=full),
        probe_item(lone, hashes=full),
        probe_item(keyed, hashes=full),
    ]
    candidate = [
        probe_item(adam, hashes=full),
        probe_item(absent, outcome="timeout"),
        probe_item(lone, hashes=full),
        probe_item(keyed, hashes={"fit": "a"}),
    ]
    code, rows, report = compare_expected_report(tmp_path, capsys, expected, baseline, candidate)
    status = {key: row["status"] for key, row in rows.items()}
    assert status[json.dumps(adam, sort_keys=True)] == "matched"
    assert status[json.dumps(absent, sort_keys=True)] == "unmeasured"
    assert rows[json.dumps(absent, sort_keys=True)]["reason"]["baseline"] == [
        {"replay": None, "outcome": "absent"}
    ]
    assert status[json.dumps(lone, sort_keys=True)] == "not_comparable"
    assert status[json.dumps(keyed, sort_keys=True)] == "mismatched"
    assert report["counts"] == {
        "matched": 1,
        "mismatched": 1,
        "unmeasured": 1,
        "not_comparable": 1,
    }
    assert code == 1 and not report["passed"]


def test_compare_hashes_expected_missing_on_both_sides_is_unmeasured(tmp_path, capsys):
    adam = {"fixture": "digits", "optimizer": "adam", "init": "recursive"}
    missing = {"fixture": "mix_1m", "optimizer": "adam", "init": "recursive"}
    expected = [
        {"backend": "cpu", "identity": probe, "hash_keys": ["fit"]} for probe in (adam, missing)
    ]
    side = [probe_item(adam, hashes={"fit": "a"})]
    code, rows, report = compare_expected_report(tmp_path, capsys, expected, side, list(side))
    assert [row["status"] for row in rows.values()] == ["matched", "unmeasured"]
    reason = list(rows.values())[1]["reason"]
    assert reason["baseline"] == reason["candidate"] == [{"replay": None, "outcome": "absent"}]
    assert code == 0 and report["passed"]


def test_compare_hashes_expected_classic_needs_two_identical_replays(tmp_path, capsys):
    lone = {"fixture": "digits", "optimizer": "compatibility", "init": "spectral"}
    expected = [{"backend": "cpu", "identity": lone, "hash_keys": ["fit"]}]
    baseline = [
        probe_item(lone, hashes={"fit": "a"}),
        probe_item(lone, hashes={"fit": "a"}, replay=1),
    ]
    candidate = [probe_item(lone, hashes={"fit": "a"})]
    code, rows, _ = compare_expected_report(tmp_path, capsys, expected, baseline, candidate)
    assert [row["status"] for row in rows.values()] == ["matched"]
    assert code == 0


VARIANT_HASHES = {"fit": "a", "transform_512": "t"}


def write_variant(root, variant, hashes=None, seed=11, **overrides):
    """Write a variant run.json and return its successful manifest item."""
    import hashlib
    import json

    from comparison.umap06 import run

    init, optimizer, library_defaults = run.VARIANT_METADATA[variant]
    path = root / variant / "out" / str(seed) / "run.json"
    path.parent.mkdir(parents=True)
    record = {
        "config": run.VARIANTS[variant],
        "library_defaults": library_defaults,
        "init": init,
        "optimizer": optimizer,
        "backend": "cpu",
        "fixture": "digits",
        "seed": seed,
        "components": 2,
        "neighbor_search": "auto",
        "precomputed_knn": False,
        "init_rescale": "isotropic",
        "epochs": 500,
        "settings": {},
        "hashes": dict(VARIANT_HASHES) if hashes is None else hashes,
        **overrides,
    }
    path.write_text(json.dumps(record))
    return variant_item(
        variant,
        seed=seed,
        record=str(path),
        record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def variant_item(variant, seed=11, outcome="success", **fields):
    return {
        "kind": "variant",
        "variant": variant,
        "backend": "cpu",
        "fixture": "digits",
        "seed": seed,
        "outcome": outcome,
        "command": ["worker", "--transform-batches", "512", "--skip-repeat-fit"],
        **fields,
    }


def compare_variants_report(tmp_path, capsys, items):
    import json
    from argparse import Namespace

    from comparison.umap06 import run

    manifest = None
    if items is not None:
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"items": items}))
    try:
        run.compare_variants(Namespace(dir=tmp_path, manifest=manifest))
        code = 0
    except SystemExit as exc:
        code = exc.code
    return code, json.loads(capsys.readouterr().out)


def test_compare_variants_matches_all_three(tmp_path, capsys):
    from comparison.umap06 import run

    items = [write_variant(tmp_path, variant) for variant in run.VARIANTS]
    code, report = compare_variants_report(tmp_path, capsys, items)
    assert code == 0 and report["passed"], report
    (row,) = report["results"]
    assert row["status"] == "matched"
    assert row["hash_keys"] == ["fit", "transform_512"]


def test_compare_variants_requires_manifest(tmp_path, capsys):
    from comparison.umap06 import run

    for variant in run.VARIANTS:
        write_variant(tmp_path, variant)
    code, report = compare_variants_report(tmp_path, capsys, None)
    assert code == 1 and not report["passed"]
    assert "--manifest is required" in report["errors"][0]


@pytest.mark.parametrize("missing", ["explicit", "auto", "library"])
def test_compare_variants_requires_all_three(tmp_path, capsys, missing):
    from comparison.umap06 import run

    items = [
        write_variant(tmp_path, variant)
        if variant != missing
        else variant_item(variant, outcome="failed")
        for variant in run.VARIANTS
    ]
    code, report = compare_variants_report(tmp_path, capsys, items)
    assert code == 1 and not report["passed"]
    (row,) = report["results"]
    assert row["status"] == "unmeasured"
    assert row["reason"] == {missing: "failed"}


def test_compare_variants_planned_identity_with_all_records_missing_fails(tmp_path, capsys):
    from comparison.umap06 import run

    items = [write_variant(tmp_path, variant) for variant in run.VARIANTS]
    lost = [
        variant_item(variant, seed=12, record=str(tmp_path / variant / "gone.json"))
        for variant in run.VARIANTS
    ]
    code, report = compare_variants_report(tmp_path, capsys, items + lost)
    assert code == 1 and not report["passed"]
    statuses = {row["key"]["seed"]: row["status"] for row in report["results"]}
    assert statuses == {11: "matched", 12: "unmeasured"}


def test_compare_variants_rejects_empty_hash_maps(tmp_path, capsys):
    from comparison.umap06 import run

    items = [write_variant(tmp_path, variant, hashes={}) for variant in run.VARIANTS]
    code, report = compare_variants_report(tmp_path, capsys, items)
    assert code == 1 and not report["passed"]
    (row,) = report["results"]
    assert row["status"] == "mismatched"
    assert len(row["reason"]) == 3


def test_compare_variants_rejects_missing_planned_transforms(tmp_path, capsys):
    from comparison.umap06 import run

    items = [write_variant(tmp_path, variant, hashes={"fit": "a"}) for variant in run.VARIANTS]
    code, report = compare_variants_report(tmp_path, capsys, items)
    assert code == 1
    (row,) = report["results"]
    assert row["status"] == "mismatched"
    assert all("transform_512" in reason for reason in row["reason"])


@pytest.mark.parametrize(
    "overrides",
    [{"init": "spectral"}, {"optimizer": "compatibility"}, {"library_defaults": False}],
)
def test_compare_variants_rejects_wrong_metadata(tmp_path, capsys, overrides):
    items = [
        write_variant(tmp_path, "explicit"),
        write_variant(tmp_path, "auto"),
        write_variant(tmp_path, "library", **overrides),
    ]
    code, report = compare_variants_report(tmp_path, capsys, items)
    assert code == 1
    (row,) = report["results"]
    assert row["status"] == "mismatched"
    assert any("library metadata" in reason for reason in row["reason"])


def test_compare_variants_rejects_tampered_record(tmp_path, capsys):
    from comparison.umap06 import run

    items = [write_variant(tmp_path, variant) for variant in run.VARIANTS]
    items[1]["record_sha256"] = "0" * 64
    code, report = compare_variants_report(tmp_path, capsys, items)
    assert code == 1
    (row,) = report["results"]
    assert row["reason"] == ["auto record digest does not match manifest"]


def test_compare_variants_rejects_mismatch_wrong_label_and_unplanned(tmp_path, capsys):
    items = [
        write_variant(tmp_path, "explicit"),
        write_variant(tmp_path, "auto", hashes={"fit": "b", "transform_512": "t"}),
        write_variant(tmp_path, "library", config="adam_recursive"),
    ]
    write_variant(tmp_path, "library", seed=12)
    code, report = compare_variants_report(tmp_path, capsys, items)
    assert code == 1
    (row,) = report["results"]
    assert row["status"] == "mismatched"
    assert any("is not 'default'" in reason for reason in row["reason"])
    assert len(report["errors"]) == 1 and "not referenced" in report["errors"][0]


def test_planned_variant_hash_keys_from_command():
    from comparison.umap06 import run

    command = ["w", "--transform-batches", "512", "4096", "--skip-repeat-fit"]
    assert run.planned_variant_hash_keys({"command": command}) == [
        "fit",
        "transform_4096",
        "transform_512",
    ]
    assert run.planned_variant_hash_keys({"command": [*command, "--skip-transform"]}) == ["fit"]
    assert run.planned_variant_hash_keys({"command": ["w"]}) is None
    assert run.planned_variant_hash_keys({"hash_keys": ["transform_1"]}) is None
    assert run.planned_variant_hash_keys({"hash_keys": ["fit"], "command": ["w"]}) == ["fit"]


def test_phase6_edge_case_generators_are_deterministic():
    import numpy as np
    from comparison.umap06 import phase6_edge_cases as edge

    first, second = edge.case_list(), edge.case_list()
    names = [name for name, _, _ in first]
    assert len(names) == len(set(names))
    assert names == [name for name, _, _ in second]
    for (name, generate, kwargs), (_, again, kwargs_again) in zip(first, second, strict=True):
        a, b, c = generate(), generate(), again()
        assert a.dtype == np.float32, name
        assert edge.digest(a) == edge.digest(b) == edge.digest(c), name
        assert kwargs == kwargs_again
    assert edge.separated_blobs().shape == (240, 8)
    duplicates = edge.with_duplicates()
    np.testing.assert_array_equal(duplicates[200:], duplicates[:20])


def test_phase7_halved_runs_pin_116_rows_in_tiers():
    from comparison.umap06 import run

    full = run.phase7_runs()
    assert len(full) == 232 and not any(row["precomputed_knn"] for row in full)
    rows = run.phase7_halved_runs()
    keys = [run.identity(dict(row, role="cuscan")) for row in rows]
    assert len(rows) == 116 and set(keys) <= {
        run.identity(dict(row, role="cuscan")) for row in full
    }
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["fixture"]] = counts.get(row["fixture"], 0) + 1
    assert counts == {
        "mix_50k": 36,
        "blobs_1k": 20,
        "digits": 20,
        "curve_20k": 20,
        "mix_200k": 12,
        "mix_1m": 8,
    }
    for row in rows:
        single = row["fixture"] == "mix_1m" or (
            row["fixture"] == "mix_200k" and row["backend"] == "cpu"
        )
        assert row["seed"] in ((11,) if single else (11, 29)), row
    groups: dict[tuple, set[int]] = {}
    for row in rows:
        key = (row["fixture"], row["config"], row["backend"], row["components"])
        groups.setdefault(key, set()).add(row["seed"])
    assert all(
        seeds == ({11} if key[0] == "mix_1m" or key[:3:2] == ("mix_200k", "cpu") else {11, 29})
        for key, seeds in groups.items()
    )
    tiers = [run.phase7_tier(row) for row in rows]
    assert tiers == sorted(tiers)
    assert {tier: tiers.count(tier) for tier in set(tiers)} == {1: 32, 2: 48, 3: 16, 4: 12, 5: 8}
    assert {row["config"] for row in rows if run.phase7_tier(row) == 1} == {
        "adam_recursive",
        "classic_spectral",
    }
    for tier in range(1, 6):
        backends = [row["backend"] for row in rows if run.phase7_tier(row) == tier]
        assert backends == sorted(backends, key=lambda backend: backend != "cuda"), tier


def test_phase7_targeted_commands_keep_quality_and_performance_scoped(tmp_path):
    from comparison.umap06 import phase7_campaign as campaign

    quality = list(
        campaign.quality_items(tmp_path, tmp_path, tmp_path / "ref.json", (3,), targeted=True)
    )
    command = quality[0]["command"]
    assert command[command.index("--seeds") + 1] == "11"
    assert command[command.index("--components") + 1 : command.index("--fixtures")] == ["2", "16"]
    assert "mix_1m" not in command and "--no-child-timeout" in command
    performance = list(
        campaign.performance_snapshot_items(tmp_path, tmp_path / "profile", targeted=True)
    )
    for item in performance:
        command = item["command"]
        assert command[command.index("--epochs") + 1] == "80"
        assert command[command.index("--init") + 1] == "recursive"
        assert command[command.index("--warm-repeats") + 1] == "2"
        assert "--transform-epochs" not in command


def test_b7_schedules_snapshot_pairs_and_separate_attribution(tmp_path):
    from comparison.umap06 import phase7_campaign as campaign

    baseline = tmp_path / "p7"
    predecessor = tmp_path / "predecessor"
    candidate = tmp_path / "candidate"
    tracing = tmp_path / "tracing"
    items = list(campaign.b7_items(tmp_path / "out", baseline, predecessor, candidate, tracing))
    assert len(items) == 42
    for index, case in enumerate(campaign.PERFORMANCE_CASES):
        name = "_".join(map(str, case))
        group = items[index * 6 : index * 6 + 6]
        assert [item["arm"] for item in group] == [
            "predecessor",
            "candidate",
            "candidate",
            "predecessor",
            "predecessor",
            "candidate",
        ]
        assert [item["pair"] for item in group] == [1, 1, 2, 2, 3, 3]
        for item in [*group, items[36 + index]]:
            command = item["command"]
            assert item["case"] == name
            assert Path(command[command.index("--load-snapshot") + 1]) == (
                baseline / "performance" / name / "snapshot"
            )
            assert command[command.index("--warm-repeats") + 1] == "2"
            assert command[command.index("--stage-barriers") + 1] == (
                "true" if item["kind"] == "b7_attributed" else "false"
            )
            assert command[command.index("--paired-modes") + 1] == (
                "true" if item["kind"] == "b7_attributed" else "false"
            )
            assert Path(command[0]) == (
                tracing
                if item["kind"] == "b7_attributed"
                else predecessor
                if item["arm"] == "predecessor"
                else candidate
            )


def test_b7_rejects_changed_and_wrong_configuration_snapshots(tmp_path):
    import json

    from comparison.umap06 import phase7_campaign as campaign

    items = []
    for backend, train, query, features, components in campaign.PERFORMANCE_CASES:
        case = f"{backend}_{train}_{query}_{features}_{components}"
        directory = tmp_path / "performance" / case / "snapshot"
        directory.mkdir(parents=True)
        manifest = directory / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "parameters": {
                        "backend": backend,
                        "train_rows": train,
                        "query_rows": query,
                        "features": features,
                        "components": components,
                        "neighbors": 15,
                        "epochs": 80,
                        "optimizer": "adam",
                        "initialization": "recursive",
                        "search": "approximate",
                    }
                }
            )
        )
        (directory / "query.bin").write_bytes(b"query")
        items.append(
            {
                "kind": "performance_snapshot",
                "case": case,
                "outcome": "success",
                "record_sha256": campaign.sha256(manifest),
            }
        )
    predecessor = {"campaign": "P7-targeted", "items": items}
    hashes = campaign.b7_snapshots(tmp_path, predecessor)
    assert len(hashes) == 6 and all(len(files) == 2 for files in hashes.values())
    first = tmp_path / "performance" / items[0]["case"] / "snapshot" / "manifest.json"
    first.write_text(first.read_text().replace('"epochs": 80', '"epochs": 200'))
    with pytest.raises(ValueError, match="missing or changed"):
        campaign.b7_snapshots(tmp_path, predecessor)
    items[0]["record_sha256"] = campaign.sha256(first)
    with pytest.raises(ValueError, match="parameters"):
        campaign.b7_snapshots(tmp_path, predecessor)


def test_q7e_dispatches_paired_archived_snapshots_and_never_approves_partial(tmp_path):
    import json

    from comparison.umap06 import phase7_campaign as campaign

    items = list(campaign.q7e_items(tmp_path / "out", tmp_path / "data", tmp_path / "p7"))
    assert len(items) == 4
    assert {item["case"] for item in items} == {
        f"{fixture}_2_{backend}_adam_spectral_11"
        for fixture in ("curve_20k", "mix_50k")
        for backend in ("cuda", "cpu")
    }
    for item in items:
        command = item["command"]
        assert "--q7e" in command and "--require-snapshot" in command
        assert Path(command[command.index("--snapshot") + 1]) == (
            tmp_path / "p7" / "snapshots" / item["case"]
        )
    record = Path(items[0]["record"])
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            [
                {"batch": 1, "mode": "forward"},
                {"batch": 1, "mode": "reverse"},
            ]
        )
    )
    items[0].update(outcome="success", record_sha256=campaign.sha256(record))
    summary = campaign.q7e_summary({"items": items})
    assert not summary["measured"]
    assert all(item["case"] in " ".join(summary["missing"]["forward"]) for item in items)
    assert not summary["eligible"]
    record.write_text(json.dumps([{"batch": 1, "mode": "forward"}]))
    items[0]["record_sha256"] = campaign.sha256(record)
    assert items[0]["case"] in " ".join(
        campaign.q7e_summary({"items": items})["missing"]["forward"]
    )


def test_q7e_eligibility_requires_evidence_and_reports_misses(tmp_path):
    import json

    from comparison.umap06 import phase7_campaign as campaign

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    items = list(campaign.q7e_items(tmp_path / "out", tmp_path / "data", tmp_path / "p7"))
    margins = {
        str(batch): {
            fixture: {"transform_trustworthiness": 0.01, "preservation_qr": 0.02}
            for fixture in ("curve_20k", "mix_50k")
        }
        for batch in (512, 4096)
    }
    reference = tmp_path / "reference.json"
    save(reference, {"acceptance_rule": {"transform_margins_by_batch": margins}})
    upstream = []
    for item in items:
        command = item["command"]
        fixture = command[command.index("--fixture") + 1]
        backend = command[command.index("--backend") + 1]
        snapshot = Path(command[command.index("--snapshot") + 1])
        snapshot.mkdir(parents=True)
        (snapshot / "fit.h5").write_bytes(b"fit")
        (snapshot / "fit.npy").write_bytes(b"embedding")
        model_hash = campaign.sha256(snapshot / "fit.h5")
        embedding_hash = campaign.sha256(snapshot / "fit.npy")
        save(
            snapshot / "manifest.json",
            {
                "identity": {
                    "backend": backend,
                    "optimizer": "adam",
                    "init": "spectral",
                    "components": 2,
                    "seed": 11,
                },
                "model_sha256": model_hash,
                "embedding_sha256": embedding_hash,
                "train_sha256": "train",
            },
        )
        rows = []
        for batch in (512, 4096):
            upstream.append(
                {
                    "fixture": fixture,
                    "seed": 11,
                    "batch": batch,
                    "n_neighbors": 15,
                    "epsilon": 0.12,
                    "tie_rule": campaign.QUERY_TIE_RULE,
                    "train_sha256": "train",
                    "query_sha256": str(batch),
                    "recall": 0.99,
                }
            )
            for mode in ("forward", "reverse"):
                rows.append(
                    {
                        "batch": batch,
                        "mode": mode,
                        "query_sha256": str(batch),
                        "train_sha256": "train",
                        "tie_rule": campaign.QUERY_TIE_RULE,
                        "model_sha256": model_hash,
                        "embedding_sha256": embedding_hash,
                        "fitted_state_before_sha256": ("a" if mode == "forward" else "b") * 64,
                        "fitted_state_after_sha256": ("a" if mode == "forward" else "b") * 64,
                        "recall": 0.99,
                        "scores": {"trustworthiness": 0.95, "preservation_qr": 0.5},
                    }
                )
        save(Path(item["record"]), rows)
        item.update(outcome="success", record_sha256=campaign.sha256(Path(item["record"])))
    # Both backends share a single upstream reference row per fixture/batch.
    upstream = [row for i, row in enumerate(upstream) if row not in upstream[:i]]
    query = tmp_path / "query.json"
    save(query, {"schema": campaign.QUERY_REFERENCE_SCHEMA, "rows": upstream})
    manifest = {
        "items": items,
        "provenance": {
            "upstream_reference": {"path": str(reference), "sha256": campaign.sha256(reference)}
        },
    }
    summary = campaign.q7e_summary(manifest, query, campaign.sha256(query))
    assert set(summary["eligible"]) == {"forward", "reverse"}
    assert "mix_1m" in summary["excluded"]
    record = Path(items[0]["record"])
    rows = json.loads(record.read_text())
    rows[1]["recall"] = 0.5
    del rows[1]["fitted_state_after_sha256"]
    save(record, rows)
    items[0]["record_sha256"] = campaign.sha256(record)
    summary = campaign.q7e_summary(manifest, query, campaign.sha256(query))
    assert summary["eligible"] == ["forward"]
    assert any("recall" in miss for miss in summary["missed"]["reverse"])
    assert any("fit invariance" in miss for miss in summary["missing"]["reverse"])
    assert not campaign.q7e_summary(manifest, query, "bad")["eligible"]


def test_q7e_replay_requires_archived_source_without_refitting(tmp_path):
    from comparison.umap06 import phase7_stages as stages

    snapshot = tmp_path / "snapshot"
    with pytest.raises(RuntimeError, match="archived Phase 7 reverse-graph source"):
        list(stages.models(snapshot, "cpu", None, None, 2, 11, q7e=True))
    assert not snapshot.exists()


def test_query_recall_uses_distance_then_id_ties_and_counts_missing():
    import numpy as np
    from comparison.umap06 import quality

    train = np.array([[1.0], [-1.0], [2.0], [3.0]])
    queries = np.array([[0.0]])
    # Rows 0 and 1 tie at distance 1; the lower training ID wins the k=1 slot.
    assert quality.query_recall(train, queries, np.array([[0]]), k=1)["recall"] == 1.0
    assert quality.query_recall(train, queries, np.array([[1]]), k=1)["recall"] == 0.0
    result = quality.query_recall(train, queries, np.array([[0, 0, -1]]), k=3)
    assert result["recall"] == pytest.approx(1 / 3)
    assert result["missing"] == 1 and result["tie_rule"] == "distance,training_id"
    exact = quality.exact_neighbors(train, queries, k=3)[0]
    np.testing.assert_array_equal(exact, [[0, 1, 2]])
    shared = quality.query_recall(train, queries, np.array([[2, 1, 0]]), k=3, exact_ids=exact)
    assert shared["recall"] == 1.0 and shared["per_query"].tolist() == [1.0]
    for invalid in (np.array([[4, 0, 1]]), np.array([[0.0, 1.0, 2.0]]), np.array([[0, 1]])):
        with pytest.raises(ValueError):
            quality.query_recall(train, queries, invalid, k=3)
    assert quality.content_sha256(train) == quality.content_sha256(train.astype(np.float32))
    assert quality.content_sha256(train) != quality.content_sha256(train[::-1])
    assert quality.content_sha256(train) != quality.content_sha256(train.reshape(2, 2))


def build_phase7(tmp_path, monkeypatch, approximate=False):
    import json

    import numpy as np
    from comparison.umap06.spec import Fixture

    run, args, path, doc, ref = build_campaign(
        tmp_path,
        monkeypatch,
        configs=("classic_spectral", "adam_recursive"),
        settings=PHASE4_SETTINGS,
    )
    rows = []
    for record in doc["records"]:
        if record["identity"]["precomputed_knn"]:
            continue
        row = dict(record["identity"])
        row.pop("role")
        row.pop("settings")
        rows.append(row)
    monkeypatch.setattr(run, "phase7_halved_runs", lambda: rows)
    monkeypatch.setattr(run, "phase7_runs", lambda: rows)
    monkeypatch.setattr(
        run,
        "phase7_frozen_worker_hash",
        lambda: run.sha256(run.ROOT / "comparison/umap06/upstream_worker.py"),
    )
    args = phase4_args(args)
    args.full_seeds = False
    args.query_reference = None
    args.query_reference_sha256 = None
    if approximate:
        monkeypatch.setitem(run.FIXTURES, "tiny", Fixture("tiny", 5000, 2, (1,), (11, 29)))
        rng = np.random.default_rng(7)
        train = rng.normal(size=(30, 2)).astype(np.float32)
        queries = rng.normal(size=(2, 2)).astype(np.float32)
        monkeypatch.setattr(
            run.datasets, "load_fixture", lambda data, name: (train, {1: queries}, None)
        )
        exact = run.quality.exact_neighbors(train, queries, k=15)[0]
        for record in doc["records"]:
            if record["identity"]["precomputed_knn"]:
                continue
            run_path = Path(record["run"])
            np.save(run_path.parent / "query_neighbors_1.npy", exact)
            stored = json.loads(run_path.read_text())
            stored["hashes"] = {"query_neighbors_1": run.query_neighbors_digest(exact)}
            run_path.write_text(json.dumps(stored))
            record["identity"] = stored
        path.write_text(json.dumps(doc))
        reference = {
            "schema": run.QUERY_REFERENCE_SCHEMA,
            "rows": [
                {
                    "fixture": "tiny",
                    "seed": seed,
                    "batch": 1,
                    "n_neighbors": 15,
                    "epsilon": 0.12,
                    "tie_rule": run.quality.QUERY_TIE_RULE,
                    "train_sha256": run.quality.content_sha256(train),
                    "query_sha256": run.quality.content_sha256(queries),
                    "recall": 0.995,
                }
                for seed in (11, 29)
            ],
        }
        query_path = tmp_path / "upstream-query.json"
        query_path.write_text(json.dumps(reference))
        args.query_reference = query_path
        args.query_reference_sha256 = run.sha256(query_path)
    return run, args, path, doc, ref


def test_phase7_historical_worker_bytes_are_verified(monkeypatch):
    from comparison.umap06 import run

    assert run.phase7_frozen_worker_hash() == run.FROZEN_UPSTREAM_WORKER_SHA256
    monkeypatch.setattr(run.subprocess, "check_output", lambda *args, **kwargs: b"tampered")
    with pytest.raises(ValueError, match="historical upstream worker sha256 mismatch"):
        run.phase7_frozen_worker_hash()


def test_phase7_rejects_tampered_frozen_worker_provenance(tmp_path, monkeypatch):
    import json

    run, args, _, _, _ = build_phase7(tmp_path, monkeypatch)
    reference = json.loads(args.upstream_reference.read_text())
    reference["provenance"]["source_hashes"]["comparison/umap06/upstream_worker.py"] = "wrong"
    args.upstream_reference.write_text(json.dumps(reference))
    args.upstream_sha256 = run.sha256(args.upstream_reference)
    result = run.phase2_acceptance(args, phase7=True)
    assert any("upstream source provenance mismatch" in error for error in result["errors"])


def test_phase7_revised_scope_retains_missing_identities(tmp_path, monkeypatch):
    import json

    run, args, path, doc, _ = build_phase7(tmp_path, monkeypatch)
    rows = run.phase7_halved_runs()
    extra = dict(rows[0], fixture="mix_1m", seed=11)
    monkeypatch.setattr(run, "phase7_halved_runs", lambda: [*rows, extra])
    args.seed_11_no_mix_1m = True
    target = next(
        record
        for record in doc["records"]
        if record["identity"]["seed"] == 11 and not record["identity"]["precomputed_knn"]
    )
    doc["records"].remove(target)
    path.write_text(json.dumps(doc))
    result = run.phase2_acceptance(args, phase7=True)
    assert result["scope"] == "seed-11-no-mix-1m"
    assert all(
        row["seed"] == 11 and row["fixture"] != "mix_1m" for row in result["expected_identities"]
    )
    assert len(result["expected_identities"]) == sum(row["seed"] == 11 for row in rows)
    assert not result["complete"] and not result["passed"]
    assert any("missing candidate scores" in item for item in result["incomplete"])


def test_phase7_acceptance_gates_transform_targets(tmp_path, monkeypatch):
    import json

    run, args, path, doc, _ = build_phase7(tmp_path, monkeypatch)
    result = run.phase2_acceptance(args, phase7=True)
    assert result["passed"], (result["errors"], result["incomplete"])
    assert result["schema"] == "cuscan-umap06-phase7-acceptance-v1"
    assert result["scope"] == "halved-fit" and not result["fit_only"]
    assert "transform_metrics" not in result
    transform = [item for item in result["comparisons"] if item["metric"].startswith("transform_")]
    assert transform and all(item["target"] == "met" and item["gating"] for item in transform)
    assert result["query_recall"]["targets"] == []
    for record in doc["records"]:
        if record["identity"]["config"] == "adam_recursive" and record["transforms"]:
            record["transforms"]["1"]["preservation_qr"] = 0.1
    path.write_text(json.dumps(doc))
    result = run.phase2_acceptance(args, phase7=True)
    assert not result["passed"]
    assert any("transform_1_preservation_qr" in error for error in result["errors"])
    missed = [item for item in result["comparisons"] if item.get("target") == "missed"]
    assert missed and all(item["group"][1] == "adam_recursive" for item in missed)


def test_phase7_unmeasured_rows_and_transforms_are_reported(tmp_path, monkeypatch):
    import json

    run, args, path, doc, _ = build_phase7(tmp_path, monkeypatch)
    target = next(
        r
        for r in doc["records"]
        if r["identity"]["config"] == "classic_spectral" and not r["identity"]["precomputed_knn"]
    )
    doc["records"].remove(target)
    path.write_text(json.dumps(doc))
    result = run.phase2_acceptance(args, phase7=True)
    assert result["scoped_quality_passed"], result["errors"]
    assert not result["complete"] and not result["passed"]
    missing = target["identity"]
    assert [
        missing["backend"],
        missing["fixture"],
        missing["config"],
        missing["components"],
        missing["precomputed_knn"],
    ] in result["not_measured"]
    assert len(result["expected_identities"]) == len(run.phase7_halved_runs())

    for record in doc["records"]:
        record["transforms"] = {}
    doc["fit_only"] = True
    path.write_text(json.dumps(doc))
    result = run.phase2_acceptance(args, phase7=True)
    assert result["scoped_quality_passed"], result["errors"]
    assert not result["passed"]
    unmeasured = [item for item in result["comparisons"] if item.get("target") == "not measured"]
    assert unmeasured and all(item["metric"].startswith("transform_") for item in unmeasured)
    assert any("transform metric not measured" in item for item in result["incomplete"])


def test_phase7_query_recall_pairs_by_content_hash(tmp_path, monkeypatch):
    import json

    run, args, _, _, _ = build_phase7(tmp_path, monkeypatch, approximate=True)
    result = run.phase2_acceptance(args, phase7=True)
    assert result["passed"], (result["errors"], result["incomplete"])
    recall = result["query_recall"]
    assert {(t["backend"], t["target"]) for t in recall["targets"]} == {
        ("cpu", "met"),
        ("cuda", "met"),
    }
    assert all(row["paired"] and row["cuscan_recall"] == 1.0 for row in recall["rows"])
    assert recall["upstream_unpaired"] == []

    reference = json.loads(args.query_reference.read_text())
    for row in reference["rows"]:
        row["recall"] = 1.0
    reference["rows"][0]["query_sha256"] = "0" * 64
    args.query_reference.write_text(json.dumps(reference))
    with pytest.raises(ValueError):
        run.load_query_reference(args.query_reference, args.query_reference_sha256)
    args.query_reference_sha256 = run.sha256(args.query_reference)
    result = run.phase2_acceptance(args, phase7=True)
    recall = result["query_recall"]
    unpaired = [row for row in recall["rows"] if not row["paired"]]
    assert unpaired and all(row["seed"] == 11 for row in unpaired)
    assert all(t["target"] == "met" for t in recall["targets"])
    assert not result["passed"] and any("partially measured" in i for i in result["incomplete"])

    args.query_reference = None
    args.query_reference_sha256 = None
    result = run.phase2_acceptance(args, phase7=True)
    assert all(t["target"] == "not measured" for t in result["query_recall"]["targets"])
    assert not result["passed"]


def test_phase7_query_recall_miss_and_tampered_ids_fail(tmp_path, monkeypatch):
    import json

    import numpy as np

    run, args, score_path, doc, _ = build_phase7(tmp_path, monkeypatch, approximate=True)
    target = next(
        r
        for r in doc["records"]
        if r["identity"]["backend"] == "cuda" and not r["identity"]["precomputed_knn"]
    )
    neighbors = Path(target["run"]).parent / "query_neighbors_1.npy"
    worse = np.full((2, 15), -1, dtype=np.int64)
    np.save(neighbors, worse)
    result = run.phase2_acceptance(args, phase7=True)
    assert any("query neighbor hash mismatch" in error for error in result["errors"])
    stored = json.loads(Path(target["run"]).read_text())
    stored["hashes"]["query_neighbors_1"] = run.query_neighbors_digest(worse)
    Path(target["run"]).write_text(json.dumps(stored))
    target["identity"] = stored
    score_path.write_text(json.dumps(doc))
    result = run.phase2_acceptance(args, phase7=True)
    assert not result["passed"]
    assert any("query recall target missed: cuda tiny" == e for e in result["errors"])
    targets = {t["backend"]: t for t in result["query_recall"]["targets"]}
    assert targets["cpu"]["target"] == "met" and targets["cuda"]["target"] == "missed"
    assert targets["cuda"]["by_batch"][0]["batch"] == 1
    assert targets["cuda"]["by_batch"][0]["target"] == "missed"


def test_accept_phase7_requires_paired_query_reference_flags(tmp_path, monkeypatch):
    run, args, _, _, _ = build_phase7(tmp_path, monkeypatch)
    args.query_reference = tmp_path / "missing.json"
    with pytest.raises(SystemExit):
        run.accept_phase7(args)
    assert not list(args.artifacts.glob("*-accept-phase7-*"))
    args.query_reference = None
    run.accept_phase7(args)
    receipts = list(args.artifacts.glob("*-accept-phase7-*/acceptance.json"))
    assert len(receipts) == 1 and run.read_json(receipts[0])["passed"]
