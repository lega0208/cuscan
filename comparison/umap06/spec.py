"""Fixture and run matrix for the UMAP 0.6.0 comparison."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Fixture:
    name: str
    rows: int
    features: int
    batches: tuple[int, ...]
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class Config:
    optimizer: str
    init: str


_SMALL_SEEDS = (11, 29, 47, 71, 101)
_SEEDS = (11, 29, 47)
FIXTURES = {
    f.name: f
    for f in (
        Fixture("blobs_1k", 1_000, 32, (200,), _SMALL_SEEDS),
        Fixture("digits", 1_497, 64, (300,), _SMALL_SEEDS),
        Fixture("curve_20k", 20_000, 32, (512, 4_096), _SEEDS),
        Fixture("mix_50k", 50_000, 32, (512, 4_096), _SEEDS),
        Fixture("mix_200k", 200_000, 32, (512, 10_000), _SEEDS),
        Fixture("mix_1m", 1_000_000, 32, (512, 10_000), _SEEDS),
        Fixture("mix_10m", 10_000_000, 32, (), _SEEDS),
    )
}
CONFIGS = {
    "classic_spectral": Config("compatibility", "spectral"),
    "classic_random": Config("compatibility", "random"),
    "adam_spectral": Config("adam", "spectral"),
    "adam_recursive": Config("adam", "recursive"),
    "momentum_spectral": Config("momentum", "spectral"),
}


def expected_runs(role: str, backend: str | None = None) -> list[dict]:
    """Return ordinary matrix runs, excluding separate determinism checks."""
    if role not in ("upstream", "cuscan"):
        raise ValueError(f"unknown role: {role}")
    if role == "upstream" and backend is not None:
        raise ValueError("upstream runs have no backend")
    if role == "cuscan" and backend not in (None, "cpu", "cuda"):
        raise ValueError(f"unknown backend: {backend}")
    backends = (None,) if role == "upstream" else ((backend,) if backend else ("cpu", "cuda"))
    runs = []
    for device in backends:

        def add(
            fixture: Fixture,
            config: str,
            components: int = 2,
            exact: bool = False,
            device: str | None = device,
        ):
            for seed in fixture.seeds:
                run = {
                    "fixture": fixture.name,
                    "config": config,
                    "seed": seed,
                    "components": components,
                    "precomputed_knn": exact,
                }
                if device is not None:
                    run["backend"] = device
                runs.append(run)

        for fixture in FIXTURES.values():
            if fixture.name == "mix_10m":
                continue
            configs = CONFIGS if role == "upstream" else ("classic_spectral", "classic_random")
            for config in configs:
                if config == "classic_random" and fixture.name in ("mix_200k", "mix_1m"):
                    continue
                add(fixture, config)
        variant_configs = (
            ("classic_spectral", "adam_spectral") if role == "upstream" else ("classic_spectral",)
        )
        for config in variant_configs:
            for components in (16, 48):
                add(FIXTURES["mix_50k"], config, components)
            add(FIXTURES["mix_50k"], config, exact=True)
    return runs


def warmup_plan(fixture: Fixture) -> dict:
    """Rows needed to compile the small and large upstream kernels."""
    fit_rows = [min(fixture.rows, 2000)]
    if fixture.rows > 1024:
        fit_rows.append(1000)
    transform_rows = [512] if fixture.batches else []
    if any(rows > 1024 for rows in fixture.batches):
        transform_rows.append(2048)
    return {"fit_rows": fit_rows, "transform_rows": transform_rows}
