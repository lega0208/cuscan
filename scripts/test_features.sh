#!/bin/bash
set -euo pipefail

echo '=== Explicit backend feature compilation ==='
cargo check -p fast-cubecl-common --lib --no-default-features
for package in fast-cubecl-common fast-umap fast-hdbscan fast-umap-python; do
    for features in cpu cuda cpu,cuda cpu,tracing cuda,tracing cpu,cuda,tracing; do
        cargo check -p "$package" --lib --no-default-features --features "$features"
    done
done
cargo check --workspace --all-targets --no-default-features --features fast-umap/cpu,fast-hdbscan/cpu,fast-umap-python/cpu

echo '=== Expected backend-free facade diagnostics and dependency isolation ==='
python - <<'PY'
import subprocess

for package in ('fast-umap', 'fast-hdbscan'):
    result = subprocess.run(
        ['cargo', 'check', '-p', package, '--lib', '--no-default-features'],
        capture_output=True, text=True,
    )
    assert result.returncode != 0, f'{package}: backend-free build unexpectedly succeeded'
    assert 'enable at least one backend feature: cpu or cuda' in result.stderr, result.stderr

for package in ('fast-umap-core', 'fast-hdbscan-core'):
    for features in ([], ['--features', 'tracing']):
        tree = subprocess.check_output(
            ['cargo', 'tree', '-p', package, '--no-default-features', '-e', 'normal',
             '--prefix', 'none', *features], text=True,
        )
        for forbidden in ('burn', 'cubecl', 'cudarc'):
            assert forbidden not in tree, (package, tree)
        if package == 'fast-hdbscan-core':
            for forbidden in ('faer', 'rand'):
                assert forbidden not in tree, (package, tree)

for features in ('cpu', 'cuda', 'cpu,cuda', 'cpu,tracing'):
    tree = subprocess.check_output(
        ['cargo', 'tree', '-p', 'fast-hdbscan', '--no-default-features',
         '--features', features, '-e', 'normal', '--prefix', 'none'], text=True,
    )
    for forbidden in ('fast-umap', 'faer'):
        assert forbidden not in {line.split()[0] for line in tree.splitlines()}, tree

for feature in ('cpu', 'cuda'):
    tree = subprocess.check_output(
        ['cargo', 'tree', '-p', 'fast-umap-python', '--no-default-features',
         '--features', feature, '-e', 'normal', '--prefix', 'none',
         '--format', '{p} features={f}'], text=True,
    )
    for facade in ('fast-umap', 'fast-hdbscan'):
        rows = [line for line in tree.splitlines() if line.startswith(f'{facade} v')]
        assert rows and feature in rows[0].split('features=')[1].split(','), tree
PY

echo '=== Backend-free host crates ==='
for features in fast-umap/cuda,fast-hdbscan/cpu fast-umap/cpu,fast-hdbscan/cuda; do
    cargo test -p fast-umap -p fast-hdbscan --lib --no-default-features --features "$features" facade_permissions_are_not_expanded_by_peer_features
done
for package in fast-umap-core fast-hdbscan-core; do
    cargo test -p "$package" --no-default-features
    cargo test -p "$package" --no-default-features --features tracing
done

echo '=== Standard CPU behavior ==='
for suite in test_standard_umap test_standard_dispatch test_standard_neighbors test_standard_serialization test_standard_ann test_standard_query test_standard_graph test_standard_layout test_standard_estimator test_standard_quality; do
    cargo test -p fast-umap --no-default-features --features cpu --test "$suite"
done

echo 'CUDA runtime lanes require usable hardware; run explicitly and report unavailable lanes as unverified.'
