#!/bin/bash
set -euo pipefail

echo '=== Runtime package build contracts ==='
for package in cuscan-common umap hdbscan cuscan-python; do
    cargo check -p "$package" --lib
    cargo check -p "$package" --lib --no-default-features
    cargo check -p "$package" --lib --no-default-features --features tracing
done

echo '=== Declared features, resolved backends, and dependency isolation ==='
python - <<'PY'
import json
import re
import subprocess

metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--format-version', '1']))
runtime = {'cuscan-common', 'umap', 'hdbscan', 'cuscan-python'}
packages = {package['name']: package for package in metadata['packages']}
for name in runtime:
    features = packages[name]['features']
    assert not {'cpu', 'cuda'} & features.keys(), (name, features)

for name in runtime:
    tree = subprocess.check_output(
        ['cargo', 'tree', '-p', name, '--no-default-features', '-e', 'features',
         '--prefix', 'none'], text=True,
    )
    for dependency in ('burn', 'burn-cubecl', 'cubecl'):
        assert re.search(rf'^{dependency} v', tree, re.MULTILINE), (name, dependency)
        for backend in ('cpu', 'cuda'):
            assert re.search(rf'^{dependency} feature "{backend}"', tree, re.MULTILINE), (name, dependency, backend, tree)

for package in ('umap-core', 'hdbscan-core'):
    for features in ([], ['--features', 'tracing']):
        tree = subprocess.check_output(
            ['cargo', 'tree', '-p', package, '--no-default-features', '-e', 'normal',
             '--prefix', 'none', *features], text=True,
        )
        for forbidden in ('burn', 'cubecl', 'cudarc'):
            assert forbidden not in tree, (package, tree)
        if package == 'hdbscan-core':
            for forbidden in ('faer', 'rand'):
                assert forbidden not in tree, (package, tree)

tree = subprocess.check_output(
    ['cargo', 'tree', '-p', 'hdbscan', '--no-default-features', '-e', 'normal', '--prefix', 'none'], text=True,
)
for forbidden in ('umap', 'faer'):
    assert forbidden not in {line.split()[0] for line in tree.splitlines()}, tree
PY

echo '=== Backend-free host crates ==='
for package in umap-core hdbscan-core; do
    cargo test -p "$package" --no-default-features
    cargo test -p "$package" --no-default-features --features tracing
done

echo '=== Standard CPU behavior ==='
for suite in test_standard_umap test_standard_dispatch test_standard_neighbors test_standard_serialization test_standard_ann test_standard_query test_standard_graph test_standard_layout test_standard_estimator test_standard_quality; do
    cargo test -p umap --test "$suite"
done

echo 'CUDA runtime lanes require usable hardware; run explicitly and report unavailable lanes as unverified.'
