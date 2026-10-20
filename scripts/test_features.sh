#!/bin/bash
set -euo pipefail

echo '=== Explicit backend feature compilation ==='
cargo check -p fast-umap --lib --no-default-features --features cpu
cargo check -p fast-umap --lib --no-default-features --features cuda,cpu
cargo check -p fast-umap --lib --no-default-features --features cuda
cargo check --workspace --all-targets --no-default-features --features fast-umap/cpu

echo '=== Expected backend-free diagnostic ==='
python - <<'PY'
import subprocess
result = subprocess.run(
    ['cargo', 'check', '-p', 'fast-umap', '--lib', '--no-default-features'], capture_output=True, text=True
)
assert result.returncode != 0, 'backend-free build unexpectedly succeeded'
assert 'enable at least one backend feature: cpu or cuda' in result.stderr, result.stderr
PY

echo '=== Backend-free host crate ==='
cargo test -p fast-umap-core
cargo test -p fast-umap-core --features tracing

echo '=== Standard CPU behavior ==='
for suite in test_standard_umap test_standard_dispatch test_standard_neighbors test_standard_serialization test_standard_ann test_standard_query test_standard_graph test_standard_layout test_standard_estimator test_standard_quality; do
    cargo test -p fast-umap --no-default-features --features cpu --test "$suite"
done

echo 'CUDA runtime lanes require usable hardware; run explicitly and report unavailable lanes as unverified.'
