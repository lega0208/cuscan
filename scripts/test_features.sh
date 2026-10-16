#!/bin/bash
set -euo pipefail

# Compilation only: parametric targets are not runtime acceptance for this migration.
echo '=== Explicit backend feature compilation ==='
cargo check --lib --no-default-features --features cpu
cargo check --lib --no-default-features --features cuda,cpu,verbose
cargo check --lib --no-default-features --features cpu,gpu
cargo check --lib --no-default-features --features gpu
cargo check --lib --no-default-features --features cuda
cargo check --all-targets --no-default-features --features cpu,parametric

echo '=== Expected backend-free diagnostic ==='
python - <<'PY'
import subprocess
result = subprocess.run(
    ['cargo', 'check', '--lib', '--no-default-features'], capture_output=True, text=True
)
assert result.returncode != 0, 'backend-free build unexpectedly succeeded'
assert 'enable at least one backend feature: cpu, cuda, or gpu' in result.stderr, result.stderr
PY

echo '=== Standard CPU behavior ==='
for suite in test_standard_umap test_standard_dispatch test_standard_ann test_standard_query test_standard_graph test_standard_layout test_standard_init test_standard_estimator test_standard_quality; do
    cargo test --no-default-features --features cpu --test "$suite"
done

echo 'GPU runtime lanes require usable hardware; run explicitly and report unavailable lanes as unverified.'
