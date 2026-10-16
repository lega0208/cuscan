#!/bin/bash
set -euo pipefail

for features in cpu cuda gpu cpu,cuda cpu,gpu cuda,cpu,verbose; do
    cargo check --no-default-features --features "$features" --lib
done
# Parametric support is compile-only, not a runtime acceptance claim.
cargo check --no-default-features --features cpu,parametric --all-targets
