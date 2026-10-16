#!/bin/bash
set -euo pipefail

cargo test --no-default-features --features cpu --lib backend_selection
cargo test --no-default-features --features cpu --test test_standard_umap
cargo test --no-default-features --features cpu --test test_standard_dispatch
