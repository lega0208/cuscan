#!/bin/bash
set -euo pipefail

cargo test -p fast-umap --no-default-features --features cpu --lib backend_selection
cargo test -p fast-umap --no-default-features --features cpu --test test_standard_umap
cargo test -p fast-umap --no-default-features --features cpu --test test_standard_dispatch
