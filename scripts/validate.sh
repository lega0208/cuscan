#!/bin/bash
set -euo pipefail

uv run maturin develop --release
uv lock --check
uv sync --locked --no-install-project --extra serialization
uv run --no-sync ruff check
uv run --no-sync ruff format --check
uv run --no-sync ty check
uv run --no-sync python crates/hdbscan/tests/fixtures/generate_fast_hdbscan.py --verify-provenance
uv run --no-sync python crates/hdbscan/tests/fixtures/generate_fast_hdbscan.py --check

cargo fmt --all --check
cargo cclippy --workspace --all-targets --locked --features umap/tracing,hdbscan/tracing,cuscan-python/tracing,hdbscan-core/test-support
cargo doc --workspace --no-deps --locked
cargo deny --locked check