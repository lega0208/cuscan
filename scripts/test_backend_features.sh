#!/bin/bash
set -euo pipefail

for features in cpu cuda cpu,cuda; do
    cargo check -p fast-umap --no-default-features --features "$features" --lib
done
