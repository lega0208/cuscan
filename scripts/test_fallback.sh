#!/bin/bash
set -euo pipefail

# Auto falls back only after confirmed CUDA absence; execution failures never retry.
cargo test -p cuscan-common --lib -- --exact device::tests::resolution_distinguishes_cuda_presence_absence_and_discovery_failure
cargo test -p cuscan-common --lib -- --exact device::tests::cpu_device_construction_failure_is_reported_without_fallback
cargo test -p umap --test test_standard_umap
cargo test -p umap --test test_standard_dispatch
