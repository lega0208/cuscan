#!/bin/bash
set -euo pipefail

cargo test -p fast-cubecl-common --lib -- --exact backend_choice::tests::choice_default_and_validation
cargo test -p fast-cubecl-common --lib -- --exact device::tests::explicit_cpu_selection_does_not_probe_cuda
cargo test -p fast-cubecl-common --lib -- --exact device::tests::explicit_cuda_selection_rejects_out_of_range_index
cargo test -p fast-umap --test test_standard_umap
cargo test -p fast-umap --test test_standard_dispatch
