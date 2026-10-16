#!/bin/bash
set -euo pipefail

# Only discovery-confirmed CUDA absence permits Auto to choose CPU.
# No kernel/execution failure is retried on another backend.
cargo test --no-default-features --features cpu --lib backend_selection
cargo test --no-default-features --features cpu --lib cpu_device_construction_failure
