#!/bin/bash
set -e

echo "=== Testing Feature-Based Compilation ==="
echo

echo "1. Testing default features (gpu + verbose):"
cargo check --example feature_demo
echo

echo "2. Testing the umap-rs CPU fallback feature:"
cargo check --example feature_demo --features cpu --no-default-features
echo

echo "3. Testing minimal features (Flex CPU backend):"
cargo check --lib --no-default-features
echo

echo "4. Testing CUDA backend without default features:"
cargo check --lib --no-default-features --features cuda
echo

echo "5. Testing GPU and CUDA backends together:"
cargo check --lib --features gpu,cuda
echo

echo "6. Testing CUDA backend tests when CUDA runtime is present:"
if command -v nvidia-smi >/dev/null 2>&1 || { command -v ldconfig >/dev/null 2>&1 && ldconfig -p | grep -q 'libcuda\.so'; }; then
    FAST_UMAP_REQUIRE_CUDA=1 cargo test --no-default-features --features cuda --lib cuda_backend -- --nocapture
else
    echo "SKIPPED: neither nvidia-smi nor libcuda was found"
fi
echo

echo "7. Testing all non-CUDA features:"
cargo check --example feature_demo --features all
echo

echo "8. Testing GPU backend explicitly:"
cargo check --example backend_choice --features gpu
echo

echo "9. Testing the Flex CPU device explicitly:"
cargo check --example backend_choice --no-default-features
echo

echo "=== Feature Compilation Tests Complete ==="
