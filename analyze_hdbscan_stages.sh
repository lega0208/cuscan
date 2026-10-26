#!/usr/bin/env bash
# Native HDBSCAN hotspot campaign; all timed execution belongs to the Python runner.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
usage() {
    printf '%s\n' 'Usage: bash analyze_hdbscan_stages.sh [--rows 20000|50000|100000] [--output-dir NEW_DIRECTORY]' \
        'Requires cargo, uv, nsys, ncu and working CPU/CUDA runtimes and counter permissions.' \
        'Default: 20000 rows; fresh directory under artifacts/hdbscan_stages.'
}
fail() { printf 'HDBSCAN analysis: %s\n' "$*" >&2; exit 1; }
rows=20000
output=''
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --rows|--output-dir)
            (($# >= 2)) || fail "Missing value for $1"
            case "$1" in --rows) rows=$2 ;; --output-dir) output=$2 ;; esac
            shift 2 ;;
        *) fail "Unknown argument: $1" ;;
    esac
done
case "$rows" in 20000) native_timeout=10; profile_timeout=10 ;; 50000|100000) native_timeout=240; profile_timeout=90 ;; *) fail '--rows must be 20000, 50000 or 100000' ;; esac
for tool in cargo uv nsys ncu; do command -v "$tool" >/dev/null || fail "Required tool not found: $tool"; done
if [[ -n "$output" ]]; then
    [[ ! -e "$output" && ! -L "$output" ]] || fail "Output directory already exists: $output"
    mkdir -p -- "$(dirname -- "$output")"
    mkdir -- "$output"
else
    mkdir -p -- "$root/artifacts/hdbscan_stages"
    output=$(mktemp -d "$root/artifacts/hdbscan_stages/run-${rows}-XXXXXXXX")
fi
output=$(cd -- "$output" && pwd)
mkdir -- "$output/raw"
printf 'Evidence directory: %s\n' "$output"
active=''
interrupt() {
    trap '' INT TERM
    if [[ -n "$active" ]]; then
        kill -TERM "$active" || true
        wait "$active" || true
    fi
    printf 'Interrupted; diagnostics retained at %s\n' "$output" >&2
    exit 130
}
trap interrupt INT TERM
run() {
    "$@" &
    active=$!
    local status=0
    wait "$active" || status=$?
    active=''
    if ((status)); then
        printf 'Phase failed (exit %s); diagnostics: %s\n' "$status" "$output" >&2
        exit "$status"
    fi
}
# Build logs stay visible; compilation has no fit-execution deadline.
run cargo build --manifest-path "$root/Cargo.toml" -p fast-hdbscan --release --features tracing --example profile_hdbscan_stages
binary="$root/target/release/examples/profile_hdbscan_stages"
# Resolve imports/environment independently of the caller working directory.
export UV_PROJECT="$root"
run uv run --no-sync python "$root/analyze/benchmark_hdbscan_stages.py" --binary "$binary" --rows "$rows" --run-dir "$output" --timeout "$native_timeout"
run uv run --no-sync python "$root/analyze/profile_hdbscan_kernels.py" --binary "$binary" --rows "$rows" --run-dir "$output" --timeout "$profile_timeout"
run uv run --no-sync python "$root/analyze/profile_hdbscan_counters.py" --binary "$binary" --binary-label current --rows "$rows" --run-dir "$output" --timeout "$profile_timeout"
run uv run --no-sync python "$root/analyze/summarize_hdbscan_stages.py" --benchmark "$output/benchmark.json" --kernels "$output/cuda_kernels.json" --counters "$output/cuda_counters.json"
