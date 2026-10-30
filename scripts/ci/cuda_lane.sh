#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    echo 'Usage: scripts/ci/cuda_lane.sh [--ref <commit>] [--rows 20000|100000]'
}

ref=HEAD
explicit_ref=false
rows=
while (($#)); do
    case "$1" in
        --ref|--rows)
            if (($# < 2)); then usage >&2; exit 2; fi
            if [[ $1 == --ref ]]; then ref=$2; explicit_ref=true; else rows=$2; fi
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ -n $rows && $rows != 20000 && $rows != 100000 ]]; then usage >&2; exit 2; fi

root=$(git rev-parse --show-toplevel)
sha=$(git -C "$root" rev-parse --verify "${ref}^{commit}")
cache=${XDG_CACHE_HOME:-"$HOME/.cache"}/cuscan-cuda
mkdir -p "$cache/logs" "$cache/target" "$cache/wt"
exec > >(tee -a "$cache/logs/$sha.log") 2>&1
printf '\nCUDA lane: %s\n' "$sha"
git -C "$root" show -s --format='Commit: %H%nAuthor: %an <%ae>%nSubject: %s' "$sha"
worktree=
worktree_added=false
finish() {
    result=$?
    trap - EXIT INT TERM HUP
    if [[ -n $worktree ]]; then
        if $worktree_added; then
            if ! git -C "$root" worktree remove --force "$worktree"; then
                echo "Failed to remove lane worktree: $worktree" >&2
                result=1
            fi
        elif ! rmdir "$worktree"; then
            echo "Failed to remove lane directory: $worktree" >&2
            result=1
        fi
    fi
    exit "$result"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

if $explicit_ref; then
    worktree=$(mktemp -d "$cache/wt/lane.XXXXXXXX")
    git -C "$root" worktree add --detach "$worktree" "$sha"
    worktree_added=true
    cd "$worktree"
else
    cd "$root"
fi
export CARGO_TARGET_DIR="$cache/target"
CUSCAN_REQUIRE_CUDA=1 cargo test --locked -p cuscan-common -p umap -p hdbscan --features hdbscan-core/test-support -- --nocapture
uv sync --locked --no-install-project --extra serialization
uv run --no-sync maturin develop --release --features tracing
CUSCAN_TEST_BACKEND=cuda CUSCAN_REQUIRE_CUDA=1 uv run --no-sync pytest python/tests
if [[ -n $rows ]]; then
    bash scripts/analyze_hdbscan_stages.sh --rows "$rows" || echo 'Diagnostic analysis failed; test result is unaffected.' >&2
fi
