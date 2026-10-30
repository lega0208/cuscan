#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    echo 'Usage: scripts/ci/act.sh list | dry <scenario> | <scenario> [act options]'
    echo 'Scenarios: pr, pr-docs, push-main, tag, dispatch, release-dry'
}
root=$(git rev-parse --show-toplevel)
cd "$root"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/cuscan-act.XXXXXXXX")
trap 'rm -rf -- "$tmp"' EXIT

run_scenario() {
    local scenario=$1 event fixture workflow
    shift
    case "$scenario" in
        pr) event=pull_request; fixture=.github/act/events/pull_request.json; workflow=ci.yml ;;
        pr-docs) event=pull_request; fixture=.github/act/events/pull_request_docs.json; workflow=ci.yml ;;
        push-main) event=push; fixture=.github/act/events/push_main.json; workflow=ci.yml ;;
        tag) event=push; fixture=.github/act/events/tag.json; workflow=release.yml ;;
        dispatch|release-dry) event=workflow_dispatch; fixture=.github/act/events/dispatch.json; workflow=release.yml ;;
        *) usage >&2; return 2 ;;
    esac
    act "$event" -W ".github/workflows/$workflow" -e "$fixture" "$@"
}

if (($# == 0)); then usage >&2; exit 2; fi
mode=$1
shift
case "$mode" in
    list)
        if (($#)); then usage >&2; exit 2; fi
        for scenario in pr pr-docs push-main tag dispatch; do
            printf '\n=== %s ===\n' "$scenario"
            run_scenario "$scenario" -l
        done ;;
    dry)
        if (($# == 0)); then usage >&2; exit 2; fi
        scenario=$1; shift
        run_scenario "$scenario" -n "$@" ;;
    release-dry)
        # This is an actual rehearsal build, not act's -n graph-only mode.
        run_scenario release-dry "$@" 2>&1 | tee "$tmp/release.log"
        python3 - "$tmp/release.log" <<'PY'
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text(errors="replace")
log = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", log)
if not re.search(r"\[[^\]]*sdist-source-build[^\]]*\].*Job succeeded", log):
    sys.exit("sdist-source-build did not report a successful execution")
PY
        ;;
    *) run_scenario "$mode" "$@" ;;
esac
