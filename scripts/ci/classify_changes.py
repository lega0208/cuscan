#!/usr/bin/env python3
"""Classify GitHub changes. Uses GITHUB_EVENT_NAME, GITHUB_EVENT_PATH and GITHUB_SHA by default.

PRs compare the checkout's merge commit against its first parent. Pushes compare
before against head; unavailable diffs fail safe to all flags. Outputs are printed
and, when set, appended to GITHUB_OUTPUT as lower-case boolean values.
For act fixtures only, an event with literal act=true may supply normalized
repository-relative ci_changed_paths instead of a Git diff.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

FLAGS = ("rust", "python_pkg", "python_tests", "fixtures", "deps", "workflows")
ALL = set(FLAGS)


def classify(paths: list[str]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        parts = path.split("/")
        flags: set[str] = set()
        if path.startswith((".github/", "scripts/ci/")) or path in {
            ".actrc",
            "version.txt",
            "release-please-config.json",
            ".release-please-manifest.json",
        }:
            return ALL.copy()
        if path.startswith(("crates/", ".cargo/")) or path in {
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            "burn.toml",
        }:
            flags.add("rust")
        if path.startswith("python/cuscan/") or path in {
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "THIRD_PARTY_NOTICES",
        }:
            flags.add("python_pkg")
        if (
            path.startswith(("python/tests/", "analyze/", "comparison/"))
            or (len(parts) == 2 and parts[0] == "scripts" and path.endswith(".py"))
            or path in {"uv.lock", ".python-version"}
        ):
            flags.add("python_tests")
        if len(parts) >= 5 and parts[0] == "crates" and parts[2:4] == ["tests", "fixtures"]:
            flags.add("fixtures")
        if parts[-1] == "Cargo.toml" or path in {"Cargo.lock", "deny.toml", "uv.lock"}:
            flags.add("deps")
        ignored = (
            path.startswith(
                ("docs/", "archive/", "references/", "artifacts/", ".vscode/", ".ignore/")
            )
            or path in {"CHANGELOG.md", "AGENTS.md", ".gitignore"}
            or (len(parts) == 1 and path.endswith(".md") and path != "README.md")
        )
        if not flags and not ignored:
            return ALL.copy()
        result.update(flags)
    return result


def changed_paths(base: str, head: str) -> list[str]:
    # -z preserves tabs, newlines, and non-ASCII filenames without Git quoting.
    data = subprocess.check_output(
        ["git", "diff", "--name-status", "-z", "--find-renames", base, head, "--"]
    )
    fields = data.decode("utf-8", errors="surrogateescape").split("\0")
    if fields[-1] != "":
        raise ValueError("unterminated git diff output")
    fields.pop()
    paths = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        if not re.fullmatch(r"(?:[ACDMRTUXB][0-9]*)", status):
            raise ValueError("invalid git diff status")
        count = 2 if status[0] in "RC" else 1
        if index + count > len(fields):
            raise ValueError("incomplete git diff output")
        paths.extend(fields[index : index + count])
        index += count
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", ""))
    parser.add_argument("--before", default=None)
    parser.add_argument("--head", default=os.environ.get("GITHUB_SHA", "HEAD"))
    args = parser.parse_args()
    flags = ALL.copy()
    if args.event in {"push", "pull_request"}:
        try:
            event_path = os.environ.get("GITHUB_EVENT_PATH")
            event = json.loads(Path(event_path).read_text(encoding="utf-8")) if event_path else {}
            if event.get("act") is True and "ci_changed_paths" in event:
                paths = event["ci_changed_paths"]
                if not isinstance(paths, list) or any(
                    not isinstance(path, str)
                    or not path
                    or "\\" in path
                    or "\0" in path
                    or any(part in {"", ".", ".."} for part in path.split("/"))
                    for path in paths
                ):
                    raise ValueError("invalid act ci_changed_paths")
                flags = classify(paths)
            else:
                before = args.before
                if args.event == "pull_request":
                    before = f"{args.head}^1"
                elif before is None:
                    before = event.get("before")
                if before and not re.fullmatch(r"0+", before):
                    flags = classify(changed_paths(before, args.head))
        except (
            OSError,
            ValueError,
            TypeError,
            AttributeError,
            subprocess.CalledProcessError,
        ) as exc:
            print(f"Unable to determine changes; running all jobs: {exc}", file=sys.stderr)
    output = "".join(f"{flag}={'true' if flag in flags else 'false'}\n" for flag in FLAGS)
    print(output, end="")
    if output_path := os.environ.get("GITHUB_OUTPUT"):
        with open(output_path, "a", encoding="utf-8") as stream:
            stream.write(output)


if __name__ == "__main__":
    main()
