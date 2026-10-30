#!/usr/bin/env python3
"""Create or update the open scheduled CI failure tracking issue using gh."""

import json
import os
import subprocess
import sys

TITLE = "Scheduled CI failure"


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, text=True, stdout=subprocess.PIPE).stdout


def main() -> None:
    if os.environ.get("ACT") == "true":
        print("Skipping GitHub issue mutation under act")
        return
    repository = os.environ["GITHUB_REPOSITORY"]
    run_id = os.environ["GITHUB_RUN_ID"]
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    sha = os.environ.get("GITHUB_SHA", "unknown")
    run_url = f"{server}/{repository}/actions/runs/{run_id}"
    body = f"Scheduled CI failed for commit `{sha}`.\n\nRun: {run_url}"

    pages = json.loads(
        gh("api", "--paginate", "--slurp", f"repos/{repository}/issues?state=open&per_page=100")
    )
    issues = [
        issue
        for page in pages
        for issue in page
        if issue.get("title") == TITLE and "pull_request" not in issue
    ]
    if issues:
        # Reuse the oldest tracking issue if duplicates already exist.
        number = min(issue["number"] for issue in issues)
        gh("issue", "comment", str(number), "--repo", repository, "--body", body)
    else:
        gh("issue", "create", "--repo", repository, "--title", TITLE, "--body", body)


if __name__ == "__main__":
    try:
        main()
    except (KeyError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"Could not report scheduled CI failure: {exc}", file=sys.stderr)
        sys.exit(1)
