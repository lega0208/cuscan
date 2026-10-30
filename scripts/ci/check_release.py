#!/usr/bin/env python3
"""Validate release inputs and emit pep440_version to stdout and GITHUB_OUTPUT.

Run with Python 3.11+ from the repository root. Production takes --tag or
GITHUB_REF_NAME and checks the checkout and origin/main ancestry. Rehearsal
requires -dev.N.
"""

import argparse
import os
import re
import subprocess
from pathlib import Path

import tomllib

SEMVER = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-(dev|alpha|beta|rc)\.(0|[1-9][0-9]*))?"
)


def pep440_version(version: str) -> str:
    match = SEMVER.fullmatch(version)
    if not match:
        raise ValueError(f"unsupported Cargo SemVer: {version}")
    major, minor, patch, kind, number = match.groups()
    suffix = {None: "", "dev": ".dev", "alpha": "a", "beta": "b", "rc": "rc"}[kind]
    return f"{major}.{minor}.{patch}{suffix}{number or ''}"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def check_ancestry(commit: str) -> None:
    subprocess.run(
        ["git", "fetch", "--no-tags", "origin", "+refs/heads/main:refs/remotes/origin/main"],
        check=True,
    )
    while True:
        result = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"])
        if result.returncode == 0:
            return
        if result.returncode != 1:
            raise ValueError("git could not check release ancestry")
        if git("rev-parse", "--is-shallow-repository") != "true":
            raise ValueError("release commit is not an ancestor of origin/main")
        old_boundary = git("rev-parse", "--git-path", "shallow")
        previous = Path(old_boundary).read_bytes()
        subprocess.run(
            [
                "git",
                "fetch",
                "--no-tags",
                "--deepen=100",
                "origin",
                "+refs/heads/main:refs/remotes/origin/main",
            ],
            check=True,
        )
        if git("rev-parse", "--is-shallow-repository") == "true":
            if Path(old_boundary).read_bytes() == previous:
                raise ValueError("unable to deepen history to establish release ancestry")


def validate(mode: str, version: str, tag: str | None, changelog: str) -> str:
    pep = pep440_version(version)
    is_dev = re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+-dev\.[0-9]+", version) is not None
    if mode == "rehearsal":
        if not is_dev:
            raise ValueError("rehearsal requires a -dev.N version")
    else:
        if is_dev:
            raise ValueError("production cannot use a -dev.N version")
        if tag != f"v{version}":
            raise ValueError("tag does not match Cargo version")
        if not re.search(rf"^## \[{re.escape(version)}\](?:\s|$)", changelog, re.MULTILINE):
            raise ValueError("CHANGELOG is missing the version heading")
    return pep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("production", "rehearsal"), required=True)
    parser.add_argument("--tag", default=os.environ.get("GITHUB_REF_NAME"))
    args = parser.parse_args()
    try:
        with open("Cargo.toml", "rb") as stream:
            version = tomllib.load(stream)["workspace"]["package"]["version"]
        changelog = (
            Path("CHANGELOG.md").read_text(encoding="utf-8") if args.mode == "production" else ""
        )
        pep = validate(args.mode, version, args.tag, changelog)
        if args.mode == "production":
            commit = git("rev-parse", "HEAD^{commit}")
            if git("rev-parse", f"refs/tags/{args.tag}^{{commit}}") != commit:
                raise ValueError("tag does not point to the checked-out commit")
            check_ancestry(commit)
        output = f"pep440_version={pep}\n"
        print(output, end="")
        if output_path := os.environ.get("GITHUB_OUTPUT"):
            with open(output_path, "a", encoding="utf-8") as stream:
                stream.write(output)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"release verification failed: {exc}\n")


if __name__ == "__main__":
    main()
