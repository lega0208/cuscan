#!/usr/bin/env python3
"""Validate release proposals and synchronize their lockfile using trusted code."""

import argparse
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

from check_release import changelog_versions

MEMBERS = {
    "crates/umap-core": "umap-core",
    "crates/umap": "umap",
    "crates/cuscan-python": "cuscan-python",
    "crates/cuscan-common": "cuscan-common",
    "crates/hdbscan-core": "hdbscan-core",
    "crates/hdbscan": "hdbscan",
}
ALLOWED = {
    "Cargo.toml",
    "Cargo.lock",
    "CHANGELOG.md",
    "version.txt",
    ".release-please-manifest.json",
}
BRANCH = "release-please--branches--main--components--cuscan"
BOOTSTRAP = "c76bb96e2284c8404b79daabe6df5fb10d11594b"
REQUIRED_CONFIG = {
    "package-name": "cuscan",
    "release-type": "simple",
    "version-file": "version.txt",
    "include-component-in-tag": False,
    "include-v-in-tag": True,
    "draft": False,
    "prerelease": False,
    "bump-minor-pre-major": True,
    "bump-patch-for-minor-pre-major": False,
    "changelog-path": "CHANGELOG.md",
    "extra-files": [
        {"type": "toml", "path": "Cargo.toml", "jsonpath": "$.workspace.package.version"}
    ],
}
BOOTSTRAP_CONFIG = {
    "$schema": "https://raw.githubusercontent.com/googleapis/release-please/v17.1.3/schemas/config.json",
    "bootstrap-sha": BOOTSTRAP,
    "packages": {
        ".": {
            **REQUIRED_CONFIG,
            "release-as": "0.1.0",
            "changelog-sections": [
                {"type": "feat", "section": "Features"},
                {"type": "fix", "section": "Bug Fixes"},
                {"type": "perf", "section": "Performance"},
                {"type": "revert", "section": "Reverts"},
                {"type": "docs", "section": "Documentation", "hidden": True},
                {"type": "chore", "section": "Miscellaneous Chores", "hidden": True},
                {"type": "test", "section": "Tests", "hidden": True},
            ],
        }
    },
}
VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
TITLE = re.compile(r"[a-z]+(?:\([a-zA-Z0-9_.\-/]+\))?!?: .+\Z")
SHA = re.compile(r"[0-9a-f]{40}")


def run(root: Path, *args: str) -> str:
    return subprocess.check_output(args, cwd=root, text=True).strip()


def git(root: Path, *args: str) -> str:
    return run(root, "git", *args)


def version(value: object) -> tuple[int, ...]:
    if not isinstance(value, str) or not VERSION.fullmatch(value):
        raise ValueError(f"invalid stable version: {value!r}")
    return tuple(map(int, value.split(".")))


def load_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def manifest(root: Path) -> dict:
    value = json.loads((root / ".release-please-manifest.json").read_text())
    if not isinstance(value, dict) or set(value) - {"."}:
        raise ValueError("invalid release manifest")
    if "." in value:
        version(value["."])
    return value


def is_proposal(base: Path, candidate: Path, pr: dict | None) -> bool:
    if pr is not None and pr["head"]["ref"] == BRANCH:
        return True
    old = base / ".release-please-manifest.json"
    return (
        old.exists()
        and old.read_bytes() != (candidate / ".release-please-manifest.json").read_bytes()
    )


def check_config(root: Path, bootstrap: bool) -> None:
    config = json.loads((root / "release-please-config.json").read_text())
    if bootstrap:
        if json.dumps(config, sort_keys=True) != json.dumps(BOOTSTRAP_CONFIG, sort_keys=True):
            raise ValueError("invalid bootstrap release configuration")
        return
    if set(config["packages"]) != {"."}:
        raise ValueError("invalid release configuration packages")
    package = config["packages"]["."]
    if any(
        key not in package or type(package[key]) is not type(value) or package[key] != value
        for key, value in REQUIRED_CONFIG.items()
    ):
        raise ValueError("invalid release configuration")


def workspace(root: Path) -> tuple[dict, str]:
    data = load_toml(root / "Cargo.toml")
    v = data["workspace"]["package"]["version"]
    version(v)
    members = data["workspace"]["members"]
    if set(members) != set(MEMBERS) or len(members) != 6:
        raise ValueError("unexpected workspace members")
    for member, name in MEMBERS.items():
        pkg = load_toml(root / member / "Cargo.toml")["package"]
        if pkg["name"] != name or pkg["version"] != {"workspace": True}:
            raise ValueError(f"invalid inherited member {member}")
    return data, v


def lock(root: Path, expected: str | None) -> dict:
    data = load_toml(root / "Cargo.lock")
    found = set()
    for package in data["package"]:
        name = package["name"]
        if name in MEMBERS.values() and "source" not in package:
            if name in found or (expected is not None and package["version"] != expected):
                raise ValueError(f"invalid local lock entry {name}")
            found.add(name)
            package["version"] = "<workspace-version>"
    if found != set(MEMBERS.values()):
        raise ValueError("missing local lock entries")
    return data


def check_clean(root: Path) -> None:
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"checkout is not clean: {root}")


def verify(base: Path, candidate: Path, proposal: bool = True, check_lock: bool = True) -> str:
    if base.resolve() == candidate.resolve():
        raise ValueError("base and candidate must be separate checkouts")
    check_clean(base)
    check_clean(candidate)
    base_sha = git(base, "rev-parse", "HEAD")
    head = git(candidate, "rev-parse", "HEAD")
    changes = set(
        git(candidate, "diff", "--name-only", "--no-renames", base_sha, head).splitlines()
    )
    if not proposal:
        _, old_v = workspace(base)
        _, new_v = workspace(candidate)
        old_manifest_path = base / ".release-please-manifest.json"
        if not old_manifest_path.exists():
            check_config(candidate, bootstrap=True)
            if old_v != "0.1.0" or manifest(candidate) != {}:
                raise ValueError("invalid initial release automation setup")
            if lock(base, old_v) != lock(candidate, new_v):
                raise ValueError("initial setup changes lockfile resolution")
        elif manifest(base) != manifest(candidate):
            raise ValueError("ordinary change modifies release manifest")
        if old_v != new_v:
            raise ValueError("ordinary change modifies release version or manifest")
        if (candidate / "version.txt").read_text().strip() != new_v:
            raise ValueError("version.txt and Cargo version disagree")
        lock(candidate, new_v)
        return new_v
    if changes - ALLOWED:
        raise ValueError(f"disallowed changes: {sorted(changes - ALLOWED)}")
    for path in ALLOWED | {f"{m}/Cargo.toml" for m in MEMBERS} | {"release-please-config.json"}:
        for root in (base, candidate):
            entry = git(root, "ls-files", "--stage", "--", path)
            if not entry.startswith("100644 "):
                raise ValueError(f"missing or non-regular tracked file: {path}")
    old, old_v = workspace(base)
    new, new_v = workspace(candidate)
    old["workspace"]["package"]["version"] = new_v
    if old != new:
        raise ValueError("workspace TOML changed beyond its version")
    if (base / "release-please-config.json").read_bytes() != (
        candidate / "release-please-config.json"
    ).read_bytes():
        raise ValueError("release configuration changed")
    if (base / "version.txt").read_text().strip() != old_v or (
        candidate / "version.txt"
    ).read_text().strip() != new_v:
        raise ValueError("version.txt and Cargo version disagree")
    old_manifest, new_manifest = manifest(base), manifest(candidate)
    if proposal:
        check_config(base, bootstrap=not old_manifest)
        if new_manifest != {".": new_v}:
            raise ValueError("proposal manifest disagrees with Cargo")
        if old_manifest:
            if version(new_v) <= version(old_manifest["."]):
                raise ValueError("release version must increase")
        else:
            if (old_v, new_v) != ("0.1.0", "0.1.0"):
                raise ValueError("invalid bootstrap proposal")
        versions = changelog_versions((candidate / "CHANGELOG.md").read_text())
        if versions.count(new_v) != 1:
            raise ValueError("missing or duplicate changelog version heading")
    elif old_manifest != new_manifest:
        raise ValueError("ordinary change modifies release manifest")
    if lock(base, old_v) != lock(candidate, new_v if check_lock else None):
        raise ValueError("lockfile changed beyond local package versions")
    return new_v


def select_pr(prs: object, output: object, repo: str, bot: str, bot_id: int) -> dict | None:
    if not isinstance(prs, list) or not isinstance(output, list):
        raise ValueError("invalid PR list or action output")
    candidates = []
    for pr in prs:
        if not isinstance(pr, dict):
            raise ValueError("invalid PR metadata")
        head = pr["head"]
        if head["ref"] != BRANCH:
            continue
        if (
            pr["state"] != "open"
            or pr["base"]["ref"] != "main"
            or pr["base"]["repo"]["full_name"] != repo
            or head["repo"]["full_name"] != repo
            or pr["user"]["login"] != bot
            or pr["user"]["id"] != bot_id
            or pr["user"]["type"] != "Bot"
            or "autorelease: pending" not in [label["name"] for label in pr["labels"]]
            or not SHA.fullmatch(head["sha"])
            or not isinstance(pr["number"], int)
        ):
            raise ValueError("release PR ownership or metadata mismatch")
        candidates.append(pr)
    if len(candidates) > 1:
        raise ValueError("multiple release PRs")
    if output:
        if (
            len(output) != 1
            or not isinstance(output[0], dict)
            or not candidates
            or output[0].get("number") != candidates[0]["number"]
        ):
            raise ValueError("action output does not identify selected PR")
    return candidates[0] if candidates else None


def sync(
    base: Path, candidate: Path, branch: str, head: str, bot_name: str, bot_email: str
) -> None:
    if branch != BRANCH or not SHA.fullmatch(head) or git(candidate, "rev-parse", "HEAD") != head:
        raise ValueError("unexpected branch or head")
    if not bot_name or not bot_email or "\n" in bot_name + bot_email:
        raise ValueError("invalid bot identity")
    proposed = verify(base, candidate, check_lock=False)
    cargo_env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"GH_TOKEN", "GITHUB_TOKEN"} and not key.startswith("GIT_CONFIG_")
    }
    subprocess.run(
        ["cargo", "update", "--workspace", "--offline"],
        cwd=candidate,
        env=cargo_env,
        check=True,
    )
    if lock(base, workspace(base)[1]) != lock(candidate, proposed):
        raise ValueError("cargo changed non-workspace lock entries")
    if git(candidate, "diff", "--name-only") not in ("", "Cargo.lock"):
        raise ValueError("cargo changed unexpected files")
    if not git(candidate, "diff", "--name-only", "--", "Cargo.lock"):
        return
    subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "add", "--", "Cargo.lock"],
        cwd=candidate,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            f"user.name={bot_name}",
            "-c",
            f"user.email={bot_email}",
            "commit",
            "-m",
            "chore: synchronize release lockfile",
        ],
        cwd=candidate,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "credential.helper=",
            "-c",
            "credential.helper=!gh auth git-credential",
            "push",
            f"--force-with-lease=refs/heads/{branch}:{head}",
            "origin",
            f"HEAD:refs/heads/{branch}",
        ],
        cwd=candidate,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "sync"):
        p = sub.add_parser(name)
        p.add_argument("--base", type=Path, required=True)
        p.add_argument("--candidate", type=Path, required=True)
        if name == "verify":
            p.add_argument("--ordinary", action="store_true")
        else:
            for arg in ("branch", "head", "bot-name", "bot-email"):
                p.add_argument(f"--{arg}", required=True)
    p = sub.add_parser("select")
    for arg in ("prs", "output", "repo", "bot"):
        p.add_argument(f"--{arg}", required=True)
    p.add_argument("--bot-id", type=int, required=True)
    p = sub.add_parser("title")
    p.add_argument("title")
    args = parser.parse_args()
    try:
        if args.command == "verify":
            print(verify(args.base, args.candidate, proposal=not args.ordinary))
        elif args.command == "sync":
            sync(args.base, args.candidate, args.branch, args.head, args.bot_name, args.bot_email)
        elif args.command == "select":
            prs = json.loads(Path(args.prs).read_text())
            output = json.loads(Path(args.output).read_text() or "[]")
            selected = select_pr(prs, output, args.repo, args.bot, args.bot_id)
            print(
                json.dumps(
                    None
                    if selected is None
                    else {
                        "number": selected["number"],
                        "branch": selected["head"]["ref"],
                        "head": selected["head"]["sha"],
                    }
                )
            )
        elif not TITLE.fullmatch(args.title):
            raise ValueError("PR title must be a Conventional Commit")
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"release PR check failed: {exc}\n")


if __name__ == "__main__":
    main()
