#!/usr/bin/env python3
"""Validate an existing release and recover publication without overwriting artifacts."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

GITHUB = "https://api.github.com"
PYPI = "https://pypi.org/pypi"


def request_json(url: str, token: str | None = None):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "cuscan-release"}
    if token:
        headers.update({"Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"})
    with urlopen(Request(url, headers=headers), timeout=30) as response:
        return json.load(response)


def release_metadata(repo: str, tag: str, sha: str, token: str, attempts: int = 6):
    base = f"{GITHUB}/repos/{repo}"
    for attempt in range(attempts):
        try:
            release = request_json(f"{base}/releases/tags/{quote(tag, safe='')}", token)
            break
        except HTTPError as exc:
            if exc.code != 404 or attempt == attempts - 1:
                raise
            time.sleep(5)
    if not isinstance(release, dict) or (
        release.get("tag_name") != tag
        or release.get("draft") is not False
        or release.get("prerelease") is not False
        or release.get("immutable") is not False
        or type(release.get("id")) is not int
    ):
        raise ValueError("release identity, stability, or mutable status is invalid")
    ref = request_json(f"{base}/git/ref/tags/{quote(tag, safe='')}", token)
    if not isinstance(ref, dict) or ref.get("ref") != f"refs/tags/{tag}":
        raise ValueError("invalid release tag reference")
    obj = ref.get("object")
    for _ in range(10):
        if not isinstance(obj, dict) or not isinstance(obj.get("sha"), str):
            raise ValueError("invalid release tag object")
        if obj.get("type") != "tag":
            break
        annotated = request_json(f"{base}/git/tags/{obj['sha']}", token)
        obj = annotated.get("object") if isinstance(annotated, dict) else None
    if not isinstance(obj, dict) or obj.get("type") != "commit" or obj.get("sha") != sha:
        raise ValueError("release tag does not resolve to the checked-out commit")
    return release


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def distributions(directory: Path) -> list[Path]:
    files = sorted(directory.iterdir())
    if (
        len(files) != 3
        or any(not path.is_file() for path in files)
        or sum(path.name.endswith(".whl") for path in files) != 2
        or sum(path.name.endswith(".tar.gz") for path in files) != 1
    ):
        raise ValueError("expected exactly two wheels and one source distribution")
    return files


def missing_pypi(files: list[Path], version: str) -> list[Path]:
    try:
        metadata = request_json(f"{PYPI}/cuscan/{quote(version, safe='')}/json")
    except HTTPError as exc:
        if exc.code == 404:
            return files
        raise
    if not isinstance(metadata, dict) or not isinstance(metadata.get("urls"), list):
        raise ValueError("invalid PyPI response")
    local = {path.name: path for path in files}
    seen = set()
    for item in metadata["urls"]:
        if not isinstance(item, dict):
            raise ValueError("invalid PyPI artifact")
        name = item.get("filename")
        hashes = item.get("digests")
        if (
            not isinstance(name, str)
            or name in seen
            or name not in local
            or not isinstance(hashes, dict)
        ):
            raise ValueError("ambiguous or unexpected PyPI artifact")
        seen.add(name)
        if hashes.get("sha256") != digest(local[name]):
            raise ValueError(f"PyPI artifact hash conflict: {name}")
    return [path for path in files if path.name not in seen]


def missing_assets(repo: str, release_id: int, token: str, files: list[Path]) -> list[Path]:
    assets = []
    page = 1
    while True:
        batch = request_json(
            f"{GITHUB}/repos/{repo}/releases/{release_id}/assets?per_page=100&page={page}", token
        )
        if not isinstance(batch, list):
            raise ValueError("invalid release assets response")
        assets.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    existing = {}
    for asset in assets:
        if not isinstance(asset, dict) or not isinstance(asset.get("name"), str):
            raise ValueError("invalid release asset")
        name = asset["name"]
        if name in existing:
            raise ValueError(f"duplicate release asset: {name}")
        existing[name] = asset
    missing = []
    for path in files:
        asset = existing.get(path.name)
        if asset is None:
            missing.append(path)
        elif asset.get("state") != "uploaded" or asset.get("digest") != f"sha256:{digest(path)}":
            raise ValueError(f"release asset hash conflict or unavailable digest: {path.name}")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "upload"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--pending", type=Path, default=Path("dist-pending"))
    args = parser.parse_args()
    if os.environ.get("ACT") == "true":
        parser.exit(1, "release API operations are forbidden under ACT\n")
    try:
        token = os.environ["GH_TOKEN"]
        files = distributions(args.dist)
        release = release_metadata(args.repo, args.tag, args.sha, token)
        if args.action == "prepare":
            missing = missing_pypi(files, args.version)
            args.pending.mkdir(exist_ok=False)
            for path in missing:
                shutil.copyfile(path, args.pending / path.name)
            if output := os.environ.get("GITHUB_OUTPUT"):
                with open(output, "a", encoding="utf-8") as stream:
                    stream.write(f"publish_needed={'true' if missing else 'false'}\n")
        else:
            missing = missing_assets(args.repo, release["id"], token, files)
            if missing:
                subprocess.run(
                    ["gh", "release", "upload", args.tag, *map(str, missing)], check=True
                )
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"release artifact verification failed: {exc}\n")


if __name__ == "__main__":
    main()
