#!/usr/bin/env python3
"""Print a uniquely locked package version; append version=... to GITHUB_OUTPUT.

Usage: python scripts/ci/locked_version.py PACKAGE [--lock uv.lock]
Requires Python 3.11 or newer.
"""

import argparse
import os
import sys
from pathlib import Path

import tomllib


def locked_version(path: Path, name: str) -> str:
    with path.open("rb") as stream:
        packages = tomllib.load(stream)["package"]
    versions = {entry["version"] for entry in packages if entry.get("name") == name}
    if len(versions) != 1:
        raise ValueError(f"expected exactly one locked version for {name}, found {len(versions)}")
    version = versions.pop()
    if not isinstance(version, str) or not version or "\n" in version or "\r" in version:
        raise ValueError(f"invalid locked version for {name}")
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package")
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    args = parser.parse_args()
    try:
        version = locked_version(args.lock, args.package)
        print(version)
        if output_path := os.environ.get("GITHUB_OUTPUT"):
            with open(output_path, "a", encoding="utf-8") as stream:
                stream.write(f"version={version}\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"cannot read locked version: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
