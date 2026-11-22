#!/usr/bin/env -S uv run --no-sync
"""Write a `cuscan-native-provenance-v1` receipt for a freshly built native worker binary.

Run immediately after the worker build, from any directory:

    uv run --no-sync comparison/native_provenance.py --source-root WORKTREE \
        --binary WORKTREE/target/release/examples/ann_construction_worker \
        --build-command "soldr cargo build -p umap --release --example ... --features nvtx" \
        --profile release --features nvtx --out target/rinnd-4/phase-0/h-provenance.json

The source hash covers `git diff --binary HEAD` over the build inputs followed by every
untracked, non-ignored file under the same paths (sorted) as `path\\0sha256\\n`, so untracked
modules are part of the identity. Existing output paths are refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

SCHEMA = "cuscan-native-provenance-v1"
SOURCE_PATHS = (
    "crates",
    "comparison",
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo/config.toml",
    "cubecl.toml",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=root)


def source_hash(root: Path) -> tuple[str, list[str]]:
    """Hash tracked changes and untracked, non-ignored files under the build inputs."""
    digest = hashlib.sha256()
    digest.update(git(root, "diff", "--binary", "HEAD", "--", *SOURCE_PATHS))
    listed = git(
        root, "ls-files", "--others", "--exclude-standard", "-z", "--", *SOURCE_PATHS
    ).split(b"\0")
    untracked = sorted(entry.decode() for entry in listed if entry)
    for relative in untracked:
        digest.update(relative.encode() + b"\0" + sha256_file(root / relative).encode() + b"\n")
    return digest.hexdigest(), untracked


def tool_output(*command: str) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def build_receipt(
    *, root: Path, binary: Path, build_command: str, profile: str, features: list[str]
) -> dict[str, Any]:
    root = root.resolve()
    binary = binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"worker binary not found: {binary}")
    base_sha = git(root, "rev-parse", "HEAD").decode().strip()
    tree_id = git(root, "rev-parse", "HEAD^{tree}").decode().strip()
    for name, value in (("base_sha", base_sha), ("tree_id", tree_id)):
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError(f"{name} is not a lowercase Git object ID: {value!r}")
    digest, untracked = source_hash(root)
    rustc = subprocess.run(
        ["rustc", "-Vv"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "schema": SCHEMA,
        "base_sha": base_sha,
        "tree_id": tree_id,
        "source_root": str(root),
        "source_paths": list(SOURCE_PATHS),
        "source_sha256": digest,
        "source_hash_rule": (
            "sha256(git diff --binary HEAD -- SOURCE_PATHS || for each sorted untracked "
            "non-ignored path: path NUL sha256 LF)"
        ),
        "untracked_paths": untracked,
        "build_command": build_command,
        "profile": profile,
        "features": features,
        "rustc": rustc,
        "nvcc": tool_output("nvcc", "--version"),
        "nvidia_driver": tool_output(
            "nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"
        ),
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
    }


def load_receipt(path: Path, binary: Path) -> dict[str, Any]:
    """Load a native receipt and refuse it unless the binary still hashes to its record."""
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} receipt")
    for key, length in (("base_sha", 40), ("tree_id", 40), ("source_sha256", 64)):
        value = receipt.get(key)
        if not isinstance(value, str) or re.fullmatch(f"[0-9a-f]{{{length}}}", value) is None:
            raise ValueError(f"provenance {key} must be a lowercase {length}-digit hash: {path}")
    actual = sha256_file(binary)
    if receipt.get("binary_sha256") != actual:
        raise ValueError(f"provenance binary_sha256 does not match {binary}: {path}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--build-command", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--features", nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {args.out}")
    receipt = build_receipt(
        root=args.source_root,
        binary=args.binary,
        build_command=args.build_command,
        profile=args.profile,
        features=args.features,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
