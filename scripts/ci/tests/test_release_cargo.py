"""Exercise cold Cargo cache priming and offline workspace lock synchronization."""

import json
import os
import subprocess
import tomllib
from pathlib import Path

MEMBERS = (
    "umap-core",
    "umap",
    "cuscan-python",
    "cuscan-common",
    "hdbscan-core",
    "hdbscan",
)


def run(root: Path, env: dict[str, str], *command: str) -> None:
    subprocess.run(command, cwd=root, env=env, check=True, capture_output=True, text=True)


def commit(root: Path, env: dict[str, str], message: str) -> None:
    run(root, env, "git", "add", "--all")
    run(
        root,
        env,
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        message,
    )


def packages(lockfile: Path) -> dict:
    with lockfile.open("rb") as stream:
        return tomllib.load(stream)


def normalize_workspace_versions(lockfile: Path, expected: str) -> dict:
    data = packages(lockfile)
    found = set()
    for package in data["package"]:
        if package["name"] in MEMBERS and "source" not in package:
            assert package["name"] not in found
            assert package["version"] == expected
            found.add(package["name"])
            package["version"] = "workspace-version"
    assert found == set(MEMBERS)
    registry = [package for package in data["package"] if "source" in package]
    assert any(package["name"] == "itoa" and "checksum" in package for package in registry)
    return data


def test_cold_fetch_then_offline_workspace_sync_and_noop(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    bootstrap_home = tmp_path / "bootstrap-cargo-home"
    cold_home = tmp_path / "cold-cargo-home"
    bootstrap_home.mkdir()
    cold_home.mkdir()
    env = os.environ.copy()
    env["CARGO_HOME"] = str(bootstrap_home)
    env.pop("CARGO_NET_OFFLINE", None)
    run(base, env, "git", "init", "-q")
    member_paths = [f"crates/{name}" for name in MEMBERS]
    (base / "Cargo.toml").write_text(
        '[workspace]\nresolver = "2"\nmembers = '
        + json.dumps(member_paths)
        + '\n[workspace.package]\nversion = "0.1.0"\nedition = "2021"\n'
    )
    for name in MEMBERS:
        member = base / "crates" / name
        (member / "src").mkdir(parents=True)
        dependency = '\n[dependencies]\nitoa = "=1.0.15"\n' if name == "umap" else ""
        (member / "Cargo.toml").write_text(
            f'[package]\nname = "{name}"\nversion.workspace = true\n'
            f"edition.workspace = true\n{dependency}"
        )
        (member / "src" / "lib.rs").write_text("")
    run(base, env, "cargo", "generate-lockfile")
    commit(base, env, "trusted base")
    candidate = tmp_path / "candidate"
    run(tmp_path, env, "git", "clone", "-q", str(base), str(candidate))

    # This independent home starts empty; fetch only trusted base manifests and lock.
    cold_env = env | {"CARGO_HOME": str(cold_home)}
    assert not list(cold_home.iterdir())
    run(base, cold_env, "cargo", "fetch", "--locked")
    original = normalize_workspace_versions(base / "Cargo.lock", "0.1.0")

    # The initial proposal has no Cargo version change and must be a real no-op.
    initial_lock = (candidate / "Cargo.lock").read_bytes()
    run(candidate, cold_env, "cargo", "update", "--workspace", "--offline")
    assert (candidate / "Cargo.lock").read_bytes() == initial_lock

    manifest = candidate / "Cargo.toml"
    manifest.write_text(manifest.read_text().replace('version = "0.1.0"', 'version = "0.1.1"'))
    commit(candidate, cold_env, "release proposal")
    run(candidate, cold_env, "cargo", "update", "--workspace", "--offline")
    assert normalize_workspace_versions(candidate / "Cargo.lock", "0.1.1") == original
    updated_lock = (candidate / "Cargo.lock").read_bytes()
    run(candidate, cold_env, "cargo", "update", "--workspace", "--offline")
    assert (candidate / "Cargo.lock").read_bytes() == updated_lock
