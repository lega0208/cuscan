import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import classify_changes as classifier


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("crates/foo/src/lib.rs", {"rust"}),
        ("Cargo.toml", {"rust", "deps"}),
        ("Cargo.lock", {"rust", "deps"}),
        ("rust-toolchain.toml", {"rust"}),
        (".cargo/config.toml", {"rust"}),
        ("burn.toml", {"rust"}),
        ("python/cuscan/__init__.py", {"python_pkg"}),
        ("pyproject.toml", {"python_pkg"}),
        ("README.md", {"python_pkg"}),
        ("LICENSE", {"python_pkg"}),
        ("THIRD_PARTY_NOTICES", {"python_pkg"}),
        ("python/tests/test_x.py", {"python_tests"}),
        ("analyze/a.py", {"python_tests"}),
        ("comparison/a.py", {"python_tests"}),
        ("scripts/a.py", {"python_tests"}),
        ("uv.lock", {"python_tests", "deps"}),
        (".python-version", {"python_tests"}),
        ("crates/foo/tests/fixtures/a.py", {"rust", "fixtures"}),
        ("crates/foo/Cargo.toml", {"rust", "deps"}),
        ("other/Cargo.toml", {"deps"}),
        ("deny.toml", {"deps"}),
        (".github/a", classifier.ALL),
        ("scripts/ci/a", classifier.ALL),
        (".actrc", classifier.ALL),
        ("docs/development.md", set()),
        ("archive/a", set()),
        ("references/a", set()),
        ("artifacts/a", set()),
        ("CHANGELOG.md", set()),
        ("AGENTS.md", set()),
        ("OTHER.md", set()),
        (".vscode/a", set()),
        (".gitignore", set()),
        (".ignore/a", set()),
        ("scripts/nested/a.py", classifier.ALL),
        ("unknown.file", classifier.ALL),
    ],
)
def test_path_rules(path, expected):
    assert classifier.classify([path]) == expected


def test_union_and_unknown_fail_safe():
    assert classifier.classify(["README.md", "Cargo.lock"]) == {"python_pkg", "rust", "deps"}
    assert classifier.classify(["README.md", "mystery"]) == classifier.ALL


def run_git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def test_real_rename_includes_old_and_new_paths(tmp_path, monkeypatch):
    run_git(tmp_path, "init")
    run_git(tmp_path, "config", "user.name", "Test")
    run_git(tmp_path, "config", "user.email", "test@example.org")
    (tmp_path / "README.md").write_text("contents\n")
    run_git(tmp_path, "add", ".")
    run_git(tmp_path, "commit", "-m", "initial")
    run_git(tmp_path, "mv", "README.md", "CHANGELOG.md")
    run_git(tmp_path, "commit", "-m", "rename")
    monkeypatch.chdir(tmp_path)
    paths = classifier.changed_paths("HEAD^", "HEAD")
    assert set(paths) == {"README.md", "CHANGELOG.md"}
    assert classifier.classify(paths) == {"python_pkg"}


@pytest.mark.parametrize("event", ["workflow_dispatch", "schedule", "merge_group"])
def test_non_diff_events_emit_all(event, tmp_path):
    output = tmp_path / "output"
    script = Path(classifier.__file__).resolve()
    env = {**os.environ, "GITHUB_OUTPUT": str(output)}
    subprocess.run([sys.executable, str(script), "--event", event], env=env, check=True)
    assert output.read_text() == "".join(f"{flag}=true\n" for flag in classifier.FLAGS)


def test_zero_before_and_unavailable_diff_emit_all(tmp_path):
    output = tmp_path / "output"
    script = Path(classifier.__file__).resolve()
    env = {**os.environ, "GITHUB_OUTPUT": str(output)}
    for before in ("0" * 40, "not-a-ref"):
        subprocess.run(
            [sys.executable, str(script), "--event", "push", "--before", before],
            cwd=tmp_path,
            env=env,
            check=True,
        )
    assert output.read_text() == 2 * "".join(f"{flag}=true\n" for flag in classifier.FLAGS)


def run_event(tmp_path, event):
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    output = tmp_path / "output"
    env = {**os.environ, "GITHUB_OUTPUT": str(output), "GITHUB_EVENT_PATH": str(event_path)}
    subprocess.run(
        [sys.executable, str(Path(classifier.__file__).resolve()), "--event", "pull_request"],
        cwd=tmp_path,
        env=env,
        check=True,
    )
    return output.read_text()


def test_act_docs_override(tmp_path):
    output = run_event(tmp_path, {"act": True, "ci_changed_paths": ["docs/development.md"]})
    assert output == "".join(f"{flag}=false\n" for flag in classifier.FLAGS)


@pytest.mark.parametrize("act", [None, False, 1, "true"])
def test_real_event_never_uses_synthetic_paths(tmp_path, act):
    output = run_event(tmp_path, {"act": act, "ci_changed_paths": ["docs/development.md"]})
    assert output == "".join(f"{flag}=true\n" for flag in classifier.FLAGS)


@pytest.mark.parametrize(
    "paths",
    [
        None,
        "docs/development.md",
        [1],
        [""],
        ["/docs/a"],
        ["docs//a"],
        ["docs/../a"],
        ["docs/./a"],
        ["docs\\a"],
        ["docs/a\0b"],
    ],
)
def test_invalid_act_paths_fail_safe(tmp_path, paths):
    output = run_event(tmp_path, {"act": True, "ci_changed_paths": paths})
    assert output == "".join(f"{flag}=true\n" for flag in classifier.FLAGS)
