import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_release as release
import locked_version


@pytest.mark.parametrize(
    ("cargo", "pep"),
    [
        ("0.1.0", "0.1.0"),
        ("0.1.1-dev.1", "0.1.1.dev1"),
        ("0.1.1-rc.1", "0.1.1rc1"),
        ("0.1.1-alpha.2", "0.1.1a2"),
        ("0.1.1-beta.3", "0.1.1b3"),
    ],
)
def test_maturin_version_mapping(cargo, pep):
    assert release.pep440_version(cargo) == pep


@pytest.mark.parametrize(
    "version", ["01.2.3", "1.2", "1.2.3-dev.01", "1.2.3-other.1", "1.2.3+meta"]
)
def test_unsupported_versions_rejected(version):
    with pytest.raises(ValueError):
        release.pep440_version(version)


def test_production_validation():
    assert release.validate("production", "1.2.3", "v1.2.3", "## [1.2.3] — release") == "1.2.3"
    with pytest.raises(ValueError, match="tag"):
        release.validate("production", "1.2.3", "v1.2.4", "## [1.2.3]")
    with pytest.raises(ValueError, match="CHANGELOG"):
        release.validate("production", "1.2.3", "v1.2.3", "## [1.2.30]")
    with pytest.raises(ValueError, match="production"):
        release.validate("production", "1.2.3-dev.1", "v1.2.3-dev.1", "## [1.2.3-dev.1]")


# Captured from release-please 17.1.3 Simple.buildReleasePullRequest.
@pytest.mark.parametrize(
    ("version", "heading"),
    [
        ("0.1.0", "## 0.1.0 (2026-10-30)"),
        (
            "0.1.1",
            "## [0.1.1](https://github.com/example/cuscan/compare/v0.1.0...v0.1.1) (2026-10-30)",
        ),
    ],
)
def test_generated_headings(version, heading):
    assert release.validate("production", version, f"v{version}", heading) == version
    with pytest.raises(ValueError, match="exactly one"):
        release.validate("production", version, f"v{version}", f"{heading}\n{heading}")


def test_duplicate_older_version_is_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        release.validate(
            "production",
            "0.1.1",
            "v0.1.1",
            "## 0.1.1 (2026-10-30)\n## 0.1.0 (2026-10-29)\n## [0.1.0] — release",
        )


@pytest.mark.parametrize("version", ["1.2.3-alpha.1", "1.2.3-beta.1", "1.2.3-rc.1"])
def test_production_rejects_prereleases(version):
    with pytest.raises(ValueError, match="stable"):
        release.validate("production", version, f"v{version}", f"## {version}")


def test_rehearsal_validation():
    assert release.validate("rehearsal", "1.2.3-dev.1", None, "") == "1.2.3.dev1"
    with pytest.raises(ValueError, match="rehearsal"):
        release.validate("rehearsal", "1.2.3", None, "")


def test_non_ancestor_rejected(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.org")
    git("checkout", "-b", "main")
    (tmp_path / "file").write_text("one")
    git("add", ".")
    git("commit", "-m", "main")
    git("remote", "add", "origin", str(tmp_path))
    git("checkout", "--orphan", "unrelated")
    (tmp_path / "file").write_text("two")
    git("add", ".")
    git("commit", "-m", "unrelated")
    commit = git("rev-parse", "HEAD")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="not an ancestor"):
        release.check_ancestry(commit)


def test_locked_version_requires_unique_version(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text('[[package]]\nname = "maturin"\nversion = "1.15.0"\n')
    assert locked_version.locked_version(lock, "maturin") == "1.15.0"
    with pytest.raises(ValueError):
        locked_version.locked_version(lock, "missing")
    with lock.open("a") as stream:
        stream.write('[[package]]\nname = "maturin"\nversion = "1.16.0"\n')
    with pytest.raises(ValueError):
        locked_version.locked_version(lock, "maturin")
