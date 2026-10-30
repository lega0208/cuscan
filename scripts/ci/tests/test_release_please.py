"""Offline tests for the trusted release proposal helper."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).parents[1] / "check_release_pr.py"
SPEC = importlib.util.spec_from_file_location("check_release_pr", HELPER)
assert SPEC and SPEC.loader
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def command(root, *args):
    return subprocess.check_output(args, cwd=root, text=True).strip()


def commit(root, message):
    command(root, "git", "add", "--all")
    command(
        root,
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        message,
    )


def fixture_repo(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    command(base, "git", "init", "-q")
    members = list(helper.MEMBERS)
    (base / "Cargo.toml").write_text(
        "[workspace]\nmembers = "
        + json.dumps(members)
        + '\n[workspace.package]\nversion = "0.1.0"\n'
    )
    packages = []
    for path, name in helper.MEMBERS.items():
        member = base / path
        member.mkdir(parents=True)
        (member / "Cargo.toml").write_text(
            f'[package]\nname = "{name}"\nversion.workspace = true\n'
        )
        packages.append(f'[[package]]\nname = "{name}"\nversion = "0.1.0"\n')
    packages.append(
        '[[package]]\nname = "external"\nversion = "1.0.0"\n'
        'source = "registry+x"\nchecksum = "abc"\n'
    )
    (base / "Cargo.lock").write_text("\n".join(packages))
    (base / "version.txt").write_text("0.1.0\n")
    (base / ".release-please-manifest.json").write_text("{}\n")
    (base / "CHANGELOG.md").write_text("# Changelog\n")
    (base / "release-please-config.json").write_text(json.dumps(helper.BOOTSTRAP_CONFIG))
    commit(base, "base")
    candidate = tmp_path / "candidate"
    command(tmp_path, "git", "clone", "-q", str(base), str(candidate))
    (candidate / ".release-please-manifest.json").write_text('{".": "0.1.0"}\n')
    (candidate / "CHANGELOG.md").write_text("# Changelog\n\n## 0.1.0 (2026-01-01)\n")
    commit(candidate, "proposal")
    return base, candidate


def test_initial_proposal_and_repeated_release(tmp_path):
    base, candidate = fixture_repo(tmp_path)
    assert helper.verify(base, candidate) == "0.1.0"
    (base / ".release-please-manifest.json").write_text('{".": "0.1.0"}\n')
    commit(base, "released")
    command(candidate, "git", "fetch", "-q", str(base), "HEAD")
    with pytest.raises(ValueError, match="increase"):
        helper.verify(base, candidate)


def test_stale_release_branch_is_routed_as_proposal(tmp_path):
    base, candidate = fixture_repo(tmp_path)
    (base / ".release-please-manifest.json").write_text('{".": "0.1.0"}\n')
    commit(base, "released")
    command(candidate, "git", "fetch", "-q", str(base), "HEAD")
    assert not helper.is_proposal(base, candidate, None)
    assert helper.is_proposal(base, candidate, sample_pr())
    with pytest.raises(ValueError, match="increase"):
        helper.verify(base, candidate)


def test_manifest_change_routes_as_proposal(tmp_path):
    base, candidate = fixture_repo(tmp_path)
    assert helper.is_proposal(base, candidate, None)
    pr = sample_pr()
    pr["head"]["ref"] = "feature"
    assert helper.is_proposal(base, candidate, pr)


def test_missing_lock_is_rejected(tmp_path):
    base, candidate = fixture_repo(tmp_path)
    (candidate / "Cargo.lock").unlink()
    commit(candidate, "remove lock")
    with pytest.raises(ValueError, match="missing or non-regular"):
        helper.verify(base, candidate)


@pytest.mark.parametrize("field", ["release-type", "extra-files", "bump-minor-pre-major"])
def test_bootstrap_rejects_tampered_base_config(tmp_path, field):
    base, candidate = fixture_repo(tmp_path)
    for root in (base, candidate):
        path = root / "release-please-config.json"
        config = json.loads(path.read_text())
        config["packages"]["."][field] = "tampered"
        path.write_text(json.dumps(config))
        commit(root, "tampered config")
    command(candidate, "git", "fetch", "-q", str(base), "HEAD")
    with pytest.raises(ValueError, match="bootstrap release configuration"):
        helper.verify(base, candidate)


def test_bootstrap_setup_and_ordinary_changes(tmp_path):
    base, candidate = fixture_repo(tmp_path)
    for root in (base, candidate):
        command(root, "git", "reset", "--hard", "-q", "HEAD~1" if root == candidate else "HEAD")
    for name in (".release-please-manifest.json", "version.txt", "release-please-config.json"):
        (base / name).unlink()
    commit(base, "before setup")
    command(candidate, "git", "fetch", "-q", str(base), "HEAD")
    assert not helper.is_proposal(base, candidate, None)
    assert helper.verify(base, candidate, proposal=False) == "0.1.0"
    (candidate / "README.md").write_text("ordinary change\n")
    commit(candidate, "ordinary")
    assert helper.verify(base, candidate, proposal=False) == "0.1.0"


def test_post_bootstrap_cleanup_and_required_config(tmp_path):
    base, candidate = fixture_repo(tmp_path)
    for root in (base, candidate):
        config_path = root / "release-please-config.json"
        config = json.loads(config_path.read_text())
        del config["bootstrap-sha"]
        del config["packages"]["."]["release-as"]
        config_path.write_text(json.dumps(config))
        (root / ".release-please-manifest.json").write_text('{".": "0.1.0"}\n')
        commit(root, "bootstrap cleanup")
    command(candidate, "git", "fetch", "-q", str(base), "HEAD")
    assert helper.verify(base, candidate, proposal=False) == "0.1.0"
    for root in (base, candidate):
        config_path = root / "release-please-config.json"
        config = json.loads(config_path.read_text())
        config["packages"]["."]["release-type"] = "rust"
        config_path.write_text(json.dumps(config))
        commit(root, "tamper strategy")
    command(candidate, "git", "fetch", "-q", str(base), "HEAD")
    (candidate / ".release-please-manifest.json").write_text('{".": "0.1.1"}\n')
    (candidate / "Cargo.toml").write_text(
        (candidate / "Cargo.toml").read_text().replace("0.1.0", "0.1.1")
    )
    (candidate / "version.txt").write_text("0.1.1\n")
    commit(candidate, "proposal")
    with pytest.raises(ValueError, match="invalid release configuration"):
        helper.verify(base, candidate, check_lock=False)


def test_proposal_rejects_unexpected_files_and_duplicate_heading(tmp_path):
    base, candidate = fixture_repo(tmp_path)
    (candidate / "CHANGELOG.md").write_text("## 0.1.0 (2026-01-01)\n## [0.1.0](url) (2026-01-02)\n")
    commit(candidate, "duplicate")
    with pytest.raises(ValueError, match="duplicate"):
        helper.verify(base, candidate)
    (candidate / "evil.py").write_text("raise SystemExit\n")
    commit(candidate, "unexpected")
    with pytest.raises(ValueError, match="disallowed"):
        helper.verify(base, candidate)


def test_lock_resolution_and_member_tampering_rejected(tmp_path):
    base, candidate = fixture_repo(tmp_path)
    lock = candidate / "Cargo.lock"
    lock.write_text(lock.read_text().replace('checksum = "abc"', 'checksum = "def"'))
    commit(candidate, "dependency tampering")
    with pytest.raises(ValueError, match="lockfile"):
        helper.verify(base, candidate)
    member = candidate / "crates/umap/Cargo.toml"
    member.write_text(member.read_text().replace("version.workspace = true", 'version = "0.1.0"'))
    commit(candidate, "member tampering")
    with pytest.raises(ValueError, match="disallowed"):
        helper.verify(base, candidate)


def sample_pr():
    return {
        "number": 10,
        "state": "open",
        "user": {"login": "release[bot]", "id": 123, "type": "Bot"},
        "labels": [{"name": "autorelease: pending"}],
        "base": {"ref": "main", "repo": {"full_name": "owner/repo"}},
        "head": {"ref": helper.BRANCH, "sha": "a" * 40, "repo": {"full_name": "owner/repo"}},
    }


def test_select_action_and_noop():
    pr = sample_pr()
    assert helper.select_pr([pr], [{"number": 10}], "owner/repo", "release[bot]", 123) == pr
    assert helper.select_pr([pr], [], "owner/repo", "release[bot]", 123) == pr
    assert helper.select_pr([], [], "owner/repo", "release[bot]", 123) is None
    with pytest.raises(ValueError):
        helper.select_pr([pr, pr], [], "owner/repo", "release[bot]", 123)
    with pytest.raises(ValueError):
        helper.select_pr([pr], [{"number": 11}], "owner/repo", "release[bot]", 123)


@pytest.mark.parametrize("mutation", ["owner", "base", "branchrepo", "label", "sha", "type"])
def test_select_rejects_spoofed_metadata(mutation):
    pr = sample_pr()
    if mutation == "owner":
        pr["user"]["id"] = 456
    elif mutation == "base":
        pr["base"]["ref"] = "other"
    elif mutation == "branchrepo":
        pr["head"]["repo"]["full_name"] = "fork/repo"
    elif mutation == "label":
        pr["labels"] = [{"name": "autorelease: tagged"}]
    elif mutation == "sha":
        pr["head"]["sha"] = "not-a-sha"
    else:
        pr["user"]["type"] = "User"
    with pytest.raises(ValueError):
        helper.select_pr([pr], [], "owner/repo", "release[bot]", 123)


def test_title_validation():
    assert helper.TITLE.fullmatch("chore(main): release 0.1.0")
    assert helper.TITLE.fullmatch("feat!: breaking change")
    assert not helper.TITLE.fullmatch("release this")


def test_sync_noop_strips_app_credentials(tmp_path, monkeypatch):
    base, candidate = fixture_repo(tmp_path)
    head = command(candidate, "git", "rev-parse", "HEAD")
    calls = []
    monkeypatch.setenv("GH_TOKEN", "secret")
    monkeypatch.setenv("GIT_CONFIG_TEST_SECRET", "secret")
    real_run = subprocess.run

    def fake_run(args, **kwargs):
        if "stdout" in kwargs:
            return real_run(args, **kwargs)
        calls.append((args, kwargs))

    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    helper.sync(base, candidate, helper.BRANCH, head, "Bot", "bot@example.com")
    assert len(calls) == 1
    assert calls[0][0] == ["cargo", "update", "--workspace", "--offline"]
    assert "GH_TOKEN" not in calls[0][1]["env"]
    assert "GIT_CONFIG_TEST_SECRET" not in calls[0][1]["env"]


def test_sync_lease_rejects_racing_push(tmp_path, monkeypatch):
    base, candidate = fixture_repo(tmp_path)
    (base / ".release-please-manifest.json").write_text('{".": "0.1.0"}\n')
    commit(base, "released")
    command(candidate, "git", "fetch", "-q", str(base), "HEAD")
    (candidate / ".release-please-manifest.json").write_text('{".": "0.1.1"}\n')
    (candidate / "Cargo.toml").write_text(
        (candidate / "Cargo.toml").read_text().replace("0.1.0", "0.1.1")
    )
    (candidate / "version.txt").write_text("0.1.1\n")
    (candidate / "CHANGELOG.md").write_text("## [0.1.1](url) (2026-01-02)\n")
    commit(candidate, "next release")
    head = command(candidate, "git", "rev-parse", "HEAD")
    calls = []
    real_run = subprocess.run

    def fake_run(args, **kwargs):
        if "stdout" in kwargs:
            return real_run(args, **kwargs)
        calls.append(args)
        if args[0] == "cargo":
            path = candidate / "Cargo.lock"
            path.write_text(path.read_text().replace('version = "0.1.0"', 'version = "0.1.1"'))
        elif "push" in args:
            raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        helper.sync(base, candidate, helper.BRANCH, head, "Bot", "bot@example.com")
    assert f"--force-with-lease=refs/heads/{helper.BRANCH}:{head}" in calls[-1]
