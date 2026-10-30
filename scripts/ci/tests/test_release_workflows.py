"""Exercise inline release workflow code without actions or remote credentials."""

import io
import json
import os
import subprocess
import sys
import types
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("act", "event", "enabled"),
    [
        ("true", {}, False),
        ("TRUE", {}, False),
        ("", {"act": True}, False),
        ("false", {"act": True}, False),
        ("", {"act": False}, True),
        ("", {}, True),
    ],
)
def test_release_please_guard_before_credentials(tmp_path, act, event, enabled):
    text = (ROOT / ".github/workflows/release-please.yml").read_text()
    assert "github.event_path" not in text
    guard = text.split("python3 - <<'PY'\n", 1)[1].split("\n          PY", 1)[0]
    code = "\n".join(line.removeprefix("          ") for line in guard.splitlines())
    fixture = tmp_path / "event.json"
    fixture.write_text(json.dumps(event))
    output = tmp_path / "output"
    subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "ACT": act,
            "GITHUB_EVENT_PATH": str(fixture),
            "GITHUB_OUTPUT": str(output),
        },
        check=True,
    )
    assert output.read_text() == f"enabled={str(enabled).lower()}\n"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("app_slug", "", "Missing App slug"),
        ("bot_login", "other[bot]", "Unexpected App bot identity"),
        ("bot_type", "User", "Unexpected App bot identity"),
    ],
)
def test_release_please_select_rejects_identity_mismatch(
    tmp_path, monkeypatch, field, value, error
):
    with pytest.raises(SystemExit, match=error):
        run_select(tmp_path, monkeypatch, {field: value})


@pytest.mark.parametrize("selected", [None, {"branch": "release-branch", "head": "abc123"}])
def test_release_please_select_anonymous_metadata(tmp_path, monkeypatch, selected):
    requests, calls = run_select(tmp_path, monkeypatch, selected=selected)
    assert [request.full_url for request in requests] == [
        "https://api.github.com/users/release-app[bot]",
    ]
    assert len(calls) == 2
    assert calls[0] == [
        "gh",
        "api",
        "--paginate",
        "--slurp",
        "/repos/owner/repo/pulls?state=open&base=main&per_page=100",
    ]
    assert calls[1] == [
        "python3",
        "trusted/scripts/ci/check_release_pr.py",
        "select",
        "--prs",
        str(tmp_path / "prs.json"),
        "--output",
        str(tmp_path / "action-prs.json"),
        "--repo",
        "owner/repo",
        "--bot",
        "release-app[bot]",
        "--bot-id",
        "42",
    ]
    assert json.loads((tmp_path / "prs.json").read_text()) == [{"number": 1}, {"number": 2}]
    assert (tmp_path / "action-prs.json").read_text() == '[{"number": 1}]'
    expected = (
        "present=false\n"
        if selected is None
        else (
            "present=true\nbranch=release-branch\nhead=abc123\n"
            "bot=release-app[bot]\nemail=42+release-app[bot]@users.noreply.github.com\n"
        )
    )
    assert (tmp_path / "output").read_text() == expected


def run_select(tmp_path, monkeypatch, overrides=None, selected=None):
    metadata = {
        "app_slug": "release-app",
        "bot_login": "release-app[bot]",
        "bot_type": "Bot",
    }
    metadata.update(overrides or {})
    monkeypatch.setenv("GH_TOKEN", "integration-secret")
    monkeypatch.setenv("REPOSITORY", "owner/repo")
    monkeypatch.setenv("APP_SLUG", metadata["app_slug"])
    monkeypatch.setenv("ACTION_PRS", '[{"number": 1}]')
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    requests = []
    calls = []

    def urlopen(request, timeout):
        assert timeout == 10
        assert request.get_header("Authorization") is None
        assert request.get_header("Accept") == "application/vnd.github+json"
        assert request.get_header("User-agent") == "release-please"
        requests.append(request)
        payload = {"id": 42, "login": metadata["bot_login"], "type": metadata["bot_type"]}
        return io.BytesIO(json.dumps(payload).encode())

    def check_output(command, text):
        assert text is True
        assert os.environ["GH_TOKEN"] == "integration-secret"
        calls.append(command)
        return (
            json.dumps([[{"number": 1}], [{"number": 2}]])
            if len(calls) == 1
            else json.dumps(selected)
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(subprocess, "check_output", check_output)
    text = (ROOT / ".github/workflows/release-please.yml").read_text()
    section = text.split("- name: Select authenticated root release PR", 1)[1]
    script = section.split("python3 - <<'PY'\n", 1)[1].split("\n          PY", 1)[0]
    exec("\n".join(line.removeprefix("          ") for line in script.splitlines()), {})
    return requests, calls


def run_ci_consistency(
    tmp_path,
    monkeypatch,
    *,
    current=None,
    event_pr=None,
    proposal=True,
    bot_id="42",
    bot_login="release-app[bot]",
):
    pr = {
        "number": 1,
        "title": "chore: release",
        "user": {"id": 42, "login": "release-app[bot]", "type": "Bot"},
        "head": {"sha": "abc", "ref": "release-branch", "repo": {"full_name": "owner/repo"}},
        "base": {"sha": "base", "ref": "main", "repo": {"full_name": "owner/repo"}},
        "labels": [{"name": "autorelease: pending"}],
    }
    current = pr if current is None else current
    event_pr = pr if event_pr is None else event_pr
    fixture = tmp_path / "event.json"
    fixture.write_text(json.dumps({"pull_request": event_pr}))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(fixture))
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("BOT_ID", bot_id)
    monkeypatch.setenv("BOT_LOGIN", bot_login)
    helper = types.ModuleType("check_release_pr")
    helper.BRANCH = "release-branch"
    helper.is_proposal = lambda base, candidate, event: proposal
    monkeypatch.setitem(sys.modules, "check_release_pr", helper)
    calls = []

    def run(command, check):
        assert check is True
        calls.append(command)

    def check_output(command, text):
        assert text is True
        calls.append(command)
        assert command == ["gh", "api", "/repos/owner/repo/pulls/1"]
        return json.dumps(current)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "check_output", check_output)
    text = (ROOT / ".github/workflows/ci.yml").read_text()
    section = text.split("- name: Validate Conventional Commit title and shared release state", 1)[
        1
    ]
    script = section.split("python3 - <<'PY'\n", 1)[1].split("\n          PY", 1)[0]
    try:
        exec("\n".join(line.removeprefix("          ") for line in script.splitlines()), {})
    finally:
        assert all("/apps/" not in str(call) and "/users/" not in str(call) for call in calls)
    return calls


def test_ci_release_bot_identity_accepted(tmp_path, monkeypatch):
    text = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "${{ vars.RELEASE_PLEASE_BOT_ID }}" in text
    assert "${{ vars.RELEASE_PLEASE_BOT_LOGIN }}" in text
    calls = run_ci_consistency(tmp_path, monkeypatch)
    assert any("verify" in call and "--ordinary" not in call for call in calls)


@pytest.mark.parametrize(
    ("bot_id", "bot_login"),
    [
        ("", "release-app[bot]"),
        ("0", "release-app[bot]"),
        ("42x", "release-app[bot]"),
        ("42", ""),
        ("42", "release-app"),
        ("42", "bad login[bot]"),
    ],
)
def test_ci_release_bot_configuration_invalid(tmp_path, monkeypatch, bot_id, bot_login):
    with pytest.raises(SystemExit, match="Invalid configured release bot identity"):
        run_ci_consistency(tmp_path, monkeypatch, bot_id=bot_id, bot_login=bot_login)


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (("user", "id"), 43, "identity/branch/label"),
        (("user", "id"), True, "identity/branch/label"),
        (("user", "login"), "other[bot]", "identity/branch/label"),
        (("user", "type"), "User", "identity/branch/label"),
        (("head", "sha"), "new", "advanced"),
        (("head", "repo", "full_name"), "fork/repo", "identity/branch/label"),
        (("base", "repo", "full_name"), "other/repo", "identity/branch/label"),
        (("base", "ref"), "other", "identity/branch/label"),
        (("head", "ref"), "release-branch-extra", "identity/branch/label"),
        (("labels",), [], "identity/branch/label"),
    ],
)
def test_ci_release_pr_rejects_mismatch(tmp_path, monkeypatch, path, value, error):
    current = {
        "number": 1,
        "user": {"id": 42, "login": "release-app[bot]", "type": "Bot"},
        "head": {"sha": "abc", "ref": "release-branch", "repo": {"full_name": "owner/repo"}},
        "base": {"ref": "main", "repo": {"full_name": "owner/repo"}},
        "labels": [{"name": "autorelease: pending"}],
    }
    target = current
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(SystemExit, match=error):
        run_ci_consistency(tmp_path, monkeypatch, current=current)


def test_ci_ordinary_pr_does_not_require_bot_configuration(tmp_path, monkeypatch):
    calls = run_ci_consistency(tmp_path, monkeypatch, proposal=False, bot_id="", bot_login="")
    assert any("verify" in call and "--ordinary" in call for call in calls)
    assert not any(call[:2] == ["gh", "api"] for call in calls)
