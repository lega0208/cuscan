import importlib.util
import json
import runpy
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "issue_on_failure.py"
spec = importlib.util.spec_from_file_location("issue_on_failure", SCRIPT)
issue_on_failure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(issue_on_failure)


@pytest.fixture
def github_env(monkeypatch):
    monkeypatch.delenv("ACT", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.example/")
    monkeypatch.setenv("GITHUB_SHA", "abc123")


def mock_gh(monkeypatch, pages):
    calls = []

    def gh(*args):
        calls.append(args)
        return json.dumps(pages) if args[0] == "api" else ""

    monkeypatch.setattr(issue_on_failure, "gh", gh)
    return calls


def test_act_skips_gh_without_required_environment(monkeypatch, capsys):
    monkeypatch.setenv("ACT", "true")
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)

    def unexpected(*args):
        pytest.fail("gh must not be called under act")

    monkeypatch.setattr(issue_on_failure, "gh", unexpected)
    issue_on_failure.main()
    assert "Skipping" in capsys.readouterr().out


def test_no_matching_issue_creates(github_env, monkeypatch):
    calls = mock_gh(monkeypatch, [[{"title": "Other", "number": 1}]])
    issue_on_failure.main()
    assert calls == [
        ("api", "--paginate", "--slurp", "repos/owner/repo/issues?state=open&per_page=100"),
        (
            "issue",
            "create",
            "--repo",
            "owner/repo",
            "--title",
            "Scheduled CI failure",
            "--body",
            "Scheduled CI failed for commit `abc123`.\n\n"
            "Run: https://github.example/owner/repo/actions/runs/123",
        ),
    ]


def test_comments_on_oldest_matching_issue(github_env, monkeypatch):
    calls = mock_gh(
        monkeypatch,
        [
            [
                {"title": "Scheduled CI failure", "number": 27},
                {"title": "Scheduled CI failure", "number": 4},
                {"title": "Other", "number": 1},
            ]
        ],
    )
    issue_on_failure.main()
    assert calls[1][:5] == ("issue", "comment", "4", "--repo", "owner/repo")
    assert len(calls) == 2


def test_pull_request_with_matching_title_is_ignored(github_env, monkeypatch):
    calls = mock_gh(
        monkeypatch,
        [
            [
                {"title": "Scheduled CI failure", "number": 2, "pull_request": {}},
            ]
        ],
    )
    issue_on_failure.main()
    assert calls[1][:2] == ("issue", "create")


def test_late_paginated_issue_is_matched(github_env, monkeypatch):
    calls = mock_gh(
        monkeypatch,
        [
            [{"title": "Other", "number": 1}],
            [],
            [{"title": "Scheduled CI failure", "number": 42}],
        ],
    )
    issue_on_failure.main()
    assert calls[0][1:3] == ("--paginate", "--slurp")
    assert calls[1][:3] == ("issue", "comment", "42")
    assert len(calls) == 2


@pytest.mark.parametrize("failed_call", [1, 2])
def test_gh_failure_propagates_from_main(github_env, monkeypatch, failed_call):
    calls = []

    def gh(*args):
        calls.append(args)
        if len(calls) == failed_call:
            raise subprocess.CalledProcessError(1, ["gh", *args])
        return "[[]]"

    monkeypatch.setattr(issue_on_failure, "gh", gh)
    with pytest.raises(subprocess.CalledProcessError):
        issue_on_failure.main()
    assert len(calls) == failed_call


def test_invalid_json_propagates_from_main(github_env, monkeypatch):
    calls = []

    def gh(*args):
        calls.append(args)
        return "not json"

    monkeypatch.setattr(issue_on_failure, "gh", gh)
    with pytest.raises(json.JSONDecodeError):
        issue_on_failure.main()
    assert len(calls) == 1


@pytest.mark.parametrize("invalid_json", [False, True])
def test_cli_errors_exit_nonzero(github_env, monkeypatch, capsys, invalid_json):
    def run(args, **kwargs):
        if invalid_json:
            return subprocess.CompletedProcess(args, 0, stdout="not json")
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exc.value.code == 1
    assert "Could not report scheduled CI failure" in capsys.readouterr().err
