import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CUDA_LANE = ROOT / "scripts/ci/cuda_lane.sh"
ACT = ROOT / "scripts/ci/act.sh"


def git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()


def executable(path, source):
    path.write_text(f"#!{sys.executable}\n" + source)
    path.chmod(0o755)


@pytest.fixture
def lane(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "CI Test")
    git(repo, "config", "user.email", "ci@example.invalid")
    (repo / "tracked.txt").write_text("committed\n")
    diagnostics = repo / "scripts/analyze_hdbscan_stages.sh"
    diagnostics.parent.mkdir()
    diagnostics.write_text('#!/bin/bash\necho "diagnostic failed"\nexit 17\n')
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "initial")
    sha = git(repo, "rev-parse", "HEAD")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    calls.mkdir()
    executable(
        bin_dir / "gh",
        "import os, pathlib, sys\n"
        "with (pathlib.Path(os.environ['MOCK_CALLS']) / 'gh').open('a') as f:\n"
        "    f.write(' '.join(sys.argv[1:]) + '\\n')\n",
    )
    executable(
        bin_dir / "cargo",
        "import os, pathlib, signal, subprocess, sys\n"
        "p = pathlib.Path(os.environ['MOCK_CALLS'])\n"
        "(p / 'cargo-cwd').write_text(os.getcwd())\n"
        "(p / 'cargo-head').write_text(subprocess.run("
        "['git', 'symbolic-ref', '-q', 'HEAD'], capture_output=True).stdout.decode())\n"
        "(p / 'cargo-env').write_text(os.environ.get('CUSCAN_REQUIRE_CUDA', ''))\n"
        "if os.environ.get('MOCK_SIGNAL'):\n"
        "    os.kill(os.getppid(), int(os.environ['MOCK_SIGNAL']))\n"
        "sys.exit(int(os.environ.get('MOCK_CARGO_EXIT', '0')))\n",
    )
    executable(
        bin_dir / "uv",
        "import os, pathlib, sys\n"
        "with (pathlib.Path(os.environ['MOCK_CALLS']) / 'uv').open('a') as f:\n"
        "    f.write(' '.join(sys.argv[1:]) + ' backend=' + "
        "os.environ.get('CUSCAN_TEST_BACKEND', '') + ' required=' + "
        "os.environ.get('CUSCAN_REQUIRE_CUDA', '') + '\\n')\n",
    )
    executable(
        bin_dir / "act",
        "import os, pathlib\n"
        "(pathlib.Path(os.environ['MOCK_CALLS']) / 'act').write_text('called')\n",
    )
    env = os.environ.copy()
    env.update(
        PATH=f"{bin_dir}{os.pathsep}{env['PATH']}",
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        MOCK_CALLS=str(calls),
    )
    return repo, sha, calls, env


def run_lane(lane, *args, **env_overrides):
    repo, _, _, env = lane
    return subprocess.run(
        ["bash", str(CUDA_LANE), *args],
        cwd=repo,
        env={**env, **env_overrides},
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_report_option_is_rejected(lane):
    _, _, calls, _ = lane
    result = run_lane(lane, "--report")
    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert not (calls / "gh").exists()
    assert not (calls / "cargo-cwd").exists()


def test_ref_uses_detached_tree_and_preserves_checkout(lane):
    repo, sha, calls, env = lane
    (repo / "tracked.txt").write_text("local changes\n")
    (repo / "untracked.txt").write_text("untracked\n")
    before = git(repo, "status", "--porcelain", "--untracked-files=all")
    result = run_lane(lane, "--ref", sha)
    assert result.returncode == 0, result.stdout + result.stderr
    assert git(repo, "status", "--porcelain", "--untracked-files=all") == before
    assert (repo / "tracked.txt").read_text() == "local changes\n"
    assert (repo / "untracked.txt").read_text() == "untracked\n"
    worktree = Path((calls / "cargo-cwd").read_text())
    assert worktree != repo
    assert not worktree.exists()
    assert (calls / "cargo-head").read_text() == ""
    assert (calls / "cargo-env").read_text() == "1"
    assert git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1
    assert not (calls / "gh").exists()
    assert (Path(env["XDG_CACHE_HOME"]) / "cuscan-cuda/logs" / f"{sha}.log").exists()
    assert "backend=cuda required=1" in (calls / "uv").read_text()
    assert "maturin develop --release --features tracing" in (calls / "uv").read_text()


def test_cargo_failure_cleans_worktree(lane):
    _, sha, calls, _ = lane
    result = run_lane(lane, "--ref", sha, MOCK_CARGO_EXIT="23")
    assert result.returncode == 23
    assert not (calls / "gh").exists()
    assert not (calls / "uv").exists()
    assert not Path((calls / "cargo-cwd").read_text()).exists()


@pytest.mark.parametrize("sig,exit_code", [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_signal_cleans_worktree(lane, sig, exit_code):
    _, sha, calls, _ = lane
    result = run_lane(lane, "--ref", sha, MOCK_SIGNAL=str(sig.value))
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert not (calls / "gh").exists()
    assert not Path((calls / "cargo-cwd").read_text()).exists()


def test_diagnostic_failure_does_not_gate_tests(lane):
    _, sha, calls, _ = lane
    result = run_lane(lane, "--ref", sha, "--rows", "20000")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Diagnostic analysis failed" in result.stderr + result.stdout
    assert not (calls / "gh").exists()


@pytest.mark.parametrize("args", [("unsupported",), ("dry", "unsupported"), ("dry",)])
def test_act_rejects_unsupported_scenarios(lane, args):
    repo, _, calls, env = lane
    result = subprocess.run(
        ["bash", str(ACT), *args], cwd=repo, env=env, text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert not (calls / "act").exists()
