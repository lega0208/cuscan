import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_campaign_deadline_reaps_nested_worker(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    destination = tmp_path / "child"
    pidfile = tmp_path / "worker.pid"
    worker = (
        "import os, pathlib, time; "
        f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    nested = (
        "import sys; from comparison.umap06.run import child; "
        f"child([sys.executable, '-c', {worker!r}], {str(destination)!r}, None)"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "comparison.umap06.campaign",
            "run",
            "--ledger",
            str(ledger),
            "--budget-seconds",
            "10",
            "--reserve-seconds",
            "2",
            "--",
            sys.executable,
            "-c",
            nested,
        ],
        cwd=Path(__file__).resolve().parents[2],
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert json.loads((destination / "child.json").read_text())["timed_out"]
    assert json.loads(ledger.read_text().splitlines()[-1])["outcome"] == "deadline"
