"""Exercise inline precredential guard code without actions or remote credentials."""

import json
import os
import subprocess
import sys
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
    ],
)
def test_release_please_guard_before_credentials(tmp_path, act, event, enabled):
    text = (ROOT / ".github/workflows/release-please.yml").read_text()
    guard = text.split("python3 - <<'PY'\n", 1)[1].split("\n          PY", 1)[0]
    code = "\n".join(line.removeprefix("          ") for line in guard.splitlines())
    fixture = tmp_path / "event.json"
    fixture.write_text(json.dumps(event))
    output = tmp_path / "output"
    subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "ACT": act, "EVENT_PATH": str(fixture), "GITHUB_OUTPUT": str(output)},
        check=True,
    )
    assert output.read_text() == f"enabled={str(enabled).lower()}\n"
