import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location(
    "ann_perf_capture", Path(__file__).resolve().parents[2] / "scripts/ann_perf_capture.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--timeout", "0"),
        ("--timeout", "-1"),
        ("--timeout", "nan"),
        ("--timeout", "inf"),
        ("--repeats", "0"),
        ("--repeats", "-1"),
        ("--slots", "0"),
        ("--slots", "-1"),
    ],
)
def test_invalid_arguments_do_not_create_output(tmp_path, monkeypatch, option, value):
    output = tmp_path / "capture"
    monkeypatch.setattr(
        sys, "argv", ["ann_perf_capture.py", "cpu", "10", "4", str(output), option, value]
    )
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("slots", [None, 123])
def test_capture_records_selected_executable(tmp_path, monkeypatch, slots):
    executable = tmp_path / "predecessor"
    contents = b"preserved executable bytes"
    executable.write_bytes(contents)
    os.utime(executable, ns=(1_600_000_000_000_000_000, 1_600_000_000_123_456_789))
    output = tmp_path / "capture"
    argv = [
        "ann_perf_capture.py",
        "cpu",
        "10",
        "4",
        str(output),
        "--executable",
        str(executable),
        "--repeats",
        "1",
    ]
    if slots is not None:
        argv.extend(["--slots", str(slots)])
    monkeypatch.setattr(sys, "argv", argv)
    process = Mock()
    process.poll.return_value = 0
    process.wait.return_value = 0
    popen = Mock(return_value=process)
    monkeypatch.setattr(module.subprocess, "Popen", popen)

    module.main()

    capture = json.loads((output / "capture.json").read_text())
    command = [str(executable.resolve()), "cpu", "10", "4", "1", str(slots or 4194304)]
    assert popen.call_args.args[0] == command
    assert capture["command"] == command
    assert capture["executable_sha256"] == hashlib.sha256(contents).hexdigest()
    assert capture["executable_mtime_ns"] == executable.stat().st_mtime_ns


def test_default_probe_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    executable = tmp_path / "target/release/examples/ann_tile_probe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"default probe")
    output = tmp_path / "capture"
    monkeypatch.setattr(sys, "argv", ["ann_perf_capture.py", "cpu", "10", "4", str(output)])
    process = Mock()
    process.poll.return_value = 0
    process.wait.return_value = 0
    popen = Mock(return_value=process)
    monkeypatch.setattr(module.subprocess, "Popen", popen)

    module.main()

    assert popen.call_args.args[0] == [str(executable), "cpu", "10", "4", "5", "4194304"]


def test_budgeted_probe_command(tmp_path, monkeypatch):
    executable = tmp_path / "probe"
    executable.write_bytes(b"probe")
    output = tmp_path / "capture"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ann_perf_capture.py",
            "cpu",
            "10",
            "4",
            str(output),
            "--executable",
            str(executable),
            "--budgeted",
            "--slots",
            "123",
        ],
    )
    process = Mock()
    process.poll.return_value = 0
    process.wait.return_value = 0
    popen = Mock(return_value=process)
    monkeypatch.setattr(module.subprocess, "Popen", popen)

    module.main()

    command = [str(executable.resolve()), "cpu", "10", "4", "5", "budget"]
    assert popen.call_args.args[0] == command
    assert json.loads((output / "capture.json").read_text())["command"] == command


def test_budgeted_still_rejects_invalid_slots(tmp_path, monkeypatch):
    output = tmp_path / "capture"
    monkeypatch.setattr(
        sys,
        "argv",
        ["ann_perf_capture.py", "cpu", "10", "4", str(output), "--budgeted", "--slots", "0"],
    )
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert not output.exists()
