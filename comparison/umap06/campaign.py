"""Enforce a shared wall-clock budget for serial measurement commands."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def timestamp(value):
    return dt.datetime.fromtimestamp(value, dt.UTC).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run")
    run.add_argument("--ledger", type=Path, required=True)
    run.add_argument("--budget-seconds", type=float)
    run.add_argument("--reserve-seconds", type=float)
    run.add_argument("command", nargs=argparse.REMAINDER)
    report = commands.add_parser("report")
    report.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("UMAP 0.6 campaign commands require POSIX file locking and process groups")
    import fcntl

    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.open("a+", encoding="utf-8") as ledger:
        # Serialize invocations, including execution, to preserve one campaign timeline.
        fcntl.flock(ledger, fcntl.LOCK_EX)
        ledger.seek(0)
        entries = [json.loads(line) for line in ledger if line.strip()]
        if args.action == "report":
            if not entries:
                parser.error("ledger is empty")
            manifest = {
                "campaign": entries[0],
                "commands": entries[1:],
                "unrun_or_killed": [
                    entry for entry in entries[1:] if entry["outcome"] == "deadline"
                ],
            }
            output = args.ledger.with_suffix(args.ledger.suffix + ".manifest.json")
            with output.open("x", encoding="utf-8") as stream:
                json.dump(manifest, stream, indent=2)
                stream.write("\n")
            print(output)
            return

        command = args.command
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            parser.error("a command is required after --")
        budget = args.budget_seconds if args.budget_seconds is not None else 1200
        reserve = args.reserve_seconds if args.reserve_seconds is not None else 30
        if budget <= 0 or reserve < 0:
            parser.error("budget must be positive and reserve nonnegative")
        if not entries:
            start = time.time()
            entries.append(
                {
                    "start": timestamp(start),
                    "deadline": start + budget,
                    "budget_seconds": budget,
                    "reserve_seconds": reserve,
                }
            )
            ledger.write(json.dumps(entries[0]) + "\n")
            ledger.flush()
        campaign = entries[0]
        if (args.budget_seconds is not None and budget != campaign["budget_seconds"]) or (
            args.reserve_seconds is not None and reserve != campaign["reserve_seconds"]
        ):
            parser.error("budget and reserve must match the existing ledger")
        started = time.time()
        remaining = campaign["deadline"] - campaign["reserve_seconds"] - started
        code = None
        outcome = "deadline"
        if remaining > 0:
            env = os.environ.copy()
            env["UMAP06_CAMPAIGN_DEADLINE"] = str(campaign["deadline"])
            try:
                process = subprocess.Popen(command, env=env, start_new_session=True)
                try:
                    code = process.wait(timeout=remaining)
                    outcome = "success" if code == 0 else "failed"
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    code = process.returncode
            except OSError as exc:
                print(f"Unable to launch command: {exc}", file=sys.stderr)
                outcome = "launch_error"
        entry = {
            "argv": command,
            "start": timestamp(started),
            "end": timestamp(time.time()),
            "exit_status": code,
            "outcome": outcome,
        }
        ledger.write(json.dumps(entry) + "\n")
        ledger.flush()
        if outcome != "success":
            raise SystemExit(code if code is not None and code > 0 else 1)


if __name__ == "__main__":
    main()
