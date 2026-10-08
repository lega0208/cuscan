"""Alternate matched CUDA fit binaries; include per-fit setup, exclude warmups."""

import argparse
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--deadline-seconds", type=float, default=600)
    args = parser.parse_args()
    if args.pairs < 5 or args.deadline_seconds <= 0:
        parser.error("Require at least five pairs and a positive deadline")
    started = time.monotonic()
    result = {"complete": False, "configuration": vars(args).copy(), "modes": {}}
    result["configuration"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result["configuration"].items()
    }
    result["configuration"]["workload"] = "5000x32/20; seed42; warmup1000; 15neighbors"
    try:
        with tempfile.TemporaryDirectory(prefix="umap-pairs-") as directory:
            for mode in ("exact", "approximate"):
                samples = {"control": [], "candidate": [], "order": []}
                result["modes"][mode] = samples
                for pair in range(args.pairs):
                    order = ("control", "candidate") if pair % 2 == 0 else ("candidate", "control")
                    samples["order"].append(list(order))
                    for arm in order:
                        remaining = args.deadline_seconds - (time.monotonic() - started)
                        if remaining <= 0:
                            raise TimeoutError("Combined benchmark deadline reached")
                        command = [
                            str(getattr(args, arm).resolve()),
                            str(args.input.resolve()),
                            "5000",
                            "32",
                            "20",
                            directory,
                            "--search",
                            mode,
                            "--warmups",
                            "1",
                            "--runs",
                            "1",
                        ]
                        run = subprocess.run(
                            command, capture_output=True, text=True, check=True, timeout=remaining
                        )
                        value = json.loads(run.stdout)["fit_seconds"][0]
                        if not 0 < value < float("inf"):
                            raise ValueError("Invalid fit duration")
                        samples[arm].append(value)
                differences = [
                    control - candidate
                    for control, candidate in zip(
                        samples["control"], samples["candidate"], strict=True
                    )
                ]
                samples["paired_savings_seconds"] = differences
                samples["control_median"] = statistics.median(samples["control"])
                samples["candidate_median"] = statistics.median(samples["candidate"])
                samples["paired_median_saving"] = statistics.median(differences)
                samples["paired_sample_std"] = statistics.stdev(differences)
                # Positive savings must exceed observed pair-to-pair variation, not just noise.
                samples["credible_gain"] = (
                    samples["candidate_median"] < samples["control_median"]
                    and samples["paired_median_saving"] > samples["paired_sample_std"]
                )
        result["complete"] = True
        result["retain_candidate"] = all(
            samples["credible_gain"] for samples in result["modes"].values()
        )
    except (subprocess.SubprocessError, TimeoutError, ValueError) as error:
        result["error"] = str(error)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
