"""Screen preserved layouts across epoch budgets without replacing the 20-epoch acceptance suite."""

import argparse
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from compare_umap_spans import _python_worker, make_clustered_input, nonself, tie_aware_recall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", action="append", required=True, help="NAME=PATH; include serial"
    )
    parser.add_argument("--epochs", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--deadline-seconds", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 2 or any(epoch < 1 for epoch in args.epochs) or args.deadline_seconds <= 0:
        parser.error("Require >=2 runs, positive epochs and deadline")
    binaries = {}
    for item in args.binary:
        name, separator, path = item.partition("=")
        if not name or not separator or name in binaries or not Path(path).is_file():
            parser.error("Each binary needs a unique NAME and existing PATH")
        binaries[name] = str(Path(path).resolve())
    if "serial" not in binaries:
        parser.error("Include the original serial reference")

    import numpy as np
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import pairwise_distances

    started = time.monotonic()
    output = {
        "complete": False,
        "configuration": {
            "rows": 5000,
            "features": 32,
            "components": 2,
            "neighbors": 15,
            "init": "random",
            "metric": "euclidean",
            "epochs": args.epochs,
            "runs": args.runs,
            "warmups": 1,
            "deadline_seconds": args.deadline_seconds,
            "cuda_seeds": "42+run",
            "python": "unseeded parallel n_jobs=-1",
            "purpose": "screening; not a replacement for five-run20epoch acceptance",
            "binaries": binaries,
        },
        "points": [],
    }

    def run(command):
        remaining = args.deadline_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Combined screen deadline reached")
        return subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=remaining
        )

    try:
        with tempfile.TemporaryDirectory(prefix="layout-epochs-") as temporary:
            directory = Path(temporary)
            input_path = directory / "input.f32le"
            make_clustered_input(5000, 32, input_path)
            matrix = np.fromfile(input_path, dtype="<f4").reshape(5000, 32)
            distances = pairwise_distances(matrix)
            masked = distances.copy()
            np.fill_diagonal(masked, np.inf)
            exact_kth = np.partition(masked, 13, axis=1)[:, 13]
            del masked
            for epochs in args.epochs:
                for mode in ("exact", "approximate"):
                    point = {"epochs": epochs, "mode": mode, "layouts": {}}
                    output["points"].append(point)
                    py_dir = directory / f"python-{epochs}-{mode}"
                    py_dir.mkdir()
                    # Existing worker keeps exact neighbors inside fit and forces approximate.
                    worker = run(
                        [
                            "uv",
                            "run",
                            "--no-sync",
                            "python",
                            str(Path(__file__).resolve()),
                            "--python-worker",
                            str(input_path),
                            "5000",
                            "32",
                            str(epochs),
                            mode,
                            str(args.runs),
                            str(py_dir),
                        ]
                    )
                    point["python"] = json.loads(worker.stdout)
                    python_trust = [
                        float(
                            trustworthiness(
                                distances,
                                np.load(py_dir / f"embedding_{index}.npy"),
                                n_neighbors=15,
                                metric="precomputed",
                            )
                        )
                        for index in range(args.runs)
                    ]
                    point["python"]["trustworthiness"] = python_trust
                    if mode == "approximate":
                        py_recalls = [
                            tie_aware_recall(
                                matrix, nonself(np.load(py_dir / f"knn_{index}.npy"), 14), exact_kth
                            )
                            for index in range(args.runs)
                        ]
                        point["python"]["fit_recall_at14"] = py_recalls
                        point["python"]["every_run_recall_ge0.90"] = all(
                            value >= 0.9 for value in py_recalls
                        )
                    python_mean = statistics.mean(python_trust)
                    for name, binary in binaries.items():
                        out_dir = directory / f"{name}-{epochs}-{mode}"
                        measured = run(
                            [
                                binary,
                                str(input_path),
                                "5000",
                                "32",
                                str(epochs),
                                str(out_dir),
                                "--search",
                                mode,
                                "--warmups",
                                "1",
                                "--runs",
                                str(args.runs),
                            ]
                        )
                        result = json.loads(measured.stdout)
                        point["layouts"][name] = result
                        scores = []
                        recalls = []
                        for index in range(args.runs):
                            if time.monotonic() - started >= args.deadline_seconds:
                                raise TimeoutError("Combined screen deadline reached")
                            embedding = np.fromfile(
                                out_dir / f"embedding_{index}.f32le", dtype="<f4"
                            ).reshape(5000, 2)
                            if not np.isfinite(embedding).all():
                                raise ValueError(f"{name}: nonfinite embedding")
                            scores.append(
                                float(
                                    trustworthiness(
                                        distances, embedding, n_neighbors=15, metric="precomputed"
                                    )
                                )
                            )
                            if mode == "approximate":
                                width = result["fit_graph_width"]
                                if width != 14:
                                    raise ValueError("Expected actual-fit nonself graph width14")
                                graph = np.fromfile(
                                    out_dir / f"graph_{index}.i32le", dtype="<i4"
                                ).reshape(5000, width)
                                recalls.append(tie_aware_recall(matrix, graph, exact_kth))
                        result["trustworthiness"] = scores
                        result["trust_mean"] = statistics.mean(scores)
                        result["trust_sample_std"] = statistics.stdev(scores)
                        result["median_fit_seconds"] = statistics.median(result["fit_seconds"])
                        result["each_within_0.02_of_python_mean"] = all(
                            abs(score - python_mean) <= 0.02 for score in scores
                        )
                        if recalls:
                            result["fit_recall_at14"] = recalls
                            result["every_run_recall_ge0.90"] = all(
                                value >= 0.9 for value in recalls
                            )
                    serial = point["layouts"]["serial"]
                    for result in point["layouts"].values():
                        result["speedup_vs_serial"] = (
                            serial["median_fit_seconds"] / result["median_fit_seconds"]
                        )
                        result["trust_delta_vs_serial"] = (
                            result["trust_mean"] - serial["trust_mean"]
                        )
            if time.monotonic() - started >= args.deadline_seconds:
                raise TimeoutError("Combined screen deadline reached")
            output["complete"] = True
    except (subprocess.SubprocessError, TimeoutError, ValueError) as error:
        output["incomplete_reason"] = str(error)
        raise
    finally:
        output["elapsed_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--python-worker":
        input_path, rows, features, epochs, mode, runs, out = sys.argv[2:]
        print(
            json.dumps(
                _python_worker(
                    Path(input_path),
                    int(rows),
                    int(features),
                    int(epochs),
                    mode,
                    int(runs),
                    Path(out),
                )
            )
        )
    else:
        main()
