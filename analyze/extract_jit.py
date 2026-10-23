import collections
import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
manifest = json.loads((ROOT / "captures.json").read_text())
out = ROOT / "jit"
out.mkdir(exist_ok=True)
rows = []
for capture in manifest["captures"]:
    name = capture["env"]["CUBECL_ENVIRONMENT"]
    source = REPO / "target" / "environment" / f"{name}.db"
    profile = json.loads((REPO / capture["profile_json"]).read_text())
    if capture["backend"] == "cpu":
        rows.append(
            {
                "profile_json": capture["profile_json"],
                "backend": "cpu",
                "database_exists": source.exists(),
                "cold_first_fit_seconds": profile["cold_first_fit_seconds"],
                "scope": "LLVM server compile_only uses a process-local HashMap; a fresh process"
                + " starts empty.No persistent CompilationRecord instrumentation is emitted by"
                + " the pinned CPU runtime. First-fit timing includes JIT/backend setup, not"
                + " isolated compilation duration.",
            }
        )
        continue
    if not source.is_file():
        raise RuntimeError(f"isolated CUDA environment store missing: {source}")
    directory = out / name
    directory.mkdir(exist_ok=True)
    database = directory / "environment.sqlite"
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
        records = connection.execute(
            "SELECT key,value,origin FROM entries WHERE namespace='records/v1/compilation' ORDER BY key"  # noqa: E501
        ).fetchall()
        with sqlite3.connect(database) as copy:
            connection.backup(copy)
    cbor = []
    for index, (_, value, _) in enumerate(records):
        path = directory / f"compilation-{index:03}.cbor"
        path.write_bytes(value)
        cbor.append(path)
    result = subprocess.run(
        [str(ROOT / "bin" / "export-jit-records"), *map(str, cbor)], text=True, capture_output=True
    )
    if result.returncode:
        raise RuntimeError(f"native compilation-record decoder failed: {result.stderr}")
    decoded = [json.loads(line) for line in result.stdout.splitlines()]
    records_json = directory / "records.json"
    records_json.write_text(json.dumps(decoded, indent=2) + "\n")
    outcomes = collections.Counter(row["record"]["record"]["outcome"] for row in decoded)
    if not outcomes["Compiled"]:
        raise RuntimeError(f"no actual fresh compilation observed: {name}")

    def seconds(value):
        return value["secs"] + value["nanos"] / 1_000_000_000

    compiled_seconds = sum(
        seconds(row["record"]["record"]["duration"])
        for row in decoded
        if row["record"]["record"]["outcome"] == "Compiled"
    )
    latest_stamp = max(seconds(row["record"]["stamp"]["offset"]) for row in decoded)
    if latest_stamp > profile["cold_first_fit_seconds"] + 0.005:
        raise RuntimeError(f"compilation observed after excluded cold fit: {name}")
    rows.append(
        {
            "profile_json": capture["profile_json"],
            "backend": "cuda",
            "source_database": str(source.relative_to(REPO)),
            "database_copy": str(database.relative_to(REPO)),
            "database_copy_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
            "records_json": str(records_json.relative_to(REPO)),
            "records_json_sha256": hashlib.sha256(records_json.read_bytes()).hexdigest(),
            "record_count": len(decoded),
            "outcomes": dict(outcomes),
            "compiled_duration_seconds": compiled_seconds,
            "latest_record_session_offset_seconds": latest_stamp,
            "cold_first_fit_seconds": profile["cold_first_fit_seconds"],
            "all_records_within_first_fit": True,
            "kernels": [row["record"]["record"]["kernel"] for row in decoded],
            "raw_records": [
                {
                    "key_hex": key.hex(),
                    "origin": origin,
                    "cbor_sha256": hashlib.sha256(value).hexdigest(),
                }
                for key, value, origin in records
            ],
        }
    )
report = {
    "retrospective": True,
    "query": "SELECT key,value,origin FROM entries WHERE namespace='records/v1/compilation' ORDER BY key",  # noqa: E501
    "decoder": "Pinned cubecl_environment::records::Stamped<cubecl_server::compiler::CompilationRecord>, ciborium CBOR decoding; no inferred/fabricated compilation durations.",  # noqa: E501
    "records_are_compilations_not_launches": True,
    "captures": rows,
    "gpu": subprocess.check_output(
        [
            "/usr/lib/wsl/lib/nvidia-smi",
            "--query-gpu=name,driver_version,uuid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip(),
    "limitations": [
        "CompilationRecord duration covers expansion/backend compilation/device load, not pure PTX compiler duration.",  # noqa: E501
        "CUDA driver PTX cache and hardware caches were not cleared.",
        "CPU only has cold first-fit timing with known fresh process-local compilation cache; isolated LLVM compilation duration is unavailable without dependency instrumentation.",  # noqa: E501
    ],
}
(ROOT / "jit-compilation-evidence.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    "Exported",
    sum(row["backend"] == "cuda" for row in rows),
    "CUDA compilation stores;",
    sum(row.get("record_count", 0) for row in rows),
    "actual compilation records",
)
for row in rows:
    if row["backend"] == "cuda":
        print(
            Path(row["profile_json"]).name,
            row["outcomes"],
            "compile seconds",
            row["compiled_duration_seconds"],
        )
