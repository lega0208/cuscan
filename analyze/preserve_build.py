import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("variant")
parser.add_argument("--preserve-only", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parent
rows = [
    json.loads(line) for line in (root / f"{args.variant}-build.jsonl").read_text().splitlines()
]
packages = {}
for name in ("fast_cubecl_common", "fast_hdbscan", "profile_hdbscan_stages"):
    row = next(
        r for r in rows if r.get("reason") == "compiler-artifact" and r["target"]["name"] == name
    )
    if row["fresh"]:
        raise RuntimeError(f"Rejecting cross-worktree reused objective artifact: {name}")
    packages[name] = {
        "source": row["target"]["src_path"],
        "fresh": row["fresh"],
        "package_id": row["package_id"],
    }
example = next(
    r
    for r in rows
    if r.get("reason") == "compiler-artifact"
    and r.get("executable")
    and r["target"]["name"] == "profile_hdbscan_stages"
)
profiler = root / "bin" / args.variant
semantics = root / "bin" / f"{args.variant}-semantics"
if not args.preserve_only:
    shutil.copy2(example["executable"], profiler)
    library = next(
        r
        for r in rows
        if r.get("reason") == "compiler-artifact" and r["target"]["name"] == "fast_hdbscan"
    )
    rlib = next(p for p in library["filenames"] if p.endswith(".rlib"))
    dependency_dirs = sorted(
        {
            str(Path(p).parent)
            for r in rows
            if r.get("reason") == "compiler-artifact"
            for p in r["filenames"]
        }
    )
    native_dirs = sorted(
        {p for r in rows if r.get("reason") == "build-script-executed" for p in r["linked_paths"]}
    )
    command = [
        "rustc",
        "--edition=2024",
        "-O",
        str(root / "capture_semantics.rs"),
        "--extern",
        f"fast_hdbscan={rlib}",
    ]
    for directory in dependency_dirs:
        command.extend(["-L", f"dependency={directory}"])
    for directory in native_dirs:
        command.extend(["-L", directory])
    command.extend(["-o", str(semantics)])
    (root / f"{args.variant}-semantic-build-command.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )
    subprocess.run(command, check=True)
worktree = Path(packages["fast_hdbscan"]["source"]).parents[3]
patch = subprocess.check_output(["git", "-C", str(worktree), "diff", "--binary"])
(root / f"{args.variant}-source.patch").write_bytes(patch)
paths = [
    "Cargo.toml",
    "Cargo.lock",
    "crates/fast-cubecl-common/src/kernels/neighbors.rs",
    "crates/fast-cubecl-common/src/neighbors.rs",
    "crates/fast-hdbscan/src/estimator.rs",
    "crates/fast-hdbscan/tests/test_estimator.rs",
    "crates/fast-hdbscan/tests/test_kdtree_search.rs",
    "crates/fast-hdbscan/examples/profile_hdbscan_stages.rs",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


record = {
    "variant": args.variant,
    "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "worktree": str(worktree),
    "head": subprocess.check_output(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True
    ).strip(),
    "status": subprocess.check_output(
        ["git", "-C", str(worktree), "status", "--short", "--untracked-files=no"], text=True
    ),
    "packages": packages,
    "source_sha256": {path: digest(worktree / path) for path in paths},
    "patch_sha256": hashlib.sha256(patch).hexdigest(),
    "binary_sha256": {str(path): digest(path) for path in (profiler, semantics)},
    "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
    "cargo": subprocess.check_output(["cargo", "-V"], text=True),
    "retrospective": True,
    "note": "Original-source reconstruction after optimization, not an authentic pre-change"
    + " capture. Objective packages forcibly cleaned/rebuilt between worktrees; initial"
    + " shared-target fresh reuse was rejected, not used as evidence.",
}
(root / f"{args.variant}-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
print(args.variant, record["head"], record["binary_sha256"])
