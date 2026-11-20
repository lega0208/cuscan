"""Phase 6 focused runtime trace of the effective fit `init` and `optimizer`.

Never touches the measured checkout's product source. Three steps:

* ``setup`` (outside any campaign ledger): creates a scratch ``git worktree``
  at ``HEAD``, overlays the measured checkout's modified, deleted and untracked
  non-ignored files, refuses to continue unless the scratch product hash equals
  the measured one, applies the generated diagnostic patch (stored as
  ``<out>/trace.patch``), and compiles every shard (Rust test targets and,
  for Python shards, a release extension in a scratch virtual environment).
* ``shards``: prints the per-shard argv the campaign owner wraps in
  ``python -m comparison.umap06.campaign run --ledger ...``.
* ``shard``: runs one shard in the scratch tree with ``CUSCAN_REQUIRE_CUDA=1``
  and ``CUSCAN_TRACE_FIT_CONFIG`` pointing at ``<out>/shards/<id>.jsonl``.
* ``diff``: compares the pin and flip trace outputs per shard.
* ``check-product``: the final search for ``CUSCAN_TRACE_FIT_CONFIG`` in the
  product tree. ``remove``: removes the scratch worktree.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRACE_ENV = "CUSCAN_TRACE_FIT_CONFIG"
PRODUCT_PATHS = ("crates", "python", "Cargo.toml", "Cargo.lock", "pyproject.toml", "uv.lock")
EXCLUDED_DIRS = {"target", "__pycache__", ".pytest_cache", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib", ".dll"}


def _trace_block(site: str, init: str, optimizer: str, rows: str) -> str:
    return f"""    if let Some(trace_path) = std::env::var_os("{TRACE_ENV}") {{
        use std::io::Write as _;
        let line = serde_json::json!({{
            "site": "{site}",
            "init": format!("{{:?}}", {init}),
            "optimizer": format!("{{:?}}", {optimizer}),
            "n_rows": {rows},
            "thread": std::thread::current().name(),
            "pytest": std::env::var("PYTEST_CURRENT_TEST").ok(),
            "pid": std::process::id(),
        }});
        let options = std::fs::OpenOptions::new().create(true).append(true).open(trace_path);
        if let Ok(mut file) = options {{
            let _ = file.write_all(format!("{{line}}\\n").as_bytes());
        }}
    }}
"""


# (file, unique anchor ending at the opening brace of the body, inserted block)
INSTRUMENTATION = (
    (
        "crates/umap-core/Cargo.toml",
        "[dependencies]\n",
        "serde_json = { workspace = true }\n",
    ),
    (
        "crates/umap/Cargo.toml",
        "[dependencies]\n",
        "serde_json = { workspace = true }\n",
    ),
    (
        "crates/umap/src/standard_umap/estimator.rs",
        "    pub fn fit(&self, data: &[Vec<f32>]) -> Result<FittedUMAP> {\n",
        _trace_block("UMAP::fit", "self.config.init", "self.config.optimizer", "data.len()"),
    ),
    (
        "crates/umap-core/src/init.rs",
        "    config: &UMAPConfig,\n    seed: u64,\n) -> Result<Vec<f32>> {\n",
        _trace_block(
            "umap_core::init::initialize", "config.init", "config.optimizer", "data.len()"
        ),
    ),
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def run(
    argv: list[str], cwd: Path, check: bool = True, **kwargs: Any
) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=cwd, check=check, **kwargs)


def git(root: Path, *args: str) -> str:
    return run(["git", *args], root, capture_output=True, text=True).stdout


def product_sha256(root: Path) -> dict[str, str]:
    """Same selection as phase5_probes.product_sha256, evaluated at ``root``."""
    output = run(
        ["git", "ls-files", "-c", "-o", "--exclude-standard", "-z", "--", *PRODUCT_PATHS],
        root,
        capture_output=True,
    ).stdout
    names = sorted({os.fsdecode(name) for name in output.split(b"\0") if name})
    return {
        name: sha256_bytes((root / name).read_bytes())
        for name in names
        if not EXCLUDED_DIRS.intersection(Path(name).parts)
        and Path(name).suffix not in EXCLUDED_SUFFIXES
        and (root / name).is_file()
    }


def tree_digest(hashes: dict[str, str]) -> str:
    return sha256_bytes(json.dumps(hashes, sort_keys=True).encode())


def working_changes(root: Path) -> tuple[list[str], list[str]]:
    """Return (copy, delete) paths from `git status --porcelain`, untracked included."""
    raw = run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        root,
        capture_output=True,
    ).stdout
    entries = [os.fsdecode(item) for item in raw.split(b"\0") if item]
    copy: list[str] = []
    delete: list[str] = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        status, path = entry[:2], entry[3:]
        index += 1
        if "R" in status or "C" in status:
            source = entries[index]
            index += 1
            if "R" in status:
                delete.append(source)
        if (root / path).is_file() or (root / path).is_symlink():
            copy.append(path)
        elif "D" in status or not (root / path).exists():
            delete.append(path)
    return sorted(set(copy)), sorted(set(delete))


# Shards -------------------------------------------------------------------


def package_name(root: Path, crate_dir: Path) -> str:
    return tomllib.loads((root / crate_dir / "Cargo.toml").read_text())["package"]["name"]


def shard_for(root: Path, path: str) -> dict[str, Any]:
    """Map an audited test file to its shard (one per Rust test target or Python file)."""
    parts = Path(path).parts
    if path.endswith(".py"):
        return {"id": "py-" + re.sub(r"[^A-Za-z0-9]+", "_", path), "kind": "python", "path": path}
    if len(parts) >= 4 and parts[0] == "crates" and path.endswith(".rs"):
        crate_dir = Path(*parts[:2])
        package = package_name(root, crate_dir)
        if parts[2] == "tests" and len(parts) == 4:
            target = ["--test", Path(parts[3]).stem]
            name = Path(parts[3]).stem
        elif parts[2] == "src":
            target = ["--lib"]
            name = "lib"
        elif parts[2] == "tests" and len(parts) > 4:
            raise SystemExit(f"{path}: test module file; list its test target instead")
        else:
            raise SystemExit(f"{path}: not a Rust test target (benches/examples are not traced)")
        return {
            "id": f"rs-{package}-{name}",
            "kind": "rust",
            "path": path,
            "package": package,
            "target": target,
        }
    raise SystemExit(f"{path}: unsupported shard file")


def shard_list(root: Path, files: list[str]) -> list[dict[str, Any]]:
    shards: dict[str, dict[str, Any]] = {}
    for path in files:
        shard = shard_for(root, path)
        existing = shards.setdefault(shard["id"], {**shard, "files": []})
        existing["files"].append(path)
    return [shards[key] for key in sorted(shards)]


def read_test_files(args: argparse.Namespace) -> list[str]:
    files = list(args.test_file or [])
    if args.test_files:
        text = args.test_files.read_text(encoding="utf-8")
        if args.test_files.suffix == ".json":
            files.extend(json.loads(text))
        else:
            files.extend(line.strip() for line in text.splitlines() if line.strip())
    if not files:
        raise SystemExit("no test files given; pass the audited test files")
    return sorted(set(files))


def scratch_env(manifest: dict[str, Any]) -> dict[str, str]:
    scratch = Path(manifest["scratch"])
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(scratch / "target")
    env["UV_PROJECT_ENVIRONMENT"] = str(scratch / ".venv")
    env.pop("VIRTUAL_ENV", None)
    # The campaign's package snapshot is uninstrumented; use the scratch install.
    env.pop("PYTHONPATH", None)
    env.pop(TRACE_ENV, None)
    return env


def cargo_test_argv(shard: dict[str, Any], no_run: bool) -> list[str]:
    argv = ["soldr", "cargo", "test", "--release", "-p", shard["package"], *shard["target"]]
    return argv + (["--no-run"] if no_run else [])


def python_test_argv(shard: dict[str, Any]) -> list[str]:
    return [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        shard["path"],
    ]


# Commands -----------------------------------------------------------------


def apply_instrumentation(scratch: Path) -> None:
    for relative, anchor, block in INSTRUMENTATION:
        path = scratch / relative
        text = path.read_text(encoding="utf-8")
        if text.count(anchor) != 1:
            raise SystemExit(f"{relative}: instrumentation anchor not found exactly once")
        path.write_text(text.replace(anchor, anchor + block), encoding="utf-8")


def patch_bytes(scratch: Path) -> bytes:
    files = [relative for relative, _, _ in INSTRUMENTATION]
    return run(["git", "diff", "--no-color", "--", *files], scratch, capture_output=True).stdout


def setup(args: argparse.Namespace) -> None:
    out: Path = args.out.resolve()
    scratch: Path = args.scratch.resolve()
    if out.exists():
        raise SystemExit(f"{out} exists; each trace run needs a fresh output directory")
    if scratch.exists():
        raise SystemExit(f"{scratch} exists; remove the previous scratch worktree first")
    shards = shard_list(ROOT, read_test_files(args))
    measured = product_sha256(ROOT)
    head = git(ROOT, "rev-parse", "HEAD").strip()
    copy, delete = working_changes(ROOT)

    out.mkdir(parents=True)
    run(["git", "worktree", "add", "--detach", str(scratch), head], ROOT)
    for relative in copy:
        target = scratch / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            target.unlink()
        shutil.copy2(ROOT / relative, target, follow_symlinks=False)
    for relative in delete:
        if (scratch / relative).is_file() or (scratch / relative).is_symlink():
            (scratch / relative).unlink()

    scratch_hashes = product_sha256(scratch)
    manifest: dict[str, Any] = {
        "head": head,
        "measured_root": str(ROOT),
        "scratch": str(scratch),
        "measured_product_sha256": tree_digest(measured),
        "scratch_product_sha256": tree_digest(scratch_hashes),
        "overlay": {"copied": copy, "deleted": delete},
        "shards": shards,
    }
    if scratch_hashes != measured:
        differing = sorted(set(measured.items()) ^ set(scratch_hashes.items()))
        manifest["product_mismatch"] = sorted({name for name, _ in differing})
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        raise SystemExit("scratch product hash differs from the measured tree; run not started")

    apply_instrumentation(scratch)
    patch = patch_bytes(scratch)
    (out / "trace.patch").write_bytes(patch)
    manifest["patch_sha256"] = sha256_bytes(patch)
    if (ROOT / "crates").exists() and run(
        ["git", "grep", "-q", TRACE_ENV, "--", "crates", "python"], ROOT, check=False
    ).returncode == 0:
        raise SystemExit(f"{TRACE_ENV} found in the measured product tree")

    env = scratch_env(manifest)
    builds = []
    for shard in shards:
        if shard["kind"] == "rust":
            argv = cargo_test_argv(shard, no_run=True)
            run(argv, scratch, env=env)
            builds.append(argv)
    if any(shard["kind"] == "python" for shard in shards):
        for argv in (
            ["uv", "sync", "--frozen"] + [f"--extra={extra}" for extra in args.uv_extra],
            ["uv", "run", "--no-sync", "maturin", "develop", "--release", "--features", "tracing"],
        ):
            run(argv, scratch, env=env)
            builds.append(argv)
    manifest["builds"] = builds
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(out / "manifest.json"), "shards": [s["id"] for s in shards]}))


def load_manifest(out: Path) -> dict[str, Any]:
    manifest = json.loads((out / "manifest.json").read_text())
    if "patch_sha256" not in manifest or "builds" not in manifest:
        raise SystemExit(f"{out}: setup did not complete")
    return manifest


def shards_command(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    manifest = load_manifest(out)
    print(
        json.dumps(
            [
                {
                    "id": shard["id"],
                    "argv": [
                        sys.executable,
                        "-m",
                        "comparison.umap06.phase6_trace",
                        "shard",
                        "--out",
                        str(out),
                        "--shard",
                        shard["id"],
                    ],
                }
                for shard in manifest["shards"]
            ],
            indent=2,
        )
    )


def shard_command(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    manifest = load_manifest(out)
    scratch = Path(manifest["scratch"])
    shard = next((s for s in manifest["shards"] if s["id"] == args.shard), None)
    if shard is None:
        raise SystemExit(f"unknown shard {args.shard}")
    if sha256_bytes(patch_bytes(scratch)) != manifest["patch_sha256"]:
        raise SystemExit("scratch instrumentation differs from the recorded patch")
    shards_dir = out / "shards"
    shards_dir.mkdir(exist_ok=True)
    trace = shards_dir / f"{shard['id']}.jsonl"
    result_path = shards_dir / f"{shard['id']}.result.json"
    if trace.exists() or result_path.exists():
        raise SystemExit(f"shard {shard['id']} already ran in {out}")
    trace.touch()
    env = scratch_env(manifest)
    env[TRACE_ENV] = str(trace)
    env["CUSCAN_REQUIRE_CUDA"] = "1"
    argv = (
        cargo_test_argv(shard, no_run=False) if shard["kind"] == "rust" else python_test_argv(shard)
    )
    returncode = subprocess.run(argv, cwd=scratch, env=env, check=False).returncode
    result_path.write_text(
        json.dumps({"id": shard["id"], "argv": argv, "returncode": returncode}, indent=2) + "\n"
    )
    sys.exit(returncode)


def test_name(record: dict[str, Any]) -> str:
    if record.get("pytest"):
        return re.sub(r" \((setup|call|teardown)\)$", "", record["pytest"])
    return record.get("thread") or "<unnamed>"


def load_run(out: Path) -> dict[str, collections.Counter]:
    """Shard id -> multiset of (site, test, init, optimizer) for completed shards."""
    completed: dict[str, collections.Counter] = {}
    for result_path in sorted((out / "shards").glob("*.result.json")):
        result = json.loads(result_path.read_text())
        if result["returncode"] != 0:
            continue
        trace = out / "shards" / f"{result['id']}.jsonl"
        lines = trace.read_text(encoding="utf-8").splitlines()
        completed[result["id"]] = collections.Counter(
            (rec["site"], test_name(rec), rec["init"], rec["optimizer"])
            for rec in map(json.loads, filter(None, lines))
        )
    return completed


def diff_command(args: argparse.Namespace) -> None:
    allow: set[str] = set()
    if args.allowlist:
        allow = {line.strip() for line in args.allowlist.read_text().splitlines() if line.strip()}
    pin_manifest, flip_manifest = (
        load_manifest(args.pin.resolve()),
        load_manifest(args.flip.resolve()),
    )
    planned = {s["id"] for s in pin_manifest["shards"]} | {s["id"] for s in flip_manifest["shards"]}
    pin, flip = load_run(args.pin.resolve()), load_run(args.flip.resolve())

    def allowed(test: str) -> bool:
        return any(test == entry or test.endswith("::" + entry) for entry in allow)

    shards: dict[str, Any] = {}
    for shard in sorted(planned):
        if shard not in pin or shard not in flip:
            shards[shard] = {"status": "unmeasured", "pin": shard in pin, "flip": shard in flip}
            continue
        differences = []
        for key in sorted(set(pin[shard]) | set(flip[shard])):
            if pin[shard][key] != flip[shard][key]:
                site, test, init, optimizer = key
                differences.append(
                    {
                        "site": site,
                        "test": test,
                        "init": init,
                        "optimizer": optimizer,
                        "pin": pin[shard][key],
                        "flip": flip[shard][key],
                        "allowlisted": allowed(test),
                    }
                )
        unpinned = [d for d in differences if not d["allowlisted"]]
        shards[shard] = {
            "status": "unpinned" if unpinned else "identical",
            "records": {"pin": sum(pin[shard].values()), "flip": sum(flip[shard].values())},
            "differences": differences,
        }
    report = {
        "pin": {
            k: pin_manifest.get(k) for k in ("head", "measured_product_sha256", "patch_sha256")
        },
        "flip": {
            k: flip_manifest.get(k) for k in ("head", "measured_product_sha256", "patch_sha256")
        },
        "allowlist": sorted(allow),
        "shards": shards,
        "unpinned": sorted(s for s, v in shards.items() if v["status"] == "unpinned"),
        "unmeasured": sorted(s for s, v in shards.items() if v["status"] == "unmeasured"),
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text, end="")
    sys.exit(1 if report["unpinned"] else 0)


def check_product(_: argparse.Namespace) -> None:
    found = run(
        ["git", "grep", "-n", TRACE_ENV, "--", *PRODUCT_PATHS],
        ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    untracked = run(
        ["git", "ls-files", "-o", "--exclude-standard", "--", *PRODUCT_PATHS],
        ROOT,
        capture_output=True,
        text=True,
    ).stdout.split()
    hits = [line for line in found.stdout.splitlines()]
    hits += [
        p
        for p in untracked
        if (ROOT / p).is_file() and TRACE_ENV.encode() in (ROOT / p).read_bytes()
    ]
    print(json.dumps({"variable": TRACE_ENV, "hits": hits}))
    sys.exit(1 if hits else 0)


def remove(args: argparse.Namespace) -> None:
    manifest = json.loads((args.out.resolve() / "manifest.json").read_text())
    run(["git", "worktree", "remove", "--force", manifest["scratch"]], ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("setup")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--test-file", action="append", help="audited test file (repeatable)")
    p.add_argument(
        "--test-files", type=Path, help="JSON list or newline-separated audited test files"
    )
    p.add_argument(
        "--uv-extra", action="append", default=[], help="uv sync extra for Python shards"
    )
    p.set_defaults(func=setup)
    p = commands.add_parser("shards")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=shards_command)
    p = commands.add_parser("shard")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard", required=True)
    p.set_defaults(func=shard_command)
    p = commands.add_parser("diff")
    p.add_argument("--pin", type=Path, required=True)
    p.add_argument("--flip", type=Path, required=True)
    p.add_argument("--allowlist", type=Path, help="default-subject test names, one per line")
    p.add_argument("--report", type=Path)
    p.set_defaults(func=diff_command)
    commands.add_parser("check-product").set_defaults(func=check_product)
    p = commands.add_parser("remove")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=remove)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
