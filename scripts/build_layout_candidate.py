"""Build an archived layout in a new, retained worktree without editing the main checkout."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT / "artifacts" / "layout-experiments"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", help="candidate key in the experiment manifest")
    parser.add_argument(
        "--workspace", type=Path, required=True, help="new directory, never overwritten"
    )
    parser.add_argument("--target-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ARCHIVES / "manifest.json").read_text())
    if args.candidate not in manifest["candidates"]:
        parser.error("Unknown candidate")
    if args.workspace.exists():
        parser.error("Workspace already exists; choose a new path to preserve its work")
    candidate = manifest["candidates"][args.candidate]
    workspace = args.workspace.resolve()
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(workspace), manifest["base_commit"]],
        cwd=ROOT,
        check=True,
    )
    patch = ROOT / candidate["interface_patch"]
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=workspace, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=workspace, check=True)
    shutil.copyfile(ARCHIVES / candidate["kernel"], workspace / manifest["kernel_target"])
    if candidate["interface"] == "legacy_csr":
        # The interface patch includes the diagnostic accessor, but not the CLI example.
        shutil.copyfile(
            ROOT / "crates/umap/examples/profile_standard_cuda.rs",
            workspace / "examples/profile_standard_cuda.rs",
        )
    if candidate.get("estimator"):
        shutil.copyfile(
            ARCHIVES / candidate["estimator"], workspace / "src/standard_umap/estimator.rs"
        )
    target = args.target_dir.resolve()
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--features",
            "tracing",
            "--example",
            "profile_standard_cuda",
            "--target-dir",
            str(target),
        ],
        cwd=workspace,
        check=True,
    )
    binary = workspace / "target" / "profile_standard_cuda"
    binary.parent.mkdir(exist_ok=True)
    shutil.copy2(target / "release/examples/profile_standard_cuda", binary)
    receipt = {
        "candidate": args.candidate,
        "base_commit": manifest["base_commit"],
        "interface": candidate["interface"],
        "workspace": str(workspace),
        "binary": str(binary),
    }
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
