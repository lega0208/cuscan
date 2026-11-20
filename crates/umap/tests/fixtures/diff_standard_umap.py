#!/usr/bin/env python3
"""Compare frozen UMAP JSON fixtures using the Phase 0 A4 acceptance rules."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np

FILES = ("standard_umap_reference.json", "ann_campaign_reference.json")
ARRAY_FILES = ("standard_umap_quality.npz", "ann_campaign_quality.npz")
GRAPH_FLOATS = {"rho", "sigma", "directed", "symmetric", "epochs_per_sample"}
PROBE_FLOATS = {
    "head_initial",
    "tail_initial",
    "final_head",
    "final_tail",
    "head_embedding",
    "tail_embedding",
    "alpha_used",
    "alpha_after_epoch",
}
MISSING = object()


def classification(path: tuple[str | int, ...]) -> str:
    root = path[0] if path else ""
    if root in {"reference", "source_basis", "quality"} or path == ("data", "input_file"):
        return "diagnostic"
    fields = {part for part in path if isinstance(part, str)}
    if root == "graph_cases" and fields & GRAPH_FLOATS:
        return "graph_tolerance"
    if root == "optimizer_probes":
        if "epochs_per_sample" in fields:
            return "graph_tolerance"
        if fields & PROBE_FLOATS:
            return "layout_tolerance"
    return "exact"


def number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def numeric_array(value: object) -> bool:
    return isinstance(value, list) and all(number(v) or numeric_array(v) for v in value)


def shape(value: object) -> object:
    if isinstance(value, list):
        return [shape(item) for item in value]
    return None


def flatten(value: object):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    else:
        yield value


def ulp_key(value: float) -> int:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    # Fold signed zero together, and order the two halves monotonically.
    magnitude = bits & 0x7FFFFFFFFFFFFFFF
    return 0x8000000000000000 - magnitude if bits >> 63 else 0x8000000000000000 + magnitude


def path_string(path: tuple[str | int, ...]) -> str:
    return "$" + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in path)


def compare(before: object, after: object) -> tuple[list[dict[str, object]], int]:
    records: list[dict[str, object]] = []
    violations = 0

    def record(path: tuple[str | int, ...], old: object, new: object) -> None:
        nonlocal violations
        rule = classification(path)
        missing = old is MISSING or new is MISSING
        same_shape = not missing and shape(old) == shape(new)
        numeric = not missing and (
            (number(old) and number(new)) or (numeric_array(old) and numeric_array(new))
        )
        equal = not missing and same_shape and old == new
        max_abs = None
        max_ulp = None
        nonfinite = False
        within = False
        if numeric and same_shape:
            pairs = list(zip(flatten(old), flatten(new), strict=True))
            within = True
            differences = []
            ulps = []
            for left, right in pairs:
                try:
                    a, b = float(left), float(right)
                except OverflowError:
                    nonfinite = True
                    within = False
                    continue
                if not math.isfinite(a) or not math.isfinite(b):
                    nonfinite = True
                    within = False
                    continue
                difference = abs(a - b)
                if not math.isfinite(difference):
                    nonfinite = True
                    within = False
                    continue
                differences.append(difference)
                ulps.append(abs(ulp_key(a) - ulp_key(b)))
                tolerance = 1e-5 + 1e-5 * abs(a) if rule == "graph_tolerance" else 1e-5
                if difference > tolerance:
                    within = False
            if not nonfinite:
                max_abs = max(differences, default=0.0)
                max_ulp = max(ulps, default=0)
        passed = (equal and not nonfinite) if rule == "exact" else within
        if rule == "diagnostic":
            passed = True
        if not passed:
            violations += 1
        records.append(
            {
                "path": path_string(path),
                "classification": rule,
                "exact_equal": equal and not nonfinite,
                "max_abs_difference": max_abs,
                "max_ulp_distance": max_ulp,
                "nonfinite_or_unrepresentable": nonfinite,
                "before_missing": old is MISSING,
                "after_missing": new is MISSING,
                "same_shape": same_shape,
                "passed": passed,
            }
        )

    def walk(path: tuple[str | int, ...], old: object, new: object) -> None:
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(old.keys() | new.keys()):
                walk((*path, key), old.get(key, MISSING), new.get(key, MISSING))
        elif (
            isinstance(old, list)
            and isinstance(new, list)
            and not (numeric_array(old) and numeric_array(new))
        ):
            if len(old) != len(new):
                record(path, old, new)
            for index in range(max(len(old), len(new))):
                walk(
                    (*path, index),
                    old[index] if index < len(old) else MISSING,
                    new[index] if index < len(new) else MISSING,
                )
        else:
            record(path, old, new)

    walk((), before, after)
    return records, violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    total = 0
    for name in FILES:
        with (args.before / name).open(encoding="utf-8") as stream:
            before = json.load(stream)
        with (args.after / name).open(encoding="utf-8") as stream:
            after = json.load(stream)
        records, violations = compare(before, after)
        results[name] = {"violations": violations, "fields": records}
        total += violations
    for name in ARRAY_FILES:
        with (
            np.load(args.before / name, allow_pickle=False) as before,
            np.load(args.after / name, allow_pickle=False) as after,
        ):
            fields = []
            violations = 0
            for key in sorted(set(before.files) | set(after.files)):
                if key not in before or key not in after:
                    fields.append({"path": key, "passed": False, "reason": "missing array"})
                    violations += 1
                    continue
                old, new = before[key], after[key]
                records, _ = compare(
                    {"quality": {key: old.tolist()}}, {"quality": {key: new.tolist()}}
                )
                finite = bool(np.isfinite(old).all() and np.isfinite(new).all())
                passed = old.shape == new.shape and old.dtype == new.dtype and finite
                fields.append(
                    {
                        "path": key,
                        "before_shape": list(old.shape),
                        "after_shape": list(new.shape),
                        "before_dtype": str(old.dtype),
                        "after_dtype": str(new.dtype),
                        "finite": finite,
                        "passed": passed,
                        "fields": records,
                    }
                )
                violations += not passed
            results[name] = {"violations": violations, "arrays": fields}
            total += violations
    report = {
        "schema": "umap-phase0-fixture-diff-v1",
        "before": str(args.before),
        "after": str(args.after),
        "ulp_format": "IEEE 754 binary64, signed zeros identified",
        "graph_tolerance": "abs(after-before) <= 1e-5 + 1e-5 * abs(before)",
        "layout_tolerance": "abs(after-before) <= 1e-5",
        "violations": total,
        "files": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
