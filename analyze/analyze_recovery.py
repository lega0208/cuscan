#!/usr/bin/env python3
"""Aggregate retrospective core-distance recovery captures and source-derived models."""

import itertools
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECOVERY_SPANS = {
    "recovery_neighbor_search",
    "recovery_neighbor_readback",
    "recovery_core_upload",
    "recovery_candidate_readback",
    "recovery_host_padding_core",
    "recovery_seed",
}


def fail(message):
    raise ValueError(message)


def number(value, label, *, integer=False):
    try:
        result = int(value) if integer else float(value)
    except (TypeError, ValueError):
        fail(f"{label}: expected {'integer' if integer else 'numeric'} value, got {value!r}")
    if isinstance(value, bool) or not math.isfinite(result):
        fail(f"{label}: expected finite numeric value, got {value!r}")
    return result


def trace_events(path):
    data = json.loads(path.read_text())
    events = data.get("traceEvents") if isinstance(data, dict) else data
    if not isinstance(events, list):
        fail(f"{path}: Chrome trace must be an array or contain traceEvents array")
    return events


def span_name(event):
    return event.get("name") or event.get("args", {}).get("name")


def span_bounds(event):
    start = number(event.get("ts"), "trace ts")
    duration = number(event.get("dur"), f"trace {span_name(event)} dur")
    if duration < 0:
        fail(f"trace {span_name(event)}: negative duration")
    return start, start + duration


def fit_partition(events, count, capture_name):
    boundaries = sorted(
        (e for e in events if span_name(e) == "input_validation" and e.get("ph") == "B"),
        key=lambda e: number(e.get("ts"), "input_validation ts"),
    )
    if len(boundaries) != count:
        fail(
            f"{capture_name}: expected {count} input_validation B fit boundaries, found"
            f" {len(boundaries)}"
        )
    starts = [number(e.get("ts"), "input_validation ts") for e in boundaries]
    if any(b <= a for a, b in itertools.pairwise(starts)):
        fail(f"{capture_name}: fit boundaries are not strictly increasing")
    return [(starts[i], starts[i + 1] if i + 1 < count else math.inf) for i in range(count)]


def in_interval(event, interval):
    t = number(event.get("ts"), "trace ts")
    return interval[0] <= t < interval[1]


def numeric_arg(event, key, context):
    args = event.get("args") or {}
    if key not in args:
        fail(f"{context}: missing args.{key}")
    return number(args[key], f"{context} args.{key}", integer=True)


def memory_stats(events, interval, capture_name, fit_index):
    records = []
    for event in events:
        args = event.get("args") or {}
        if (
            event.get("ph") != "i"
            or args.get("message") != "recovery_memory"
            or not in_interval(event, interval)
        ):
            continue
        label = str(args.get("label", "")).strip('"')
        record = {
            key: numeric_arg(event, key, f"{capture_name} fit {fit_index} memory {label}")
            for key in (
                "bytes_in_use",
                "bytes_padding",
                "bytes_reserved",
                "number_allocs",
            )
        }
        record["label"] = label
        records.append(record)
    if not records:
        fail(f"{capture_name} fit {fit_index}: no recovery_memory samples")
    return {
        "samples": records,
        "sampled_maxima": {
            key: max(r[key] for r in records)
            for key in (
                "bytes_in_use",
                "bytes_padding",
                "bytes_reserved",
                "number_allocs",
            )
        },
    }


def fit_recovery(events, interval, capture_name, fit_index):
    stages = {name: 0.0 for name in RECOVERY_SPANS}
    counts = {name: 0 for name in RECOVERY_SPANS}
    payloads = {
        "recovery_neighbor_readback": 0,
        "recovery_candidate_readback": 0,
        "recovery_core_upload": 0,
    }
    performed = []
    spans = complete_spans(events, capture_name)
    for event in spans:
        if not in_interval(event, interval):
            continue
        name = span_name(event)
        if name in RECOVERY_SPANS:
            duration = (
                number(event.get("dur"), f"{capture_name} fit {fit_index} {name} dur") / 1_000_000
            )
            stages[name] += duration
            counts[name] += 1
            if name in payloads:
                payloads[name] += numeric_arg(
                    event,
                    "known_payload_bytes",
                    f"{capture_name} fit {fit_index} {name}",
                )
            if name == "recovery_core_upload":
                value = (event.get("args") or {}).get("performed")
                performed.append(str(value).lower() == "true" or value is True)
    if sum(counts.values()) == 0:
        fail(f"{capture_name} fit {fit_index}: no recovery spans")
    if counts["recovery_core_upload"] > 1:
        fail(f"{capture_name} fit {fit_index}: duplicate recovery_core_upload spans")
    if counts["recovery_core_upload"] == 1 and performed[0] != (
        payloads["recovery_core_upload"] > 0
    ):
        fail(f"{capture_name} fit {fit_index}: recovery_core_upload performed flag/bytes disagree")
    return {
        "stage_seconds": stages,
        "stage_counts": counts,
        "logical_payload_bytes": payloads,
        "core_upload_performed": performed[0] if performed else None,
        "memory": memory_stats(events, interval, capture_name, fit_index),
    }


def complete_spans(events, capture_name):
    """Return complete X spans and pair B/E spans, rejecting unmatched recoveries."""
    completed = [dict(event) for event in events if event.get("ph") == "X"]
    open_spans = {}
    for event in events:
        phase = event.get("ph")
        if phase not in ("B", "E"):
            continue
        name = span_name(event)
        key = (event.get("pid"), event.get("tid"), name)
        if phase == "B":
            open_spans.setdefault(key, []).append(event)
            continue
        stack = open_spans.get(key)
        if not stack:
            if name in RECOVERY_SPANS:
                fail(f"{capture_name}: unmatched recovery span E event for {name}")
            continue
        begin = stack.pop()
        start, end = number(begin.get("ts"), "span B ts"), number(event.get("ts"), "span E ts")
        if end < start:
            fail(f"{capture_name}: negative {name} B/E span")
        combined = dict(begin)
        combined["ph"] = "X"
        combined["dur"] = end - start
        completed.append(combined)
    for (_, _, name), stack in open_spans.items():
        if stack and name in RECOVERY_SPANS:
            fail(f"{capture_name}: unclosed recovery span B event for {name}")
    return completed


def kd_nodes(n, leaf=40):
    if n <= leaf:
        return 1
    left = n // 2
    return 1 + kd_nodes(left, leaf) + kd_nodes(n - left, leaf)


def source_model(capture):
    c = capture["case"]
    n, d = (number(c[key], f"case.{key}", integer=True) for key in ("n", "d"))
    ms = number(c["min_samples"], "case.min_samples", integer=True)
    if n < 2 or d < 1 or ms < 1:
        fail(f"{c.get('name')}: require n>=2, d>=1, min_samples>=1")
    k = min(ms, n - 1)
    search, variant, backend = capture["search"], capture["variant"], capture["backend"]
    if search == "brute-force":
        batch = min(max(n - 1, 1), 64 if variant == "original" else 256)
        batches = (n + batch - 1) // batch
        tiles = math.ceil(n / 128)
        groups = min(tiles, 4)
        neighbor_launches = (
            batches
            if backend == "cpu" or variant == "original"
            else batches * (math.ceil(tiles / groups) + tiles + 1)
        )
        neighbor_search = {
            "batch_rows": batch,
            "batches": batches,
            "neighbor_launches": neighbor_launches,
            "readback_transactions": batches * (2 if variant == "original" else 1),
        }
        tree_nodes = None
        core_payload = 4 * n if variant == "original" else 0
        grouped_scratch = (
            8 * batch * k * (groups + 2) if backend == "cuda" and variant == "current" else 0
        )
        neighbor_output = 8 * batch * k
    elif search == "kd-tree":
        batch, batches, tree_nodes = None, 1, kd_nodes(n)
        neighbor_search = {
            "batch_rows": n,
            "batches": 1,
            "neighbor_launches": 1,
            "readback_transactions": 2 if variant == "original" else 1,
        }
        core_payload = 4 * n if variant == "original" else 0
        grouped_scratch, neighbor_output = 0, 8 * n * k
    else:
        fail(f"unsupported search {search!r}")
    if variant not in ("original", "current") or backend not in ("cpu", "cuda"):
        fail("variant/backend must be original|current and cpu|cuda")
    rounds_value = capture.get("round_count", capture.get("boruvka_rounds"))
    if rounds_value is None:
        fail(f"{c.get('name')}: capture must provide round_count or boruvka_rounds (R)")
    rounds = number(rounds_value, "R", integer=True)
    if rounds < 0:
        fail("R must be nonnegative")
    if search == "brute-force":
        tiles_count = math.ceil(n / 256)
        candidate_block = min(n, (64 * 1024 * 1024) // (8 * tiles_count))
        candidate = {
            "tile_width": 256,
            "scratch_budget_bytes": 64 * 1024 * 1024,
            "kernel_launches": 2 * math.ceil(n / candidate_block) * rounds,
            "candidate_readback_transactions": (rounds * (2 if variant == "original" else 1)),
            "candidate_readback_logical_bytes": 8 * n * rounds,
            "component_upload_logical_bytes": 4 * n * rounds,
            "node_component_upload_logical_bytes": 0,
            "candidate_scratch_model": {
                "tiles": tiles_count,
                "block_rows": candidate_block,
                "bytes": 8 * candidate_block * tiles_count,
                "is_observed_peak": False,
            },
        }
    else:
        candidate = {
            "tree_leaf_size": 40,
            "kd_tree_nodes": tree_nodes,
            "kernel_launches": rounds,
            "candidate_readback_transactions": (rounds * (2 if variant == "original" else 1)),
            "candidate_readback_logical_bytes": 8 * n * rounds,
            "component_upload_logical_bytes": 4 * n * rounds,
            "node_component_upload_logical_bytes": 4 * tree_nodes * rounds,
        }
    candidate["core_upload_logical_bytes"] = core_payload if rounds > 0 else 0
    kd_bytes = (4 * n + (16 + 8 * d) * tree_nodes) if search == "kd-tree" else 0
    if search == "kd-tree":
        candidate["tree_upload_logical_bytes"] = kd_bytes
    padding = 8 * n * ms
    read_rows = batch if search == "brute-force" else n
    if variant == "original":
        host_phases = {
            "neighbor_materialization": {
                "compact_neighbor_arrays": 8 * n * k,
                "readback_and_conversion_staging_upper_bound": 8 * read_rows * k,
            },
            "padding_and_core_phase": {
                "compact_neighbor_arrays": 8 * n * k,
                "padded_neighbor_arrays": padding,
                "core_distances": 4 * n,
            },
            "later_seed_phase": {
                "padded_neighbor_arrays": padding,
                "core_distances": 4 * n,
                "fitted_id_clone": 4 * n * ms,
            },
        }
    else:
        host_phases = {
            "neighbor_materialization": {
                "compact_neighbor_arrays": 8 * n * k,
                "paired_readback_staging_upper_bound": 8 * read_rows * k,
                "extra_conversion_vec_upper_bound": (
                    4 * read_rows * k if search == "brute-force" else 0
                ),
            },
            "seed_phase": {
                "compact_neighbor_arrays": 8 * n * k,
                "seed_core_distances": 4 * n,
            },
            "after_seed": {
                "compact_neighbor_ids": 4 * n * k,
                "seed_core_distances": 4 * n,
            },
            "late_public_padding": {
                "compact_neighbor_ids": 4 * n * k,
                "extra_public_neighbor_ids_if_ms_differs_from_k": (4 * n * ms if ms != k else 0),
            },
        }
    core_device_phase = {}
    if variant == "current":
        core_device_phase = (
            {
                "core_parts_upper_bound": 4 * n + 4 * k,
                "core_concat_result": 4 * n,
                "query_slice_materialization_upper_bound": 4 * batch * d,
                "core_column_copy_upper_bound": 4 * batch,
            }
            if search == "brute-force"
            else {"resident_core_column_copy": 4 * n}
        )
    device_buffers = {
        "input": 4 * n * d,
        "neighbor_outputs_per_batch_or_tree": neighbor_output,
        "boruvka_core": 4 * n if variant == "current" or rounds else 0,
        "boruvka_candidate_outputs": 8 * n if rounds else 0,
        "boruvka_point_components": 4 * n if rounds else 0,
    }
    if search == "kd-tree":
        device_buffers["kd_tree_upload"] = kd_bytes
        device_buffers["extra_round_node_components"] = 4 * tree_nodes if rounds else 0
    neighbor_scratch = {
        "grouped_cuda_scratch_logical_bytes": grouped_scratch,
        "grouped_cuda_scratch_present": grouped_scratch > 0,
        "source_aligned_budget_limit_bytes": 32 * 1024 * 1024,
        "alignment_note": (
            "Logical bytes exclude alignment. Source separately sums"
            " 2*align(4*B*k*G)+4*align(4*B*k), checks each allocation against max_page"
            " and the sum against 32 MiB, and shrinks B if required. Actual native"
            " launch counts independently validate unshrunk B for these captures."
        ),
    }
    return {
        "n": n,
        "d": d,
        "k": k,
        "boruvka_rounds_R": rounds,
        "source_anchors": {
            "estimator": (
                f"target/core-distance-recovery-{variant}/crates/fast-hdbscan/src/estimator.rs"
            ),
            "neighbor_kernel": (
                f"target/core-distance-recovery-{variant}/crates/fast-cubecl-common/src/kernels/neighbors.rs"
            ),
            "host_conversion": (
                "burn-std-0.22.1/src/data/tensor/conversion.rs::try_to_vec uses"
                " as_slice()?.to_vec()"
            ),
            "neighbor_batching_and_readback": (
                "estimator.rs core_distances/exact_neighbors orchestration;"
                " source-derived batch and transaction counts"
            ),
            "boruvka": (
                "estimator.rs spanning_tree; 256-row tiles, 64 MiB scratch, round"
                " uploads/readbacks and core upload"
            ),
            "kd_tree": (
                "estimator.rs KD-tree construction/upload; balanced floor/ceil"
                " recursion to leaf size 40"
            ),
        },
        "neighbor_search": neighbor_search,
        "boruvka": candidate,
        "logical_neighbor_readback_bytes_per_fit": 8 * n * k,
        "kd_tree": (
            {"nodes": tree_nodes, "logical_upload_bytes": kd_bytes, "leaf_size": 40}
            if tree_nodes
            else None
        ),
        "live_buffer_accounting_bytes": {
            "host_phase_components": host_phases,
            "device_core_phase_components": core_device_phase,
            "device_neighbor_output_per_batch_or_tree": device_buffers,
            "neighbor_scratch": neighbor_scratch,
            "candidate_scratch": candidate.get("candidate_scratch_model"),
            "kd_tree_upload_bytes": kd_bytes if search == "kd-tree" else 0,
            "interpretation": (
                "Logical payload components are phase-specific estimates/bounds"
                " excluding allocator alignment, not a sum of mutually exclusive"
                " buffers and not an exact allocation peak. Host result extraction and"
                " complete MST/DSU/tree/runtime allocation accounting are excluded."
            ),
        },
    }


def aggregate(capture):
    name = capture.get("case", {}).get("name", "unnamed")
    profile_path, trace_path = (
        ROOT / capture["profile_json"],
        ROOT / capture["trace_json"],
    )
    profile = json.loads(profile_path.read_text())
    events = trace_events(trace_path)
    warm_timings = profile.get("fit_seconds")
    warm = profile.get("recovery_stage_seconds", {}).get("warm_fits")
    if (
        not isinstance(warm_timings, list)
        or len(warm_timings) != 3
        or not isinstance(warm, list)
        or len(warm) != 3
    ):
        fail(f"{name}: require three warm fit_seconds and recovery_stage_seconds.warm_fits records")
    warm_timings = [number(v, f"{name} warm fit_seconds", integer=False) for v in warm_timings]
    cold = number(profile.get("cold_first_fit_seconds"), f"{name} cold_first_fit_seconds")
    if cold <= 0 or any(v <= 0 for v in warm_timings):
        fail(f"{name}: full-fit timings must be positive")
    intervals = fit_partition(events, 4, name)
    records = [fit_recovery(events, intervals[i], name, i) for i in range(4)]
    for i, record in enumerate(records):
        declared = profile["recovery_stage_seconds"]["cold_first_fit"] if i == 0 else warm[i - 1]
        if set(declared) < {"stage_seconds", "memory_events"}:
            fail(f"{name} profile recovery record {i}: missing stage_seconds/memory_events")
        if not isinstance(declared["memory_events"], list):
            fail(f"{name} profile recovery record {i}: memory_events must be an array")
        trace_memory = records[i]["memory"]["samples"]
        if len(declared["memory_events"]) != len(trace_memory):
            fail(
                f"{name} fit {i}: profile/trace memory event counts differ"
                f" ({len(declared['memory_events'])} vs {len(trace_memory)})"
            )
        for memory_index, (profile_event, trace_event) in enumerate(
            zip(declared["memory_events"], trace_memory, strict=True)
        ):
            profile_args = profile_event.get("args", profile_event)
            for key in (
                "bytes_in_use",
                "bytes_padding",
                "bytes_reserved",
                "number_allocs",
            ):
                if (
                    number(
                        profile_args.get(key),
                        f"{name} profile fit {i} memory {memory_index} {key}",
                        integer=True,
                    )
                    != trace_event[key]
                ):
                    fail(
                        f"{name} fit {i} memory event {memory_index}: profile/trace {key} mismatch"
                    )
        for stage in RECOVERY_SPANS:
            stage_values = declared["stage_seconds"].get(stage, [])
            if not isinstance(stage_values, list):
                fail(f"{name} fit {i} stage {stage}: expected per-span timing array")
            values = [number(duration, f"{name} profile {stage}") for duration in stage_values]
            if any(value < 0 for value in values):
                fail(f"{name}: negative recovery stage timing")
            observed_count = record["stage_counts"][stage]
            observed_sum = record["stage_seconds"][stage]
            if len(values) != observed_count or not math.isclose(
                sum(values), observed_sum, rel_tol=0.05, abs_tol=2e-6
            ):
                fail(
                    f"{name} fit {i} stage {stage}: profile has count/sum"
                    f" {len(values)}/{sum(values):.9g}s; trace has"
                    f" {observed_count}/{observed_sum:.9g}s"
                )
    gran = profile.get("stage_seconds")
    if not isinstance(gran, dict):
        fail(f"{name}: missing granular stage_seconds")
    gran = {key: [number(x, f"{name} stage {key}") for x in vals] for key, vals in gran.items()}
    if any(v < 0 for values in gran.values() for v in values):
        fail(f"{name}: negative granular stage timing")
    boruvka_samples = profile.get("boruvka_samples")
    if not isinstance(boruvka_samples, list) or len(boruvka_samples) != 3:
        fail(f"{name}: expected three warm boruvka_samples")
    warm_rounds = [
        number(sample.get("round_count"), f"{name} boruvka round_count", integer=True)
        for sample in boruvka_samples
    ]
    if len(set(warm_rounds)) != 1:
        fail(f"{name}: warm Boruvka round count varies: {warm_rounds}")
    spans = complete_spans(events, name)
    cold_rounds = sum(
        1
        for event in spans
        if span_name(event) == "boruvka_round" and in_interval(event, intervals[0])
    )
    rounds = [cold_rounds, *warm_rounds]
    if len(set(rounds)) != 1:
        fail(f"{name}: cold/warm Boruvka round count varies: {rounds}")
    model = source_model({**capture, "round_count": rounds[0]})
    expected_neighbor = model["logical_neighbor_readback_bytes_per_fit"]
    expected_core = model["boruvka"]["core_upload_logical_bytes"]
    for i, record in enumerate(records):
        actual = record["logical_payload_bytes"]
        if actual["recovery_neighbor_readback"] != expected_neighbor:
            fail(
                f"{name} fit {i}: neighbor readback metadata"
                f" {actual['recovery_neighbor_readback']} != source prediction"
                f" {expected_neighbor}"
            )
        if actual["recovery_core_upload"] != (
            expected_core if capture["variant"] == "original" else 0
        ):
            fail(
                f"{name} fit {i}: core upload metadata {actual['recovery_core_upload']}"
                f" != source prediction {expected_core}"
            )
        if (
            actual["recovery_candidate_readback"]
            != model["boruvka"]["candidate_readback_logical_bytes"]
        ):
            fail(
                f"{name} fit {i}: candidate readback metadata disagrees with n*8*R"
                " source prediction"
            )
        if capture["variant"] == "original" and record["stage_counts"]["recovery_core_upload"] != 1:
            fail(f"{name} fit {i}: original-source capture missing recovery_core_upload span")
        if capture["variant"] == "current" and actual["recovery_core_upload"] != 0:
            fail(f"{name} fit {i}: current-source capture unexpectedly reports core-upload bytes")
        expected_search_spans = model["neighbor_search"]["batches"]
        if record["stage_counts"]["recovery_neighbor_search"] != expected_search_spans:
            fail(
                f"{name} fit {i}: expected {expected_search_spans}"
                " recovery_neighbor_search spans, found"
                f" {record['stage_counts']['recovery_neighbor_search']}"
            )
        if record["stage_counts"]["recovery_neighbor_readback"] != expected_search_spans:
            fail(
                f"{name} fit {i}: expected {expected_search_spans} neighbor readback"
                f" spans, found {record['stage_counts']['recovery_neighbor_readback']}"
            )
        if record["stage_counts"]["recovery_candidate_readback"] != rounds[i]:
            fail(
                f"{name} fit {i}: expected {rounds[i]} candidate readback spans, found"
                f" {record['stage_counts']['recovery_candidate_readback']}"
            )
    warm_records = records[1:]
    return {
        "identity": {key: capture[key] for key in ("variant", "backend", "search", "case")},
        "snapshot": capture.get("snapshot"),
        "env": capture.get("env"),
        "measurements": {
            "warm_fit_seconds": warm_timings,
            "cold_first_fit_seconds": cold,
            "warm_full_fit_median_seconds": statistics.median(warm_timings),
            "granular_stage_seconds": {
                key: {"samples": values, "median_seconds": statistics.median(values)}
                for key, values in gran.items()
            },
            "recovery_fits": records,
            "recovery_stage_sums_seconds": {
                key: sum(r["stage_seconds"][key] for r in records) for key in sorted(RECOVERY_SPANS)
            },
            "recovery_stage_counts": {
                key: sum(r["stage_counts"][key] for r in records) for key in sorted(RECOVERY_SPANS)
            },
            "warm_sampled_allocator_maxima_per_fit": [
                r["memory"]["sampled_maxima"] for r in warm_records
            ],
        },
        "derived_accounting": model,
    }


def main():
    if len(sys.argv) != 3:
        fail("usage: analyze_recovery.py INPUT_MANIFEST OUTPUT_JSON")
    manifest_path, output_path = (Path(arg) for arg in sys.argv[1:])
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest.get("captures"), list) or not manifest["captures"]:
        fail("manifest must contain a non-empty captures array")
    output = {
        "provenance": manifest.get("provenance"),
        "methodology": (
            "Retrospective original-source capture analysis; instrumented timings"
            " include synchronization and sampled-memory observation overhead and do"
            " not replace immutable warmed Criterion distributions."
        ),
        "captures": [aggregate(capture) for capture in manifest["captures"]],
        "accounting_method": (
            "Phase-specific logical payload components/bounds exclude allocator"
            " alignment and are never summed across mutually exclusive lifetimes as a"
            " peak. Current BF output is 8*B*k; grouped CUDA scratch logical bytes are"
            " 8*B*k*(G+2), G=min(ceil(n/128),4). Source sizing separately sums aligned"
            " buffers and enforces its 32 MiB/per-page limits. KD output is 8*n*k. BF"
            " candidate scratch uses tiles=ceil(n/256), block=min(n,64 MiB//(8*tiles)),"
            " bytes=8*block*tiles. Host phases include original compact/padded"
            " neighbors/ID clone, current paired-readback staging, try_to_vec copies"
            " and late public padding. Only current BF has core parts/cat; KD extracts"
            " a single core column. KD upload is 4*n+(16+8*d)*N for balanced"
            " nonconstant representative inputs and leaf size 40. These are not global"
            " liveness/physical peak models."
        ),
        "limitations": [
            (
                "Allocator values are sampled runtime allocator reports, not actual"
                " physical VRAM, process-wide/global peak, or continuous maxima."
            ),
            (
                "Logical payload bytes are source-level metadata, not measured physical"
                " PCIe traffic."
            ),
            (
                "Memory models explicitly exclude complete accounting for"
                " input/tree/common DSU/MST buffers and host result extraction; they"
                " are not global peak estimates."
            ),
            (
                "Source-derived scratch bytes are logical payloads, not aligned"
                " physical allocations or observed peaks; CubeCL allocator samples"
                " cover only explicit boundaries."
            ),
            (
                "Instrumented retrospective timings are not acceptance-performance"
                " replacements; original immutable warmed Criterion distributions"
                " remain authoritative."
            ),
        ],
    }
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"analyze_recovery: {error}", file=sys.stderr)
        sys.exit(2)
