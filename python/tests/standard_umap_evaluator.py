"""Score frozen standard-UMAP outputs against the pinned reference suite.

Expected implementation archive keys are ``{case}__seed{seed}__fit`` and
``{case}__seed{seed}__transform`` (2-D embeddings), plus optional
``{case}__seed{seed}__fit_neighbors`` and
``{case}__seed{seed}__query_neighbors`` (integer neighbor IDs). The latter
are scored against exact input-space neighbors using the frozen tie-aware rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.manifold import trustworthiness

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "crates/umap/tests/fixtures/standard_umap_reference.json"
QUALITY = ROOT / "crates/umap/tests/fixtures/standard_umap_quality.npz"
TRUST_GAP_LIMIT = 0.025


def exact_neighbors(
    query: np.ndarray, reference: np.ndarray, k: int, *, exclude_self: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic exact kNN IDs and kth radii, ordered by (distance, ID)."""
    distances = np.linalg.norm(query[:, None, :] - reference[None, :, :], axis=2)
    if exclude_self:
        if query.shape[0] != reference.shape[0]:
            raise ValueError("fit self-exclusion requires matching query/reference rows")
        np.fill_diagonal(distances, np.inf)
    ids = np.empty((query.shape[0], k), dtype=np.int64)
    radii = np.empty(query.shape[0], dtype=np.float64)
    indices = np.arange(reference.shape[0])
    for row in range(query.shape[0]):
        order = np.lexsort((indices, distances[row]))[:k]
        ids[row] = order
        radii[row] = distances[row, order[-1]]
    return ids, radii


def _recall(
    actual_ids: np.ndarray, query: np.ndarray, reference: np.ndarray, k: int, exclude_self: bool
) -> float:
    if query.shape[0] == 0 or reference.shape[0] <= k:
        raise ValueError("recall requires queries and at least k+1 fit rows")
    ids = np.asarray(actual_ids)
    if ids.ndim != 2 or ids.shape[0] != query.shape[0]:
        raise ValueError("neighbor IDs must have shape (query_rows, returned_neighbors)")
    if ids.shape[1] > k:
        raise ValueError(f"neighbor output has {ids.shape[1]} columns, exceeding fixed k={k}")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("neighbor IDs must be integers")
    _, radii = exact_neighbors(query, reference, k, exclude_self=exclude_self)
    scores = []
    for row, candidates in enumerate(ids):
        if np.any((candidates < 0) | (candidates >= len(reference))):
            raise ValueError("neighbor ID outside reference row range")
        unique = np.unique(candidates)
        if exclude_self:
            unique = unique[unique != row]
        distances = np.linalg.norm(reference[unique] - query[row], axis=1)
        scores.append(float(np.count_nonzero(distances <= radii[row])) / k)
    return float(np.mean(scores))


def _coverage_pass(actual_keys: set[str], required_keys: set[str]) -> tuple[bool, list[str]]:
    missing = sorted(required_keys - actual_keys)
    return not missing, missing


def _non_inferiority_pass(reference_minus_candidate: float | None) -> bool:
    return (
        reference_minus_candidate is not None
        and np.isfinite(reference_minus_candidate)
        and reference_minus_candidate <= TRUST_GAP_LIMIT
    )


def self_check() -> None:
    """Exercise scoring boundaries, recall tie acceptance, validation, and coverage."""
    assert _non_inferiority_pass(-0.026), "an improvement beyond the margin must pass"
    assert _non_inferiority_pass(0.025), "the exact non-inferiority boundary must pass"
    assert not _non_inferiority_pass(0.026), "degradation beyond the margin must fail"
    assert not _non_inferiority_pass(float("nan")), "non-finite gaps must fail"
    assert not _non_inferiority_pass(float("inf")), "infinite gaps must fail"
    query = np.asarray([[0.0]])
    reference = np.asarray([[0.0], [1.0], [-1.0], [2.0]])
    tied = _recall(np.asarray([[1, 2]]), query, reference, 2, False)
    assert tied == 1.0, f"kth-distance ties should be accepted, got {tied}"
    duplicated = _recall(np.asarray([[1, 1]]), query, reference, 2, False)
    assert duplicated == 0.5, (
        f"duplicate IDs should count once with fixed k denominator, got {duplicated}"
    )
    try:
        _recall(np.asarray([[1, 2, 3]]), query, reference, 2, False)
    except ValueError:
        pass
    else:
        raise AssertionError("neighbor output wider than k must be rejected")
    try:
        _recall(np.asarray([[1, 9]]), query, reference, 2, False)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range neighbor IDs must be rejected")
    required = {"clusters__seed11__fit_neighbors", "clusters__seed11__query_neighbors"}
    passed, missing = _coverage_pass({"clusters__seed11__fit_neighbors"}, required)
    assert not passed and missing == ["clusters__seed11__query_neighbors"], (
        "recall coverage must fail when any case/seed surface is absent"
    )


def evaluate(actual_path: Path) -> dict[str, object]:
    specification = json.loads(REFERENCE.read_text(encoding="utf-8"))
    trust_k = int(specification["controls"]["trustworthiness_k"])
    fit_k = int(specification["controls"]["n_neighbors"]) - 1
    query_k = int(specification["controls"]["n_neighbors"])
    cases = specification["quality"]["cases"]
    seeds = specification["controls"]["seeds"]
    scores = specification["quality"]["scores"]
    with (
        np.load(QUALITY, allow_pickle=False) as reference,
        np.load(actual_path, allow_pickle=False) as actual,
    ):
        result: dict[str, object] = {
            "archive": str(actual_path),
            "trustworthiness_k": trust_k,
            "fit_recall_k": fit_k,
            "query_recall_k": query_k,
            "cases": {},
            "missing": [],
        }
        expected_embeddings = len(cases) * len(seeds) * 2
        required_fit_recall = {
            f"{case}__seed{seed}__fit_neighbors" for case in cases for seed in seeds
        }
        required_query_recall = {
            f"{case}__seed{seed}__query_neighbors" for case in cases for seed in seeds
        }
        actual_keys = set(actual.files)
        fit_recall_coverage, missing_fit_recall = _coverage_pass(actual_keys, required_fit_recall)
        query_recall_coverage, missing_query_recall = _coverage_pass(
            actual_keys, required_query_recall
        )
        result["missing_recall"] = {
            "fit": missing_fit_recall,
            "query": missing_query_recall,
        }
        fit_scores: list[float] = []
        fit_reference_scores: list[float] = []
        query_scores: list[float] = []
        query_reference_scores: list[float] = []
        fit_recalls: list[tuple[float, int]] = []
        query_recalls: list[tuple[float, int]] = []
        for case in cases:
            train = reference[f"{case}__train"]
            heldout = reference[f"{case}__heldout"]
            per_seed = []
            for seed, frozen in zip(seeds, scores[case], strict=True):
                fit_key = f"{case}__seed{seed}__fit"
                transform_key = f"{case}__seed{seed}__transform"
                if fit_key not in actual or transform_key not in actual:
                    result["missing"].extend(
                        key for key in (fit_key, transform_key) if key not in actual
                    )
                    continue
                fit_embedding = np.asarray(actual[fit_key])
                transform_embedding = np.asarray(actual[transform_key])
                if fit_embedding.shape != (len(train), 2) or transform_embedding.shape != (
                    len(heldout),
                    2,
                ):
                    raise ValueError(f"wrong embedding shape for {case}, seed {seed}")
                if (
                    not np.isfinite(fit_embedding).all()
                    or not np.isfinite(transform_embedding).all()
                ):
                    raise ValueError(f"non-finite embedding for {case}, seed {seed}")
                fit_quality = float(trustworthiness(train, fit_embedding, n_neighbors=trust_k))
                transform_quality = float(
                    trustworthiness(heldout, transform_embedding, n_neighbors=trust_k)
                )
                fit_reference = float(frozen["fit_trustworthiness_at_15"])
                transform_reference = float(frozen["transform_trustworthiness_at_15"])
                fit_gap = fit_reference - fit_quality
                transform_gap = transform_reference - transform_quality
                fit_scores.append(fit_quality)
                fit_reference_scores.append(fit_reference)
                query_scores.append(transform_quality)
                query_reference_scores.append(transform_reference)
                entry: dict[str, object] = {
                    "seed": seed,
                    "fit_trustworthiness": fit_quality,
                    "fit_reference": fit_reference,
                    "fit_signed_acceptance_gap_reference_minus_candidate": fit_gap,
                    "fit_absolute_gap_diagnostic": abs(fit_gap),
                    "transform_trustworthiness": transform_quality,
                    "transform_reference": transform_reference,
                    "transform_signed_acceptance_gap_reference_minus_candidate": transform_gap,
                    "transform_absolute_gap_diagnostic": abs(transform_gap),
                }
                fit_neighbor_key = f"{case}__seed{seed}__fit_neighbors"
                query_neighbor_key = f"{case}__seed{seed}__query_neighbors"
                if fit_neighbor_key in actual:
                    fit_recall = _recall(actual[fit_neighbor_key], train, train, fit_k, True)
                    entry["fit_tie_aware_recall"] = fit_recall
                    fit_recalls.append((fit_recall, len(train)))
                if query_neighbor_key in actual:
                    query_recall = _recall(
                        actual[query_neighbor_key], heldout, train, query_k, False
                    )
                    entry["query_tie_aware_recall"] = query_recall
                    query_recalls.append((query_recall, len(heldout)))
                per_seed.append(entry)
            result["cases"][case] = per_seed
        fit_mean_gap = (
            float(np.mean(fit_reference_scores)) - float(np.mean(fit_scores))
            if fit_scores
            else None
        )
        query_mean_gap = (
            float(np.mean(query_reference_scores)) - float(np.mean(query_scores))
            if query_scores
            else None
        )
        case_fit_gaps = []
        case_transform_gaps = []
        case_fit_signed_gaps = []
        case_transform_signed_gaps = []
        for case_results in result["cases"].values():
            if len(case_results) != len(seeds):
                continue
            fit_signed_gap = float(
                np.mean([entry["fit_reference"] for entry in case_results])
            ) - float(np.mean([entry["fit_trustworthiness"] for entry in case_results]))
            transform_signed_gap = float(
                np.mean([entry["transform_reference"] for entry in case_results])
            ) - float(np.mean([entry["transform_trustworthiness"] for entry in case_results]))
            case_fit_signed_gaps.append(fit_signed_gap)
            case_transform_signed_gaps.append(transform_signed_gap)
            case_fit_gaps.append(abs(fit_signed_gap))
            case_transform_gaps.append(abs(transform_signed_gap))
        result["summary"] = {
            "implementation_mean_fit_trustworthiness": float(np.mean(fit_scores))
            if fit_scores
            else None,
            "reference_mean_fit_trustworthiness": float(np.mean(fit_reference_scores))
            if fit_reference_scores
            else None,
            "fit_signed_acceptance_gap_reference_minus_candidate": fit_mean_gap,
            "absolute_gap_of_mean_fit_trustworthiness_diagnostic": abs(fit_mean_gap)
            if fit_mean_gap is not None
            else None,
            "implementation_mean_transform_trustworthiness": float(np.mean(query_scores))
            if query_scores
            else None,
            "reference_mean_transform_trustworthiness": float(np.mean(query_reference_scores))
            if query_reference_scores
            else None,
            "transform_signed_acceptance_gap_reference_minus_candidate": query_mean_gap,
            "absolute_gap_of_mean_transform_trustworthiness_diagnostic": abs(query_mean_gap)
            if query_mean_gap is not None
            else None,
            "per_case_fit_absolute_mean_gaps_diagnostic": case_fit_gaps,
            "per_case_transform_absolute_mean_gaps_diagnostic": case_transform_gaps,
            "per_case_fit_signed_mean_gaps": case_fit_signed_gaps,
            "per_case_transform_signed_mean_gaps": case_transform_signed_gaps,
            "trustworthiness_gate_scope": (
                "pooled across all case/seed scores; accept reference mean minus candidate mean "
                "<= 0.025; per-case signed (reference minus candidate) and absolute mean gaps "
                "are diagnostic only"
            ),
            "fit_trustworthiness_gate_pass": bool(
                len(fit_scores) == expected_embeddings // 2
                and not result["missing"]
                and len(case_fit_gaps) == len(cases)
                and _non_inferiority_pass(fit_mean_gap)
            ),
            "transform_trustworthiness_gate_pass": bool(
                len(query_scores) == expected_embeddings // 2
                and not result["missing"]
                and len(case_transform_gaps) == len(cases)
                and _non_inferiority_pass(query_mean_gap)
            ),
            "fit_tie_aware_recall": (
                float(
                    sum(score * rows for score, rows in fit_recalls)
                    / sum(rows for _, rows in fit_recalls)
                )
                if fit_recalls
                else None
            ),
            "query_tie_aware_recall": (
                float(
                    sum(score * rows for score, rows in query_recalls)
                    / sum(rows for _, rows in query_recalls)
                )
                if query_recalls
                else None
            ),
            "recall_gate_pass": bool(
                fit_recall_coverage
                and query_recall_coverage
                and len(fit_recalls) == len(required_fit_recall)
                and len(query_recalls) == len(required_query_recall)
                and sum(score * rows for score, rows in fit_recalls)
                / sum(rows for _, rows in fit_recalls)
                >= 0.90
                and sum(score * rows for score, rows in query_recalls)
                / sum(rows for score, rows in query_recalls)
                >= 0.90
            ),
        }
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("implementation_archive", type=Path, nargs="?")
    parser.add_argument("--output", type=Path, help="write JSON report here (default: stdout)")
    parser.add_argument(
        "--self-check", action="store_true", help="run built-in recall contract assertions and exit"
    )
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("standard UMAP evaluator self-check passed")
        return
    if args.implementation_archive is None:
        parser.error("implementation_archive is required unless --self-check is used")
    report = json.dumps(evaluate(args.implementation_archive), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
