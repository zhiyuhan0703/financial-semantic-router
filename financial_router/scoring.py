"""Deterministic metrics for frozen routing decisions."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Mapping

from financial_router.contract import BUSINESS_ROUTES, Decision


@dataclass(frozen=True)
class PredictionRecord:
    decision: Decision | None
    latency_ms: float
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    llm_calls: int = 0
    error: str | None = None


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _classification(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def score_predictions(
    gold: Mapping[str, Decision],
    predictions: Mapping[str, PredictionRecord],
) -> dict[str, object]:
    if set(gold) != set(predictions):
        raise ValueError("gold and prediction case_id sets must match exactly")

    total = len(gold)
    missing_prediction = 0
    subject_correct = 0
    entity_correct = 0
    route_correct = 0
    joint_correct = 0
    wrong_binding = 0
    clarify_tp = clarify_fp = clarify_fn = 0
    module_counts = {
        module: {"tp": 0, "fp": 0, "fn": 0} for module in BUSINESS_ROUTES
    }

    for case_id, expected in gold.items():
        predicted = predictions[case_id].decision
        if predicted is None:
            missing_prediction += 1
            predicted_ids: tuple[str, ...] = ()
            predicted_route: tuple[str, ...] = ()
            predicted_action = None
        else:
            predicted_ids = predicted.company_ids
            predicted_route = predicted.route
            predicted_action = predicted.subject_action

        subject_match = predicted_action == expected.subject_action
        entity_match = predicted is not None and predicted_ids == expected.company_ids
        route_match = predicted_route == expected.route
        subject_correct += int(subject_match)
        entity_correct += int(entity_match)
        route_correct += int(route_match)
        joint_correct += int(subject_match and entity_match and route_match)
        wrong_binding += int(bool(set(predicted_ids) - set(expected.company_ids)))

        expected_clarify = expected.route == ("clarify",)
        predicted_clarify = predicted_route == ("clarify",)
        clarify_tp += int(expected_clarify and predicted_clarify)
        clarify_fp += int(not expected_clarify and predicted_clarify)
        clarify_fn += int(expected_clarify and not predicted_clarify)

        for module in BUSINESS_ROUTES:
            expected_has = module in expected.route
            predicted_has = module in predicted_route
            module_counts[module]["tp"] += int(expected_has and predicted_has)
            module_counts[module]["fp"] += int(not expected_has and predicted_has)
            module_counts[module]["fn"] += int(expected_has and not predicted_has)

    latencies = [float(record.latency_ms) for record in predictions.values()]
    input_usage = [record.input_tokens for record in predictions.values()]
    output_usage = [record.output_tokens for record in predictions.values()]
    input_tokens = sum(value for value in input_usage if value is not None)
    output_tokens = sum(value for value in output_usage if value is not None)
    missing_input = any(value is None for value in input_usage)
    missing_output = any(value is None for value in output_usage)
    usage_missing = sum(
        record.input_tokens is None or record.output_tokens is None
        for record in predictions.values()
    )
    llm_calls = sum(record.llm_calls for record in predictions.values())

    def exact_block(correct: int) -> dict[str, int | float]:
        return {"correct": correct, "total": total, "accuracy": _ratio(correct, total)}

    return {
        "counts": {"total": total, "missing_prediction": missing_prediction},
        "subject_action": exact_block(subject_correct),
        "entity_exact": exact_block(entity_correct),
        "wrong_binding": {
            "count": wrong_binding,
            "total": total,
            "rate": _ratio(wrong_binding, total),
        },
        "route_exact": exact_block(route_correct),
        "joint_exact": exact_block(joint_correct),
        "clarify": _classification(clarify_tp, clarify_fp, clarify_fn),
        "modules": {
            module: _classification(**counts) for module, counts in module_counts.items()
        },
        "runtime": {
            "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
            "p95_latency_ms": _nearest_rank(latencies, 0.95) if latencies else 0.0,
            "average_input_tokens": None if missing_input else _ratio(input_tokens, total),
            "average_output_tokens": None if missing_output else _ratio(output_tokens, total),
            "average_total_tokens": (
                None if usage_missing else _ratio(input_tokens + output_tokens, total)
            ),
            "usage_missing_cases": usage_missing,
            "usage_observed_cases": total - usage_missing,
            "known_input_tokens": input_tokens,
            "known_output_tokens": output_tokens,
            "llm_call_rate": _ratio(llm_calls, total),
        },
    }


__all__ = ["PredictionRecord", "score_predictions"]
