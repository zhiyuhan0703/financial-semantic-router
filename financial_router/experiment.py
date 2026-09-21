"""Shared execution and reporting for the four routing experiment arms."""

from __future__ import annotations

import time
from typing import Any, Sequence

from financial_router.contract import Case, Company
from financial_router.hybrid_router import HybridRouter
from financial_router.llm_router import DecisionModel, LLMOnlyRouter
from financial_router.rule_router import RuleRouter
from financial_router.scoring import PredictionRecord, score_predictions


ROUTE_NAMES = (
    "pure_rule",
    "llm_only",
    "hybrid_no_verifier",
    "hybrid",
)


def _build_router(
    route_name: str,
    companies: Sequence[Company],
    model: DecisionModel | None,
) -> Any:
    if route_name == "pure_rule":
        return RuleRouter(companies)
    if model is None:
        raise ValueError(f"route {route_name} requires a model")
    if route_name == "llm_only":
        return LLMOnlyRouter(model)
    if route_name == "hybrid_no_verifier":
        return HybridRouter(companies, model, use_verifier=False)
    if route_name == "hybrid":
        return HybridRouter(companies, model, use_verifier=True)
    raise ValueError(f"unknown route: {route_name}")


def run_experiment(
    *,
    route_name: str,
    split: str,
    cases: Sequence[Case],
    companies: Sequence[Company],
    repeats: int,
    model: DecisionModel | None = None,
) -> dict[str, Any]:
    if route_name not in ROUTE_NAMES:
        raise ValueError(f"unknown route: {route_name}")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if route_name != "pure_rule" and model is None:
        raise ValueError(f"route {route_name} requires a model")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("duplicate case_id values would overwrite scoring records")

    runs: list[dict[str, Any]] = []
    for run_index in range(1, repeats + 1):
        router = _build_router(route_name, companies, model)
        gold = {}
        predictions = {}
        outcomes = []

        for case in cases:
            started = time.perf_counter_ns()
            result = router.route(case.router_input(companies))
            measured_latency_ms = (time.perf_counter_ns() - started) / 1_000_000
            latency_ms = float(getattr(result, "latency_ms", measured_latency_ms))
            decision = result.decision
            # No normalisation here. Business-route order is normalised once, at each arm's
            # own decision boundary (parse_decision for model output, the rule arm's own
            # Decision construction), so the scoring layer must not alter a prediction.
            record = PredictionRecord(
                decision=decision,
                latency_ms=latency_ms,
                input_tokens=getattr(result, "input_tokens", 0),
                output_tokens=getattr(result, "output_tokens", 0),
                llm_calls=int(getattr(result, "llm_calls", 0)),
                error=getattr(result, "error", None),
            )
            gold[case.case_id] = case.expected
            predictions[case.case_id] = record
            outcomes.append(
                {
                    "case_id": case.case_id,
                    "expected": case.expected.to_dict(),
                    "predicted": decision.to_dict() if decision is not None else None,
                    "correct": decision == case.expected,
                    "latency_ms": latency_ms,
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                    "llm_calls": record.llm_calls,
                    "model": getattr(result, "model", None),
                    "system_fingerprint": getattr(result, "system_fingerprint", None),
                    "audit_reason": getattr(result, "audit_reason", None),
                    "verifier_codes": [
                        item.code
                        for item in getattr(result, "verifier_violations", ())
                    ],
                    "raw_content": getattr(result, "raw_content", None),
                    "error": record.error,
                }
            )

        runs.append(
            {
                "run_index": run_index,
                "cases": outcomes,
                "metrics": score_predictions(gold, predictions),
            }
        )

    return {
        "route": route_name,
        "split": split,
        "repeats": repeats,
        "runs": runs,
    }


__all__ = ["ROUTE_NAMES", "run_experiment"]
