"""Programmatic invariants and fail-closed behavior for routing decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from financial_router.contract import BUSINESS_ROUTES, ROUTES, SUBJECT_ACTIONS, Decision


ROUTE_ONLY_CODES = frozenset(
    {
        "invalid_route_length",
        "invalid_route_value",
        "duplicate_route",
        "clarify_not_exclusive",
        "noncanonical_route_order",
        "required_route_missing",
        "unsupported_composition",
    }
)


@dataclass(frozen=True)
class Violation:
    code: str
    message: str


@dataclass(frozen=True)
class VerificationContext:
    known_company_ids: tuple[str, ...]
    allowed_company_ids: tuple[str, ...]
    required_company_ids: tuple[str, ...] = ()
    required_routes: tuple[str, ...] = ()
    history_company_ids: tuple[str, ...] = ()
    subject_must_clarify: bool = False
    required_subject_action: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VerificationContext":
        return cls(
            known_company_ids=tuple(value.get("known_company_ids", [])),
            allowed_company_ids=tuple(value.get("allowed_company_ids", [])),
            required_company_ids=tuple(value.get("required_company_ids", [])),
            required_routes=tuple(value.get("required_routes", [])),
            history_company_ids=tuple(value.get("history_company_ids", [])),
            subject_must_clarify=bool(value.get("subject_must_clarify", False)),
            required_subject_action=value.get("required_subject_action"),
        )


def verify_decision(
    decision: Decision, context: VerificationContext
) -> tuple[Violation, ...]:
    violations: list[Violation] = []
    known = set(context.known_company_ids)
    allowed = set(context.allowed_company_ids)
    predicted_ids = set(decision.company_ids)

    if decision.subject_action not in SUBJECT_ACTIONS:
        violations.append(Violation("invalid_subject_action", "unknown subject_action"))

    if len(predicted_ids) != len(decision.company_ids):
        violations.append(Violation("duplicate_company_id", "company IDs must be unique"))

    if context.required_subject_action is not None:
        if decision.subject_action != context.required_subject_action:
            violations.append(
                Violation("fixed_subject_action_changed", "deterministically fixed action was changed")
            )
        if decision.company_ids != context.required_company_ids:
            violations.append(
                Violation("fixed_company_order_changed", "fixed company sequence was changed")
            )

    if predicted_ids - known:
        violations.append(Violation("unknown_company_id", "company ID is not in authority table"))

    if predicted_ids - allowed:
        violations.append(
            Violation("company_outside_allowed_set", "company ID was not recalled or inherited")
        )

    if not 1 <= len(decision.route) <= 3:
        violations.append(Violation("invalid_route_length", "route must contain one to three items"))

    if any(route not in ROUTES for route in decision.route):
        violations.append(Violation("invalid_route_value", "route contains an unknown module"))

    if len(set(decision.route)) != len(decision.route):
        violations.append(Violation("duplicate_route", "route modules must be unique"))

    if "clarify" in decision.route and decision.route != ("clarify",):
        violations.append(Violation("clarify_not_exclusive", "clarify must be the only route"))

    if (
        decision.route
        and all(route in BUSINESS_ROUTES for route in decision.route)
        and len(set(decision.route)) == len(decision.route)
    ):
        canonical = tuple(route for route in BUSINESS_ROUTES if route in decision.route)
        if decision.route != canonical:
            violations.append(
                Violation("noncanonical_route_order", "business routes are not in canonical order")
            )

    if decision.subject_action == "clarify" and decision.company_ids:
        violations.append(
            Violation("clarify_subject_has_company", "ambiguous subject must not bind a company")
        )

    if decision.subject_action == "clarify" and decision.route != ("clarify",):
        violations.append(
            Violation("clarify_subject_route", "clarified subject must route to clarify")
        )

    if decision.subject_action in {"new_entity", "reuse"} and not decision.company_ids:
        violations.append(
            Violation("subject_requires_company", "resolved or reused subject needs a company ID")
        )

    if decision.subject_action == "reuse" and not predicted_ids.issubset(
        set(context.history_company_ids)
    ):
        violations.append(
            Violation("reuse_without_history_subject", "reused company is absent from accepted history")
        )

    if set(context.required_company_ids) - predicted_ids:
        violations.append(
            Violation("required_company_missing", "an explicitly required company was omitted")
        )

    if (
        decision.subject_action != "clarify"
        and set(context.required_routes) - set(decision.route)
    ):
        violations.append(
            Violation("required_route_missing", "an explicitly required route was omitted")
        )

    if context.subject_must_clarify and decision.subject_action != "clarify":
        violations.append(
            Violation("unsafe_subject_binding", "ambiguous or unknown subject must not be bound")
        )

    business_count = sum(route in BUSINESS_ROUTES for route in decision.route)
    if len(decision.company_ids) > 1 and business_count > 1:
        violations.append(
            Violation(
                "unsupported_composition",
                "multiple companies with multiple business modules are out of scope",
            )
        )

    return tuple(violations)


def fail_closed(
    decision: Decision,
    violations: tuple[Violation, ...],
    context: VerificationContext,
) -> Decision:
    if not violations:
        return decision

    codes = {violation.code for violation in violations}
    ids_are_safe = (
        decision.subject_action in {"new_entity", "reuse"}
        and bool(decision.company_ids)
        and set(decision.company_ids).issubset(set(context.known_company_ids))
        and set(decision.company_ids).issubset(set(context.allowed_company_ids))
    )
    if codes.issubset(ROUTE_ONLY_CODES) and ids_are_safe:
        return Decision(decision.subject_action, decision.company_ids, ("clarify",))
    return Decision("clarify", (), ("clarify",))


__all__ = [
    "VerificationContext",
    "Violation",
    "fail_closed",
    "verify_decision",
]
