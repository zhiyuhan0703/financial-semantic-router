"""Pure keyword/regex/alias routing baseline with explicit safe fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from financial_router.candidate_recall import RecallResult, recall_candidates
from financial_router.contract import BUSINESS_ROUTES, Company, Decision
from financial_router.verifier import (
    VerificationContext,
    Violation,
    fail_closed,
    verify_decision,
)


KEYWORDS = {
    "financial": (
        "营业收入",
        "营收",
        "净利润",
        "资产负债率",
        "现金流",
        "财报",
        "财务",
        "毛利率",
    ),
    "equity": (
        "第一大股东",
        "控股股东",
        "股东",
        "股权",
        "持股",
        "实际控制人",
        "实控人",
        "控制链",
    ),
    "event": (
        "重大事项",
        "公告",
        "披露",
        "处罚",
        "诉讼",
        "重组",
        "经营事项",
    ),
}
UNSUPPORTED_KEYWORDS = (
    "股价",
    "行情",
    "走势",
    "涨跌",
    "估值",
    "目标价",
    "买入",
    "卖出",
    "投资建议",
    "技术分析",
)
COMPARISON_PATTERN = re.compile(r"对比|比较|相比|分别|谁更|哪家")
REUSE_PATTERN = re.compile(r"它|该公司|这家公司|其|上述公司|这家|那它|前者|后者")


@dataclass(frozen=True)
class RuleRouteResult:
    decision: Decision
    recall: RecallResult
    detected_routes: tuple[str, ...]
    verifier_violations: tuple[Violation, ...]
    audit_reason: str | None = None


def _history_company_ids(history: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    ids: list[str] = []
    for turn in history:
        accepted = turn.get("accepted_decision", {})
        if accepted.get("subject_action") != "clarify":
            ids.extend(str(item) for item in accepted.get("company_ids", []))
    return tuple(dict.fromkeys(ids))


def _recent_subject(history: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    for turn in reversed(history):
        accepted = turn.get("accepted_decision", {})
        ids = tuple(str(item) for item in accepted.get("company_ids", []))
        if accepted.get("subject_action") != "clarify" and ids:
            return ids
    return ()


def _recent_business_routes(history: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    for turn in reversed(history):
        accepted = turn.get("accepted_decision", {})
        routes = tuple(
            route for route in accepted.get("route", []) if route in BUSINESS_ROUTES
        )
        if routes:
            return routes
    return ()


def _detect_routes(query: str) -> tuple[str, ...]:
    return tuple(
        route
        for route in BUSINESS_ROUTES
        if any(keyword in query for keyword in KEYWORDS[route])
    )


class RuleRouter:
    def __init__(self, companies: Sequence[Company]):
        self._companies = tuple(companies)
        self._known_ids = tuple(company.company_id for company in companies)

    def route(self, payload: Mapping[str, Any]) -> RuleRouteResult:
        query = str(payload["query"])
        history = tuple(payload.get("history", []))
        recall = recall_candidates(query, self._companies)
        explicit_ids = list(recall.resolved_ids)
        history_ids = _history_company_ids(history)
        recent_ids = _recent_subject(history)
        comparison = bool(COMPARISON_PATTERN.search(query))
        subject_is_unsafe = bool(recall.ambiguous_mentions or recall.unknown_mentions)

        if subject_is_unsafe:
            action = "clarify"
            company_ids: tuple[str, ...] = ()
            audit_reason = (
                "subject_ambiguous" if recall.ambiguous_mentions else "unknown_entity"
            )
        elif explicit_ids:
            if comparison and len(explicit_ids) == 1:
                explicit_ids.extend(item for item in recent_ids if item not in explicit_ids)
            action = "new_entity"
            company_ids = tuple(explicit_ids)
            audit_reason = None
        elif recent_ids and (REUSE_PATTERN.search(query) or _detect_routes(query)):
            action = "reuse"
            company_ids = recent_ids
            audit_reason = None
        else:
            action = "clarify"
            company_ids = ()
            audit_reason = "subject_missing"

        detected_routes = _detect_routes(query)
        if not detected_routes and history and (comparison or REUSE_PATTERN.search(query)):
            detected_routes = _recent_business_routes(history)

        if action == "clarify":
            routes = ("clarify",)
        elif any(keyword in query for keyword in UNSUPPORTED_KEYWORDS):
            routes = ("clarify",)
            audit_reason = "unsupported_business"
        elif not detected_routes:
            routes = ("clarify",)
            audit_reason = "intent_ambiguous"
        elif len(company_ids) > 1 and len(detected_routes) > 1:
            routes = ("clarify",)
            audit_reason = "unsupported_composition"
        else:
            routes = detected_routes

        # The rule arm reads no model output, so it has no parse boundary; normalise here so
        # all four arms share one rule for business-route order (see parse_decision).
        preliminary = Decision(action, company_ids, routes).canonicalized()
        allowed_ids = tuple(dict.fromkeys((*recall.candidate_ids, *history_ids)))
        required_ids = company_ids if action == "new_entity" else ()
        context = VerificationContext(
            known_company_ids=self._known_ids,
            allowed_company_ids=allowed_ids,
            required_company_ids=required_ids,
            history_company_ids=history_ids,
            subject_must_clarify=subject_is_unsafe,
        )
        violations = verify_decision(preliminary, context)
        final = fail_closed(preliminary, violations, context)
        return RuleRouteResult(
            decision=final,
            recall=recall,
            detected_routes=detected_routes,
            verifier_violations=violations,
            audit_reason=audit_reason,
        )


__all__ = ["RuleRouteResult", "RuleRouter"]
