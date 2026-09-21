"""Deterministic candidate recall plus constrained LLM adjudication."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from financial_router.candidate_recall import RecallResult, _find_hits, recall_candidates
from financial_router.contract import Company, Decision
from financial_router.llm_router import (
    DecisionModel,
    ModelRequest,
    SHARED_ROUTING_GUIDANCE,
    parse_decision,
)
from financial_router.rule_router import (
    COMPARISON_PATTERN,
    KEYWORDS,
    REUSE_PATTERN,
    UNSUPPORTED_KEYWORDS,
    _detect_routes,
    _history_company_ids,
    _recent_subject,
)
from financial_router.verifier import (
    VerificationContext,
    Violation,
    fail_closed,
    verify_decision,
)


HYBRID_SYSTEM_PROMPT = """你是金融问答混合路由中的受约束裁决器。请只输出一个 JSON 对象，且恰好包含：
{"subject_action":"reuse|new_entity|clarify","company_ids":["公司ID"],"route":["financial|equity|event|clarify"]}

只能从 allowed_company_ids 中选择公司；fixed_subject 非空时必须原样保留其主体动作和公司 ID。detected_routes 非空时表示完整简单请求中确认的模块，不得漏掉；为空不表示没有业务意图，应阅读完整问题判断。subject_must_clarify 为 true 时不得绑定主体。route 的业务模块顺序固定为 financial、equity、event，clarify 必须独占。单公司允许一至三个业务模块；多公司只允许同类单模块，多公司多模块必须保留主体并 route=["clarify"]。无法在候选内安全确定主体时 abstain：subject_action="clarify"、company_ids=[]、route=["clarify"]。主体已固定但意图不清时保留主体，仅 route=["clarify"]。不要回答问题，不要输出解释、Markdown 或额外字段。""" + "\n\n" + SHARED_ROUTING_GUIDANCE

# These cues suspend a subject lock; they do not decide what a negation refers to.
SUBJECT_SCOPE_CUES = re.compile(r"不|非|别|除|只|回到|之前|最初|前者|后者|第一家|第二家|那家|原来")
PLURAL_REFERENCE = re.compile(r"它们|这两家|上述公司")
# A demonstrative pronoun carries no new entity information: "000001 这家银行…" points back at
# the already-resolved company, so it must not be treated as an unknown company mention.
#
# The mention must be *entirely* a demonstrative plus common noun, so callers must use
# ``fullmatch`` -- a prefix match would also swallow "该公司科技", an unknown company name that
# merely starts with a demonstrative, and that one has to stay blocking.
REFERENTIAL_MENTION = re.compile(
    r"^(?:这|那|该|上述|前述)(?:\d+|两|几|多)?(?:家|个|些|种|类|支|只)?"
    r"(?:公司|企业|银行|厂商|集团|机构|单位|标的|股票|证券|证券商|酒企|券商|主体|对象|简称)$"
)
# "最后确认的公司" names the last *successfully* resolved subject, so a failed (clarify) final
# turn must not block binding it -- unlike a bare pronoun, which stays ambiguous after failure.
EXPLICIT_CONFIRMED_SUBJECT = re.compile(r"最后(?:确认|确定)")


def _direct_routes(query: str, companies: Sequence[Company]) -> tuple[str, ...] | None:
    """Recognize complete simple requests, not an arbitrary keyword-bearing sentence."""
    masked = query
    for hit in reversed(_find_hits(query, companies)):
        masked = masked[:hit.start] + "@" + masked[hit.end:]
    masked = re.sub(r"\s+", "", masked)
    terms = {word for words in KEYWORDS.values() for word in words}
    terms.discard("披露")  # A reporting verb need not request an event query.
    terms.update(("股权结构", "持股比例", *UNSUPPORTED_KEYWORDS))
    target = "(?:" + "|".join(re.escape(word) for word in sorted(terms, key=lambda x: (-len(x), x))) + ")"
    subject = r"(?:@(?:[和与、]@)*|它们|它|该公司|这家公司|这两家|上述公司)"
    pattern = (
        rf"(?:请)?(?:查询|查一下|查)?{subject}(?:的)?(?:20\d{{2}}年(?:的)?)?"
        rf"(?:最近|近期)?(?:披露了哪些)?(?P<targets>{target}(?:(?:和|以及|、){target})*)"
        r"(?:分别)?(?:是多少|是谁|是什么|有哪些|怎么样)?[？?。]*"
    )
    match = re.fullmatch(pattern, masked)
    return _detect_routes(match["targets"]) if match else None


@dataclass(frozen=True)
class HybridRouteResult:
    decision: Decision | None
    recall: RecallResult
    detected_routes: tuple[str, ...]
    verifier_violations: tuple[Violation, ...]
    audit_reason: str | None
    latency_ms: float
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    llm_calls: int = 0
    model: str | None = None
    system_fingerprint: str | None = None
    raw_content: str | None = None
    error: str | None = None


def _fixed_fallback(action: str | None, company_ids: tuple[str, ...]) -> Decision:
    if action in {"new_entity", "reuse"} and company_ids:
        return Decision(action, company_ids, ("clarify",))
    return Decision("clarify", (), ("clarify",))


class HybridRouter:
    def __init__(
        self,
        companies: Sequence[Company],
        model: DecisionModel,
        *,
        use_verifier: bool,
    ):
        self._companies = tuple(companies)
        self._company_by_id = {company.company_id: company for company in companies}
        self._known_ids = tuple(self._company_by_id)
        self._model = model
        self._use_verifier = use_verifier

    def route(self, payload: Mapping[str, Any]) -> HybridRouteResult:
        started = time.perf_counter_ns()
        query = str(payload.get("query", ""))
        history = tuple(payload.get("history", []))
        recall = recall_candidates(query, self._companies)
        direct_routes = _direct_routes(query, self._companies)
        detected_routes = direct_routes or ()
        history_ids = _history_company_ids(history)
        recent_ids = _recent_subject(history)
        comparison = bool(COMPARISON_PATTERN.search(query))
        unsupported = direct_routes is not None and any(
            keyword in query for keyword in UNSUPPORTED_KEYWORDS
        )

        # Only a genuine unknown entity blocks routing. A referential mention (这家银行 / 该公司)
        # adds no new entity, so it must not trigger fail-closed when a company was resolved.
        # ``fullmatch`` is required: the mention must be *entirely* a demonstrative + common noun,
        # otherwise "该公司科技" (an unknown company name) would be swallowed too.
        blocking_unknowns = tuple(
            mention for mention in recall.unknown_mentions
            if not REFERENTIAL_MENTION.fullmatch(mention)
        )
        if blocking_unknowns:
            return self._direct_result(
                _fixed_fallback(None, ()), recall, detected_routes, "unknown_entity", started,
                history_ids,
            )

        explicit_ids = list(recall.resolved_ids)
        fixed_action: str | None = None
        fixed_ids: tuple[str, ...] = ()
        subject_is_ambiguous = bool(recall.ambiguous_mentions)
        subject_scope_uncertain = direct_routes is None and bool(SUBJECT_SCOPE_CUES.search(query))
        latest_unresolved = bool(history) and (
            history[-1].get("accepted_decision", {}).get("subject_action") == "clarify"
        )
        # A failed final turn blocks a bare pronoun ("它"), but not an explicit reference to the
        # last *confirmed* subject -- that phrase exists precisely to skip the failed turn.
        confirmed_subject_reference = latest_unresolved and bool(
            EXPLICIT_CONFIRMED_SUBJECT.search(query)
        )
        unresolved_reference = not explicit_ids and not subject_is_ambiguous and (
            (latest_unresolved and not confirmed_subject_reference)
            or (len(recent_ids) > 1 and not PLURAL_REFERENCE.search(query))
        )
        subject_must_clarify = (
            not subject_scope_uncertain and unresolved_reference and bool(REUSE_PATTERN.search(query))
        ) or (
            subject_is_ambiguous and not subject_scope_uncertain and (
                latest_unresolved or any(
                    len(set(mention.candidate_ids).intersection(recent_ids)) != 1
                    for mention in recall.ambiguous_mentions
                )
            )
        )

        if explicit_ids and not subject_is_ambiguous and not subject_scope_uncertain:
            if comparison and direct_routes is None and len(explicit_ids) == 1:
                explicit_ids.extend(item for item in recent_ids if item not in explicit_ids)
            fixed_action = "new_entity"
            fixed_ids = tuple(explicit_ids)
        elif (
            not subject_is_ambiguous and not subject_scope_uncertain
            and not unresolved_reference and recent_ids
            and (REUSE_PATTERN.search(query) or detected_routes)
        ):
            fixed_action = "reuse"
            fixed_ids = recent_ids
        elif not recall.candidate_ids and not history_ids:
            return self._direct_result(
                _fixed_fallback(None, ()), recall, detected_routes, "subject_missing", started,
                history_ids,
            )

        if unsupported:
            reason = "subject_ambiguous" if subject_is_ambiguous else "unsupported_business"
            return self._direct_result(
                _fixed_fallback(fixed_action, fixed_ids), recall, detected_routes, reason, started,
                history_ids,
            )

        if fixed_ids and detected_routes and direct_routes is not None:
            if len(fixed_ids) > 1 and len(detected_routes) > 1:
                decision = Decision(fixed_action, fixed_ids, ("clarify",))
                reason = "unsupported_composition"
            else:
                decision = Decision(fixed_action, fixed_ids, detected_routes)
                reason = None
            return self._direct_result(
                decision, recall, detected_routes, reason, started, history_ids
            )

        allowed_ids = (
            fixed_ids
            if fixed_ids
            else tuple(dict.fromkeys((*recall.candidate_ids, *history_ids)))
        )
        request = ModelRequest(
            HYBRID_SYSTEM_PROMPT,
            {
                "query": query,
                "history": list(history),
                "companies": [
                    self._company_by_id[company_id].to_router_dict()
                    for company_id in allowed_ids
                    if company_id in self._company_by_id
                ],
                "allowed_company_ids": list(allowed_ids),
                "fixed_subject": (
                    {
                        "subject_action": fixed_action,
                        "company_ids": list(fixed_ids),
                    }
                    if fixed_action
                    else None
                ),
                "detected_routes": list(detected_routes),
                "subject_must_clarify": subject_must_clarify,
            },
        )

        try:
            reply = self._model.complete(request)
        except Exception as error:  # provider failures are experiment outcomes
            message = f"model_error: {type(error).__name__}: {error}"
            return HybridRouteResult(
                decision=None,
                recall=recall,
                detected_routes=detected_routes,
                verifier_violations=(),
                audit_reason="model_error",
                input_tokens=None,
                output_tokens=None,
                latency_ms=(time.perf_counter_ns() - started) / 1_000_000,
                llm_calls=getattr(error, "attempts", 1),
                error=message,
            )

        try:
            # parse_decision already normalises the business-route order at the parse
            # boundary; do not normalise a second time here.
            preliminary = parse_decision(reply.content)
        except (TypeError, ValueError) as error:
            message = f"invalid_output: {type(error).__name__}: {error}"
            return self._invalid_model_result(
                message,
                recall,
                detected_routes,
                fixed_action,
                fixed_ids,
                started,
                input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens,
                model=reply.model,
                system_fingerprint=reply.system_fingerprint,
                raw_content=reply.content,
            )

        context = self._verification_context(
            allowed_ids=allowed_ids,
            required_ids=fixed_ids,
            required_routes=detected_routes,
            history_ids=history_ids,
            required_action=fixed_action,
            subject_must_clarify=subject_must_clarify,
        )
        violations = verify_decision(preliminary, context) if self._use_verifier else ()
        if violations:
            final = fail_closed(preliminary, violations, context)
            if fixed_ids and final.subject_action == "clarify":
                final = _fixed_fallback(fixed_action, fixed_ids)
            reason = "invalid_output"
        else:
            final = preliminary
            reason = None

        return HybridRouteResult(
            decision=final,
            recall=recall,
            detected_routes=detected_routes,
            verifier_violations=violations,
            audit_reason=reason,
            latency_ms=(time.perf_counter_ns() - started) / 1_000_000,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            llm_calls=1,
            model=reply.model,
            system_fingerprint=reply.system_fingerprint,
            raw_content=reply.content,
        )

    def _direct_result(
        self,
        preliminary: Decision,
        recall: RecallResult,
        detected_routes: tuple[str, ...],
        audit_reason: str | None,
        started: int,
        history_ids: tuple[str, ...],
    ) -> HybridRouteResult:
        allowed_ids = tuple(dict.fromkeys((*recall.candidate_ids, *preliminary.company_ids)))
        context = self._verification_context(
            allowed_ids=allowed_ids,
            required_ids=preliminary.company_ids if preliminary.subject_action != "clarify" else (),
            required_routes=(
                detected_routes if preliminary.route != ("clarify",) else ()
            ),
            history_ids=history_ids,
        )
        violations = verify_decision(preliminary, context) if self._use_verifier else ()
        final = fail_closed(preliminary, violations, context) if violations else preliminary
        return HybridRouteResult(
            decision=final,
            recall=recall,
            detected_routes=detected_routes,
            verifier_violations=violations,
            audit_reason=audit_reason,
            latency_ms=(time.perf_counter_ns() - started) / 1_000_000,
        )

    def _verification_context(
        self,
        *,
        allowed_ids: tuple[str, ...],
        required_ids: tuple[str, ...],
        required_routes: tuple[str, ...],
        history_ids: tuple[str, ...],
        required_action: str | None = None,
        subject_must_clarify: bool = False,
    ) -> VerificationContext:
        return VerificationContext(
            known_company_ids=self._known_ids,
            allowed_company_ids=allowed_ids,
            required_company_ids=required_ids,
            required_routes=required_routes,
            history_company_ids=history_ids,
            required_subject_action=required_action,
            subject_must_clarify=subject_must_clarify,
        )

    def _invalid_model_result(
        self,
        error: str,
        recall: RecallResult,
        detected_routes: tuple[str, ...],
        fixed_action: str | None,
        fixed_ids: tuple[str, ...],
        started: int,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        model: str | None = None,
        system_fingerprint: str | None = None,
        raw_content: str | None = None,
    ) -> HybridRouteResult:
        decision = _fixed_fallback(fixed_action, fixed_ids) if self._use_verifier else None
        violations = (
            (Violation("invalid_output", "model output could not be used"),)
            if self._use_verifier
            else ()
        )
        return HybridRouteResult(
            decision=decision,
            recall=recall,
            detected_routes=detected_routes,
            verifier_violations=violations,
            audit_reason="invalid_output",
            latency_ms=(time.perf_counter_ns() - started) / 1_000_000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            llm_calls=1,
            model=model,
            system_fingerprint=system_fingerprint,
            raw_content=raw_content,
            error=error,
        )


__all__ = ["HybridRouteResult", "HybridRouter"]
