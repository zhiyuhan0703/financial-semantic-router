"""Provider-neutral LLM-only routing with deterministic result parsing."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from financial_router.contract import ContractError, Decision


SHARED_ROUTING_GUIDANCE = """模块定义（两条模型路线共用）：
- financial：财报项目、财务指标、指标变化及同一指标的跨公司比较。
- equity：股东、持股比例、实际控制人、控制链和股权关系；不把股价、估值、行情归入 equity。
- event：公司公告、处罚、诉讼、重组和重大经营事项；披露财务数字不必然要求 event。
- clarify：主体歧义、未知实体、意图不足、业务或组合不受支持；不执行真实查询。
行情、估值、技术分析、交易建议、研报等不受支持。实际请求只要含未支持部分，整轮 clarify，不部分执行。
主体动作：当前问题显式给出并成功解析公司时为 new_entity（即使历史提过）；无可信新主体且回指明确时为 reuse；主体无法确认时为 clarify。
公司 ID 去重且保序，当前主要/新引入公司在前、历史比较参照在后。被明确排除的公司不进入 company_ids；否定事件不等于排除公司，不能机械见“不”就删除实体。
共享简称必须有更完整名称或历史证据，不能按常识热度任选候选。多主体后的单数指代、失败轮后的指代均不自动继承；明确回到更早主体可结合完整有限历史判断。
主体明确而意图不足时保留主体，仅 route=["clarify"]；不得对“怎么样”擅自补造业务请求。"""


LLM_ONLY_SYSTEM_PROMPT = """你是金融问答系统的语义路由器。请只输出一个 JSON 对象，且恰好包含：
{"subject_action":"reuse|new_entity|clarify","company_ids":["公司ID"],"route":["financial|equity|event|clarify"]}

规则：
1. 只能使用输入公司表中的 company_id；结合当前问题和最多 10 轮已确认历史判断主体。
2. route 的业务模块顺序固定为 financial、equity、event；clarify 必须独占。
3. 单公司可选择一至三个业务模块；多公司只支持同类单模块，多公司多模块必须保留已确认公司并 route=["clarify"]。
4. 主体歧义或未知实体时 subject_action="clarify"、company_ids=[]、route=["clarify"]。
5. 主体明确但意图不清、业务不支持或组合超范围时，保留主体动作和公司 ID，仅 route=["clarify"]。
不要回答金融问题，不要输出解释、Markdown 或额外字段。""" + "\n\n" + SHARED_ROUTING_GUIDANCE


@dataclass(frozen=True)
class ModelRequest:
    system_prompt: str
    user_payload: Mapping[str, Any]


@dataclass(frozen=True)
class ModelReply:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    system_fingerprint: str | None = None


class DecisionModel(Protocol):
    def complete(self, request: ModelRequest) -> ModelReply: ...


@dataclass(frozen=True)
class LLMRouteResult:
    decision: Decision | None
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    llm_calls: int
    model: str | None
    system_fingerprint: str | None
    raw_content: str | None
    error: str | None = None


def _semantic_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "query": payload.get("query", ""),
        "history": payload.get("history", []),
        "companies": payload.get("companies", []),
    }


def parse_decision(content: str) -> Decision:
    """Parse one model reply into a Decision, normalising the business-route order.

    Route order is a presentation convention, not a semantic field: the contract fixes
    ``financial → equity → event`` precisely so that one module set has a single gold
    sequence. Normalising at this parse boundary keeps that a representation concern
    instead of an answer-correctness requirement, and applies the *same* rule to every
    arm that reads model output (LLM-only and both hybrids).

    Only a permutation of distinct business routes is normalised. Duplicates, unknown
    module names and ``clarify`` mixed with business routes are returned untouched, so
    the verifier still rejects them.
    """
    value = json.loads(content)
    if not isinstance(value, Mapping):
        raise ContractError("model output must be a JSON object")
    return Decision.from_mapping(value).canonicalized()


class LLMOnlyRouter:
    def __init__(self, model: DecisionModel):
        self._model = model

    def route(self, payload: Mapping[str, Any]) -> LLMRouteResult:
        request = ModelRequest(LLM_ONLY_SYSTEM_PROMPT, _semantic_input(payload))
        started = time.perf_counter_ns()
        try:
            reply = self._model.complete(request)
        except Exception as error:  # provider failures are experiment outcomes
            return LLMRouteResult(
                decision=None,
                latency_ms=(time.perf_counter_ns() - started) / 1_000_000,
                input_tokens=None,
                output_tokens=None,
                llm_calls=getattr(error, "attempts", 1),
                model=None,
                system_fingerprint=None,
                raw_content=None,
                error=f"model_error: {type(error).__name__}: {error}",
            )

        try:
            decision = parse_decision(reply.content)
            parse_error = None
        except (json.JSONDecodeError, ContractError, TypeError, ValueError) as error:
            decision = None
            parse_error = f"invalid_output: {type(error).__name__}: {error}"

        return LLMRouteResult(
            decision=decision,
            latency_ms=(time.perf_counter_ns() - started) / 1_000_000,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            llm_calls=1,
            model=reply.model,
            system_fingerprint=reply.system_fingerprint,
            raw_content=reply.content,
            error=parse_error,
        )


__all__ = [
    "DecisionModel",
    "LLMOnlyRouter",
    "LLMRouteResult",
    "ModelReply",
    "ModelRequest",
    "parse_decision",
]
