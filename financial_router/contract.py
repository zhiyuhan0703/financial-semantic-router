"""Input and output contracts for the routing experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


OUTPUT_FIELDS = frozenset({"subject_action", "company_ids", "route"})
SUBJECT_ACTIONS = ("reuse", "new_entity", "clarify")
BUSINESS_ROUTES = ("financial", "equity", "event")
ROUTES = BUSINESS_ROUTES + ("clarify",)


class ContractError(ValueError):
    """Raised when a value cannot be represented by the experiment contract."""


def _string_array(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ContractError(f"{field} must be an array of strings")
    return tuple(value)


@dataclass(frozen=True)
class Decision:
    subject_action: str
    company_ids: tuple[str, ...]
    route: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Decision":
        if set(value) != OUTPUT_FIELDS:
            raise ContractError("decision must contain exactly subject_action, company_ids, route")
        action = value["subject_action"]
        if not isinstance(action, str):
            raise ContractError("subject_action must be a string")
        return cls(
            subject_action=action,
            company_ids=_string_array(value["company_ids"], "company_ids"),
            route=_string_array(value["route"], "route"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_action": self.subject_action,
            "company_ids": list(self.company_ids),
            "route": list(self.route),
        }

    def canonicalized(self) -> "Decision":
        """Return the same decision with business routes in the contract's canonical order.

        Route order is a presentation convention, not a semantic field: the contract fixes
        ``financial → equity → event`` precisely so that one module set has a single gold
        sequence. Normalising here keeps that convention a representation concern instead of
        an answer-correctness requirement, and applies identically to every arm.
        """
        if (
            len(set(self.route)) != len(self.route)
            or any(item not in BUSINESS_ROUTES for item in self.route)
        ):
            return self
        canonical = tuple(item for item in BUSINESS_ROUTES if item in self.route)
        if canonical == self.route:
            return self
        return Decision(self.subject_action, self.company_ids, canonical)


@dataclass(frozen=True)
class HistoryTurn:
    query: str
    accepted_decision: Decision

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HistoryTurn":
        if set(value) != {"query", "accepted_decision"}:
            raise ContractError("history turn must contain exactly query and accepted_decision")
        if not isinstance(value["query"], str):
            raise ContractError("history query must be a string")
        return cls(
            query=value["query"],
            accepted_decision=Decision.from_mapping(value["accepted_decision"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "accepted_decision": self.accepted_decision.to_dict(),
        }


@dataclass(frozen=True)
class Alias:
    name: str
    alias_type: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Alias":
        return cls(name=str(value["name"]), alias_type=str(value["alias_type"]))

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "alias_type": self.alias_type}


@dataclass(frozen=True)
class Company:
    company_id: str
    exchange: str
    security_code: str
    legal_name: str
    security_short_name: str
    aliases: tuple[Alias, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Company":
        return cls(
            company_id=str(value["company_id"]),
            exchange=str(value["exchange"]),
            security_code=str(value["security_code"]),
            legal_name=str(value["legal_name"]),
            security_short_name=str(value["security_short_name"]),
            aliases=tuple(Alias.from_mapping(item) for item in value.get("aliases", [])),
        )

    def to_router_dict(self) -> dict[str, Any]:
        return {
            "company_id": self.company_id,
            "exchange": self.exchange,
            "security_code": self.security_code,
            "legal_name": self.legal_name,
            "security_short_name": self.security_short_name,
            "aliases": [alias.to_dict() for alias in self.aliases],
        }


@dataclass(frozen=True)
class Case:
    case_id: str
    split: str
    query: str
    history: tuple[HistoryTurn, ...]
    expected: Decision

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Case":
        query = value.get("query")
        history = value.get("history")
        expected = value.get("expected")
        if not isinstance(query, str):
            raise ContractError("case query must be a string")
        if not isinstance(history, list) or len(history) > 10:
            raise ContractError("case history must be an array with at most 10 turns")
        if not isinstance(expected, Mapping):
            raise ContractError("case gold must be nested under expected")
        return cls(
            case_id=str(value["case_id"]),
            split=str(value["split"]),
            query=query,
            history=tuple(HistoryTurn.from_mapping(item) for item in history),
            expected=Decision.from_mapping(expected),
        )

    def router_input(self, companies: Sequence[Company]) -> dict[str, Any]:
        return {
            "query": self.query,
            "history": [turn.to_dict() for turn in self.history],
            "companies": [company.to_router_dict() for company in companies],
        }


__all__ = [
    "Alias",
    "BUSINESS_ROUTES",
    "Case",
    "Company",
    "ContractError",
    "Decision",
    "HistoryTurn",
    "OUTPUT_FIELDS",
    "ROUTES",
    "SUBJECT_ACTIONS",
]
