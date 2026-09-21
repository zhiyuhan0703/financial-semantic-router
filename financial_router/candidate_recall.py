"""Deterministic company candidate recall from the static authority table."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from financial_router.contract import Company


COMPANY_LIKE_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,10}?(?:科技|集团|银行|食品|药业)")


@dataclass(frozen=True)
class AmbiguousMention:
    mention: str
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class RecallResult:
    resolved_ids: tuple[str, ...]
    ambiguous_mentions: tuple[AmbiguousMention, ...]
    unknown_mentions: tuple[str, ...]

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        ids = list(self.resolved_ids)
        for mention in self.ambiguous_mentions:
            ids.extend(mention.candidate_ids)
        return tuple(dict.fromkeys(ids))


@dataclass(frozen=True)
class _SurfaceHit:
    start: int
    end: int
    surface: str
    company_ids: tuple[str, ...]


def _surfaces(companies: Sequence[Company]) -> dict[str, tuple[str, ...]]:
    ordinal = {company.company_id: index for index, company in enumerate(companies)}
    index: dict[str, set[str]] = {}
    for company in companies:
        names = {
            company.company_id,
            company.security_code,
            company.legal_name,
            company.security_short_name,
            *(alias.name for alias in company.aliases),
        }
        for name in names:
            if name:
                index.setdefault(name, set()).add(company.company_id)
    return {
        surface: tuple(sorted(ids, key=ordinal.__getitem__))
        for surface, ids in index.items()
    }


def _find_hits(query: str, companies: Sequence[Company]) -> list[_SurfaceHit]:
    hits: list[_SurfaceHit] = []
    for surface, company_ids in _surfaces(companies).items():
        for match in re.finditer(re.escape(surface), query, flags=re.IGNORECASE):
            hits.append(_SurfaceHit(match.start(), match.end(), surface, company_ids))

    selected: list[_SurfaceHit] = []
    for hit in sorted(hits, key=lambda item: (-(item.end - item.start), item.start)):
        overlaps = any(hit.start < other.end and other.start < hit.end for other in selected)
        if not overlaps:
            selected.append(hit)
    return sorted(selected, key=lambda item: item.start)


def _unique_in_order(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def recall_candidates(query: str, companies: Sequence[Company]) -> RecallResult:
    hits = _find_hits(query, companies)
    resolved: list[str] = []
    ambiguous: list[AmbiguousMention] = []
    for hit in hits:
        if len(hit.company_ids) == 1:
            resolved.append(hit.company_ids[0])
        else:
            ambiguous.append(AmbiguousMention(hit.surface, hit.company_ids))

    masked = list(query)
    for hit in hits:
        masked[hit.start : hit.end] = " " * (hit.end - hit.start)
    unknown = [match.group(0) for match in COMPANY_LIKE_PATTERN.finditer("".join(masked))]

    return RecallResult(
        resolved_ids=_unique_in_order(resolved),
        ambiguous_mentions=tuple(ambiguous),
        unknown_mentions=_unique_in_order(unknown),
    )


__all__ = ["AmbiguousMention", "RecallResult", "recall_candidates"]
