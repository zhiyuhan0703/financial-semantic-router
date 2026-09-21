"""Hash-verified loading of a frozen routing dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from financial_router.contract import Case, Company


@dataclass(frozen=True)
class FrozenDataset:
    freeze_id: str
    manifest_path: Path
    companies: tuple[Company, ...]
    dev_cases: tuple[Case, ...]
    test_cases: tuple[Case, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _verify_files(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    by_role: dict[str, Path] = {}
    for item in manifest["files"]:
        path = (manifest_path.parent / item["path"]).resolve()
        if not path.is_file():
            raise ValueError(f"frozen file is missing: {path}")
        if path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"frozen byte size mismatch: {path}")
        if _sha256(path) != str(item["sha256"]).upper():
            raise ValueError(f"frozen SHA-256 mismatch: {path}")
        by_role[str(item["role"])] = path
    return by_role


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    cases.append(Case.from_mapping(json.loads(line)))
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"invalid case at {path}:{line_number}: {error}") from error
    return cases


def _select_cases(cases: list[Case], ids: list[str], label: str) -> tuple[Case, ...]:
    by_id = {case.case_id: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("duplicate case_id in frozen cases")
    missing = [case_id for case_id in ids if case_id not in by_id]
    if missing:
        raise ValueError(f"missing {label} case ids: {missing}")
    return tuple(by_id[case_id] for case_id in ids)


def load_frozen_dataset(manifest_path: str | Path) -> FrozenDataset:
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path)
    files = _verify_files(manifest_path, manifest)

    cases_path = files.get("cases_and_nested_gold")
    companies_path = files.get("static_company_candidates")
    if cases_path is None or companies_path is None:
        raise ValueError("freeze manifest lacks cases or company candidates")

    company_document = _read_json(companies_path)
    companies = tuple(Company.from_mapping(item) for item in company_document["companies"])
    cases = _read_cases(cases_path)
    dev_ids = list(manifest.get("dev_case_ids", []))
    test_ids = list(manifest["test_case_ids"])
    if len(test_ids) != int(manifest["test_denominator"]):
        raise ValueError("freeze manifest split counts do not match its declared denominator")
    if "dev_denominator" in manifest and len(dev_ids) != int(manifest["dev_denominator"]):
        raise ValueError("freeze manifest split counts do not match its declared denominator")
    dataset_id = manifest.get("freeze_id", manifest.get("dataset_id"))
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("manifest must declare freeze_id or dataset_id")

    return FrozenDataset(
        freeze_id=dataset_id,
        manifest_path=manifest_path,
        companies=companies,
        dev_cases=_select_cases(cases, dev_ids, "DEV"),
        test_cases=_select_cases(cases, test_ids, "TEST"),
    )


__all__ = ["FrozenDataset", "load_frozen_dataset"]
