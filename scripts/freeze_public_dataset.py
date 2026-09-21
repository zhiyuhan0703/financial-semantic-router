"""Build the byte-size and SHA-256 manifest for the public synthetic freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "frozen" / "public-v1.0"
FILES = (
    ("cases_and_nested_gold", "cases.jsonl"),
    ("static_company_candidates", "companies.json"),
    ("data_origin_declaration", "sources.json"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    rows = [
        json.loads(line)
        for line in (FREEZE / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dev_ids = [row["case_id"] for row in rows if row["split"] == "dev"]
    test_ids = [row["case_id"] for row in rows if row["split"] == "test"]
    manifest = {
        "freeze_id": "financial-router-public-v1.0",
        "frozen_at": "2026-09-21",
        "freeze_scope": "Newly authored public synthetic routing cases",
        "contract_version": "1.0",
        "dev_denominator": len(dev_ids),
        "test_denominator": len(test_ids),
        "dev_case_ids": dev_ids,
        "test_case_ids": test_ids,
        "router_input_fields": ["query", "history", "companies"],
        "excluded_from_router": ["expected", "case_family", "source_type"],
        "files": [],
    }
    for role, name in FILES:
        path = FREEZE / name
        manifest["files"].append(
            {
                "role": role,
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    (FREEZE / "freeze-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
