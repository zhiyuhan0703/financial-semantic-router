"""Run the pure-rule baseline on DEV only and print reproducible diagnostics."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

from financial_router.data import load_frozen_dataset
from financial_router.rule_router import RuleRouter
from financial_router.scoring import PredictionRecord, score_predictions


def _repository_root() -> Path:
    """Locate the repository root by marker, independent of the current directory."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "financial_router" / "data.py").is_file():
            return candidate
    raise RuntimeError("repository root not found from %s" % __file__)


def display_manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(_repository_root()).as_posix()
    except ValueError:
        return str(resolved)


MANIFEST = _repository_root() / "frozen" / "public-v1.0" / "freeze-manifest.json"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=MANIFEST,
        help="Explicit freeze manifest path; defaults to frozen/public-v1.0/freeze-manifest.json",
    )
    args = parser.parse_args(argv)
    dataset = load_frozen_dataset(args.manifest)
    router = RuleRouter(dataset.companies)
    gold = {}
    predictions = {}
    cases = []
    for case in dataset.dev_cases:
        started = perf_counter()
        result = router.route(case.router_input(dataset.companies))
        latency_ms = (perf_counter() - started) * 1000
        gold[case.case_id] = case.expected
        predictions[case.case_id] = PredictionRecord(
            decision=result.decision,
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            llm_calls=0,
        )
        cases.append(
            {
                "case_id": case.case_id,
                "expected": case.expected.to_dict(),
                "predicted": result.decision.to_dict(),
                "correct": result.decision == case.expected,
                "audit_reason": result.audit_reason,
                "verifier_codes": [item.code for item in result.verifier_violations],
            }
        )

    print(
        json.dumps(
            {
                "scope": "DEV only; not a formal TEST result",
                "freeze_id": dataset.freeze_id,
                "manifest_path": display_manifest_path(dataset.manifest_path),
                "route": "pure_rule",
                "cases": cases,
                "metrics": score_predictions(gold, predictions),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
