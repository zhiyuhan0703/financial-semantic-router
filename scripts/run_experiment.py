"""Run one experiment arm with explicit paid-API and TEST gates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from financial_router.data import load_frozen_dataset
from financial_router.deepseek import (
    DeepSeekChatModel, DeepSeekConfig, DeepSeekConfigurationError,
)
from financial_router.experiment import ROUTE_NAMES, run_experiment


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "financial_router" / "data.py").is_file():
            return candidate
    raise RuntimeError(f"repository root not found from {__file__}")


def display_manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(_repository_root()).as_posix()
    except ValueError:
        return str(resolved)


MANIFEST = _repository_root() / "frozen" / "public-v1.0" / "freeze-manifest.json"
LLM_ROUTES = frozenset(ROUTE_NAMES) - {"pure_rule"}
KEY_FILE = _repository_root() / ".env"


def load_model_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Load only this project's key; never search paths or change os.environ."""
    key_name = "DEEPSEEK_API_KEY"
    if source.get(key_name, "").strip():
        return {key_name: source[key_name].strip()}
    try:
        content = KEY_FILE.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError):
        raise DeepSeekConfigurationError("Cannot read the configured key file") from None

    result: dict[str, str] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if not separator or name.strip() != key_name:
            continue
        if key_name in result:
            raise DeepSeekConfigurationError("Duplicate API key assignment in key file")
        value = value.strip()
        if value[:1] in ("'", '"'):
            if len(value) < 2 or value[-1] != value[0]:
                raise DeepSeekConfigurationError("Unmatched quote in key file")
            value = value[1:-1].strip()
        result[key_name] = value
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", required=True, choices=ROUTE_NAMES)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--manifest", type=Path, default=MANIFEST,
        help="Explicit freeze manifest path; defaults to frozen/public-v1.0/freeze-manifest.json",
    )
    parser.add_argument("--runs", type=int)
    parser.add_argument("--model", help="Exact provider model ID; no implicit default")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--max-retries", type=int, choices=(0,), default=0)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--allow-paid-api", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this path")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.split == "test" and not args.allow_test:
        parser.error("TEST is locked; pass --allow-test only after gold review")

    model = None
    if args.route in LLM_ROUTES:
        if not args.allow_paid_api:
            parser.error("real LLM routes require explicit --allow-paid-api")
        required = {
            "--model": args.model,
            "--runs": args.runs,
            "--temperature": args.temperature,
            "--timeout-seconds": args.timeout_seconds,
            "--max-tokens": args.max_tokens,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"real LLM routes require explicit values for {', '.join(missing)}")
        if args.runs <= 0:
            parser.error("--runs must be positive")
        config = DeepSeekConfig(
            model=args.model,
            thinking_enabled=False,
            temperature=args.temperature,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
        )
        source = os.environ if environ is None else environ
        model = DeepSeekChatModel(config, environ=load_model_environment(source))
    else:
        if args.runs is None:
            args.runs = 1

    if args.runs is None or args.runs <= 0:
        parser.error("--runs must be positive")

    dataset = load_frozen_dataset(args.manifest)
    cases = dataset.dev_cases if args.split == "dev" else dataset.test_cases
    report = run_experiment(
        route_name=args.route,
        split=args.split,
        cases=cases,
        companies=dataset.companies,
        repeats=args.runs,
        model=model,
    )
    report["freeze_id"] = dataset.freeze_id
    report["manifest_path"] = display_manifest_path(dataset.manifest_path)
    report["configuration"] = {
        "model": args.model if model is not None else None,
        "thinking_enabled": False if model is not None else None,
        "temperature": args.temperature if model is not None else None,
        "timeout_seconds": args.timeout_seconds if model is not None else None,
        "max_retries": args.max_retries if model is not None else None,
        "max_tokens": args.max_tokens if model is not None else None,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


if __name__ == "__main__":
    main()
