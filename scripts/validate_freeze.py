"""Load and validate the hash-pinned dataset without producing route output."""

import json
from pathlib import Path

from financial_router.data import load_frozen_dataset


def _repository_root() -> Path:
    """Locate the repository root by marker, independent of the current directory."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "financial_router" / "data.py").is_file():
            return candidate
    raise RuntimeError("repository root not found from %s" % __file__)


MANIFEST = _repository_root() / "frozen" / "public-v1.0" / "freeze-manifest.json"


def main() -> None:
    dataset = load_frozen_dataset(MANIFEST)
    print(
        json.dumps(
            {
                "freeze_id": dataset.freeze_id,
                "companies": len(dataset.companies),
                "dev_cases": len(dataset.dev_cases),
                "test_cases": len(dataset.test_cases),
                "status": "valid",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
