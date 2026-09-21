import json
import tempfile
import unittest
from pathlib import Path

from financial_router.contract import Decision, ContractError
from financial_router.data import load_frozen_dataset


def _repository_root() -> Path:
    """Locate the repository root by marker, independent of the current directory."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "financial_router" / "data.py").is_file():
            return candidate
    raise RuntimeError("repository root not found from %s" % __file__)


MANIFEST = _repository_root() / "frozen" / "public-v1.0" / "freeze-manifest.json"


class DecisionContractTests(unittest.TestCase):
    def test_valid_decision_round_trips_as_three_fields(self):
        decision = Decision.from_mapping(
            {
                "subject_action": "new_entity",
                "company_ids": ["SH600519"],
                "route": ["financial"],
            }
        )

        self.assertEqual(
            decision.to_dict(),
            {
                "subject_action": "new_entity",
                "company_ids": ["SH600519"],
                "route": ["financial"],
            },
        )

    def test_unknown_top_level_field_is_rejected(self):
        with self.assertRaisesRegex(ContractError, "exactly"):
            Decision.from_mapping(
                {
                    "subject_action": "new_entity",
                    "company_ids": ["SH600519"],
                    "route": ["financial"],
                    "reason": "not part of scored output",
                }
            )

    def test_non_array_company_ids_is_rejected(self):
        with self.assertRaisesRegex(ContractError, "company_ids"):
            Decision.from_mapping(
                {
                    "subject_action": "new_entity",
                    "company_ids": "SH600519",
                    "route": ["financial"],
                }
            )


class FrozenDatasetTests(unittest.TestCase):
    def test_manifest_loads_eight_dev_and_twenty_four_test_cases(self):
        dataset = load_frozen_dataset(MANIFEST)

        self.assertEqual(len(dataset.dev_cases), 8)
        self.assertEqual(len(dataset.test_cases), 24)
        self.assertEqual(dataset.test_cases[0].case_id, "TEST-01")
        self.assertEqual(dataset.test_cases[-1].case_id, "TEST-24")
        self.assertEqual(dataset.test_cases[0].expected.to_dict().keys(), {
            "subject_action", "company_ids", "route"
        })

    def test_router_input_exposes_only_semantic_whitelist(self):
        dataset = load_frozen_dataset(MANIFEST)
        case = dataset.test_cases[0]

        payload = case.router_input(dataset.companies)

        self.assertEqual(set(payload), {"query", "history", "companies"})
        serialized = json.dumps(payload, ensure_ascii=False)
        for forbidden in (
            "expected",
            "annotation",
            "provenance",
            "review",
            "dataset_version",
            "source_urls",
            "verified_at",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_modified_source_file_fails_hash_verification(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            changed = root / "cases.jsonl"
            changed.write_text("{}\n", encoding="utf-8")
            manifest["files"] = [
                {
                    "role": "cases_and_nested_gold",
                    "path": "cases.jsonl",
                    "bytes": changed.stat().st_size,
                    "sha256": "0" * 64,
                }
            ]
            local_manifest = root / "freeze-manifest.json"
            local_manifest.write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_frozen_dataset(local_manifest)

    def test_external_manifest_without_dev_cases_uses_dataset_id(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest.pop("freeze_id")
        manifest.pop("dev_case_ids")
        manifest.pop("dev_denominator")
        manifest["dataset_id"] = "external-manifest-test"
        manifest["test_case_ids"] = [manifest["test_case_ids"][0]]
        manifest["test_denominator"] = 1
        for item in manifest["files"]:
            item["path"] = str((MANIFEST.parent / item["path"]).resolve())

        with tempfile.TemporaryDirectory() as directory:
            local_manifest = Path(directory) / "external-manifest.json"
            local_manifest.write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
            )
            dataset = load_frozen_dataset(local_manifest)

        self.assertEqual(dataset.freeze_id, "external-manifest-test")
        self.assertEqual(dataset.dev_cases, ())
        self.assertEqual([case.case_id for case in dataset.test_cases], ["TEST-01"])

    def test_manifest_without_dataset_identity_is_rejected_clearly(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest.pop("freeze_id")
        for item in manifest["files"]:
            item["path"] = str((MANIFEST.parent / item["path"]).resolve())

        with tempfile.TemporaryDirectory() as directory:
            local_manifest = Path(directory) / "identity-missing.json"
            local_manifest.write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "freeze_id or dataset_id"):
                load_frozen_dataset(local_manifest)

    def test_declared_dev_denominator_is_enforced(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["dev_denominator"] = len(manifest["dev_case_ids"]) - 1
        for item in manifest["files"]:
            item["path"] = str((MANIFEST.parent / item["path"]).resolve())

        with tempfile.TemporaryDirectory() as directory:
            local_manifest = Path(directory) / "dev-count-mismatch.json"
            local_manifest.write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "split counts"):
                load_frozen_dataset(local_manifest)


if __name__ == "__main__":
    unittest.main()
