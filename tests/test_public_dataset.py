import json
import unittest
from pathlib import Path

from financial_router.contract import BUSINESS_ROUTES, ROUTES, SUBJECT_ACTIONS
from financial_router.data import load_frozen_dataset


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "frozen" / "public-v1.0" / "freeze-manifest.json"
CASES = ROOT / "frozen" / "public-v1.0" / "cases.jsonl"


class PublicDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_frozen_dataset(MANIFEST)
        cls.rows = [
            json.loads(line)
            for line in CASES.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_public_freeze_has_declared_shape(self):
        self.assertEqual(self.dataset.freeze_id, "financial-router-public-v1.0")
        self.assertEqual(len(self.dataset.companies), 5)
        self.assertEqual(len(self.dataset.dev_cases), 8)
        self.assertEqual(len(self.dataset.test_cases), 24)
        self.assertEqual(
            [case.case_id for case in self.dataset.dev_cases],
            [f"DEV-{number:02d}" for number in range(1, 9)],
        )
        self.assertEqual(
            [case.case_id for case in self.dataset.test_cases],
            [f"TEST-{number:02d}" for number in range(1, 25)],
        )

    def test_every_case_is_new_public_synthetic_data(self):
        self.assertEqual(len(self.rows), 32)
        for row in self.rows:
            self.assertEqual(row["source_type"], "public_synthetic")
            self.assertIn(row["case_family"], {f"F{number}" for number in range(1, 9)})
            self.assertIn("expected", row)
            self.assertNotIn("provenance", row)
            self.assertNotIn("annotation", row)

    def test_gold_contract_uses_only_supported_values(self):
        for row in self.rows:
            expected = row["expected"]
            self.assertIn(expected["subject_action"], SUBJECT_ACTIONS)
            self.assertTrue(expected["route"])
            self.assertTrue(set(expected["route"]).issubset(ROUTES))
            if "clarify" in expected["route"]:
                self.assertEqual(expected["route"], ["clarify"])
            else:
                self.assertEqual(
                    expected["route"],
                    [route for route in BUSINESS_ROUTES if route in expected["route"]],
                )

    def test_company_ids_are_known_or_empty(self):
        known = {company.company_id for company in self.dataset.companies}
        for row in self.rows:
            self.assertTrue(set(row["expected"]["company_ids"]).issubset(known))

    def test_router_input_excludes_dataset_metadata(self):
        case = self.dataset.test_cases[0]
        payload = case.router_input(self.dataset.companies)
        rendered = json.dumps(payload, ensure_ascii=False)
        self.assertEqual(set(payload), {"query", "history", "companies"})
        self.assertNotIn("source_type", rendered)
        self.assertNotIn("case_family", rendered)
        self.assertNotIn("expected", rendered)


if __name__ == "__main__":
    unittest.main()
