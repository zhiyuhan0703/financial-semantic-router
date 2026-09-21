import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "pure_rule-public-v1.0.json"


class PublicBaselineReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report_text = REPORT.read_text(encoding="utf-8")
        cls.report = json.loads(cls.report_text)

    def test_report_identity_and_denominator(self):
        self.assertEqual(
            self.report["freeze_id"],
            "financial-router-public-v1.0",
        )
        self.assertEqual(
            self.report["manifest_path"],
            "frozen/public-v1.0/freeze-manifest.json",
        )
        self.assertEqual(self.report["route"], "pure_rule")
        self.assertEqual(self.report["split"], "test")
        self.assertEqual(len(self.report["runs"]), 1)
        self.assertEqual(
            self.report["runs"][0]["metrics"]["counts"]["total"],
            24,
        )

    def test_report_has_no_model_use_or_absolute_windows_path(self):
        runtime = self.report["runs"][0]["metrics"]["runtime"]
        self.assertEqual(runtime["llm_call_rate"], 0)
        self.assertIsNone(self.report["configuration"]["model"])
        self.assertNotIn(":\\", self.report_text)


if __name__ == "__main__":
    unittest.main()
