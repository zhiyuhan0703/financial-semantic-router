import unittest
from pathlib import Path

from financial_router.data import load_frozen_dataset
from financial_router.rule_router import RuleRouter


def _repository_root() -> Path:
    """Locate the repository root by marker, independent of the current directory."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "financial_router" / "data.py").is_file():
            return candidate
    raise RuntimeError("repository root not found from %s" % __file__)


MANIFEST = _repository_root() / "frozen" / "public-v1.0" / "freeze-manifest.json"


class RuleRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_frozen_dataset(MANIFEST)
        cls.router = RuleRouter(cls.dataset.companies)

    def test_all_eight_dev_cases_match_the_frozen_expected_decisions(self):
        mismatches = []
        for case in self.dataset.dev_cases:
            result = self.router.route(case.router_input(self.dataset.companies))
            if result.decision != case.expected:
                mismatches.append(
                    (case.case_id, case.expected.to_dict(), result.decision.to_dict())
                )

        self.assertEqual(mismatches, [])

    def test_known_company_with_vague_intent_keeps_identity_and_clarifies_route(self):
        result = self.router.route(
            {"query": "你认为美的集团怎么样", "history": [], "companies": []}
        )

        self.assertEqual(
            result.decision.to_dict(),
            {
                "subject_action": "new_entity",
                "company_ids": ["SZ000333"],
                "route": ["clarify"],
            },
        )

    def test_one_company_can_route_to_multiple_supported_modules(self):
        result = self.router.route(
            {
                "query": "美的集团的营业收入、第一大股东和重大事项分别是什么？",
                "history": [],
                "companies": [],
            }
        )

        self.assertEqual(
            result.decision.route, ("financial", "equity", "event")
        )

    def test_multi_company_multi_module_is_clarified_without_dropping_identities(self):
        result = self.router.route(
            {
                "query": "贵州茅台营业收入是多少，美的集团第一大股东是谁？",
                "history": [],
                "companies": [],
            }
        )

        self.assertEqual(result.decision.subject_action, "new_entity")
        self.assertEqual(result.decision.company_ids, ("SH600519", "SZ000333"))
        self.assertEqual(result.decision.route, ("clarify",))

    def test_unsupported_market_request_clarifies_even_with_known_company(self):
        result = self.router.route(
            {"query": "贵州茅台现在适合买入吗？", "history": [], "companies": []}
        )

        self.assertEqual(result.decision.company_ids, ("SH600519",))
        self.assertEqual(result.decision.route, ("clarify",))


if __name__ == "__main__":
    unittest.main()
