import unittest
from pathlib import Path

from financial_router.candidate_recall import recall_candidates
from financial_router.data import load_frozen_dataset


def _repository_root() -> Path:
    """Locate the repository root by marker, independent of the current directory."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "financial_router" / "data.py").is_file():
            return candidate
    raise RuntimeError("repository root not found from %s" % __file__)


MANIFEST = _repository_root() / "frozen" / "public-v1.0" / "freeze-manifest.json"


class CandidateRecallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.companies = load_frozen_dataset(MANIFEST).companies

    def test_official_short_name_resolves_one_company(self):
        result = recall_candidates("贵州茅台2025年的营业收入是多少？", self.companies)

        self.assertEqual(result.resolved_ids, ("SH600519",))
        self.assertEqual(result.ambiguous_mentions, ())

    def test_shared_alias_is_retained_as_two_way_ambiguity(self):
        result = recall_candidates("平安的股权结构怎么样？", self.companies)

        self.assertEqual(result.resolved_ids, ())
        self.assertEqual(result.ambiguous_mentions[0].mention, "平安")
        self.assertEqual(
            set(result.ambiguous_mentions[0].candidate_ids),
            {"SH601318", "SZ000001"},
        )

    def test_longest_name_suppresses_contained_shared_alias(self):
        result = recall_candidates("平安银行的第一大股东是谁？", self.companies)

        self.assertEqual(result.resolved_ids, ("SZ000001",))
        self.assertEqual(result.ambiguous_mentions, ())

    def test_multiple_explicit_companies_keep_text_order(self):
        result = recall_candidates("茅台和美的集团最近分别披露了什么？", self.companies)

        self.assertEqual(result.resolved_ids, ("SH600519", "SZ000333"))

    def test_security_code_is_a_deterministic_identity_surface(self):
        result = recall_candidates("600036 的控股股东是谁？", self.companies)

        self.assertEqual(result.resolved_ids, ("SH600036",))

    def test_unmatched_company_like_name_is_reported_without_binding(self):
        result = recall_candidates("火星科技的控股股东是谁？", self.companies)

        self.assertEqual(result.resolved_ids, ())
        self.assertEqual(result.unknown_mentions, ("火星科技",))


if __name__ == "__main__":
    unittest.main()
