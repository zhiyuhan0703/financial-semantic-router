"""Regression tests for business-route order normalisation and referential mentions.

The behavior under test is:
  * only a permutation of *distinct business routes* may be normalised;
  * duplicates, unknown modules and ``clarify`` mixed with business routes are left
    untouched so the verifier still rejects them;
  * the normalisation lives at each arm's decision boundary, not in the scoring layer;
  * all four arms obey the same rule;
  * a demonstrative + common noun is a referential mention, but a company name that merely
    starts with a demonstrative ("该火星科技") is not.
"""

import re
import unittest
from pathlib import Path

from financial_router.contract import (
    BUSINESS_ROUTES,
    Case,
    Company,
    Decision,
)
from financial_router.experiment import ROUTE_NAMES, run_experiment
from financial_router.hybrid_router import REFERENTIAL_MENTION, HybridRouter
from financial_router.llm_router import ModelReply, parse_decision
from financial_router.rule_router import RuleRouter
from financial_router.verifier import VerificationContext, fail_closed, verify_decision


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "financial_router" / "data.py").is_file():
            return candidate
    raise RuntimeError("repository root not found")


def _decision(route, company_ids=("SZ000001",), subject_action="new_entity"):
    return Decision(subject_action, tuple(company_ids), tuple(route))


def _reply(route, company_ids=("SZ000001",), subject_action="new_entity"):
    payload = {
        "subject_action": subject_action,
        "company_ids": list(company_ids),
        "route": list(route),
    }
    import json

    return ModelReply(content=json.dumps(payload), input_tokens=10, output_tokens=5)


class ScriptedModel:
    """Returns queued replies; records requests so call counts stay assertable."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        if not self.replies:
            raise AssertionError("ScriptedModel ran out of replies")
        return self.replies.pop(0)


class CanonicalRouteOrderUnitTests(unittest.TestCase):
    """The contract helper itself: normalise permutations, leave everything else alone."""

    def test_every_valid_permutation_normalises_to_the_canonical_sequence(self):
        canonical = tuple(BUSINESS_ROUTES)
        self.assertEqual(canonical, ("financial", "equity", "event"))
        permutations = [
            ("financial", "equity", "event"),
            ("financial", "event", "equity"),
            ("equity", "financial", "event"),
            ("equity", "event", "financial"),
            ("event", "financial", "equity"),
            ("event", "equity", "financial"),
        ]
        for route in permutations:
            with self.subTest(route=route):
                self.assertEqual(_decision(route).canonicalized().route, canonical)

    def test_every_valid_subset_normalises_to_canonical_order(self):
        for route in [
            ("equity", "event"),
            ("event", "equity"),
            ("event", "financial"),
            ("equity", "financial"),
            ("financial",),
            ("equity",),
            ("event",),
        ]:
            with self.subTest(route=route):
                canonical = tuple(item for item in BUSINESS_ROUTES if item in route)
                self.assertEqual(_decision(route).canonicalized().route, canonical)

    def test_duplicate_modules_are_not_normalised(self):
        for route in [
            ("financial", "financial"),
            ("equity", "event", "equity"),
            ("financial", "financial", "equity"),
        ]:
            with self.subTest(route=route):
                self.assertEqual(_decision(route).canonicalized().route, route)

    def test_unknown_modules_are_not_normalised(self):
        for route in [
            ("financial", "sentiment"),
            ("sentiment", "equity"),
            ("equity", "financial", "unknown"),
        ]:
            with self.subTest(route=route):
                self.assertEqual(_decision(route).canonicalized().route, route)

    def test_clarify_mixed_with_business_routes_is_not_normalised(self):
        for route in [
            ("clarify", "financial"),
            ("financial", "clarify"),
            ("clarify",),
        ]:
            with self.subTest(route=route):
                self.assertEqual(_decision(route).canonicalized().route, route)

    def test_normalisation_preserves_subject_action_and_company_ids(self):
        original = _decision(("event", "financial"), ("SH600519", "SZ000333"), "reuse")
        normalised = original.canonicalized()
        self.assertEqual(normalised.route, ("financial", "event"))
        self.assertEqual(normalised.company_ids, original.company_ids)
        self.assertEqual(normalised.subject_action, original.subject_action)

    def test_verifier_still_rejects_noncanonical_order_when_it_sees_one(self):
        context = VerificationContext(
            known_company_ids=("SZ000001",),
            allowed_company_ids=("SZ000001",),
        )
        codes = {
            violation.code
            for violation in verify_decision(_decision(("equity", "financial")), context)
        }
        self.assertIn("noncanonical_route_order", codes)
        self.assertEqual(
            {
                violation.code
                for violation in verify_decision(_decision(("financial", "equity")), context)
            },
            set(),
        )

    def test_verifier_still_rejects_duplicate_unknown_and_mixed_clarify(self):
        context = VerificationContext(
            known_company_ids=("SZ000001",),
            allowed_company_ids=("SZ000001",),
        )
        cases = {
            "duplicate_route": ("financial", "financial"),
            "invalid_route_value": ("financial", "sentiment"),
            "clarify_not_exclusive": ("clarify", "financial"),
        }
        for expected_code, route in cases.items():
            with self.subTest(route=route):
                codes = {
                    violation.code
                    for violation in verify_decision(_decision(route), context)
                }
                self.assertIn(expected_code, codes)

    def test_fail_closed_downgrades_a_noncanonical_prediction(self):
        context = VerificationContext(
            known_company_ids=("SZ000001",),
            allowed_company_ids=("SZ000001",),
        )
        decision = _decision(("equity", "financial"))
        violations = verify_decision(decision, context)
        self.assertTrue(violations)
        self.assertEqual(fail_closed(decision, violations, context).route, ("clarify",))


class ParseDecisionBoundaryTests(unittest.TestCase):
    """The parse boundary normalises; nothing downstream needs to."""

    def test_parse_decision_normalises_a_permutation(self):
        parsed = parse_decision(
            '{"subject_action":"new_entity","company_ids":["SZ000001"],'
            '"route":["event","financial"]}'
        )
        self.assertEqual(parsed.route, ("financial", "event"))

    def test_parse_decision_leaves_duplicates_unknown_and_mixed_clarify(self):
        payloads = [
            ('{"subject_action":"new_entity","company_ids":["SZ000001"],'
             '"route":["financial","financial"]}', ("financial", "financial")),
            ('{"subject_action":"new_entity","company_ids":["SZ000001"],'
             '"route":["financial","sentiment"]}', ("financial", "sentiment")),
            ('{"subject_action":"new_entity","company_ids":["SZ000001"],'
             '"route":["clarify","financial"]}', ("clarify", "financial")),
        ]
        for content, expected_route in payloads:
            with self.subTest(route=expected_route):
                self.assertEqual(parse_decision(content).route, expected_route)


class ReferentialMentionBoundaryTests(unittest.TestCase):
    """Demonstrative + common noun is referential; a name starting with one is not.

    The pattern must be applied with ``fullmatch``: the whole mention has to be a demonstrative
    plus a common noun, otherwise "该公司科技" would be swallowed by a prefix match.
    """

    def test_demonstrative_plus_common_noun_is_referential(self):
        for mention in [
            "这家银行",
            "这家公司",
            "该公司",
            "那家公司",
            "上述公司",
            "前述企业",
            "这两家公司",
            "那些股票",
        ]:
            with self.subTest(mention=mention):
                self.assertTrue(REFERENTIAL_MENTION.fullmatch(mention))

    def test_company_names_starting_with_a_demonstrative_are_not_referential(self):
        for mention in [
            "该公司科技",
            "该公司集团",
            "该火星科技",
            "这火星科技",
            "那火星科技",
            "该平安证券",
            "该茅台",
        ]:
            with self.subTest(mention=mention):
                self.assertIsNone(REFERENTIAL_MENTION.fullmatch(mention))

    def test_dollar_anchor_is_what_keeps_names_blocking(self):
        """Regression guard: without the trailing ``$`` a prefix match swallows unknown names.

        The real pattern is anchored, so it is safe with either ``match`` or ``fullmatch``.
        This test pins the *reason* for the anchor by rebuilding the unanchored variant.
        """
        unanchored = re.compile(REFERENTIAL_MENTION.pattern.rstrip("$"))
        self.assertIsNotNone(unanchored.match("该公司科技"), "premise: prefix match is too wide")
        self.assertIsNone(REFERENTIAL_MENTION.match("该公司科技"))
        self.assertIsNone(REFERENTIAL_MENTION.fullmatch("该公司科技"))
        # The anchored pattern and the explicit fullmatch agree on every case.
        for mention in ["该公司", "这家银行", "该公司科技", "该公司集团"]:
            with self.subTest(mention=mention):
                self.assertEqual(
                    bool(REFERENTIAL_MENTION.match(mention)),
                    bool(REFERENTIAL_MENTION.fullmatch(mention)),
                )


class ArmConsistencyTests(unittest.TestCase):
    """All four arms answer one noncanonical query with the same canonical route order."""

    @classmethod
    def setUpClass(cls):
        cls.companies = (
            Company.from_mapping(
                {
                    "company_id": "SZ000001",
                    "exchange": "SZSE",
                    "security_code": "000001",
                    "legal_name": "平安银行股份有限公司",
                    "security_short_name": "平安银行",
                    "aliases": [{"name": "平安", "alias_type": "test_alias"}],
                    "source_urls": [],
                    "verified_at": None,
                }
            ),
        )
        cls.case = Case.from_mapping(
            {
                "case_id": "ORDER-01",
                "split": "dev",
                "query": "平安银行先看近期重大事项，再看第一大股东，最后看经营现金流。",
                "history": [],
                "expected": {
                    "subject_action": "new_entity",
                    "company_ids": ["SZ000001"],
                    "route": ["financial", "equity", "event"],
                },
                "annotation": {},
                "provenance": {},
                "review": {},
                "dataset_version": "route-order-regression",
            }
        )

    def test_rule_arm_is_canonical_even_though_it_has_no_parse_boundary(self):
        result = RuleRouter(self.companies).route(self.case.router_input(self.companies))
        self.assertEqual(result.decision.route, ("financial", "equity", "event"))
        self.assertEqual(result.decision, self.case.expected)

    def test_hybrid_arms_are_canonical_for_a_noncanonical_model_reply(self):
        """The model really is consulted here, so this exercises the reply path, not a fallback."""
        for use_verifier in (False, True):
            with self.subTest(use_verifier=use_verifier):
                model = ScriptedModel(
                    _reply(("event", "financial", "equity"), ("SZ000001",))
                )
                router = HybridRouter(self.companies, model, use_verifier=use_verifier)
                result = router.route(self.case.router_input(self.companies))
                # Assert the model was used exactly once: a silent deterministic fallback would
                # make this test pass without exercising the parse boundary at all.
                self.assertEqual(result.llm_calls, 1)
                self.assertEqual(len(model.requests), 1)
                self.assertEqual(result.decision.route, ("financial", "equity", "event"))
                self.assertEqual(result.decision, self.case.expected)

    def test_all_four_arms_agree_on_the_same_query(self):
        observations = {}
        call_counts = {}
        for route_name in ROUTE_NAMES:
            model = None
            if route_name != "pure_rule":
                model = ScriptedModel(
                    _reply(("event", "financial", "equity"), ("SZ000001",))
                )
            report = run_experiment(
                route_name=route_name,
                split="dev",
                cases=[self.case],
                companies=self.companies,
                repeats=1,
                model=model,
            )
            outcome = report["runs"][0]["cases"][0]
            observations[route_name] = outcome["predicted"]
            call_counts[route_name] = outcome["llm_calls"]

        expected = {
            "subject_action": "new_entity",
            "company_ids": ["SZ000001"],
            "route": ["financial", "equity", "event"],
        }
        for route_name, predicted in observations.items():
            with self.subTest(route=route_name):
                self.assertEqual(predicted, expected)

        # The three model-reading arms each consult the model once; the rule arm cannot.
        self.assertEqual(call_counts["pure_rule"], 0)
        for route_name in ("llm_only", "hybrid_no_verifier", "hybrid"):
            with self.subTest(calls=route_name):
                self.assertEqual(call_counts[route_name], 1)


class ScoringLayerIsPassiveTests(unittest.TestCase):
    """The scoring layer must not rewrite a prediction; only the arm boundary may."""

    @classmethod
    def setUpClass(cls):
        cls.companies = (
            Company.from_mapping(
                {
                    "company_id": "SZ000001",
                    "exchange": "SZSE",
                    "security_code": "000001",
                    "legal_name": "平安银行股份有限公司",
                    "security_short_name": "平安银行",
                    "aliases": [{"name": "平安", "alias_type": "test_alias"}],
                    "source_urls": [],
                    "verified_at": None,
                }
            ),
        )

    def _case(self, case_id, expected_route):
        return Case.from_mapping(
            {
                "case_id": case_id,
                "split": "dev",
                "query": "平安银行先看近期重大事项，再看第一大股东，最后看经营现金流。",
                "history": [],
                "expected": {
                    "subject_action": "new_entity",
                    "company_ids": ["SZ000001"],
                    "route": list(expected_route),
                },
                "annotation": {},
                "provenance": {},
                "review": {},
                "dataset_version": "route-order-regression",
            }
        )

    def test_recorded_prediction_is_exactly_what_the_arm_returned(self):
        """The scoring layer records the arm's decision verbatim -- no repair on the way in."""
        case = self._case("ORDER-02", ("equity", "event"))
        direct = RuleRouter(self.companies).route(case.router_input(self.companies))
        report = run_experiment(
            route_name="pure_rule",
            split="dev",
            cases=[case],
            companies=self.companies,
            repeats=1,
        )
        outcome = report["runs"][0]["cases"][0]
        self.assertEqual(outcome["predicted"], direct.decision.to_dict())

    def test_a_noncanonical_gold_route_is_recorded_verbatim_not_repaired(self):
        noncanonical_gold = ("equity", "event")
        case = self._case("ORDER-03", noncanonical_gold)
        report = run_experiment(
            route_name="pure_rule",
            split="dev",
            cases=[case],
            companies=self.companies,
            repeats=1,
        )
        outcome = report["runs"][0]["cases"][0]
        self.assertEqual(outcome["expected"]["route"], list(noncanonical_gold))
        self.assertNotEqual(outcome["predicted"]["route"], list(noncanonical_gold))
        self.assertFalse(outcome["correct"])

    def test_canonical_order_is_not_treated_as_an_answer_error(self):
        canonical_gold = ("financial", "equity", "event")
        case = self._case("ORDER-04", canonical_gold)
        report = run_experiment(
            route_name="pure_rule",
            split="dev",
            cases=[case],
            companies=self.companies,
            repeats=1,
        )
        outcome = report["runs"][0]["cases"][0]
        self.assertEqual(outcome["predicted"]["route"], list(canonical_gold))
        self.assertTrue(outcome["correct"])


if __name__ == "__main__":
    unittest.main()
