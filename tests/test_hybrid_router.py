import json
import unittest
from pathlib import Path

from financial_router.data import load_frozen_dataset
from financial_router.hybrid_router import HybridRouter
from financial_router.llm_router import ModelReply


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "financial_router" / "data.py").is_file():
            return candidate
    raise RuntimeError("repository root not found")


MANIFEST = _repository_root() / "frozen" / "public-v1.0" / "freeze-manifest.json"


class ScriptedModel:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return self.replies.pop(0)


class HybridRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_frozen_dataset(MANIFEST)

    def test_clear_subject_and_intent_take_zero_call_deterministic_path(self):
        model = ScriptedModel()
        router = HybridRouter(self.dataset.companies, model, use_verifier=True)

        result = router.route(
            {"query": "贵州茅台的营业收入是多少？", "history": [], "companies": []}
        )

        self.assertEqual(
            result.decision.to_dict(),
            {
                "subject_action": "new_entity",
                "company_ids": ["SH600519"],
                "route": ["financial"],
            },
        )
        self.assertEqual(result.llm_calls, 0)
        self.assertEqual(model.requests, [])
        self.assertEqual(result.verifier_violations, ())

    def test_bare_ambiguous_alias_cannot_be_guessed_even_inside_candidate_set(self):
        model = ScriptedModel(
            ModelReply(
                content=(
                    '{"subject_action":"new_entity",'
                    '"company_ids":["SZ000001"],"route":["equity"]}'
                ),
                input_tokens=80,
                output_tokens=16,
            )
        )
        router = HybridRouter(self.dataset.companies, model, use_verifier=True)

        result = router.route(
            {"query": "平安的第一大股东是谁？", "history": [], "companies": []}
        )

        self.assertEqual(result.decision.company_ids, ())
        self.assertEqual(result.decision.route, ("clarify",))
        self.assertIn("unsafe_subject_binding", [item.code for item in result.verifier_violations])
        self.assertEqual(result.llm_calls, 1)
        self.assertEqual(
            set(model.requests[0].user_payload["allowed_company_ids"]),
            {"SH601318", "SZ000001"},
        )
        self.assertEqual(
            {item["company_id"] for item in model.requests[0].user_payload["companies"]},
            {"SH601318", "SZ000001"},
        )

    def test_unknown_company_fails_closed_without_calling_model(self):
        model = ScriptedModel()
        router = HybridRouter(self.dataset.companies, model, use_verifier=True)

        result = router.route(
            {"query": "火星科技的控股股东是谁？", "history": [], "companies": []}
        )

        self.assertEqual(
            result.decision.to_dict(),
            {"subject_action": "clarify", "company_ids": [], "route": ["clarify"]},
        )
        self.assertEqual(result.llm_calls, 0)
        self.assertEqual(result.audit_reason, "unknown_entity")

    def test_explicit_intent_reuses_recent_confirmed_subject_without_model(self):
        model = ScriptedModel()
        router = HybridRouter(self.dataset.companies, model, use_verifier=True)
        history = [
            {
                "query": "招商银行的营业收入是多少？",
                "accepted_decision": {
                    "subject_action": "new_entity",
                    "company_ids": ["SH600036"],
                    "route": ["financial"],
                },
            }
        ]

        result = router.route(
            {"query": "它的资产负债率是多少？", "history": history, "companies": []}
        )

        self.assertEqual(
            result.decision.to_dict(),
            {
                "subject_action": "reuse",
                "company_ids": ["SH600036"],
                "route": ["financial"],
            },
        )
        self.assertEqual(result.llm_calls, 0)
        self.assertEqual(result.verifier_violations, ())

    def test_verifier_blocks_company_outside_ambiguous_candidate_set(self):
        model = ScriptedModel(
            ModelReply(
                content=(
                    '{"subject_action":"new_entity",'
                    '"company_ids":["SH600519"],"route":["equity"]}'
                )
            )
        )
        router = HybridRouter(self.dataset.companies, model, use_verifier=True)

        result = router.route(
            {"query": "平安的第一大股东是谁？", "history": [], "companies": []}
        )

        self.assertEqual(
            result.decision.to_dict(),
            {"subject_action": "clarify", "company_ids": [], "route": ["clarify"]},
        )
        self.assertIn(
            "company_outside_allowed_set",
            [item.code for item in result.verifier_violations],
        )

    def test_ambiguous_subject_can_abstain_despite_explicit_route_signal(self):
        model = ScriptedModel(
            ModelReply(
                content=(
                    '{"subject_action":"clarify",'
                    '"company_ids":[],"route":["clarify"]}'
                )
            )
        )
        router = HybridRouter(self.dataset.companies, model, use_verifier=True)

        result = router.route(
            {"query": "平安的第一大股东是谁？", "history": [], "companies": []}
        )

        self.assertEqual(
            result.decision.to_dict(),
            {"subject_action": "clarify", "company_ids": [], "route": ["clarify"]},
        )
        self.assertEqual(result.verifier_violations, ())

    def test_no_verifier_ablation_keeps_same_out_of_set_model_decision(self):
        model = ScriptedModel(
            ModelReply(
                content=(
                    '{"subject_action":"new_entity",'
                    '"company_ids":["SH600519"],"route":["equity"]}'
                )
            )
        )
        router = HybridRouter(self.dataset.companies, model, use_verifier=False)

        result = router.route(
            {"query": "平安的第一大股东是谁？", "history": [], "companies": []}
        )

        self.assertEqual(result.decision.company_ids, ("SH600519",))
        self.assertEqual(result.decision.route, ("equity",))
        self.assertEqual(result.verifier_violations, ())

    def test_invalid_model_output_preserves_deterministic_subject_and_clarifies(self):
        model = ScriptedModel(ModelReply(content="not-json"))
        router = HybridRouter(self.dataset.companies, model, use_verifier=True)

        result = router.route(
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
        self.assertEqual([item.code for item in result.verifier_violations], ["invalid_output"])
        self.assertIn("invalid_output", result.error)

    def test_provider_failure_is_reported_without_synthesizing_a_fallback(self):
        class FailingModel:
            def complete(self, request):
                raise TimeoutError("deadline exceeded")

        router = HybridRouter(self.dataset.companies, FailingModel(), use_verifier=True)

        result = router.route(
            {"query": "你认为美的集团怎么样", "history": [], "companies": []}
        )

        self.assertIsNone(result.decision)
        self.assertEqual(result.verifier_violations, ())
        self.assertEqual(result.audit_reason, "model_error")
        self.assertIn("model_error: TimeoutError", result.error)
        self.assertIsNone(result.input_tokens)
        self.assertIsNone(result.output_tokens)

    def test_model_cannot_change_a_fixed_subject_action_or_order(self):
        for query, fixed_ids, action, model_ids in (
            ("你认为美的集团怎么样", ["SZ000333"], "reuse", ["SZ000333"]),
            ("你认为贵州茅台和美的集团怎么样", ["SH600519", "SZ000333"],
             "new_entity", ["SZ000333", "SH600519"]),
        ):
            with self.subTest(query=query):
                history = [{"query": "先认识这些公司", "accepted_decision": {
                    "subject_action": "new_entity", "company_ids": fixed_ids, "route": ["financial"],
                }}]
                model = ScriptedModel(ModelReply(json.dumps({
                    "subject_action": action, "company_ids": model_ids, "route": ["event"],
                })))
                result = HybridRouter(self.dataset.companies, model, use_verifier=True).route({
                    "query": query, "history": history,
                })
                self.assertEqual(result.decision.subject_action, "new_entity")
                self.assertEqual(result.decision.company_ids, tuple(fixed_ids))
                self.assertEqual(result.decision.route, ("clarify",))
                self.assertTrue(result.verifier_violations)

    def test_negation_and_partially_recognized_intents_are_not_locked(self):
        scenarios = (
            ("不要查贵州茅台，查美的集团的营业收入", ["SZ000333"], ["financial"]),
            ("美的集团的营业收入和谁说了算", ["SZ000333"], ["financial", "equity"]),
            ("不问美的集团的股价，只问第一大股东", ["SZ000333"], ["equity"]),
            ("美的集团披露的营业收入是多少", ["SZ000333"], ["financial"]),
        )
        for query, ids, routes in scenarios:
            with self.subTest(query=query):
                model = ScriptedModel(ModelReply(json.dumps({
                    "subject_action": "new_entity", "company_ids": ids, "route": routes,
                })))
                result = HybridRouter(self.dataset.companies, model, use_verifier=True).route({
                    "query": query, "history": [],
                })
                self.assertEqual(result.llm_calls, 1)
                self.assertEqual(result.decision.company_ids, tuple(ids))
                self.assertEqual(result.decision.route, tuple(routes))
                self.assertEqual(model.requests[0].user_payload["detected_routes"], [])
                if query.startswith(("不要", "不问")):
                    self.assertIsNone(model.requests[0].user_payload["fixed_subject"])

    def test_ordinal_reference_is_not_forced_to_reuse_every_recent_company(self):
        history = [{"query": "贵州茅台和美的集团的营业收入", "accepted_decision": {
            "subject_action": "new_entity", "company_ids": ["SH600519", "SZ000333"],
            "route": ["financial"],
        }}]
        model = ScriptedModel(ModelReply(
            '{"subject_action":"reuse","company_ids":["SH600519"],"route":["equity"]}'
        ))
        result = HybridRouter(self.dataset.companies, model, use_verifier=True).route({
            "query": "前者的第一大股东是谁？", "history": history,
        })
        self.assertEqual(result.llm_calls, 1)
        self.assertIsNone(model.requests[0].user_payload["fixed_subject"])
        self.assertEqual(result.decision.company_ids, ("SH600519",))

    def test_singular_pronoun_with_multiple_or_unresolved_subjects_must_clarify(self):
        accepted = {"query": "贵州茅台和美的集团的营业收入", "accepted_decision": {
            "subject_action": "new_entity", "company_ids": ["SH600519", "SZ000333"],
            "route": ["financial"],
        }}
        unresolved = {"query": "火星科技的营业收入", "accepted_decision": {
            "subject_action": "clarify", "company_ids": [], "route": ["clarify"],
        }}
        for history in ([accepted], [accepted, unresolved]):
            with self.subTest(history_length=len(history)):
                model = ScriptedModel(ModelReply(
                    '{"subject_action":"reuse","company_ids":["SH600519"],"route":["equity"]}'
                ))
                result = HybridRouter(self.dataset.companies, model, use_verifier=True).route({
                    "query": "它的第一大股东是谁？", "history": history,
                })
                self.assertEqual(result.decision.subject_action, "clarify")
                self.assertEqual(result.decision.company_ids, ())

    def test_supported_arrays_and_composition_boundary_match_in_both_ablations(self):
        scenarios = (
            ("美的集团的营业收入和第一大股东和重大事项", ["SZ000333"],
             ["financial", "equity", "event"]),
            ("贵州茅台和美的集团的重大事项有哪些？", ["SH600519", "SZ000333"], ["event"]),
            ("贵州茅台和美的集团的营业收入和第一大股东", ["SH600519", "SZ000333"], ["clarify"]),
            ("美的集团的营业收入和股价", ["SZ000333"], ["clarify"]),
        )
        for use_verifier in (True, False):
            for query, ids, routes in scenarios:
                with self.subTest(query=query, use_verifier=use_verifier):
                    model = ScriptedModel()
                    result = HybridRouter(self.dataset.companies, model, use_verifier=use_verifier).route({
                        "query": query, "history": [],
                    })
                    self.assertIsNotNone(result.decision)
                    self.assertEqual(result.decision.company_ids, tuple(ids))
                    self.assertEqual(result.decision.route, tuple(routes))
                    self.assertEqual(result.llm_calls, 0)

    def test_respectively_in_a_single_company_request_does_not_add_old_subject(self):
        history = [{"query": "贵州茅台的营业收入", "accepted_decision": {
            "subject_action": "new_entity", "company_ids": ["SH600519"], "route": ["financial"],
        }}]
        result = HybridRouter(self.dataset.companies, ScriptedModel(), use_verifier=True).route({
            "query": "美的集团的营业收入和第一大股东分别是什么？", "history": history,
        })
        self.assertIsNotNone(result.decision)
        self.assertEqual(result.decision.company_ids, ("SZ000333",))
        self.assertEqual(result.decision.route, ("financial", "equity"))


if __name__ == "__main__":
    unittest.main()
