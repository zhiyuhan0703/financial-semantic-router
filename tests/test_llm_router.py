import unittest

from financial_router.llm_router import (
    LLM_ONLY_SYSTEM_PROMPT,
    LLMOnlyRouter,
    ModelReply,
)
from financial_router.hybrid_router import HYBRID_SYSTEM_PROMPT


class ScriptedModel:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return self.replies.pop(0)


class LLMOnlyRouterTests(unittest.TestCase):
    def test_both_model_routes_receive_the_same_business_definitions(self):
        for prompt in (LLM_ONLY_SYSTEM_PROMPT, HYBRID_SYSTEM_PROMPT):
            for definition in (
                "financial：财报项目", "equity：股东", "event：公司公告",
                "clarify：主体歧义", "不把股价、估值、行情归入 equity",
                "被明确排除的公司不进入 company_ids",
            ):
                with self.subTest(definition=definition):
                    self.assertIn(definition, prompt)
        self.assertIn("subject_must_clarify", HYBRID_SYSTEM_PROMPT)

    def test_valid_json_uses_one_call_and_preserves_usage(self):
        model = ScriptedModel(
            ModelReply(
                content=(
                    '{"subject_action":"new_entity",'
                    '"company_ids":["SH600519"],"route":["financial"]}'
                ),
                input_tokens=120,
                output_tokens=18,
                model="fake-model",
                system_fingerprint="fake-fingerprint",
            )
        )
        router = LLMOnlyRouter(model)
        payload = {
            "query": "贵州茅台的营业收入是多少？",
            "history": [],
            "companies": [{"company_id": "SH600519", "legal_name": "贵州茅台酒股份有限公司"}],
            "expected": {"must_not": "leak"},
            "provenance": {"must_not": "leak"},
        }

        result = router.route(payload)

        self.assertEqual(
            result.decision.to_dict(),
            {
                "subject_action": "new_entity",
                "company_ids": ["SH600519"],
                "route": ["financial"],
            },
        )
        self.assertEqual(result.llm_calls, 1)
        self.assertEqual(result.input_tokens, 120)
        self.assertEqual(result.output_tokens, 18)
        self.assertEqual(result.model, "fake-model")
        self.assertEqual(result.system_fingerprint, "fake-fingerprint")
        self.assertGreaterEqual(result.latency_ms, 0.0)
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(set(model.requests[0].user_payload), {"query", "history", "companies"})
        self.assertIn("JSON", model.requests[0].system_prompt)

    def test_invalid_json_is_recorded_as_missing_prediction(self):
        router = LLMOnlyRouter(ScriptedModel(ModelReply(content="not-json")))

        result = router.route({"query": "平安怎么样？", "history": [], "companies": []})

        self.assertIsNone(result.decision)
        self.assertEqual(result.llm_calls, 1)
        self.assertIn("invalid_output", result.error)

    def test_semantically_invalid_but_structured_output_is_not_verified(self):
        router = LLMOnlyRouter(
            ScriptedModel(
                ModelReply(
                    content=(
                        '{"subject_action":"invented",'
                        '"company_ids":["NOT_IN_TABLE"],"route":["unknown"]}'
                    )
                )
            )
        )

        result = router.route({"query": "随便问", "history": [], "companies": []})

        self.assertEqual(result.decision.subject_action, "invented")
        self.assertEqual(result.decision.company_ids, ("NOT_IN_TABLE",))
        self.assertEqual(result.decision.route, ("unknown",))
        self.assertIsNone(result.error)

    def test_provider_failure_is_recorded_without_raising(self):
        class FailingModel:
            def complete(self, request):
                raise TimeoutError("deadline exceeded")

        result = LLMOnlyRouter(FailingModel()).route(
            {"query": "贵州茅台怎么样？", "history": [], "companies": []}
        )

        self.assertIsNone(result.decision)
        self.assertEqual(result.llm_calls, 1)
        self.assertIn("model_error: TimeoutError", result.error)
        self.assertIsNone(result.input_tokens)
        self.assertIsNone(result.output_tokens)


if __name__ == "__main__":
    unittest.main()
