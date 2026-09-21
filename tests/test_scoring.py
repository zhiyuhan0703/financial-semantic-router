import unittest

from financial_router.contract import Decision
from financial_router.scoring import PredictionRecord, score_predictions


def decision(action, company_ids, route):
    return Decision(action, tuple(company_ids), tuple(route))


class ScoringTests(unittest.TestCase):
    def test_scores_exact_sets_clarification_cost_and_runtime_separately(self):
        gold = {
            "C1": decision("new_entity", ["A"], ["financial"]),
            "C2": decision("clarify", [], ["clarify"]),
            "C3": decision("new_entity", ["A"], ["financial", "equity"]),
        }
        predictions = {
            "C1": PredictionRecord(
                decision("new_entity", ["A"], ["financial"]),
                latency_ms=10,
                input_tokens=0,
                output_tokens=0,
                llm_calls=0,
            ),
            "C2": PredictionRecord(
                decision("new_entity", ["B"], ["equity"]),
                latency_ms=20,
                input_tokens=30,
                output_tokens=10,
                llm_calls=1,
            ),
            "C3": PredictionRecord(
                decision("new_entity", ["A"], ["financial"]),
                latency_ms=100,
                input_tokens=60,
                output_tokens=30,
                llm_calls=1,
            ),
        }

        result = score_predictions(gold, predictions)

        self.assertEqual(result["counts"]["total"], 3)
        self.assertEqual(result["subject_action"]["correct"], 2)
        self.assertAlmostEqual(result["subject_action"]["accuracy"], 2 / 3)
        self.assertEqual(result["entity_exact"]["correct"], 2)
        self.assertEqual(result["wrong_binding"]["count"], 1)
        self.assertAlmostEqual(result["wrong_binding"]["rate"], 1 / 3)
        self.assertEqual(result["route_exact"]["correct"], 1)
        self.assertEqual(result["joint_exact"]["correct"], 1)
        self.assertEqual(result["clarify"], {
            "tp": 0, "fp": 0, "fn": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0
        })
        self.assertEqual(result["modules"]["financial"]["tp"], 2)
        self.assertEqual(result["modules"]["financial"]["f1"], 1.0)
        self.assertEqual(result["modules"]["equity"]["tp"], 0)
        self.assertEqual(result["runtime"]["p50_latency_ms"], 20)
        self.assertEqual(result["runtime"]["p95_latency_ms"], 100)
        self.assertEqual(result["runtime"]["average_total_tokens"], 130 / 3)
        self.assertEqual(result["runtime"]["llm_call_rate"], 2 / 3)

    def test_order_mismatch_is_not_exact_but_is_not_a_wrong_company_binding(self):
        gold = {"C1": decision("new_entity", ["A", "B"], ["financial", "equity"])}
        predictions = {
            "C1": PredictionRecord(
                decision("new_entity", ["B", "A"], ["equity", "financial"]),
                latency_ms=1,
            )
        }

        result = score_predictions(gold, predictions)

        self.assertEqual(result["entity_exact"]["correct"], 0)
        self.assertEqual(result["route_exact"]["correct"], 0)
        self.assertEqual(result["wrong_binding"]["count"], 0)

    def test_missing_prediction_counts_as_failure_without_crashing(self):
        gold = {"C1": decision("clarify", [], ["clarify"])}
        predictions = {
            "C1": PredictionRecord(None, latency_ms=50, error="timeout")
        }

        result = score_predictions(gold, predictions)

        self.assertIn("missing_prediction", result["counts"])
        self.assertEqual(result["counts"]["missing_prediction"], 1)
        self.assertEqual(result["entity_exact"]["correct"], 0)
        self.assertEqual(result["joint_exact"]["correct"], 0)
        self.assertEqual(result["clarify"]["fn"], 1)

    def test_unknown_api_usage_is_not_reported_as_zero_cost(self):
        gold = {
            "C1": decision("clarify", [], ["clarify"]),
            "C2": decision("new_entity", ["A"], ["financial"]),
        }
        predictions = {
            "C1": PredictionRecord(None, 50, input_tokens=None, output_tokens=None,
                                   llm_calls=1, error="timeout"),
            "C2": PredictionRecord(gold["C2"], 1),
        }
        runtime = score_predictions(gold, predictions)["runtime"]
        self.assertIsNone(runtime["average_total_tokens"])
        self.assertEqual(runtime["usage_missing_cases"], 1)
        self.assertEqual(runtime["usage_observed_cases"], 1)
        self.assertEqual(runtime["known_input_tokens"], 0)
        self.assertEqual(runtime["known_output_tokens"], 0)

    def test_case_id_sets_must_match(self):
        with self.assertRaisesRegex(ValueError, "case_id"):
            score_predictions(
                {"C1": decision("clarify", [], ["clarify"])},
                {},
            )

    def test_illegal_but_parseable_decision_is_wrong_not_missing(self):
        gold = {"C1": decision("new_entity", ["A"], ["financial"])}
        predicted = decision("invented", ["A"], ["invented"])
        result = score_predictions(gold, {"C1": PredictionRecord(predicted, 1)})
        self.assertIn("missing_prediction", result["counts"])
        self.assertEqual(result["counts"]["missing_prediction"], 0)
        self.assertEqual(result["joint_exact"]["correct"], 0)


if __name__ == "__main__":
    unittest.main()
