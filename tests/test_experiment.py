import json
import unittest
from pathlib import Path

from financial_router.data import load_frozen_dataset
from financial_router.experiment import run_experiment
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


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_frozen_dataset(MANIFEST)

    def test_pure_rule_dev_report_uses_shared_scorer(self):
        report = run_experiment(
            route_name="pure_rule",
            split="dev",
            cases=self.dataset.dev_cases,
            companies=self.dataset.companies,
            repeats=1,
        )

        self.assertEqual(report["route"], "pure_rule")
        self.assertEqual(report["split"], "dev")
        self.assertEqual(len(report["runs"]), 1)
        self.assertEqual(report["runs"][0]["metrics"]["counts"]["total"], 8)
        self.assertEqual(report["runs"][0]["metrics"]["runtime"]["llm_call_rate"], 0)
        self.assertEqual(len(report["runs"][0]["cases"]), 8)

    def test_llm_only_report_keeps_gold_out_of_model_request(self):
        case = self.dataset.dev_cases[0]
        model = ScriptedModel(
            ModelReply(
                content=json.dumps(case.expected.to_dict()),
                input_tokens=10,
                output_tokens=5,
            )
        )

        report = run_experiment(
            route_name="llm_only",
            split="dev",
            cases=(case,),
            companies=self.dataset.companies,
            repeats=1,
            model=model,
        )

        self.assertTrue(report["runs"][0]["cases"][0]["correct"])
        self.assertEqual(
            set(model.requests[0].user_payload), {"query", "history", "companies"}
        )
        self.assertEqual(report["runs"][0]["metrics"]["runtime"]["llm_call_rate"], 1)

    def test_llm_route_requires_a_model(self):
        with self.assertRaisesRegex(ValueError, "model"):
            run_experiment(
                route_name="hybrid",
                split="dev",
                cases=self.dataset.dev_cases[:1],
                companies=self.dataset.companies,
                repeats=1,
            )

    def test_duplicate_case_ids_are_rejected_before_any_model_call(self):
        case = self.dataset.dev_cases[0]
        model = ScriptedModel(
            ModelReply(json.dumps(case.expected.to_dict())),
            ModelReply(json.dumps(case.expected.to_dict())),
        )
        with self.assertRaisesRegex(ValueError, "duplicate case_id"):
            run_experiment(
                route_name="llm_only", split="dev", cases=(case, case),
                companies=self.dataset.companies, repeats=1, model=model,
            )
        self.assertEqual(model.requests, [])

    def test_unknown_route_and_nonpositive_repeats_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "route"):
            run_experiment(
                route_name="invented",
                split="dev",
                cases=self.dataset.dev_cases[:1],
                companies=self.dataset.companies,
                repeats=1,
            )
        with self.assertRaisesRegex(ValueError, "repeats"):
            run_experiment(
                route_name="pure_rule",
                split="dev",
                cases=self.dataset.dev_cases[:1],
                companies=self.dataset.companies,
                repeats=0,
            )


if __name__ == "__main__":
    unittest.main()
