import json
import unittest
from pathlib import Path

from financial_router.contract import Decision
from financial_router.verifier import (
    VerificationContext,
    fail_closed,
    verify_decision,
)


FAULTS = Path(__file__).parents[1] / "fixtures" / "verifier_faults.jsonl"


def decision(action, company_ids, route):
    return Decision(action, tuple(company_ids), tuple(route))


class VerifierTests(unittest.TestCase):
    def test_fixed_subject_action_and_company_order_cannot_be_changed(self):
        context = VerificationContext.from_mapping({
            "known_company_ids": ["A", "B"],
            "allowed_company_ids": ["A", "B"],
            "required_company_ids": ["A", "B"],
            "required_subject_action": "new_entity",
            "history_company_ids": ["A", "B"],
        })
        for value, code in (
            (decision("reuse", ["A", "B"], ["event"]), "fixed_subject_action_changed"),
            (decision("new_entity", ["B", "A"], ["event"]), "fixed_company_order_changed"),
        ):
            with self.subTest(code=code):
                violations = verify_decision(value, context)
                self.assertIn(code, [item.code for item in violations])
                self.assertEqual(fail_closed(value, violations, context).company_ids, ())

    def test_duplicate_company_ids_are_rejected(self):
        context = VerificationContext(known_company_ids=("A",), allowed_company_ids=("A",))
        value = decision("new_entity", ["A", "A"], ["financial"])
        violations = verify_decision(value, context)
        self.assertIn("duplicate_company_id", [item.code for item in violations])

    def test_valid_supported_shapes_pass(self):
        contexts_and_decisions = [
            (
                VerificationContext(
                    known_company_ids=("A", "B"),
                    allowed_company_ids=("A",),
                    required_company_ids=("A",),
                ),
                decision("new_entity", ["A"], ["financial", "equity", "event"]),
            ),
            (
                VerificationContext(
                    known_company_ids=("A", "B"),
                    allowed_company_ids=("A", "B"),
                    required_company_ids=("A", "B"),
                ),
                decision("new_entity", ["A", "B"], ["event"]),
            ),
            (
                VerificationContext(
                    known_company_ids=("A", "B"),
                    allowed_company_ids=("A", "B"),
                    required_company_ids=("A", "B"),
                ),
                decision("new_entity", ["A", "B"], ["clarify"]),
            ),
            (
                VerificationContext(
                    known_company_ids=("A",),
                    allowed_company_ids=("A",),
                    history_company_ids=("A",),
                ),
                decision("reuse", ["A"], ["financial"]),
            ),
        ]

        for context, value in contexts_and_decisions:
            with self.subTest(value=value):
                self.assertEqual(verify_decision(value, context), ())

    def test_fault_injection_set_is_rejected_and_fails_closed(self):
        rows = [
            json.loads(line)
            for line in FAULTS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertGreaterEqual(len(rows), 12)

        for row in rows:
            with self.subTest(name=row["name"]):
                value = Decision.from_mapping(row["decision"])
                context = VerificationContext.from_mapping(row["context"])
                violations = verify_decision(value, context)

                self.assertEqual(
                    [violation.code for violation in violations], row["expected_codes"]
                )
                self.assertEqual(
                    fail_closed(value, violations, context).to_dict(),
                    row["expected_fail_closed"],
                )

    def test_missing_explicit_route_is_rejected_as_route_only_failure(self):
        context = VerificationContext(
            known_company_ids=("A",),
            allowed_company_ids=("A",),
            required_company_ids=("A",),
            required_routes=("financial", "equity"),
        )
        value = decision("new_entity", ["A"], ["financial"])

        violations = verify_decision(value, context)

        self.assertEqual([item.code for item in violations], ["required_route_missing"])
        self.assertEqual(
            fail_closed(value, violations, context).to_dict(),
            {
                "subject_action": "new_entity",
                "company_ids": ["A"],
                "route": ["clarify"],
            },
        )

    def test_subject_abstention_may_clarify_even_when_query_has_a_route_signal(self):
        context = VerificationContext(
            known_company_ids=("A", "B"),
            allowed_company_ids=("A", "B"),
            required_routes=("equity",),
        )
        value = decision("clarify", [], ["clarify"])

        self.assertEqual(verify_decision(value, context), ())


if __name__ == "__main__":
    unittest.main()
