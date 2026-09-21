import unittest

from scripts.audit_public_release import (
    FORBIDDEN_MARKERS,
    is_secret_path,
    text_findings,
)


class AuditPublicReleaseTests(unittest.TestCase):
    def test_secret_path_classification(self):
        for name in (
            ".env",
            ".env.local",
            "config.env",
            "config.env.dev",
            "id.pem",
            "secrets/token.txt",
        ):
            with self.subTest(name=name):
                self.assertTrue(is_secret_path(name))
        self.assertFalse(is_secret_path(".env.example"))
        self.assertFalse(is_secret_path("README.md"))

    def test_internal_markers_are_detected(self):
        self.assertEqual(text_findings("public documentation"), [])
        for marker in FORBIDDEN_MARKERS:
            with self.subTest(marker=marker):
                self.assertEqual(
                    text_findings("prefix " + marker + " suffix"),
                    ["internal_marker:" + marker],
                )

    def test_secret_values_are_detected_without_storing_a_live_secret(self):
        assignment = "DEEPSEEK_API_KEY" + "=" + "synthetic-value"
        quoted_assignment = (
            "DEEPSEEK_API_KEY" + '="' + "synthetic-value" + '"'
        )
        token = "sk-" + "A" * 16
        self.assertIn(
            "secret_pattern:nonempty_deepseek_key",
            text_findings(assignment),
        )
        self.assertIn(
            "secret_pattern:nonempty_deepseek_key",
            text_findings(quoted_assignment),
        )
        self.assertIn(
            "secret_pattern:openai_style_key",
            text_findings(token),
        )


if __name__ == "__main__":
    unittest.main()
