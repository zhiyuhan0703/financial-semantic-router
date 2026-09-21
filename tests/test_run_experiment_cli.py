import io
import json
import tempfile
import traceback
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from financial_router.deepseek import DeepSeekConfigurationError
from scripts import run_experiment as cli
from scripts.run_experiment import main


def key_assignment(value, quote=""):
    return "DEEPSEEK_API_KEY" + "=" + quote + value + quote + "\n"


class KeyFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.key_file = Path(self.directory.name) / "test-only.env"
        self.path_patch = patch.object(cli, "KEY_FILE", self.key_file, create=True)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def test_environment_key_wins_without_opening_file(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("unexpected read")):
            result = cli.load_model_environment({"DEEPSEEK_API_KEY": "test-env-key"})
        self.assertEqual(result, {"DEEPSEEK_API_KEY": "test-env-key"})

    def test_missing_file_is_skipped_without_creating_it(self):
        self.assertEqual(cli.load_model_environment({}), {})
        self.assertFalse(self.key_file.exists())

    def test_file_loads_only_the_named_key_with_comments_quotes_and_bom(self):
        self.key_file.write_text(
            "# test data only\n\nUNRELATED=ignored\n "
            + "DEEPSEEK_API_KEY"
            + ' = "test-file-key" \n',
            encoding="utf-8-sig",
        )
        source = {"DEEPSEEK_API_KEY": "   ", "UNRELATED": "ignored"}
        self.assertEqual(cli.load_model_environment(source), {"DEEPSEEK_API_KEY": "test-file-key"})
        self.assertEqual(source["DEEPSEEK_API_KEY"], "   ")

    def test_single_quotes_are_supported_and_missing_key_is_skipped(self):
        self.key_file.write_text(
            key_assignment("test-file-key", quote="'"),
            encoding="utf-8",
        )
        self.assertEqual(cli.load_model_environment({}), {"DEEPSEEK_API_KEY": "test-file-key"})
        self.key_file.write_text("UNRELATED=ignored\n", encoding="utf-8")
        self.assertEqual(cli.load_model_environment({}), {})

    def test_malformed_value_and_duplicate_key_do_not_echo_test_values(self):
        for content in (
            "DEEPSEEK_API_KEY" + '="' + "test-sensitive-marker\n",
            key_assignment("test-sensitive-marker") + key_assignment("another"),
        ):
            with self.subTest(content_type="malformed_or_duplicate"):
                self.key_file.write_text(content, encoding="utf-8")
                with self.assertRaises(DeepSeekConfigurationError) as raised:
                    cli.load_model_environment({})
                self.assertNotIn("test-sensitive-marker", str(raised.exception))

    def test_pure_rule_cli_does_not_load_keys_or_emit_them_in_report(self):
        output = io.StringIO()
        with patch.object(cli, "load_model_environment", create=True) as loader:
            with redirect_stdout(output):
                main(["--route", "pure_rule"], environ={"DEEPSEEK_API_KEY": "test-marker"})
            loader.assert_not_called()
        report = json.loads(output.getvalue())
        self.assertEqual(report["route"], "pure_rule")
        self.assertNotIn("test-marker", output.getvalue())

    def test_read_error_does_not_echo_contents_or_exception_chain(self):
        for error in (OSError("test-sensitive-marker"), UnicodeError("test-sensitive-marker")):
            with self.subTest(error_type=type(error).__name__):
                with patch.object(Path, "read_text", side_effect=error):
                    with self.assertRaises(DeepSeekConfigurationError) as raised:
                        cli.load_model_environment({})
                rendered = "".join(traceback.format_exception(raised.exception))
                self.assertNotIn("test-sensitive-marker", rendered)

    def test_authorized_llm_cli_passes_file_key_to_provider_without_reporting_it(self):
        # Only the CLI wiring is under test. The network provider and runner never execute.
        self.key_file.write_text(key_assignment("test-file-key"), encoding="utf-8")
        args = ["--route", "llm_only", "--allow-paid-api", "--model", "deepseek-flash",
                "--temperature", "0", "--timeout-seconds", "30", "--max-tokens", "256",
                "--runs", "1"]
        output = io.StringIO()
        with patch.object(cli, "DeepSeekChatModel") as provider:
            with patch.object(cli, "run_experiment", return_value={"runs": []}):
                with redirect_stdout(output):
                    main(args, environ={})
        self.assertEqual(provider.call_args.kwargs["environ"], {"DEEPSEEK_API_KEY": "test-file-key"})
        self.assertNotIn("test-file-key", output.getvalue())


class ExperimentCliTests(unittest.TestCase):
    def test_output_file_uses_relative_manifest_path_and_matches_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "report.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                main(["--route", "pure_rule", "--output", str(output_path)], environ={})
            printed = json.loads(stdout.getvalue())
            saved_text = output_path.read_text(encoding="utf-8")
            saved = json.loads(saved_text)

        self.assertEqual(saved, printed)
        self.assertEqual(
            saved["manifest_path"],
            "frozen/public-v1.0/freeze-manifest.json",
        )
        self.assertNotIn(":\\", saved_text)

    def test_llm_route_requires_explicit_paid_api_gate(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["--route", "llm_only", "--split", "dev"], environ={})

    def test_test_split_requires_separate_explicit_gate(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["--route", "pure_rule", "--split", "test"], environ={})

    def test_allowed_llm_route_still_requires_environment_key(self):
        arguments = [
            "--route",
            "llm_only",
            "--split",
            "dev",
            "--allow-paid-api",
            "--model",
            "deepseek-flash",
            "--temperature",
            "0",
            "--timeout-seconds",
            "30",
            "--max-retries",
            "0",
            "--max-tokens",
            "256",
            "--runs",
            "1",
        ]

        with patch.object(cli, "load_model_environment", return_value={}, create=True):
            with self.assertRaises(DeepSeekConfigurationError):
                main(arguments, environ={})

    def test_missing_model_is_rejected_before_provider_construction(self):
        arguments = [
            "--route", "llm_only", "--allow-paid-api", "--runs", "1",
            "--temperature", "0", "--timeout-seconds", "30", "--max-tokens", "256",
        ]
        with patch.object(cli, "load_model_environment") as loader:
            with patch("scripts.run_experiment.DeepSeekChatModel") as provider:
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        main(arguments, environ={})
                provider.assert_not_called()
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
