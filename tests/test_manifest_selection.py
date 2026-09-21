import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts import run_dev, run_experiment


class ManifestSelectionTests(unittest.TestCase):
    DEFAULT_MANIFEST_RELATIVE_PATH = Path("frozen/public-v1.0/freeze-manifest.json")

    def report(self, entrypoint, arguments):
        output = io.StringIO()
        with redirect_stdout(output):
            entrypoint(arguments)
        return json.loads(output.getvalue())

    def default_manifest(self, module):
        return module._repository_root() / self.DEFAULT_MANIFEST_RELATIVE_PATH

    def displayed_manifest(self, module, manifest):
        resolved = manifest.resolve()
        try:
            return resolved.relative_to(module._repository_root()).as_posix()
        except ValueError:
            return str(resolved)

    def check_default_and_explicit_v1(self, entrypoint, module, arguments):
        expected_manifest = self.default_manifest(module)
        self.assertEqual(module.MANIFEST, expected_manifest)
        for selection in ([], ["--manifest", str(expected_manifest)]):
            with self.subTest(selection=selection):
                report = self.report(entrypoint, arguments + selection)
                self.assertEqual(report["freeze_id"], "financial-router-public-v1.0")
                self.assertEqual(
                    report["manifest_path"],
                    self.displayed_manifest(module, expected_manifest),
                )

    def check_explicit_manifest(self, entrypoint, module, arguments):
        # A deliberately unversioned filename: identity comes from the manifest.
        source_manifest = self.default_manifest(module)
        document = json.loads(source_manifest.read_text(encoding="utf-8"))
        document["freeze_id"] = "manifest-selection-test"
        for item in document["files"]:
            item["path"] = str(source_manifest.parent / item["path"])
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "selection.json"
            manifest.write_text(json.dumps(document), encoding="utf-8")
            report = self.report(entrypoint, arguments + ["--manifest", str(manifest)])
        self.assertEqual(report["freeze_id"], "manifest-selection-test")
        self.assertEqual(
            report["manifest_path"],
            self.displayed_manifest(module, manifest),
        )
        cases = report["cases"] if "cases" in report else report["runs"][0]["cases"]
        self.assertEqual([case["case_id"] for case in cases], document["dev_case_ids"])

    def check_missing_manifest(self, entrypoint, arguments):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                self.report(entrypoint, arguments + ["--manifest", str(manifest)])

    def test_experiment_default_and_explicit_v1_remain_available(self):
        self.check_default_and_explicit_v1(
            run_experiment.main, run_experiment, ["--route", "pure_rule"]
        )

    def test_dev_default_and_explicit_v1_remain_available(self):
        self.check_default_and_explicit_v1(run_dev.main, run_dev, [])

    def test_experiment_uses_explicit_manifest_not_directory_name(self):
        self.check_explicit_manifest(
            run_experiment.main, run_experiment, ["--route", "pure_rule"]
        )

    def test_dev_uses_explicit_manifest_not_directory_name(self):
        self.check_explicit_manifest(run_dev.main, run_dev, [])

    def test_experiment_missing_manifest_does_not_fall_back(self):
        self.check_missing_manifest(run_experiment.main, ["--route", "pure_rule"])

    def test_dev_missing_manifest_does_not_fall_back(self):
        self.check_missing_manifest(run_dev.main, [])


if __name__ == "__main__":
    unittest.main()
