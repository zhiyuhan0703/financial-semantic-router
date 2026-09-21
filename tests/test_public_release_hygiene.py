import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ENV = ".env.example"
FORBIDDEN = (
    "D:" + chr(92) + "个人" + "文件",
    "D:" + "/" + "个人" + "文件",
    "我的" + "分享工作台",
    "项目" + "经验库",
    "Truth" + "Net",
    "mail." + "ustc.edu.cn",
    "seminar" + " routing experiment",
)


def tracked_files():
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8")
    return [ROOT / name for name in output.split("\0") if name]


class PublicReleaseHygieneTests(unittest.TestCase):
    def test_no_secret_file_is_tracked(self):
        names = [path.relative_to(ROOT).as_posix() for path in tracked_files()]
        suspicious = [
            name
            for name in names
            if Path(name).name.startswith(".env")
            and Path(name).name != ALLOWED_ENV
        ]
        self.assertEqual(suspicious, [])

    def test_tracked_text_has_no_internal_markers(self):
        findings = []
        for path in tracked_files():
            if "docs/superpowers/" in path.relative_to(ROOT).as_posix():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for marker in FORBIDDEN:
                if marker in text:
                    findings.append((path.relative_to(ROOT).as_posix(), marker))
        self.assertEqual(findings, [])

    def test_git_author_uses_noreply_address(self):
        emails = subprocess.check_output(
            ["git", "log", "--format=%ae"], cwd=ROOT, text=True
        ).splitlines()
        self.assertTrue(emails)
        self.assertTrue(
            all(
                email == "zhiyuhan0703@users.noreply.github.com"
                for email in emails
            )
        )


if __name__ == "__main__":
    unittest.main()
