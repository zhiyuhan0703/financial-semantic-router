import re
import unittest
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "DATA_NOTICE.md",
    ROOT / "SECURITY.md",
    ROOT / "docs" / "architecture.md",
    ROOT / "docs" / "data-design.md",
    ROOT / "docs" / "evaluation.md",
)


class MarkdownLinkTests(unittest.TestCase):
    def test_local_markdown_links_resolve(self):
        missing = []
        for document in DOCUMENTS:
            text = document.read_text(encoding="utf-8")
            for target in LINK.findall(text):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                relative = unquote(target.split("#", 1)[0])
                if relative and not (document.parent / relative).resolve().exists():
                    missing.append((document.relative_to(ROOT).as_posix(), target))
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
