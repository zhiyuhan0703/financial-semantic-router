"""Audit the tracked public release surface without reading untracked files."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ENV = ".env.example"
EXPECTED_EMAIL = "zhiyuhan0703@users.noreply.github.com"
EXPECTED_PUBLIC_ROOT = "2d00713a26816720597cb3a6bcb174ef881c26e4"
FORBIDDEN_MARKERS = (
    "D:" + chr(92) + "个人" + "文件",
    "D:" + "/" + "个人" + "文件",
    "我的" + "分享工作台",
    "项目" + "经验库",
    "Truth" + "Net",
    "mail." + "ustc.edu.cn",
    "seminar" + " routing experiment",
)
SECRET_PATTERNS = (
    (
        "nonempty_deepseek_key",
        re.compile(
            r"DEEPSEEK_API_KEY\s*=\s*(?:\"[^\"\r\n]+\"|'[^'\r\n]+'|[^\s'\"]+)"
        ),
    ),
    (
        "openai_style_key",
        re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    ),
    (
        "github_token",
        re.compile(r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    (
        "aws_access_key",
        re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    ),
    (
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
)


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT
    ).decode("utf-8")


def tracked_names() -> list[str]:
    return [name for name in git_output("ls-files", "-z").split("\0") if name]


def is_secret_path(name: str) -> bool:
    path = PurePosixPath(name)
    base = path.name
    if base == ALLOWED_ENV:
        return False
    if base == ".env" or base.startswith(".env."):
        return True
    if base.endswith(".env") or ".env." in base:
        return True
    if base.endswith((".key", ".pem")) or base.startswith("secrets."):
        return True
    return "secrets" in path.parts


def text_findings(text: str) -> list[str]:
    findings = [
        f"internal_marker:{marker}"
        for marker in FORBIDDEN_MARKERS
        if marker in text
    ]
    findings.extend(
        f"secret_pattern:{label}"
        for label, pattern in SECRET_PATTERNS
        if pattern.search(text)
    )
    return findings


def audit(final: bool) -> dict[str, object]:
    findings: list[str] = []
    names = tracked_names()
    for name in names:
        if is_secret_path(name):
            findings.append(f"secret_path:{name}")
        if name.startswith("docs/superpowers/"):
            if final:
                findings.append(f"internal_planning_path:{name}")
            else:
                continue
        try:
            text = (ROOT / name).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        findings.extend(f"{name}:{item}" for item in text_findings(text))

    emails = [
        email
        for email in git_output("log", "--format=%ae").splitlines()
        if email
    ]
    findings.extend(
        f"author_email:{email}"
        for email in emails
        if email != EXPECTED_EMAIL
    )
    reachable_commits = int(git_output("rev-list", "--all", "--count").strip())
    root_commits = [
        commit
        for commit in git_output("rev-list", "--max-parents=0", "--all").splitlines()
        if commit
    ]
    if final and root_commits != [EXPECTED_PUBLIC_ROOT]:
        findings.append("public_roots:" + ",".join(root_commits))
    if final and git_output("status", "--porcelain").strip():
        findings.append("working_tree:not_clean")

    return {
        "status": "ok" if not findings else "failed",
        "tracked_files": len(names),
        "reachable_commits": reachable_commits,
        "public_roots": root_commits,
        "findings": findings,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args(argv)
    report = audit(args.final)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
