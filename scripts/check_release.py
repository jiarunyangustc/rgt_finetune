#!/usr/bin/env python3
"""Run repository checks that do not require field data or a GPU."""

import argparse
import compileall
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "requirements.txt",
    "environment.yml",
    "docs/data_format.md",
    "docs/reproduction.md",
    "examples/make_synthetic_case.py",
    "tests/test_core.py",
]
HAN = re.compile(r"[\u3400-\u9fff]")
PLACEHOLDERS = ("REPLACE_WITH_ACCOUNT", "TO_BE_ADDED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict", action="store_true", help="treat release placeholders as errors"
    )
    args = parser.parse_args()
    errors = []
    warnings = []

    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    for source in ROOT.rglob("*.py"):
        if ".git" in source.parts:
            continue
        text = source.read_text(encoding="utf-8")
        if HAN.search(text):
            errors.append(f"non-English source text: {source.relative_to(ROOT)}")

    for relative in ("README.md", "CITATION.cff", "docs/reproduction.md"):
        path = ROOT / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for placeholder in PLACEHOLDERS:
            if placeholder in text:
                message = f"release placeholder {placeholder!r} remains in {relative}"
                (errors if args.strict else warnings).append(message)

    if not compileall.compile_dir(str(ROOT), quiet=1):
        errors.append("Python byte-code compilation failed")

    for message in warnings:
        print(f"WARNING: {message}")
    for message in errors:
        print(f"ERROR: {message}")
    if errors:
        return 1
    print("Release checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
