#!/usr/bin/env python
"""
Swap *text‑only* uses of LLaVA for Mistral in your project code.

• Skips the virtual‑env at .venv/
• Skips the few files where LLaVA must remain (vision captioning)
• Creates .bak backups for every file it rewrites
Run from the repo root:
    python tools/swap_llava_to_mistral.py
"""

import pathlib
import re
import shutil

# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# files that must stay LLaVA
SKIP_FILES = {
    "captioner/llava_text.py",
    "captioner/captioner.py",
    "captioner/mistral_text.py",
}

IMPORT_RE = re.compile(
    r"from\s+captioner\.llava_text\s+import\s+query_llava_text"
)
CALL_RE = re.compile(r"\bquery_llava_text\(")
MODEL_RE = re.compile(r'model\s*=\s*["\']llava["\']')

# ---------------------------------------------------------------------------


def should_skip(file: pathlib.Path) -> bool:
    """
    Decide if a file should be left untouched.
    """
    rel = file.relative_to(REPO_ROOT).as_posix()
    return (
        rel.startswith(".venv/")       # ← skip everything inside the venv
        or rel in SKIP_FILES
        or file.suffix != ".py"
    )


def patch_text(text: str) -> str:
    """
    Apply the three regex replacements to the file content.
    """
    text = IMPORT_RE.sub(
        "from captioner.mistral_text import query_mistral_text", text
    )
    text = CALL_RE.sub("query_mistral_text(", text)

    # swap model="llava" → model="mistral" only on lines NOT sending images
    lines = []
    for line in text.splitlines():
        if "image" not in line:
            line = MODEL_RE.sub('model="mistral"', line)
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    changed = 0

    for file in REPO_ROOT.rglob("*.py"):
        if should_skip(file):
            continue

        original = file.read_text(encoding="utf-8", errors="ignore")
        patched = patch_text(original)

        if patched != original:
            backup = file.with_suffix(file.suffix + ".bak")
            # keep the original as backup
            shutil.copyfile(file, backup)
            # write the patched version
            file.write_text(patched, encoding="utf-8")
            print(
                f"✔  Patched {file.relative_to(REPO_ROOT)} "
                f"(backup → {backup.name})"
            )
            changed += 1

    print(
        f"\nDone. {changed} file(s) updated.\n"
        "Run a quick manual test or your test suite before committing."
    )


if __name__ == "__main__":  # pragma: no cover
    main()