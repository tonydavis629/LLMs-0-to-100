"""Step 1: load_text()

Run by src/main.py. You do NOT need to edit this file.
"""

from __future__ import annotations


from tests.check import Check, bad, ok


def check_load_text(text: str) -> list[Check]:
    """The Gutenberg wrapper should be gone and the whole book kept."""
    checks = []

    # The header ends with the START marker; none of it should survive
    checks.append(
        ok("the Project Gutenberg header is removed")
        if "PROJECT GUTENBERG EBOOK" not in text.upper()[:2000]
        else bad("the Project Gutenberg header is removed",
                  "the text still begins with the Gutenberg header")
    )

    # The license footer follows the END marker; it should be gone too
    checks.append(
        ok("the Project Gutenberg license footer is removed")
        if "PROJECT GUTENBERG" not in text.upper()[-2000:]
        else bad("the Project Gutenberg license footer is removed",
                  "the text still ends with the Gutenberg license")
    )

    # The book body itself must still be there, start to finish
    has_start = "Alice" in text[:500]
    has_end = "THE END" in text[-500:]
    checks.append(
        ok("the book runs from the title page to THE END")
        if has_start and has_end
        else bad("the book runs from the title page to THE END",
                  f"title page found: {has_start}, 'THE END' found: {has_end}")
    )
    return checks
