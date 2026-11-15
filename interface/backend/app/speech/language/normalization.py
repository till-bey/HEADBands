"""Non-destructive Swiss German normalization helper.

Preserves the original Swiss German transcription intact and produces an optional
Standard German gloss ONLY in a separate field (`normalized_text`).
"""

from __future__ import annotations

import re

_LEXICAL_GLOSS_MAP: dict[str, str] = {
    "grüezi": "Guten Tag",
    "grüessech": "Guten Tag",
    "mitenand": "zusammen",
    "zäme": "zusammen",
    "chasch": "kannst",
    "nöd": "nicht",
    "nid": "nicht",
    "öd": "nicht",
    "hüt": "heute",
    "wätter": "Wetter",
    "isch": "ist",
    "gsi": "gewesen",
    "gsii": "gewesen",
    "öppis": "etwas",
    "öpper": "jemand",
    "luege": "schauen",
    "lueg": "schau",
    "schaffe": "arbeiten",
    "häsch": "hast",
    "händ": "haben",
    "hei": "haben",
    "güet": "gut",
    "wier": "wir",
    "mier": "wir",
    "ig": "ich",
    "embrüf": "hinauf",
    "embri": "hinunter",
}

_WORD_BOUNDARY_RE = re.compile(r"\b([a-zA-ZäöüÄÖÜ]+)\b")


def build_swiss_german_gloss(dialect_text: str) -> str | None:
    """Create a lightweight Standard German gloss without modifying the original dialect text."""
    cleaned = dialect_text.strip()
    if not cleaned:
        return None

    changed = False

    def _replace(match: re.Match[str]) -> str:
        nonlocal changed
        word = match.group(1)
        replacement = _LEXICAL_GLOSS_MAP.get(word.lower())
        if replacement is None:
            return word
        changed = True
        if word[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement

    glossed = _WORD_BOUNDARY_RE.sub(_replace, cleaned)
    return glossed if changed else None
