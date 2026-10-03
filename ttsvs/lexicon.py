"""Pronunciation dictionary: whole-word replacements applied before synthesis."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from . import config


def _norm(word: str) -> str:
    """Trim and collapse inner whitespace so "New  York" and "New York" are one key."""
    return " ".join(word.split())


class Lexicon:
    """word -> "say as" text, kept in a small JSON file.

    The file is re-read on every call, so the CLI and the UI (separate
    processes) always see each other's edits. Matching is case-insensitive
    and keys are unique ignoring case.
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else config.lexicon_path()

    # -- storage ---------------------------------------------------------

    def _load(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {_norm(k): v for k, v in data.items() if isinstance(k, str) and _norm(k) and isinstance(v, str)}

    def _save(self, data: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ordered = dict(sorted(data.items(), key=lambda kv: (kv[0].lower(), kv[0])))
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    @staticmethod
    def _find_key(data: dict[str, str], word: str) -> str | None:
        low = word.lower()
        for key in data:
            if key.lower() == low:
                return key
        return None

    # -- public API ------------------------------------------------------

    def entries(self) -> dict[str, str]:
        """All entries, sorted by word (case-insensitive)."""
        data = self._load()
        return dict(sorted(data.items(), key=lambda kv: (kv[0].lower(), kv[0])))

    def set(self, word: str, say_as: str) -> None:
        """Add or replace an entry (a word differing only in case replaces the old one)."""
        word = _norm(word)
        if not word:
            raise ValueError("The word to replace must not be empty.")
        data = self._load()
        existing = self._find_key(data, word)
        if existing is not None:
            del data[existing]
        data[word] = str(say_as).strip()
        self._save(data)

    def remove(self, word: str) -> None:
        """Delete an entry; KeyError if it is not there."""
        data = self._load()
        existing = self._find_key(data, _norm(word))
        if existing is None:
            raise KeyError(word)
        del data[existing]
        self._save(data)

    def apply(self, text: str) -> str:
        """Replace whole words, case-insensitively, longest entry first.

        A match may not touch another word character on either side, so "cat"
        does not match inside "concatenate", while keys such as "C++" still
        work. Replacement text is inserted literally.
        """
        data = self._load()
        if not data or not text:
            return text
        lookup = {key.lower(): value for key, value in data.items()}
        keys = sorted(data, key=lambda k: (-len(k), k.lower()))
        alternatives = (r"\s+".join(re.escape(part) for part in key.split()) for key in keys)
        pattern = re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.IGNORECASE)

        def replace(match: re.Match) -> str:
            found = _norm(match.group(0)).lower()
            return lookup.get(found, match.group(0))

        return pattern.sub(replace, text)
