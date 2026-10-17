"""Split long text into chunks a TTS model can handle."""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?…。！？])[\"')\]”’]*\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:，；：])\s+")


def _pack(pieces: list[str], max_chars: int) -> list[str]:
    """Greedily merge small pieces into chunks no longer than max_chars."""
    out: list[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + 1 + len(piece) <= max_chars:
            cur = f"{cur} {piece}"
        else:
            if cur:
                out.append(cur)
            cur = piece
    if cur:
        out.append(cur)
    return out


def _split_words(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    for word in text.split():
        while len(word) > max_chars:  # a single absurdly long token
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        pieces.append(word)
    return _pack(pieces, max_chars)


def _split_sentence(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    clauses = [c for c in _CLAUSE_END.split(sentence) if c.strip()]
    out: list[str] = []
    for clause in _pack(clauses, max_chars):
        out.extend([clause] if len(clause) <= max_chars else _split_words(clause, max_chars))
    return out


def split_text(text: str, max_chars: int = 300) -> list[str]:
    """Split on sentences, then commas, then words; each chunk <= max_chars.

    Paragraph breaks always start a new chunk. Blank chunks are dropped.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    chunks: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        flat = " ".join(paragraph.split())
        if not flat:
            continue
        sentences = [s.strip() for s in _SENTENCE_END.split(flat) if s.strip()]
        pieces: list[str] = []
        for s in sentences:
            pieces.extend(_split_sentence(s, max_chars))
        chunks.extend(_pack(pieces, max_chars))
    return chunks
