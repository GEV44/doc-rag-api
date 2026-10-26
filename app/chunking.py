"""Sentence-aware chunking with character overlap."""
import re
from typing import List

_SENT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


_HEADING = re.compile(r"(?m)^(#{1,6}\s.*?)[ \t]*$")


def split_sentences(text: str) -> List[str]:
    # Markdown headings become their own "sentence" instead of gluing onto the next paragraph.
    text = _HEADING.sub(lambda m: m.group(1).rstrip(".:") + ".\n\n", text)
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]


def _hard_split(unit: str, size: int) -> List[str]:
    """Split an over-long unit on word boundaries."""
    out, cur = [], ""
    for w in unit.split():
        if cur and len(cur) + 1 + len(w) > size:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, size: int = 500, overlap: int = 100) -> List[str]:
    """Pack sentences into chunks of <= ~size chars; each new chunk starts with
    trailing sentences of the previous one totalling <= overlap chars."""
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and < size")
    units: List[str] = []
    for s in split_sentences(text):
        units.extend(_hard_split(s, size) if len(s) > size else [s])
    chunks: List[str] = []
    cur: List[str] = []
    for u in units:
        if cur and len(" ".join(cur + [u])) > size:
            chunks.append(" ".join(cur))
            carry: List[str] = []
            n = 0
            for prev in reversed(cur):
                if n + len(prev) + 1 > overlap:
                    break
                carry.insert(0, prev)
                n += len(prev) + 1
            cur = carry
            # drop carry if it would still overflow with the new unit
            while cur and len(" ".join(cur + [u])) > size:
                cur.pop(0)
        cur.append(u)
    if cur:
        tail = " ".join(cur)
        if not chunks or tail != chunks[-1]:
            chunks.append(tail)
    return chunks
