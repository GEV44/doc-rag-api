"""Sentence-aware chunking with character overlap."""
import re
from typing import List

_SENT = re.compile(r"(?<=[.!?։])\s+|\n{2,}")


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
            held = None
            if len(cur) > 1 and cur[-1].startswith("#"):
                held = cur.pop()  # never strand a heading at the end of a chunk; it leads the next one
            chunks.append(" ".join(cur))
            if held:
                cur = [held]
                while cur and len(" ".join(cur + [u])) > size:
                    cur.pop(0)
                cur.append(u)
                continue
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


_MD_HEADING = re.compile(r"(?m)^(#{1,6})\s+(.*?)\s*$")


def chunk_segments(segments: List[dict], size: int = 500, overlap: int = 100) -> List[dict]:
    """Chunk loader segments and attach citation metadata: {"text", "page", "section"}.
    `section` is the nearest Markdown/DOCX heading at or before the chunk (carried across chunks and
    pages); None when the document has no headings."""
    out: List[dict] = []
    section = None
    for seg in segments:
        heads = [(f"{m.group(1)} {m.group(2).rstrip('.:')}", m.group(2).rstrip(".:"))
                 for m in _MD_HEADING.finditer(seg["text"])]
        for piece in chunk_text(seg["text"], size, overlap):
            best = max(((piece.rfind(h), t) for h, t in heads if h in piece), default=(-1, None))
            if best[0] >= 0:
                section = best[1]
            out.append({"text": piece, "page": seg.get("page"), "section": section})
    return out
