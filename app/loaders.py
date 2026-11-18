"""Turn uploaded bytes into text segments. Supported: .txt .md .pdf (text layer) .docx.

A segment is {"text": str, "page": int | None}; PDFs yield one segment per page so chunks can cite
page numbers. DOCX headings are emitted as Markdown headings so chunks can cite sections.
"""
import io
import os
from typing import Dict, List

ALLOWED = {".txt", ".md", ".pdf", ".docx"}


class UnsupportedType(ValueError):
    pass


class EmptyDocument(ValueError):
    pass


def extension(filename: str) -> str:
    return os.path.splitext(filename or "")[1].lower()


def _clean(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def _docx_text(data: bytes) -> str:
    from docx import Document
    try:
        doc = Document(io.BytesIO(data))
    except Exception as e:
        raise EmptyDocument(f"Could not read DOCX ({type(e).__name__})")
    lines = []
    for p in doc.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        style = (p.style.name or "") if p.style is not None else ""
        if style == "Title":
            lines.append(f"# {t}")
        elif style.startswith("Heading"):
            digits = "".join(ch for ch in style if ch.isdigit())
            lines.append(f"{'#' * min(int(digits or 1), 6)} {t}")
        else:
            lines.append(t)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                lines.append(" | ".join(cells))
    return "\n\n".join(lines)


def load_segments(filename: str, data: bytes) -> List[Dict]:
    ext = extension(filename)
    if ext not in ALLOWED:
        raise UnsupportedType(f"Unsupported file type '{ext or 'none'}'. Allowed: .txt, .md, .pdf, .docx")
    if ext == ".pdf":
        if not data.lstrip()[:5] == b"%PDF-":
            raise EmptyDocument("File content is not a PDF (missing %PDF header)")
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(data))
            segs = [{"text": _clean(p.extract_text() or ""), "page": i + 1} for i, p in enumerate(reader.pages)]
        except Exception as e:  # pypdf raises several error types on malformed files
            raise EmptyDocument(f"Could not read PDF ({type(e).__name__})")
        segs = [s for s in segs if s["text"]]
    elif ext == ".docx":
        if data[:2] != b"PK":
            raise EmptyDocument("File content is not a DOCX (not a zip container)")
        segs = [{"text": _clean(_docx_text(data)), "page": None}]
    else:
        segs = [{"text": _clean(data.decode("utf-8", errors="replace")), "page": None}]
        segs = [s for s in segs if s["text"]]
    if not segs:
        raise EmptyDocument("No extractable text (empty file, or a scanned PDF - OCR is not supported)")
    return segs


def load_text(filename: str, data: bytes) -> str:
    return "\n\n".join(s["text"] for s in load_segments(filename, data))
