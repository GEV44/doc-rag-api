"""Turn uploaded bytes into plain text. Supported: .txt .md .pdf (text layer only)."""
import io
import os

ALLOWED = {".txt", ".md", ".pdf"}


class UnsupportedType(ValueError):
    pass


class EmptyDocument(ValueError):
    pass


def extension(filename: str) -> str:
    return os.path.splitext(filename or "")[1].lower()


def load_text(filename: str, data: bytes) -> str:
    ext = extension(filename)
    if ext not in ALLOWED:
        raise UnsupportedType(f"Unsupported file type '{ext or 'none'}'. Allowed: .txt, .md, .pdf")
    if ext == ".pdf":
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as e:  # pypdf raises several error types on malformed files
            raise EmptyDocument(f"Could not read PDF ({type(e).__name__})")
    else:
        text = data.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").strip()
    if not text:
        raise EmptyDocument("No extractable text (empty file, or a scanned PDF - OCR is not supported)")
    return text
