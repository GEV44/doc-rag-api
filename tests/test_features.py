"""Tests for: page/section citations, multilingual (hy/ru/en), DOCX upload, rate limiting."""
import io

import pytest

from app.config import Settings
from app.langdetect import detect_language
from app.loaders import load_segments
from app.ratelimit import RateLimiter
from app.retrieval import tokenize
from conftest import make_client


def make_pdf_pages(pages):
    """Multi-page PDF with a text layer (no extra deps)."""
    n = len(pages)
    objs = [b"<</Type/Catalog/Pages 2 0 R>>",
            b"<</Type/Pages/Kids[" + b" ".join(b"%d 0 R" % (3 + 2 * i) for i in range(n)) + b"]/Count %d>>" % n]
    for i, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objs.append(b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents %d 0 R"
                    b"/Resources<</Font<</F1 %d 0 R>>>>>>" % (4 + 2 * i, 3 + 2 * n))
        objs.append(b"<</Length %d>>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>")
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offs:
        out += b"%010d 00000 n \n" % o
    out += b"trailer<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, xref)
    return out


def make_docx(blocks):
    from docx import Document
    d = Document()
    for kind, text in blocks:
        d.add_heading(text, level=1) if kind == "h" else d.add_paragraph(text)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


# ---- citations: page / section ------------------------------------------------


def test_pdf_answer_cites_page_number(client):
    pdf = make_pdf_pages(["Intro page about nothing special.",
                          "The vault code rotates every Friday morning.",
                          "Closing remarks and thanks."])
    assert client.post("/ingest", files={"files": ("manual.pdf", pdf)}).status_code == 200
    s = client.post("/ask", json={"question": "When does the vault code rotate?"}).json()["sources"][0]
    assert s["source"] == "manual.pdf" and s["page"] == 2
    assert s["citation"] == "manual.pdf, p. 2"


def test_markdown_answer_cites_section(client):
    md = ("# Policy\n\n## Refunds\n\nRefunds are issued within seven days.\n\n"
          "## Shipping\n\nParcels travel by courier and arrive in three days.\n")
    client.post("/ingest", files={"files": ("policy.md", md.encode())})
    s = client.post("/ask", json={"question": "How fast do parcels arrive by courier?"}).json()["sources"][0]
    assert s["section"] == "Shipping"
    assert s["page"] is None
    assert s["citation"] == "policy.md, section 'Shipping'"


def test_plain_text_citation_is_just_the_file(client):
    client.post("/ingest", files={"files": ("n.txt", b"The gate opens at dawn every day.")})
    s = client.post("/ask", json={"question": "When does the gate open?"}).json()["sources"][0]
    assert s["citation"] == "n.txt"


def test_load_segments_pdf_pages():
    segs = load_segments("a.pdf", make_pdf_pages(["one", "two"]))
    assert [s["page"] for s in segs] == [1, 2]


# ---- DOCX upload + validation ----------------------------------------------------


def test_docx_upload_with_heading_citation(client):
    docx = make_docx([("h", "Onboarding"), ("p", "New hires receive a laptop on day one."),
                      ("h", "Benefits"), ("p", "Gym membership is reimbursed monthly.")])
    r = client.post("/ingest", files={"files": ("hr.docx", docx)})
    assert r.status_code == 200, r.text
    s = client.post("/ask", json={"question": "Is the gym membership reimbursed?"}).json()["sources"][0]
    assert s["source"] == "hr.docx" and s["section"] == "Benefits"


def test_docx_with_wrong_content_rejected(client):
    assert client.post("/ingest", files={"files": ("fake.docx", b"just text")}).status_code == 422


def test_pdf_magic_bytes_checked(client):
    assert client.post("/ingest", files={"files": ("fake.pdf", b"PK\x03\x04 zip")}).status_code == 422


def test_unsupported_type_message_lists_docx(client):
    r = client.post("/ingest", files={"files": ("x.exe", b"MZ")})
    assert r.status_code == 415 and ".docx" in r.json()["detail"]


# ---- multilingual -------------------------------------------------------------------


@pytest.mark.parametrize("text,lang", [
    ("How long is the warranty?", "en"),
    ("Сколько длится гарантия на товар?", "ru"),
    ("Որքա՞ն է երաշխիքի ժամկետը", "hy"),
    ("12345 ???", "unknown"),
])
def test_detect_language(text, lang):
    assert detect_language(text) == lang


def test_russian_inflections_match_after_stemming():
    assert set(tokenize("возврат товаров")) & set(tokenize("Возврат товара возможен"))
    assert tokenize("отпуска") == tokenize("отпуск")


def test_armenian_inflections_match_after_stemming():
    assert tokenize("երաշխիքի") == tokenize("երաշխիքը")


def test_english_tokenizer_unchanged():
    assert tokenize("returns") == tokenize("return")


RU_DOC = ("# Гарантия\n\nГарантия на технику действует два года с даты покупки.\n\n"
          "# Возврат\n\nВозврат товара возможен в течение тридцати дней при наличии чека.\n")
HY_DOC = ("# Երաշխիք\n\nՏեխնիկայի երաշխիքը գործում է երկու տարի գնման օրվանից։\n\n"
          "# Վերադարձ\n\nԱպրանքի վերադարձը հնարավոր է երեսուն օրվա ընթացքում։\n")


def test_russian_question_over_russian_document(tmp_path):
    client = make_client(tmp_path, chunk_size=90, chunk_overlap=0)
    client.post("/ingest", files={"files": ("ru.md", RU_DOC.encode("utf-8"))})
    j = client.post("/ask", json={"question": "Сколько действует гарантия на технику?"}).json()
    assert j["language"] == "ru"
    assert j["sources"][0]["section"] == "Гарантия"


def test_armenian_question_over_armenian_document(tmp_path):
    client = make_client(tmp_path, chunk_size=90, chunk_overlap=0)
    client.post("/ingest", files={"files": ("hy.md", HY_DOC.encode("utf-8"))})
    j = client.post("/ask", json={"question": "Որքա՞ն ժամանակ է գործում երաշխիքը"}).json()
    assert j["language"] == "hy"
    assert j["sources"][0]["section"] == "Երաշխիք"


def test_no_match_reply_is_localized(client):
    client.post("/ingest", files={"files": ("ru.md", RU_DOC.encode("utf-8"))})
    j = client.post("/ask", json={"question": "Кто выиграл чемпионат мира по футболу?"}).json()
    assert j["mode"] == "no_match" and j["language"] == "ru"
    assert "не нашёл" in j["answer"].lower() or "не найд" in j["answer"].lower()


# ---- rate limiting ----------------------------------------------------------------


def test_rate_limiter_sliding_window():
    now = [0.0]
    rl = RateLimiter(2, window=60, clock=lambda: now[0])
    assert rl.check("k") == (True, 0)
    assert rl.check("k")[0]
    ok, retry = rl.check("k")
    assert not ok and retry > 0
    assert rl.check("other")[0]          # separate bucket per client
    now[0] = 61
    assert rl.check("k")[0]              # window slid


def test_rate_limit_disabled_by_default(client):
    for _ in range(5):
        assert client.post("/ask", json={"question": "hello"}).status_code == 200


def test_rate_limit_returns_429_with_retry_after(tmp_path):
    c = make_client(tmp_path, rate_limit_per_min=2)
    for _ in range(2):
        assert c.post("/ask", json={"question": "hello"}).status_code == 200
    r = c.post("/ask", json={"question": "hello"})
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    assert c.get("/health").status_code == 200   # health is never limited


def test_rate_limit_is_per_api_key(tmp_path):
    c = make_client(tmp_path, api_key="s3cret", rate_limit_per_min=1)
    h = {"X-API-Key": "s3cret"}
    assert c.post("/ask", json={"question": "hi"}, headers=h).status_code == 200
    assert c.post("/ask", json={"question": "hi"}, headers=h).status_code == 429
    assert c.post("/ask", json={"question": "hi"}).status_code == 401   # auth still checked first


def test_settings_has_rate_limit_field():
    assert Settings().rate_limit_per_min == 0
