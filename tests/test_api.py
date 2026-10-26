import json

import httpx
import pytest

from app.answer import llm_answer
from app.chunking import chunk_text
from app.config import Settings
from app.retrieval import Retriever, embed
from conftest import make_client, upload_samples


def make_pdf(text: str) -> bytes:
    """Minimal one-page PDF with a text layer (no extra deps)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<</Type/Catalog/Pages 2 0 R>>", b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
            b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
            b"<</Length %d>>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>"]
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


# ---- chunking -------------------------------------------------------------
def test_chunking_respects_size_and_overlaps():
    text = " ".join(f"Sentence number {i} says something useful." for i in range(40))
    chunks = chunk_text(text, size=200, overlap=60)
    assert len(chunks) > 3
    assert all(len(c) <= 200 for c in chunks)
    # overlap: the last sentence of chunk n reappears at the start of chunk n+1
    for a, b in zip(chunks, chunks[1:]):
        assert a.split(". ")[-1].rstrip(".") in b


def test_chunking_short_text_and_long_sentence():
    assert chunk_text("Just one sentence.", 500, 100) == ["Just one sentence."]
    assert chunk_text("   ", 500, 100) == []
    long = " ".join(["word"] * 300)  # one 1499-char sentence
    assert all(len(c) <= 200 for c in chunk_text(long, 200, 20))


def test_chunking_invalid_params():
    with pytest.raises(ValueError):
        chunk_text("x", 100, 100)


# ---- ingest ---------------------------------------------------------------
def test_ingest_samples_and_health(client):
    r = upload_samples(client)
    assert r.status_code == 200
    body = r.json()
    assert {d["source"] for d in body["ingested"]} == {
        "clinic_faq.md", "returns_policy.md", "remote_work_handbook.txt", "product_manual.md"}
    assert body["total_chunks"] == sum(d["chunks"] for d in body["ingested"]) > 4
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["documents"] == 4 and h["answer_mode"] == "extractive"


def test_ingest_pdf(client):
    r = client.post("/ingest", files={"files": ("note.pdf", make_pdf("The vault code rotates every Friday."))})
    assert r.status_code == 200, r.text
    a = client.post("/ask", json={"question": "When does the vault code rotate?"}).json()
    assert a["sources"][0]["source"] == "note.pdf"


def test_reingest_replaces_document(client):
    client.post("/ingest", files={"files": ("a.txt", b"Alpha is the first letter.")})
    client.post("/ingest", files={"files": ("a.txt", b"Omega is the last letter.")})
    docs = client.get("/documents").json()
    assert docs == [{"source": "a.txt", "chunks": 1}]
    assert client.post("/ask", json={"question": "alpha"}).json()["mode"] == "no_match"


def test_bad_file_type_rejected(client):
    r = client.post("/ingest", files={"files": ("virus.exe", b"MZ...")})
    assert r.status_code == 415
    assert client.get("/health").json()["chunks"] == 0


def test_empty_and_corrupt_files_rejected(client):
    assert client.post("/ingest", files={"files": ("e.txt", b"  \n")}).status_code == 422
    assert client.post("/ingest", files={"files": ("bad.pdf", b"not a pdf")}).status_code == 422


def test_oversize_rejected(tmp_path):
    c = make_client(tmp_path, max_upload_mb=1)
    r = c.post("/ingest", files={"files": ("big.txt", b"a" * (1024 * 1024 + 1))})
    assert r.status_code == 413


# ---- ask ------------------------------------------------------------------
def test_ask_empty_index(client):
    r = client.post("/ask", json={"question": "anything?"})
    assert r.status_code == 200
    assert r.json()["mode"] == "empty_index" and r.json()["sources"] == []


@pytest.mark.parametrize("question,expected_source,expected_text", [
    ("What are the clinic opening hours on Saturday?", "clinic_faq.md", "9:00 to 13:00"),
    ("How long do I have to return an item for a refund?", "returns_policy.md", "30 days"),
    ("How many vacation days do employees get?", "remote_work_handbook.txt", "24 paid vacation days"),
    ("How often should I replace the cartridge?", "product_manual.md", "every 6 months"),
])
def test_ask_cites_correct_source(loaded, question, expected_source, expected_text):
    r = loaded.post("/ask", json={"question": question})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "extractive"
    top = body["sources"][0]
    assert top["source"] == expected_source and top["score"] > 0 and top["ref"] == 1
    assert expected_text in body["answer"]
    assert "[1]" in body["answer"]


def test_answer_is_sentence_not_heading(loaded):
    a = loaded.post("/ask", json={"question": "clinic opening hours"}).json()["answer"]
    assert not a.startswith("#") and "Monday to Friday" in a


def test_ask_unrelated_returns_no_match(loaded):
    body = loaded.post("/ask", json={"question": "Who won the 1998 football world cup?"}).json()
    assert body["mode"] == "no_match" and body["sources"] == []


def test_ask_validation(loaded):
    assert loaded.post("/ask", json={"question": ""}).status_code == 422
    assert loaded.post("/ask", json={"question": "x", "top_k": 0}).status_code == 422


def test_top_k_limits_sources(loaded):
    body = loaded.post("/ask", json={"question": "days", "top_k": 1}).json()
    assert len(body["sources"]) <= 1


def test_index_persists_across_restart(tmp_path):
    c1 = make_client(tmp_path)
    upload_samples(c1)
    c2 = make_client(tmp_path)  # new app instance, same DATA_DIR
    assert c2.get("/health").json()["chunks"] == c1.get("/health").json()["chunks"] > 0
    assert c2.post("/ask", json={"question": "free parking"}).json()["sources"][0]["source"] == "clinic_faq.md"


def test_delete_document(loaded):
    assert loaded.delete("/documents/clinic_faq.md").status_code == 200
    assert loaded.delete("/documents/clinic_faq.md").status_code == 404
    assert loaded.get("/health").json()["documents"] == 3


# ---- auth -----------------------------------------------------------------
def test_optional_api_key(tmp_path):
    c = make_client(tmp_path, api_key="s3cret")
    assert c.get("/health").status_code == 200
    assert c.post("/ask", json={"question": "hi"}).status_code == 401
    assert c.post("/ask", json={"question": "hi"}, headers={"X-API-Key": "s3cret"}).status_code == 200


# ---- optional LLM / embeddings (mocked HTTP; NOT tested against a real model) ----
def test_llm_answer_payload_and_parse():
    seen = {}

    def handler(req: httpx.Request):
        seen["body"] = json.loads(req.content)
        seen["path"] = req.url.path
        return httpx.Response(200, json={"choices": [{"message": {"content": " Open 8-18 [1]. "}}]})

    hits = [{"chunk": {"source": "clinic_faq.md", "text": "Open 8:00 to 18:00."}, "score": 1.0}]
    out = llm_answer("hours?", hits, "http://llm.local/v1", "m", transport=httpx.MockTransport(handler))
    assert out == "Open 8-18 [1]." and seen["path"] == "/v1/chat/completions"
    assert "[1] (clinic_faq.md)" in seen["body"]["messages"][1]["content"]


def test_llm_failure_falls_back_to_extractive(tmp_path):
    c = make_client(tmp_path, llm_base_url="http://127.0.0.1:9/v1", llm_model="none")  # nothing listens
    upload_samples(c)
    body = c.post("/ask", json={"question": "How long do I have to return an item for a refund?"}).json()
    assert body["mode"] == "extractive" and "warning" in body and "30 days" in body["answer"]


def test_embeddings_hybrid_ranks_synonym():
    """Fake 2-d embeddings put 'car' and 'automobile' together; BM25 alone cannot match them."""
    def handler(req: httpx.Request):
        texts = json.loads(req.content)["input"]
        vec = lambda t: [1.0, 0.0] if ("car" in t or "automobile" in t) else [0.0, 1.0]
        return httpx.Response(200, json={"data": [{"index": i, "embedding": vec(t)} for i, t in enumerate(texts)]})

    t = httpx.MockTransport(handler)
    chunks = [{"id": "a#0", "source": "a", "chunk_index": 0, "text": "The automobile needs servicing."},
              {"id": "b#0", "source": "b", "chunk_index": 0, "text": "The garden needs watering."}]
    vecs = embed([c["text"] for c in chunks], "http://e/v1", "m", transport=t)
    for c, v in zip(chunks, vecs):
        c["embedding"] = v
    r = Retriever(chunks)
    assert r.search("my car", 2) == []  # lexical only: miss
    qv = embed(["my car"], "http://e/v1", "m", transport=t)[0]
    hits = r.search("my car", 2, qv, emb_weight=0.5)
    assert hits and hits[0]["chunk"]["source"] == "a"
