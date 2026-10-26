# Document Q&A (RAG) API - Overview for Osmium review

Author: Gevorg Hovhannisyan (Python/ML engineer, Yerevan; GitHub GEV44). Version 0.1.0. This is a new, unreleased package: it has no customers, users, or production deployments.

## 1. What buyers receive
- Source code: FastAPI service (`/ingest`, `/ask`, `/health`, plus `/documents` list/delete), about 400 lines of Python, in a plain package layout.
- `requirements.txt` (pinned), `requirements-dev.txt`, `Dockerfile`, `.env.example` (no keys), README with quickstart and curl examples.
- 24 pytest tests, `scripts/eval.py` (retrieval check), four fictional sample documents.
- Behaviour: upload `.txt`/`.md`/`.pdf` -> sentence-aware chunking with overlap -> BM25 index persisted to `DATA_DIR/index.json` -> `/ask` returns an answer plus cited chunks with scores. Without an LLM the answer is extractive (best-matching sentences, tagged `[n]`).
- Optional: LLM answers and hybrid semantic retrieval through any OpenAI-compatible endpoint (e.g. local Ollama) via env vars.

## 2. Install and dependencies
Python 3.11 (developed on 3.11.9, Windows 11). `pip install -r requirements.txt` installs fastapi 0.162.0, uvicorn 0.52.0, python-multipart 0.0.32, rank-bm25 0.2.2, pypdf 6.14.2, httpx 0.28.1. No torch, no vector DB, no paid API, no keys needed. Run: `uvicorn app.main:create_app --factory --port 8000`. API keys are only needed if the buyer chooses to point `LLM_*` / `EMBEDDINGS_*` at a hosted provider; the buyer pays that provider directly.

## 3. Sample input and output
Input: the four bundled sample files (4 docs, 11 chunks), then `POST /ask {"question":"How long do I have to return an item for a refund?","top_k":2}`.
Output (captured from a real run, `text` fields omitted):
`"answer": "Unused items can be returned within 30 days of delivery for a full refund. [1] Original shipping costs are not refunded unless the return is due to our error. [2]"`, sources: `returns_policy.md` chunk 0 (score 5.9404), `returns_policy.md` chunk 1 (score 5.8196). An out-of-scope question returns `mode: "no_match"` and no sources. More examples are in README.md.

## 4. Testing evidence (real run, 2026-10-27)
- `python -m pytest -q`: **24 passed in 3.61 s** (Python 3.11.9). Covers chunking (size, overlap, long sentences, bad params), ingest (txt/md/pdf, re-ingest replaces, bad type -> 415, empty/corrupt -> 422, oversize -> 413), ask with citations on 4 corpora, empty index, no-match, validation, persistence across restart, delete, optional API key, LLM fallback, mocked LLM payload, mocked hybrid embeddings.
- `python scripts/eval.py` on the sample corpus (lexical mode): 13/13 questions that share vocabulary with the documents had the correct file as top source; 1/3 paraphrased questions did (the misses are listed in the script output); 2/2 out-of-scope questions correctly returned nothing. In-process `/ask` latency about 10-13 ms median on this laptop.
- A live uvicorn run with curl produced the outputs shown in README.md.
- Caveat: this is a tiny corpus (about 4,000 characters) written by the author, so these numbers show the pipeline works, not how it performs on a buyer's documents.
- Not done: Docker image not built or run; LLM/embeddings tested only against mocked HTTP, not a live Ollama/OpenAI server; PDF tested with a small synthetic text PDF, not real-world layouts; no load or concurrency testing.

## 5. Known limitations
- Default retrieval is lexical (BM25 with a very light English suffix stripper): it misses synonyms and paraphrases (see eval: 1/3) and can return a weakly related chunk when a question shares one common word. Enable embeddings for semantic matching.
- Extractive answers are quoted sentences, not synthesized prose; multi-part questions may be answered partially.
- English-oriented tokenization and stop words; other languages work only as plain word matching.
- PDF: text layer only. Scans are not OCR'd (rejected with 422 if no text); tables and multi-column layouts may extract in odd order.
- Single-tenant: one shared index, no per-user separation. Auth is only an optional shared `X-API-Key`; no rate limiting, no TLS (put it behind a reverse proxy).
- Index is one JSON file loaded into memory and rebuilt on each ingest: fine for hundreds of documents, not designed for very large corpora or heavy concurrent writes.
- Markdown headings are folded into chunk text; chunk text is whitespace-normalized.
- With an LLM enabled, answer quality and faithfulness depend on the chosen model; the prompt asks it to use only the context, which is not guaranteed.

## 6. Support and updates (proposed terms)
- Bug fixes for defects in the delivered code for 3 months from purchase.
- Email support (response within about 2 business days) for installation and configuration questions during that period.
- Compatibility patches if a pinned dependency breaks the install within the 3 months.
- Not included: custom features, integration into the buyer's systems, hosting, model/provider costs, or support for modified code. Custom work can be discussed separately.
