"""FastAPI app: /ingest, /ask, /health (+ /documents)."""
import os
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import __version__
from .answer import NO_ANSWER, extractive_answer, llm_answer
from .chunking import chunk_text
from .config import Settings, load_settings
from .loaders import EmptyDocument, UnsupportedType, load_text
from .retrieval import Retriever, embed
from .store import Store


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(None, ge=1, le=20)


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    cfg = settings or load_settings()
    store = Store(cfg.data_dir)
    state = {"retriever": Retriever(store.chunks)}
    app = FastAPI(title="Document Q&A (RAG) API", version=__version__)

    def auth(x_api_key: str = Header(default="")):
        if cfg.api_key and x_api_key != cfg.api_key:
            raise HTTPException(401, "Missing or invalid X-API-Key")

    def rebuild():
        state["retriever"] = Retriever(store.chunks)

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "documents": len(store.documents()),
                "chunks": len(store.chunks),
                "retrieval": "bm25+embeddings" if cfg.emb_enabled else "bm25",
                "answer_mode": "llm" if cfg.llm_enabled else "extractive"}

    @app.get("/documents", dependencies=[Depends(auth)])
    def documents():
        return store.documents()

    @app.delete("/documents/{source}", dependencies=[Depends(auth)])
    def delete_document(source: str):
        n = store.delete_document(source)
        if not n:
            raise HTTPException(404, f"No document named '{source}'")
        rebuild()
        return {"deleted": source, "chunks_removed": n}

    @app.post("/ingest", dependencies=[Depends(auth)])
    async def ingest(files: List[UploadFile] = File(...)):
        results = []
        for f in files:
            name = os.path.basename(f.filename or "")
            data = await f.read()
            if len(data) > cfg.max_upload_mb * 1024 * 1024:
                raise HTTPException(413, f"{name}: larger than {cfg.max_upload_mb} MB")
            try:
                text = load_text(name, data)
            except UnsupportedType as e:
                raise HTTPException(415, f"{name}: {e}")
            except EmptyDocument as e:
                raise HTTPException(422, f"{name}: {e}")
            pieces = chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)
            rows = [{"id": f"{name}#{i}", "source": name, "chunk_index": i, "text": t}
                    for i, t in enumerate(pieces)]
            if cfg.emb_enabled:
                try:
                    vecs = embed([r["text"] for r in rows], cfg.emb_base_url, cfg.emb_model, cfg.emb_api_key)
                except Exception as e:
                    raise HTTPException(502, f"Embeddings endpoint failed: {type(e).__name__}")
                for r, v in zip(rows, vecs):
                    r["embedding"] = v
            store.replace_document(name, rows)
            results.append({"source": name, "chunks": len(rows), "characters": len(text)})
        rebuild()
        return {"ingested": results, "total_chunks": len(store.chunks)}

    @app.post("/ask", dependencies=[Depends(auth)])
    def ask(req: AskRequest):
        if not store.chunks:
            return {"answer": "No documents have been ingested yet. POST files to /ingest first.",
                    "mode": "empty_index", "sources": []}
        k = req.top_k or cfg.top_k
        qemb = None
        if cfg.emb_enabled:
            try:
                qemb = embed([req.question], cfg.emb_base_url, cfg.emb_model, cfg.emb_api_key)[0]
            except Exception:
                qemb = None  # fall back to lexical-only
        hits = state["retriever"].search(req.question, k, qemb, cfg.emb_weight)
        if not hits:
            return {"answer": NO_ANSWER, "mode": "no_match", "sources": []}
        mode, warning = "extractive", None
        if cfg.llm_enabled:
            try:
                answer = llm_answer(req.question, hits, cfg.llm_base_url, cfg.llm_model, cfg.llm_api_key)
                mode = "llm"
            except Exception as e:
                answer = extractive_answer(req.question, hits)
                warning = f"LLM call failed ({type(e).__name__}); returned extractive answer instead"
        else:
            answer = extractive_answer(req.question, hits)
        out = {"answer": answer, "mode": mode,
               "sources": [{"ref": i + 1, "source": h["chunk"]["source"],
                            "chunk_index": h["chunk"]["chunk_index"],
                            "score": round(h["score"], 4), "text": h["chunk"]["text"]}
                           for i, h in enumerate(hits)]}
        if warning:
            out["warning"] = warning
        return out

    return app
