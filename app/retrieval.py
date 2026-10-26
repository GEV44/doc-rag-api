"""BM25 retrieval (default) with optional embedding re-scoring."""
import math
import re
from typing import Dict, List, Optional, Sequence

import httpx
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
STOP = set("a an the of to in on at for and or is are was were be been it its this that these those "
           "do does did can could i you we they he she my your our their with as by from if so not no "
           "what which who how when where why will would should may".split())


def stem(w: str) -> str:
    """Tiny suffix stripper (English). Not a real stemmer; matches plurals/tenses."""
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    return w[:-1] if len(w) > 3 and w.endswith("e") else w


def tokenize(text: str) -> List[str]:
    return [stem(t) for t in (m.lower() for m in _TOKEN.findall(text)) if t not in STOP]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if not na or not nb else sum(x * y for x, y in zip(a, b)) / (na * nb)


def embed(texts: List[str], base_url: str, model: str, api_key: str = "", timeout: float = 60,
          transport: Optional[httpx.BaseTransport] = None) -> List[List[float]]:
    """Call an OpenAI-compatible /embeddings endpoint (OpenAI, Ollama, vLLM, LM Studio...)."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(timeout=timeout, transport=transport) as c:
        r = c.post(f"{base_url}/embeddings", json={"model": model, "input": texts}, headers=headers)
        r.raise_for_status()
        rows = sorted(r.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in rows]


class PositiveBM25(BM25Okapi):
    """BM25Okapi with the Lucene IDF, log(1 + (N-n+.5)/(n+.5)), which is always > 0.
    Stock rank_bm25 gives zero/negative scores for terms in >= half the chunks, so tiny
    corpora (e.g. a single document) would return nothing."""

    def _calc_idf(self, nd):
        self.idf = {w: math.log(1 + (self.corpus_size - n + 0.5) / (n + 0.5)) for w, n in nd.items()}
        self.average_idf = sum(self.idf.values()) / max(len(self.idf), 1)


class Retriever:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.bm25 = PositiveBM25([tokenize(c["text"]) or ["_"] for c in chunks]) if chunks else None

    def search(self, query: str, top_k: int = 3, query_emb: Optional[List[float]] = None,
               emb_weight: float = 0.5) -> List[Dict]:
        if not self.chunks:
            return []
        q = tokenize(query)
        lex = list(self.bm25.get_scores(q)) if q else [0.0] * len(self.chunks)
        scores = lex
        if query_emb is not None and all(c.get("embedding") for c in self.chunks):
            mx = max(lex) or 1.0
            sem = [max(cosine(query_emb, c["embedding"]), 0.0) for c in self.chunks]
            scores = [(1 - emb_weight) * (l / mx) + emb_weight * s for l, s in zip(lex, sem)]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [{"chunk": self.chunks[i], "score": float(scores[i])} for i in order[:top_k] if scores[i] > 0]
