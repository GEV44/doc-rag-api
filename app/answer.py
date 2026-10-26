"""Answer generation: extractive by default, optional OpenAI-compatible LLM."""
from typing import Dict, List, Optional

import httpx

from .chunking import split_sentences
from .retrieval import tokenize

NO_ANSWER = "I could not find anything relevant to that question in the ingested documents."


def extractive_answer(question: str, hits: List[Dict], max_sentences: int = 2) -> str:
    q = set(tokenize(question))
    cands = []
    for rank, h in enumerate(hits):
        for pos, s in enumerate(split_sentences(h["chunk"]["text"])):
            if s.startswith(("#", ">")):  # headings / notes are context, not answers
                continue
            overlap = len(q & set(tokenize(s)))
            if overlap:
                cands.append((overlap, -rank, -pos, rank, pos, s))
    if not cands:
        return NO_ANSWER
    best = sorted(cands, reverse=True)[:max_sentences]
    best.sort(key=lambda c: (c[3], c[4]))  # reading order
    seen, parts = set(), []
    for c in best:
        rank, s = c[3], c[5]
        if s not in seen:
            seen.add(s)
            parts.append(f"{s} [{rank + 1}]")
    return " ".join(parts)


def llm_answer(question: str, hits: List[Dict], base_url: str, model: str, api_key: str = "",
               timeout: float = 120, transport: Optional[httpx.BaseTransport] = None) -> str:
    ctx = "\n\n".join(f"[{i + 1}] ({h['chunk']['source']}) {h['chunk']['text']}" for i, h in enumerate(hits))
    messages = [
        {"role": "system", "content": "Answer ONLY from the numbered context. Cite sources like [1]. "
                                      "If the context does not contain the answer, say you don't know."},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"},
    ]
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(timeout=timeout, transport=transport) as c:
        r = c.post(f"{base_url}/chat/completions", headers=headers,
                   json={"model": model, "messages": messages, "temperature": 0})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
