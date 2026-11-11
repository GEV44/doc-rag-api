"""Evaluate the RAG pipeline on a dataset: retrieval, citations, abstention, latency.

Uses the project's own FastAPI app in-process (POST /ingest, POST /ask), so it measures
exactly what a client would get. Offline by default with BM25; embeddings / LLM are used
only if their env vars are set (see .env.example) and --offline is not passed.

Usage:
  python eval/run_eval.py                          # all bundled datasets
  python eval/run_eval.py --dataset warranty       # one bundled dataset
  python eval/run_eval.py --dataset path/to/mine   # your own: docs/ + questions.jsonl
  python eval/run_eval.py --check                  # fail if test hit@3 < eval/baseline.json
  python eval/run_eval.py --write-baseline         # record current numbers as the baseline

A dataset folder holds docs/*.{md,txt,pdf} and questions.jsonl, one object per line:
  {"id", "split": "dev"|"test", "type", "question", "answer",
   "sources": [{"doc": "file.md", "section": "...", "evidence": "verbatim text from the doc"}]}
An empty "sources" list marks an unanswerable question (gold answer: "not in documents").
"""
import argparse
import dataclasses
import json
import os
import random
import re
import sys
import tempfile
import time
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from app.config import load_settings  # noqa: E402
from app.main import create_app  # noqa: E402
from eval.metrics import (abstention_stats, hit_at_k, mean, percentile,  # noqa: E402
                          precision_recall, reciprocal_rank, tune_threshold)

SEED = 0
RETRIEVAL_K = 5
KS = (1, 3, 5)
DATASETS_DIR = os.path.join(ROOT, "eval", "datasets")
REPORTS_DIR = os.path.join(ROOT, "eval", "reports")
BASELINE = os.path.join(ROOT, "eval", "baseline.json")
BASELINE_MARGIN = 0.05
_REF = re.compile(r"\[(\d+)\]")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def load_dataset(path: str):
    docs_dir = os.path.join(path, "docs")
    docs = {n: open(os.path.join(docs_dir, n), "rb").read() for n in sorted(os.listdir(docs_dir))}
    with open(os.path.join(path, "questions.jsonl"), encoding="utf-8") as f:
        qs = [json.loads(line) for line in f if line.strip()]
    for q in qs:
        for s in q["sources"]:
            if s["doc"] not in docs:
                raise SystemExit(f"{q['id']}: unknown doc {s['doc']}")
            if norm(s["evidence"]) not in norm(docs[s["doc"]].decode("utf-8")):
                raise SystemExit(f"{q['id']}: evidence not found verbatim in {s['doc']}: {s['evidence']!r}")
    return docs, qs


def is_gold(chunk: Dict, q: Dict) -> bool:
    """A chunk is gold if it is from a gold doc and contains a gold evidence passage."""
    return any(chunk["source"] == s["doc"] and norm(s["evidence"]) in norm(chunk["text"])
               for s in q["sources"])


def ask_all(docs: Dict[str, bytes], qs: List[Dict], offline: bool) -> List[Dict]:
    """Ingest through POST /ingest, then POST /ask for every question (no threshold: raw scores)."""
    cfg = dataclasses.replace(load_settings(), min_score=0.0)
    if offline:
        cfg = dataclasses.replace(cfg, llm_base_url="", emb_base_url="")
    with tempfile.TemporaryDirectory() as d:
        client = TestClient(create_app(dataclasses.replace(cfg, data_dir=d)))
        files = [("files", (n, b)) for n, b in docs.items()]
        r = client.post("/ingest", files=files)
        r.raise_for_status()
        mode = client.get("/health").json()
        rows = []
        for q in qs:
            t = time.perf_counter()
            resp = client.post("/ask", json={"question": q["question"], "top_k": RETRIEVAL_K}).json()
            ms = (time.perf_counter() - t) * 1000
            srcs = resp["sources"]
            flags = [is_gold({"source": s["source"], "text": s["text"]}, q) for s in srcs]
            cited_refs = {int(m) for m in _REF.findall(resp["answer"])}
            cited = [s for s in srcs if s["ref"] in cited_refs]
            cited_ids = {f"{s['source']}#{s['chunk_index']}" for s in cited}
            gold_ids = {f"{s['source']}#{s['chunk_index']}" for s in srcs if is_gold(
                {"source": s["source"], "text": s["text"]}, q)}
            # recall is over gold *evidence passages*: each must be covered by a cited chunk
            covered = sum(any(c["source"] == e["doc"] and norm(e["evidence"]) in norm(c["text"])
                              for c in cited) for e in q["sources"])
            prec = (sum(f"{c['source']}#{c['chunk_index']}" in gold_ids for c in cited) / len(cited_ids)
                    if cited_ids else 0.0)
            rows.append({"id": q["id"], "split": q["split"], "type": q["type"],
                         "answerable": bool(q["sources"]), "flags": flags,
                         "top_score": srcs[0]["score"] if srcs else 0.0,
                         "cit_precision": prec,
                         "cit_recall": covered / len(q["sources"]) if q["sources"] else 1.0,
                         "latency_ms": ms})
    return rows, mode


def summarize(rows: List[Dict], threshold: float) -> Dict:
    ans = [r for r in rows if r["answerable"]]
    out = {"n_questions": len(rows), "n_answerable": len(ans), "n_unanswerable": len(rows) - len(ans)}
    for k in KS:
        out[f"hit@{k}"] = round(mean([hit_at_k(r["flags"], k) for r in ans]), 4)
    out["mrr"] = round(mean([reciprocal_rank(r["flags"]) for r in ans]), 4)
    out["citation_precision"] = round(mean([r["cit_precision"] for r in ans]), 4)
    out["citation_recall"] = round(mean([r["cit_recall"] for r in ans]), 4)
    out["abstention"] = {k: round(v, 4) for k, v in abstention_stats(rows, threshold).items()}
    by_type = {}
    for t in sorted({r["type"] for r in ans}):
        sub = [r for r in ans if r["type"] == t]
        by_type[t] = {"n": len(sub), "hit@1": round(mean([hit_at_k(r["flags"], 1) for r in sub]), 4),
                      "hit@3": round(mean([hit_at_k(r["flags"], 3) for r in sub]), 4)}
    out["by_type"] = by_type
    return out


def evaluate(name: str, path: str, offline: bool, min_score: Optional[float]) -> Dict:
    random.seed(SEED)  # nothing here is random; fixed anyway so results never depend on it
    docs, qs = load_dataset(path)
    rows, mode = ask_all(docs, qs, offline)
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    tuned = tune_threshold(dev)
    threshold = tuned if min_score is None else min_score
    lat = [r["latency_ms"] for r in rows]
    res = {"dataset": name, "seed": SEED, "retrieval": mode["retrieval"], "answer_mode": mode["answer_mode"],
           "documents": len(docs), "chunks": mode["chunks"], "retrieval_k": RETRIEVAL_K,
           "threshold": {"min_score": threshold, "tuned_on_dev": tuned, "source": "cli" if min_score is not None else "dev"},
           "dev": summarize(dev, threshold), "test": summarize(test, threshold),
           "dev_threshold_sweep_note": "threshold = best balanced accuracy on dev (ties: lowest)",
           "latency_ms": {"p50": round(percentile(lat, 50), 2), "p95": round(percentile(lat, 95), 2),
                          "n": len(lat)}}
    # verify the abstention numbers against the live app gate (MIN_SCORE) for the test split
    res["test"]["abstention_live_check"] = live_check(docs, qs, test, offline, threshold)
    res["_rows"] = rows
    return res


def live_check(docs, qs, test_rows, offline, threshold) -> Dict:
    """Re-run the test questions through the real app with MIN_SCORE set, and count no_match modes."""
    cfg = dataclasses.replace(load_settings(), min_score=threshold)
    if offline:
        cfg = dataclasses.replace(cfg, llm_base_url="", emb_base_url="")
    ids = {r["id"] for r in test_rows}
    with tempfile.TemporaryDirectory() as d:
        client = TestClient(create_app(dataclasses.replace(cfg, data_dir=d)))
        client.post("/ingest", files=[("files", (n, b)) for n, b in docs.items()]).raise_for_status()
        ok = n = 0
        for q in qs:
            if q["id"] not in ids:
                continue
            resp = client.post("/ask", json={"question": q["question"], "top_k": RETRIEVAL_K}).json()
            abstained = resp["mode"] == "no_match"
            ok += abstained == (not q["sources"])
            n += 1
    return {"accuracy": round(ok / n, 4) if n else 0.0, "n": n}


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def markdown(res: Dict) -> str:
    t, d, th = res["test"], res["dev"], res["threshold"]
    ab = t["abstention"]
    lines = [
        f"# Evaluation: {res['dataset']}", "",
        f"Retrieval: `{res['retrieval']}` | answers: `{res['answer_mode']}` | {res['documents']} documents, "
        f"{res['chunks']} chunks | seed {res['seed']} | top-{res['retrieval_k']} retrieved per question", "",
        f"Confidence threshold `MIN_SCORE = {th['min_score']}` ({'tuned on the dev split' if th['source'] == 'dev' else 'set on the command line'}). "
        f"Numbers below are for the **test split** ({t['n_questions']} questions: {t['n_answerable']} answerable, "
        f"{t['n_unanswerable']} unanswerable); the dev split ({d['n_questions']} questions) is used only to pick the threshold.", "",
        "| Metric | Test | Dev |", "|---|---|---|",
    ]
    for k in KS:
        lines.append(f"| Retrieval hit@{k} | {pct(t[f'hit@{k}'])} | {pct(d[f'hit@{k}'])} |")
    lines += [
        f"| MRR | {t['mrr']:.3f} | {d['mrr']:.3f} |",
        f"| Citation precision | {pct(t['citation_precision'])} | {pct(d['citation_precision'])} |",
        f"| Citation recall | {pct(t['citation_recall'])} | {pct(d['citation_recall'])} |",
        f"| Answerability accuracy | {pct(ab['accuracy'])} | {pct(d['abstention']['accuracy'])} |",
        f"| - answers when it should | {pct(ab['answer_rate_on_answerable'])} | {pct(d['abstention']['answer_rate_on_answerable'])} |",
        f"| - abstains when it should | {pct(ab['abstain_rate_on_unanswerable'])} | {pct(d['abstention']['abstain_rate_on_unanswerable'])} |",
        "", f"Latency per `/ask` call (in-process, all {res['latency_ms']['n']} questions): "
            f"p50 {res['latency_ms']['p50']} ms, p95 {res['latency_ms']['p95']} ms.", "",
        f"Live check: the running app with `MIN_SCORE={th['min_score']}` abstains correctly on "
        f"{pct(t['abstention_live_check']['accuracy'])} of the {t['abstention_live_check']['n']} test questions.", "",
        "## Test split by question type", "", "| Type | n | hit@1 | hit@3 |", "|---|---|---|---|",
    ]
    for ty, v in t["by_type"].items():
        lines.append(f"| {ty} | {v['n']} | {pct(v['hit@1'])} | {pct(v['hit@3'])} |")
    lines += ["", "## Definitions", "",
              "- A retrieved chunk is **gold** if it comes from a gold document and contains the gold evidence passage.",
              "- **hit@k**: share of answerable questions with at least one gold chunk in the top k. **MRR**: mean 1/rank of the first gold chunk.",
              "- **Citation precision / recall**: the chunks the answer cites with `[n]` against the gold chunks (recall counts gold passages covered by a cited chunk; precision is 0 if nothing is cited).",
              "- **Answerability**: a question is answered when the top score is at least `MIN_SCORE`, otherwise the app returns `no_match`.",
              "- Latency is wall-clock and varies by machine; every other number is deterministic.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", action="append", help="bundled name or a folder path (repeatable); default: all bundled")
    ap.add_argument("--min-score", type=float, default=None, help="use this threshold instead of tuning on dev")
    ap.add_argument("--offline", action="store_true", help="force BM25 + extractive even if env vars are set")
    ap.add_argument("--out-dir", default=REPORTS_DIR)
    ap.add_argument("--check", action="store_true", help="exit 1 if test hit@3 is below eval/baseline.json")
    ap.add_argument("--write-baseline", action="store_true", help="write eval/baseline.json from this run")
    a = ap.parse_args(argv)

    names = a.dataset or sorted(n for n in os.listdir(DATASETS_DIR)
                                if os.path.isdir(os.path.join(DATASETS_DIR, n)))
    os.makedirs(a.out_dir, exist_ok=True)
    results = {}
    for n in names:
        path = n if os.path.isdir(n) else os.path.join(DATASETS_DIR, n)
        name = os.path.basename(os.path.normpath(path))
        res = evaluate(name, path, a.offline, a.min_score)
        res.pop("_rows")
        results[name] = res
        with open(os.path.join(a.out_dir, f"{name}.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write(markdown(res))
        with open(os.path.join(a.out_dir, f"{name}.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=2)
            f.write("\n")
        t = res["test"]
        print(f"{name}: hit@1 {pct(t['hit@1'])} hit@3 {pct(t['hit@3'])} hit@5 {pct(t['hit@5'])} MRR {t['mrr']:.3f} "
              f"cit P/R {pct(t['citation_precision'])}/{pct(t['citation_recall'])} "
              f"answerability {pct(t['abstention']['accuracy'])} (MIN_SCORE {res['threshold']['min_score']}) "
              f"p50 {res['latency_ms']['p50']} ms p95 {res['latency_ms']['p95']} ms")

    if a.write_baseline:
        base = {"metric": "test hit@3", "margin": BASELINE_MARGIN,
                "datasets": {n: {"measured": r["test"]["hit@3"],
                                 "min_hit@3": round(max(r["test"]["hit@3"] - BASELINE_MARGIN, 0.0), 4)}
                             for n, r in results.items()}}
        with open(BASELINE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(base, f, indent=2)
            f.write("\n")
        print(f"wrote {BASELINE}")
    if a.check:
        with open(BASELINE, encoding="utf-8") as f:
            base = json.load(f)["datasets"]
        failed = False
        for n, r in results.items():
            if n not in base:
                continue
            got, need = r["test"]["hit@3"], base[n]["min_hit@3"]
            ok = got >= need
            failed |= not ok
            print(f"check {n}: test hit@3 {got:.4f} vs baseline {need:.4f} -> {'ok' if ok else 'FAIL'}")
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
