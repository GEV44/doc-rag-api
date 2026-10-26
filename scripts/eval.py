"""Tiny retrieval check on the bundled SAMPLE corpus (lexical mode, no external services).
Usage:  python scripts/eval.py
Each case = (question, expected source file or None for 'should find nothing')."""
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

CASES = [
    # direct vocabulary overlap
    ("What are the clinic opening hours on Saturday?", "clinic_faq.md"),
    ("How early should new patients arrive?", "clinic_faq.md"),
    ("How long until blood test results are ready?", "clinic_faq.md"),
    ("Is there free parking?", "clinic_faq.md"),
    ("How long do I have to return an item for a refund?", "returns_policy.md"),
    ("Which items cannot be returned?", "returns_policy.md"),
    ("How do I exchange for a different size?", "returns_policy.md"),
    ("How many vacation days do employees get?", "remote_work_handbook.txt"),
    ("What is the home office equipment budget?", "remote_work_handbook.txt"),
    ("When must a lost laptop be reported?", "remote_work_handbook.txt"),
    ("How often should I replace the cartridge?", "product_manual.md"),
    ("What does the red light mean?", "product_manual.md"),
    ("Is the filter housing under warranty?", "product_manual.md"),
    # paraphrases with little vocabulary overlap (expected weakness of lexical retrieval)
    ("Can I get my money back on stuff I bought?", "returns_policy.md"),
    ("What if the doctor's office is shut on weekends?", "clinic_faq.md"),
    ("Am I allowed to work from a cafe?", "remote_work_handbook.txt"),
    # out of scope: should return no sources
    ("Who won the 1998 football world cup?", None),
    ("What is the capital of Peru?", None),
]


def main():
    with tempfile.TemporaryDirectory() as d:
        c = TestClient(create_app(Settings(data_dir=d)))
        sd = os.path.join(ROOT, "sample_data")
        files = [("files", (n, open(os.path.join(sd, n), "rb"))) for n in sorted(os.listdir(sd))]
        c.post("/ingest", files=files)
        ok_direct = n_direct = ok_para = n_para = ok_oos = n_oos = 0
        lat = []
        for i, (q, exp) in enumerate(CASES):
            t = time.perf_counter()
            r = c.post("/ask", json={"question": q}).json()
            lat.append((time.perf_counter() - t) * 1000)
            got = r["sources"][0]["source"] if r["sources"] else None
            hit = got == exp
            kind = "out-of-scope" if exp is None else ("direct" if i < 13 else "paraphrase")
            print(f"{'PASS' if hit else 'MISS'} [{kind}] {q!r} -> {got}")
            if kind == "direct":
                n_direct += 1; ok_direct += hit
            elif kind == "paraphrase":
                n_para += 1; ok_para += hit
            else:
                n_oos += 1; ok_oos += hit
        lat.sort()
        print(f"\ndirect top-1 source: {ok_direct}/{n_direct}")
        print(f"paraphrase top-1 source: {ok_para}/{n_para}")
        print(f"out-of-scope correctly empty: {ok_oos}/{n_oos}")
        print(f"/ask latency (in-process): median {lat[len(lat)//2]:.1f} ms, max {lat[-1]:.1f} ms")


if __name__ == "__main__":
    main()
