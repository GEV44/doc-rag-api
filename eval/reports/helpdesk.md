# Evaluation: helpdesk

Retrieval: `bm25` | answers: `extractive` | 8 documents, 18 chunks | seed 0 | top-5 retrieved per question

Confidence threshold `MIN_SCORE = 7.0578` (tuned on the dev split). Numbers below are for the **test split** (20 questions: 17 answerable, 3 unanswerable); the dev split (16 questions) is used only to pick the threshold.

| Metric | Test | Dev |
|---|---|---|
| Retrieval hit@1 | 76.5% | 84.6% |
| Retrieval hit@3 | 94.1% | 92.3% |
| Retrieval hit@5 | 94.1% | 92.3% |
| MRR | 0.853 | 0.885 |
| Citation precision | 79.4% | 73.1% |
| Citation recall | 85.3% | 88.5% |
| Answerability accuracy | 55.0% | 68.8% |
| - answers when it should | 47.1% | 61.5% |
| - abstains when it should | 100.0% | 100.0% |

Latency per `/ask` call (in-process, all 36 questions): p50 3.14 ms, p95 4.65 ms.

Live check: the running app with `MIN_SCORE=7.0578` abstains correctly on 55.0% of the 20 test questions.

## Test split by question type

| Type | n | hit@1 | hit@3 |
|---|---|---|---|
| direct | 10 | 90.0% | 100.0% |
| multihop | 2 | 100.0% | 100.0% |
| paraphrase | 5 | 40.0% | 80.0% |

## Definitions

- A retrieved chunk is **gold** if it comes from a gold document and contains the gold evidence passage.
- **hit@k**: share of answerable questions with at least one gold chunk in the top k. **MRR**: mean 1/rank of the first gold chunk.
- **Citation precision / recall**: the chunks the answer cites with `[n]` against the gold chunks (recall counts gold passages covered by a cited chunk; precision is 0 if nothing is cited).
- **Answerability**: a question is answered when the top score is at least `MIN_SCORE`, otherwise the app returns `no_match`.
- Latency is wall-clock and varies by machine; every other number is deterministic.
