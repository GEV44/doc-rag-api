# Evaluation: hr_policy

Retrieval: `bm25` | answers: `extractive` | 7 documents, 16 chunks | seed 0 | top-5 retrieved per question

Confidence threshold `MIN_SCORE = 5.5221` (tuned on the dev split). Numbers below are for the **test split** (21 questions: 18 answerable, 3 unanswerable); the dev split (16 questions) is used only to pick the threshold.

| Metric | Test | Dev |
|---|---|---|
| Retrieval hit@1 | 83.3% | 76.9% |
| Retrieval hit@3 | 100.0% | 92.3% |
| Retrieval hit@5 | 100.0% | 92.3% |
| MRR | 0.907 | 0.821 |
| Citation precision | 75.0% | 53.8% |
| Citation recall | 88.9% | 73.1% |
| Answerability accuracy | 66.7% | 68.8% |
| - answers when it should | 61.1% | 61.5% |
| - abstains when it should | 100.0% | 100.0% |

Latency per `/ask` call (in-process, all 37 questions): p50 3.28 ms, p95 5.61 ms.

Live check: the running app with `MIN_SCORE=5.5221` abstains correctly on 66.7% of the 21 test questions.

## Test split by question type

| Type | n | hit@1 | hit@3 |
|---|---|---|---|
| direct | 11 | 100.0% | 100.0% |
| multihop | 2 | 100.0% | 100.0% |
| paraphrase | 5 | 40.0% | 100.0% |

## Definitions

- A retrieved chunk is **gold** if it comes from a gold document and contains the gold evidence passage.
- **hit@k**: share of answerable questions with at least one gold chunk in the top k. **MRR**: mean 1/rank of the first gold chunk.
- **Citation precision / recall**: the chunks the answer cites with `[n]` against the gold chunks (recall counts gold passages covered by a cited chunk; precision is 0 if nothing is cited).
- **Answerability**: a question is answered when the top score is at least `MIN_SCORE`, otherwise the app returns `no_match`.
- Latency is wall-clock and varies by machine; every other number is deterministic.
