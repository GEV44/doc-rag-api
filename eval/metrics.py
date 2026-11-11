"""Pure metric functions for the evaluation suite (no I/O, no network)."""
import math
from typing import Dict, List, Sequence, Set


def hit_at_k(relevant_flags: Sequence[bool], k: int) -> float:
    """1.0 if any of the first k ranked results is relevant, else 0.0."""
    if k <= 0:
        return 0.0
    return 1.0 if any(relevant_flags[:k]) else 0.0


def reciprocal_rank(relevant_flags: Sequence[bool]) -> float:
    """1/rank of the first relevant result (rank starts at 1); 0.0 if none."""
    for i, rel in enumerate(relevant_flags):
        if rel:
            return 1.0 / (i + 1)
    return 0.0


def precision_recall(cited: Set[str], gold: Set[str]) -> Dict[str, float]:
    """Set-based citation precision/recall.
    precision = |cited & gold| / |cited| (0.0 when nothing is cited)
    recall    = |cited & gold| / |gold|  (1.0 when gold is empty: nothing to find)"""
    inter = len(cited & gold)
    return {"precision": inter / len(cited) if cited else 0.0,
            "recall": inter / len(gold) if gold else 1.0}


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100); 0.0 for an empty list."""
    if not values:
        return 0.0
    s = sorted(values)
    rank = max(1, math.ceil(p / 100.0 * len(s)))
    return s[min(rank, len(s)) - 1]


def abstention_stats(rows: List[Dict], threshold: float) -> Dict[str, float]:
    """rows: [{'answerable': bool, 'top_score': float}]. The system abstains when
    top_score < threshold (no retrieved chunk). Returns accuracy and its two parts."""
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    answered_ok = sum(r["top_score"] >= threshold for r in ans)
    abstained_ok = sum(r["top_score"] < threshold for r in una)
    n = len(rows)
    ans_rate = answered_ok / len(ans) if ans else 0.0
    abs_rate = abstained_ok / len(una) if una else 0.0
    return {"accuracy": (answered_ok + abstained_ok) / n if n else 0.0,
            "answer_rate_on_answerable": ans_rate,
            "abstain_rate_on_unanswerable": abs_rate,
            "balanced_accuracy": (ans_rate + abs_rate) / 2}


def tune_threshold(rows: List[Dict]) -> float:
    """Pick the abstention threshold with the best balanced accuracy on `rows`
    (the dev split). Candidates: 0 and midpoints between consecutive distinct top scores.
    Ties resolve to the lowest threshold, so the system never abstains more than needed."""
    scores = sorted({r["top_score"] for r in rows})
    cands = [0.0] + [(a + b) / 2 for a, b in zip(scores, scores[1:])]
    best_t, best = 0.0, -1.0
    for t in cands:
        b = abstention_stats(rows, t)["balanced_accuracy"]
        if b > best + 1e-12:
            best_t, best = t, b
    return round(best_t, 4)
