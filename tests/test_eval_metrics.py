import os

import pytest

from eval.metrics import (abstention_stats, hit_at_k, mean, percentile, precision_recall,
                          reciprocal_rank, tune_threshold)
from eval.run_eval import DATASETS_DIR, load_dataset


def test_hit_at_k_basic_and_edges():
    flags = [False, False, True, False]
    assert hit_at_k(flags, 1) == 0.0
    assert hit_at_k(flags, 3) == 1.0
    assert hit_at_k(flags, 10) == 1.0          # k larger than the list
    assert hit_at_k(flags, 0) == 0.0           # k = 0
    assert hit_at_k([], 3) == 0.0              # nothing retrieved


def test_reciprocal_rank():
    assert reciprocal_rank([True]) == 1.0
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([False, True, True]) == 0.5   # first relevant counts
    assert reciprocal_rank([False, False]) == 0.0
    assert reciprocal_rank([]) == 0.0


def test_precision_recall():
    assert precision_recall({"a", "b"}, {"a"}) == {"precision": 0.5, "recall": 1.0}
    assert precision_recall({"a"}, {"a", "b"}) == {"precision": 1.0, "recall": 0.5}
    assert precision_recall({"x"}, {"a"}) == {"precision": 0.0, "recall": 0.0}


def test_precision_recall_empty_sets():
    assert precision_recall(set(), {"a"}) == {"precision": 0.0, "recall": 0.0}   # nothing cited
    assert precision_recall({"a"}, set()) == {"precision": 0.0, "recall": 1.0}   # nothing to find
    assert precision_recall(set(), set()) == {"precision": 0.0, "recall": 1.0}


def test_percentile_nearest_rank():
    v = [10, 20, 30, 40, 50]
    assert percentile(v, 50) == 30
    assert percentile(v, 95) == 50
    assert percentile(v, 0) == 10
    assert percentile([7], 95) == 7
    assert percentile([], 50) == 0.0
    assert percentile([5, 1, 3], 50) == 3      # unsorted input


def test_mean_empty():
    assert mean([]) == 0.0
    assert mean([1, 2]) == 1.5


ROWS = [
    {"answerable": True, "top_score": 8.0}, {"answerable": True, "top_score": 5.0},
    {"answerable": False, "top_score": 2.0}, {"answerable": False, "top_score": 4.0},
]


def test_abstention_stats():
    s = abstention_stats(ROWS, 4.5)
    assert s["answer_rate_on_answerable"] == 1.0
    assert s["abstain_rate_on_unanswerable"] == 1.0
    assert s["accuracy"] == 1.0
    s0 = abstention_stats(ROWS, 0.0)            # never abstain
    assert s0["accuracy"] == 0.5 and s0["abstain_rate_on_unanswerable"] == 0.0
    assert abstention_stats([], 1.0)["accuracy"] == 0.0


def test_tune_threshold_separates_classes_and_is_deterministic():
    t = tune_threshold(ROWS)
    assert 4.0 < t <= 5.0
    assert abstention_stats(ROWS, t)["accuracy"] == 1.0
    assert tune_threshold(ROWS) == t
    assert tune_threshold([{"answerable": True, "top_score": 3.0}]) == 0.0   # nothing to abstain on


@pytest.mark.parametrize("name", sorted(os.listdir(DATASETS_DIR)))
def test_bundled_datasets_are_consistent(name):
    docs, qs = load_dataset(os.path.join(DATASETS_DIR, name))   # also checks evidence is verbatim
    assert 30 <= len(qs) <= 50
    assert {q["split"] for q in qs} == {"dev", "test"}
    assert len({q["id"] for q in qs}) == len(qs)
    unans = [q for q in qs if not q["sources"]]
    assert 0.1 <= len(unans) / len(qs) <= 0.25
    assert all(q["answer"] == "not in documents" for q in unans)
