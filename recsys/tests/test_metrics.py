"""Metric correctness against hand-computed values.

Every expected value here was worked out by hand before the implementation was
trusted. That matters more than it sounds: a broken metric does not raise an
error, it returns a plausible number, and every result downstream is quietly
wrong. These are the assertions that would catch a regression.
"""

import math

import pytest

from recsys.evaluation import metrics as m

# Recommended list and the relevant set used throughout.
# Hits fall at positions 2, 4 and 5.
REC = [5, 3, 9, 1, 7]
REL = {3, 1, 7}


def test_precision_at_k():
    # Top 3 are [5, 3, 9]; only 3 is relevant -> 1/3
    assert m.precision_at_k(REC, REL, 3) == pytest.approx(1 / 3)
    # Top 5 contains all three relevant items -> 3/5
    assert m.precision_at_k(REC, REL, 5) == pytest.approx(0.6)


def test_precision_handles_zero_k():
    assert m.precision_at_k(REC, REL, 0) == 0.0


def test_recall_at_k():
    # One of three relevant items found in the top 3
    assert m.recall_at_k(REC, REL, 3) == pytest.approx(1 / 3)
    # All three found in the top 5
    assert m.recall_at_k(REC, REL, 5) == pytest.approx(1.0)


def test_recall_with_no_relevant_items():
    assert m.recall_at_k(REC, set(), 5) == 0.0


def test_dcg_hand_computed():
    # 1/log2(3) + 1/log2(5) + 1/log2(6)
    expected = 1 / math.log2(3) + 1 / math.log2(5) + 1 / math.log2(6)
    assert m.dcg_at_k(REC, REL, 5) == pytest.approx(expected)
    assert m.dcg_at_k(REC, REL, 5) == pytest.approx(1.4485, abs=1e-4)


def test_ndcg_hand_computed():
    # IDCG uses min(|R|, k) = 3 positions: 1 + 1/log2(3) + 1/log2(4) = 2.1309
    # NDCG = 1.4485 / 2.1309 = 0.6797
    assert m.ndcg_at_k(REC, REL, 5) == pytest.approx(0.6797, abs=1e-4)


def test_ndcg_perfect_ordering_is_exactly_one():
    """The single most important assertion in this file.

    If the IDCG denominator uses k instead of min(|R|, k), this returns 0.4
    rather than 1.0 and every reported NDCG is deflated for users with few
    relevant items - which is most users.
    """
    perfect = [3, 1, 7, 5, 9]
    assert m.ndcg_at_k(perfect, REL, 5) == pytest.approx(1.0)


def test_ndcg_with_no_hits_is_zero():
    assert m.ndcg_at_k([100, 200, 300], REL, 3) == 0.0


def test_ndcg_rewards_earlier_hits():
    """NDCG must distinguish orderings that precision cannot."""
    early = [3, 1, 7, 99, 98]
    late = [99, 98, 3, 1, 7]
    assert m.precision_at_k(early, REL, 5) == m.precision_at_k(late, REL, 5)
    assert m.ndcg_at_k(early, REL, 5) > m.ndcg_at_k(late, REL, 5)


def test_average_precision_hand_computed():
    # Hits at ranks 2, 4, 5 -> precisions 1/2, 2/4, 3/5 = 1.6, over 3 relevant
    assert m.average_precision_at_k(REC, REL, 5) == pytest.approx(1.6 / 3)
    assert m.average_precision_at_k(REC, REL, 5) == pytest.approx(0.5333, abs=1e-4)


def test_reciprocal_rank():
    # First hit is at position 2
    assert m.reciprocal_rank(REC, REL) == pytest.approx(0.5)
    assert m.reciprocal_rank([3, 5, 9], REL) == pytest.approx(1.0)
    assert m.reciprocal_rank([100, 200], REL) == 0.0


def test_hit_rate():
    assert m.hit_rate_at_k(REC, REL, 2) == 1.0
    assert m.hit_rate_at_k(REC, REL, 1) == 0.0


def test_catalogue_coverage():
    # Items {1, 2, 3} shown out of a catalogue of 10
    assert m.catalogue_coverage([[1, 2], [2, 3]], 10) == pytest.approx(0.3)
    assert m.catalogue_coverage([], 10) == 0.0
    assert m.catalogue_coverage([[1]], 0) == 0.0


def test_coverage_exposes_popularity_collapse():
    """Coverage is what reveals a model recommending the same items to everyone."""
    concentrated = [[1, 2, 3] for _ in range(100)]
    spread = [[i, i + 1, i + 2] for i in range(1, 100, 3)]
    assert m.catalogue_coverage(concentrated, 200) < m.catalogue_coverage(spread, 200)


def test_gini_hand_computed():
    # Item 1 appears twice, item 2 once. Sorted [1, 2], n = 2:
    # (2 * (1*1 + 2*2)) / (2 * 3) - 3/2 = 10/6 - 1.5 = 0.1667
    assert m.gini_coefficient([[1, 1, 2]]) == pytest.approx(0.1667, abs=1e-4)


def test_gini_is_zero_when_exposure_is_equal():
    assert m.gini_coefficient([[1, 2, 3]]) == pytest.approx(0.0, abs=1e-9)


def test_gini_on_empty_input():
    assert m.gini_coefficient([]) == 0.0