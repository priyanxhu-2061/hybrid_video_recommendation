"""Fusion correctness.

The property that matters is that agreement across sources beats a strong score
from a single source. If that breaks, the hybrid silently degrades into
whichever retriever happens to produce the largest numbers.
"""

import pytest

from recsys.hybrid.blender import HybridBlender

# Item 2 is ranked second by content and first by collaborative.
# Item 1 is ranked first by content and third by collaborative.
# Item 3 appears in one list only, despite a high score there.
CANDIDATES = {
    "content": [(1, 0.9), (2, 0.8), (3, 0.7)],
    "collaborative": [(2, 5.0), (4, 4.0), (1, 3.0)],
}


def _order(pool):
    return [p["video_id"] for p in pool]


def test_rrf_rewards_cross_source_agreement():
    """The central property of rank fusion.

    Item 2 is not top-ranked in either list on its own merits alone, but it
    appears near the top of both. It must outrank item 1, which leads one list.
    """
    pool = HybridBlender(strategy="rank_fusion").merge(CANDIDATES)
    assert _order(pool)[0] == 2
    assert _order(pool).index(2) < _order(pool).index(1)


def test_rrf_scores_are_hand_computed():
    # Item 2: 1/(60+2) + 1/(60+1) = 0.016129 + 0.016393 = 0.032522
    pool = HybridBlender(strategy="rank_fusion").merge(CANDIDATES)
    item2 = next(p for p in pool if p["video_id"] == 2)
    assert item2["score"] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_records_every_contributing_source():
    """Source attribution is what makes recommendations explainable."""
    pool = HybridBlender(strategy="rank_fusion").merge(CANDIDATES)
    item2 = next(p for p in pool if p["video_id"] == 2)
    assert set(item2["sources"]) == {"content", "collaborative"}

    item3 = next(p for p in pool if p["video_id"] == 3)
    assert item3["sources"] == ["content"]


def test_rrf_ignores_raw_score_scale():
    """The point of rank fusion: only ordinal position matters.

    Multiplying one source's scores by a thousand must not change the result,
    which is exactly what weighted fusion cannot promise without calibration.
    """
    inflated = {
        "content": [(1, 0.9), (2, 0.8), (3, 0.7)],
        "collaborative": [(2, 5000.0), (4, 4000.0), (1, 3000.0)],
    }
    a = HybridBlender(strategy="rank_fusion").merge(CANDIDATES)
    b = HybridBlender(strategy="rank_fusion").merge(inflated)
    assert _order(a) == _order(b)


def test_rrf_source_weights_shift_the_ranking():
    weighted = HybridBlender(strategy="rank_fusion",
                             weights={"content": 5.0, "collaborative": 0.1})
    pool = weighted.merge(CANDIDATES)
    # With content heavily favoured, its top item should lead.
    assert _order(pool)[0] == 1


def test_weighted_fusion_normalises_into_unit_range():
    """Min-max normalisation must bound every contribution.

    Without it, whichever source has the largest raw scores dominates the blend
    regardless of the weights.
    """
    blender = HybridBlender(strategy="weighted",
                            weights={"content": 0.5, "collaborative": 0.5})
    pool = blender.merge(CANDIDATES)
    for entry in pool:
        assert 0.0 <= entry["score"] <= 1.0


def test_pool_is_sorted_descending():
    pool = HybridBlender(strategy="rank_fusion").merge(CANDIDATES)
    scores = [p["score"] for p in pool]
    assert scores == sorted(scores, reverse=True)


def test_empty_input_produces_empty_pool():
    assert HybridBlender(strategy="rank_fusion").merge({}) == []


def test_empty_source_list_is_skipped():
    pool = HybridBlender(strategy="rank_fusion").merge(
        {"content": [(1, 0.9)], "collaborative": []})
    assert _order(pool) == [1]


def test_unknown_strategy_raises():
    with pytest.raises(NotImplementedError):
        HybridBlender(strategy="nonsense").merge(CANDIDATES)