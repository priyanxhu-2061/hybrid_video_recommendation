"""Ranking and beyond-accuracy metrics.

Every function takes a ranked list of item ids and a set of relevant ids, and
returns a single float for one user. The evaluator averages across users.

Report the beyond-accuracy metrics alongside the accuracy ones. A model that
wins on NDCG while covering 2% of the catalogue is not the model you want in
production, and reporting only NDCG hides that completely.
"""

import math

import numpy as np


# ---------------------------------------------------------------- accuracy

def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Of the top k we showed, what fraction were relevant?"""
    if k <= 0:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Of everything relevant, what fraction did we surface in the top k?"""
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def dcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Discounted cumulative gain, binary relevance.

    Gain is 1 for a hit, 0 otherwise. The discount is 1/log2(rank+1), so a hit
    at position 1 is worth 1.0, at position 2 about 0.63, at position 10 about
    0.29. That decay is the whole point: NDCG cares where the hits landed,
    precision does not.
    """
    return sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(recommended[:k], start=1)
        if item in relevant
    )


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """DCG normalised by the best achievable DCG for this user.

    The IDCG denominator uses min(len(relevant), k) items, NOT k. If a user has
    3 relevant items and k is 10, perfect ranking means hits at positions 1-3
    and nothing more is possible - dividing by a 10-hit ideal would cap that
    user's score at 0.4 no matter what. Using k here silently deflates scores
    for exactly the users who have the least data, which is most of them.
    """
    if not relevant:
        return 0.0
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    if ideal == 0:
        return 0.0
    return dcg_at_k(recommended, relevant, k) / ideal


def average_precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Mean of the precision values measured at each hit position.

    Rewards getting hits early and clustered. Averaged over users this is MAP@k.
    """
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), k)


def reciprocal_rank(recommended: list[int], relevant: set[int]) -> float:
    """1 / position of the first hit. Averaged over users this is MRR.

    Useful when the user only needs one good result - search, or the top slot
    of a feed. Ignores everything after the first hit.
    """
    for rank, item in enumerate(recommended, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Did we get at least one hit in the top k? Blunt, but easy to explain."""
    return 1.0 if set(recommended[:k]) & relevant else 0.0


# ------------------------------------------------------- beyond accuracy

def catalogue_coverage(all_recommendations: list[list[int]], n_items: int) -> float:
    """Fraction of the catalogue that appears in at least one recommendation.

    This is the metric that exposes popularity collapse. A model recommending
    the same 50 blockbusters to everyone can post a respectable NDCG while
    coverage sits near zero.
    """
    if n_items <= 0:
        return 0.0
    seen = {item for rec in all_recommendations for item in rec}
    return len(seen) / n_items


def novelty(recommended: list[int], popularity: dict[int, int], n_users: int) -> float:
    """Mean self-information of recommended items: -log2(p_i).

    An item watched by everyone carries almost no information; a long-tail item
    carries a lot. High novelty means you are surfacing things users would not
    have found on their own - which is the actual job of a recommender.
    """
    if not recommended or n_users <= 0:
        return 0.0
    scores = []
    for item in recommended:
        count = popularity.get(item, 0)
        # Unseen items get treated as seen once, so the log stays finite.
        p = max(count, 1) / n_users
        scores.append(-math.log2(p))
    return float(np.mean(scores))


def intra_list_diversity(recommended: list[int], item_vectors, id_to_row: dict) -> float:
    """Mean pairwise distance (1 - cosine) within one recommendation list.

    Low values mean ten near-identical films. Worth watching because relevance
    optimisation drives this down by default - the safest way to be accurate is
    to recommend more of the same, and users hate it.
    """
    rows = [id_to_row[i] for i in recommended if i in id_to_row]
    if len(rows) < 2:
        return 0.0

    vectors = item_vectors[rows]
    if hasattr(vectors, "toarray"):        # sparse from TF-IDF
        vectors = vectors.toarray()
    vectors = np.asarray(vectors)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = vectors / norms
    sims = normed @ normed.T

    # Upper triangle only - each pair counted once, self-similarity excluded.
    iu = np.triu_indices(len(rows), k=1)
    return float(1.0 - sims[iu].mean())


def gini_coefficient(all_recommendations: list[list[int]]) -> float:
    """How unevenly recommendation slots are spread across items.

    0 means every recommended item appears equally often; 1 means one item
    takes every slot. Complements coverage: coverage says how many items got
    shown at all, Gini says whether the exposure was concentrated in a few.
    """
    counts = {}
    for rec in all_recommendations:
        for item in rec:
            counts[item] = counts.get(item, 0) + 1
    if not counts:
        return 0.0

    values = np.sort(np.array(list(counts.values()), dtype=float))
    n = len(values)
    index = np.arange(1, n + 1)
    return float((2 * (index * values).sum()) / (n * values.sum()) - (n + 1) / n)