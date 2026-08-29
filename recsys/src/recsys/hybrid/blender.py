"""Merges candidate lists from every retriever into one scored pool.

Four strategies, config-selectable:

weighted     normalise each source's scores to [0,1], then a weighted sum.
             Simple, interpretable, needs score calibration across sources.

switching    pick ONE source per request based on a rule (history length,
             item age). Avoids blending incomparable scores; loses ensembling.

cascade      content narrows to a broad pool, CF orders within it. Good when
             one signal is far more trustworthy than the other.

rank_fusion  Reciprocal Rank Fusion: sum 1/(k + rank). Ignores raw scores
             entirely, so nothing needs calibrating. Strong default.

Start with rank_fusion, move to `ranking/` once you have enough logged
impressions to train a reranker. The learned ranker eventually replaces most of
what this file does.
"""

from collections import defaultdict


class HybridBlender:
    def __init__(self, strategy: str = "rank_fusion", weights: dict | None = None, k: int = 60):
        self.strategy = strategy
        self.weights = weights or {}
        self.k = k

    def merge(self, candidates: dict[str, list[tuple[int, float]]]) -> list[dict]:
        """candidates: {source_name: [(video_id, score), ...]} ordered best first."""
        if self.strategy == "rank_fusion":
            return self._rrf(candidates)
        if self.strategy == "weighted":
            return self._weighted(candidates)
        raise NotImplementedError(self.strategy)

    def _rrf(self, candidates):
        scores, sources = defaultdict(float), defaultdict(list)
        for source, items in candidates.items():
            weight = self.weights.get(source, 1.0)
            for rank, (video_id, _) in enumerate(items, start=1):
                scores[video_id] += weight / (self.k + rank)
                sources[video_id].append(source)
        return self._to_pool(scores, sources)

    def _weighted(self, candidates):
        scores, sources = defaultdict(float), defaultdict(list)
        for source, items in candidates.items():
            if not items:
                continue
            raw = [s for _, s in items]
            lo, hi = min(raw), max(raw)
            span = (hi - lo) or 1.0
            weight = self.weights.get(source, 0.0)
            for video_id, score in items:
                scores[video_id] += weight * (score - lo) / span
                sources[video_id].append(source)
        return self._to_pool(scores, sources)

    @staticmethod
    def _to_pool(scores, sources):
        pool = [
            {"video_id": vid, "score": score, "sources": sources[vid]}
            for vid, score in scores.items()
        ]
        pool.sort(key=lambda x: x["score"], reverse=True)
        return pool
