"""Post-ranking reordering.

A pure relevance ranking hands back twenty near-identical films. That is not a
bug in the ranker - it is the ranker doing its job. The safest way to be
accurate is to recommend more of what the user already liked, and the result is
a feed nobody wants to scroll.

MMR trades a measured amount of relevance for spread. The caps are blunter: no
more than N items from one category or creator, regardless of score.

Order matters. This runs AFTER ranking, never before. Diversifying the candidate
pool just starves the ranker of good options to choose between.
"""

import numpy as np


class Diversifier:
    def __init__(
        self,
        lambda_: float = 0.75,
        max_per_creator: int = 2,
        max_per_category: int = 4,
    ):
        """lambda_: 1.0 is pure relevance, 0.0 is pure diversity.

        0.7-0.8 is the usual working range. Below about 0.5 the feed starts
        surfacing things the user plainly does not want, and the accuracy cost
        stops being worth the variety.
        """
        self.lambda_ = lambda_
        self.max_per_creator = max_per_creator
        self.max_per_category = max_per_category

    def apply(
        self,
        ranked: list[dict],
        item_vectors=None,
        id_to_row: dict | None = None,
        item_meta: dict | None = None,
        top_k: int = 20,
    ) -> list[dict]:
        """Greedy Maximal Marginal Relevance with hard caps.

        ranked:       [{"video_id": int, "score": float, ...}], best first
        item_vectors: matrix used for similarity (TF-IDF, genome, ALS factors)
        id_to_row:    video_id -> row index in item_vectors
        item_meta:    video_id -> {"category": str, "creator_id": str}
        """
        if not ranked:
            return []

        item_meta = item_meta or {}
        id_to_row = id_to_row or {}

        # Without vectors there is nothing to measure similarity against, so
        # fall back to caps only. Still useful - most of the visible benefit
        # comes from the caps anyway.
        if item_vectors is None or not id_to_row:
            return self._caps_only(ranked, item_meta, top_k)

        relevance = self._normalise([r["score"] for r in ranked])

        vectors = self._dense_rows(ranked, item_vectors, id_to_row)
        sims = vectors @ vectors.T  # rows are L2-normalised, so this is cosine

        selected: list[int] = []          # indices into `ranked`
        remaining = set(range(len(ranked)))
        category_counts: dict[str, int] = {}
        creator_counts: dict[str, int] = {}

        while remaining and len(selected) < top_k:
            best_idx, best_value = None, -np.inf

            for idx in remaining:
                meta = item_meta.get(ranked[idx]["video_id"], {})
                if self._would_exceed_caps(meta, category_counts, creator_counts):
                    continue

                if selected:
                    max_sim = sims[idx, selected].max()
                else:
                    max_sim = 0.0

                value = self.lambda_ * relevance[idx] - (1 - self.lambda_) * max_sim
                if value > best_value:
                    best_idx, best_value = idx, value

            if best_idx is None:
                # Every remaining candidate is capped out. Relax rather than
                # return a short list - a feed of twelve items when the caller
                # asked for twenty is a worse failure than a slightly
                # repetitive one.
                return self._fill_remainder(ranked, selected, top_k)

            selected.append(best_idx)
            remaining.discard(best_idx)

            meta = item_meta.get(ranked[best_idx]["video_id"], {})
            self._increment(meta, category_counts, creator_counts)

        return [dict(ranked[i], mmr_position=pos) for pos, i in enumerate(selected)]

    # ------------------------------------------------------------ internals

    def _would_exceed_caps(self, meta, category_counts, creator_counts) -> bool:
        category = meta.get("category")
        creator = meta.get("creator_id")
        if category and category_counts.get(category, 0) >= self.max_per_category:
            return True
        if creator and creator_counts.get(creator, 0) >= self.max_per_creator:
            return True
        return False

    @staticmethod
    def _increment(meta, category_counts, creator_counts) -> None:
        category = meta.get("category")
        creator = meta.get("creator_id")
        if category:
            category_counts[category] = category_counts.get(category, 0) + 1
        if creator:
            creator_counts[creator] = creator_counts.get(creator, 0) + 1

    @staticmethod
    def _normalise(scores: list[float]) -> np.ndarray:
        """Min-max to [0, 1].

        Required, not cosmetic. Similarity is bounded in [0, 1]; if relevance
        arrives on some other scale - ALS dot products can run past 5 - then
        lambda no longer balances the two terms and MMR silently degenerates
        into pure relevance.
        """
        arr = np.asarray(scores, dtype=float)
        lo, hi = arr.min(), arr.max()
        span = hi - lo
        if span <= 0:
            return np.ones_like(arr)
        return (arr - lo) / span

    @staticmethod
    def _dense_rows(ranked, item_vectors, id_to_row) -> np.ndarray:
        """Pull the candidate rows out and L2-normalise them.

        Items missing from id_to_row get a zero row: similarity zero to
        everything, so they are never penalised for being unknown.
        """
        n_features = item_vectors.shape[1]
        out = np.zeros((len(ranked), n_features), dtype=np.float32)

        for i, item in enumerate(ranked):
            row = id_to_row.get(item["video_id"])
            if row is None:
                continue
            vector = item_vectors[row]
            if hasattr(vector, "toarray"):
                vector = vector.toarray()
            out[i] = np.asarray(vector, dtype=np.float32).ravel()

        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def _caps_only(self, ranked, item_meta, top_k) -> list[dict]:
        """Relevance order, skipping anything that busts a cap."""
        selected, category_counts, creator_counts = [], {}, {}

        for item in ranked:
            if len(selected) >= top_k:
                break
            meta = item_meta.get(item["video_id"], {})
            if self._would_exceed_caps(meta, category_counts, creator_counts):
                continue
            selected.append(item)
            self._increment(meta, category_counts, creator_counts)

        if len(selected) < top_k:
            chosen = {i["video_id"] for i in selected}
            for item in ranked:
                if len(selected) >= top_k:
                    break
                if item["video_id"] not in chosen:
                    selected.append(item)

        return [dict(item, mmr_position=pos) for pos, item in enumerate(selected)]

    @staticmethod
    def _fill_remainder(ranked, selected, top_k) -> list[dict]:
        """Top up from the highest-scoring unselected items, caps ignored."""
        out = list(selected)
        for idx in range(len(ranked)):
            if len(out) >= top_k:
                break
            if idx not in out:
                out.append(idx)
        return [dict(ranked[i], mmr_position=pos) for pos, i in enumerate(out)]