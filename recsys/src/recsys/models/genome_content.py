"""Content model built on the MovieLens tag genome.

The TF-IDF model treats tags as text: a bag of words scraped from what users
happened to type. The genome is different in kind - 1,128 curated dimensions,
each a continuous relevance score in [0, 1] produced from thousands of user
judgements. It is already a dense embedding, no training required.

Expect this to beat TF-IDF substantially. TF-IDF over short tag strings has
almost nothing to work with: most films carry a handful of tokens, so the
vectors are sparse and near-orthogonal, and cosine similarity between two films
that share no exact tag is zero even when they are obviously alike.

Limitation worth stating: the genome only covers ~10,400 of MovieLens's 27,000
films. Anything outside it falls back to a zero vector and is never recommended
by this model - so keep the TF-IDF model in the ensemble rather than replacing
it outright.
"""

import numpy as np

from recsys.models.base import BaseRecommender


class GenomeContentRecommender(BaseRecommender):
    name = "content_genome"

    def __init__(self, genome):
        """genome: the (movie_ids, matrix, tag_names) tuple from load_genome."""
        movie_ids, matrix, tag_names = genome
        self.item_ids = movie_ids
        self.tag_names = tag_names
        self.id_to_row = {int(v): i for i, v in enumerate(movie_ids)}

        # L2-normalise once so every later dot product is a cosine.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.item_vectors = (matrix / norms).astype(np.float32)

    def fit(self, *args, **kwargs) -> "GenomeContentRecommender":
        """Nothing to fit - the genome is already trained. Present so this
        drops into the pipeline wherever the TF-IDF model goes."""
        return self

    def build_user_profile(self, history, weights=None):
        pairs = [(self.id_to_row[v], i) for i, v in enumerate(history) if v in self.id_to_row]
        if not pairs:
            return None

        rows = [r for r, _ in pairs]
        vectors = self.item_vectors[rows]

        if weights:
            w = np.array([weights[i] for _, i in pairs], dtype=np.float32)
            total = w.sum()
            profile = vectors.mean(axis=0) if total <= 0 else (vectors * (w / total)[:, None]).sum(axis=0)
        else:
            profile = vectors.mean(axis=0)

        norm = np.linalg.norm(profile)
        return profile / norm if norm > 0 else profile

    def recommend(self, history, n: int = 200, weights=None, exclude_seen: bool = True):
        profile = self.build_user_profile(history, weights)
        if profile is None:
            return []

        scores = self.item_vectors @ profile

        if exclude_seen:
            for v in history:
                row = self.id_to_row.get(v)
                if row is not None:
                    scores[row] = -np.inf

        k = min(n, len(scores) - 1)
        top = np.argpartition(-scores, k)[:n]
        top = top[np.argsort(-scores[top])]
        return [(int(self.item_ids[i]), float(scores[i])) for i in top if np.isfinite(scores[i])]

    def similar_items(self, video_id: int, n: int = 10):
        row = self.id_to_row.get(video_id)
        if row is None:
            return []
        scores = self.item_vectors @ self.item_vectors[row]
        scores[row] = -np.inf
        k = min(n, len(scores) - 1)
        top = np.argpartition(-scores, k)[:n]
        top = top[np.argsort(-scores[top])]
        return [(int(self.item_ids[i]), float(scores[i])) for i in top]