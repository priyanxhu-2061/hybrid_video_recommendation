"""Content-based retriever.

Builds an item vector from title, genres and community tags, then scores by
similarity to a profile built from what the user actually watched.

Strengths: works for brand-new items (the vector exists from metadata alone),
explainable, no item cold start.
Weakness: filter bubble. It only ever finds more of what the user already saw,
which is precisely the failure collaborative filtering exists to cover.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recsys.models.base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    name = "content"

    def __init__(self, max_features: int = 20_000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            min_df=2,
        )
        self.item_vectors = None       # L2-normalised, so dot product is cosine
        self.item_ids: np.ndarray | None = None
        self.id_to_row: dict[int, int] = {}

    @staticmethod
    def _document(row) -> str:
        """Title is repeated deliberately - it carries more signal per token
        than the tag soup, and TF-IDF has no other way to know that."""
        tags = row["tags"] if isinstance(row["tags"], list) else []
        return f"{row['title']} {row['title']} {row['category']} {' '.join(tags)} {row['description']}"

    def fit(self, videos_df) -> "ContentBasedRecommender":
        docs = videos_df.apply(self._document, axis=1).tolist()
        self.item_vectors = normalize(self.vectorizer.fit_transform(docs))
        self.item_ids = videos_df["video_id"].to_numpy()
        self.id_to_row = {int(v): i for i, v in enumerate(self.item_ids)}
        return self

    def build_user_profile(self, history: list[int], weights: list[float] | None = None):
        """Weighted mean of watched item vectors.

        Weight by engagement, not a flat average - a film the user rated 1 star
        should not pull the profile as hard as one they rated 5.
        """
        pairs = [(self.id_to_row[v], i) for i, v in enumerate(history) if v in self.id_to_row]
        if not pairs:
            return None

        rows = [r for r, _ in pairs]
        vectors = self.item_vectors[rows]

        if weights:
            w = np.array([weights[i] for _, i in pairs], dtype=float)
            total = w.sum()
            if total <= 0:
                profile = vectors.mean(axis=0)
            else:
                profile = vectors.multiply((w / total).reshape(-1, 1)).sum(axis=0)
        else:
            profile = vectors.mean(axis=0)

        return normalize(np.asarray(profile))

    def recommend(self, history, n: int = 200, weights=None, exclude_seen: bool = True):
        profile = self.build_user_profile(history, weights)
        if profile is None:
            return []

        scores = np.asarray(self.item_vectors @ profile.T).ravel()

        if exclude_seen:
            for v in history:
                row = self.id_to_row.get(v)
                if row is not None:
                    scores[row] = -np.inf

        k = min(n, len(scores) - 1)
        top = np.argpartition(-scores, k)[:n]
        top = top[np.argsort(-scores[top])]
        return [
            (int(self.item_ids[i]), float(scores[i]))
            for i in top if np.isfinite(scores[i])
        ]

    def similar_items(self, video_id: int, n: int = 10):
        row = self.id_to_row.get(video_id)
        if row is None:
            return []
        scores = np.asarray((self.item_vectors @ self.item_vectors[row].T).todense()).ravel()
        scores[row] = -np.inf
        k = min(n, len(scores) - 1)
        top = np.argpartition(-scores, k)[:n]
        top = top[np.argsort(-scores[top])]
        return [(int(self.item_ids[i]), float(scores[i])) for i in top]