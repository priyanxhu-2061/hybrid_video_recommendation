"""Item-item co-watch similarity.

Often beats matrix factorisation on short histories, and it is trivially
explainable - "people who watched this also watched". Cheap enough to keep
alongside ALS rather than choosing between them.

The shrink term is what makes it work. Without it, a pair co-rated by two users
scores as highly as a pair co-rated by two thousand, and the neighbour lists
fill up with noise from obscure films.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from recsys.models.base import BaseRecommender


class ItemKNNRecommender(BaseRecommender):
    name = "item_knn"

    def __init__(self, k: int = 50, shrink: float = 100.0):
        self.k = k
        self.shrink = shrink
        self.similarity: sp.csr_matrix | None = None
        self.item_index: dict[int, int] = {}
        self.index_to_item: np.ndarray | None = None

    def fit(self, weights_df) -> "ItemKNNRecommender":
        items = weights_df["video_id"].unique()
        users = weights_df["user_id"].unique()
        self.item_index = {v: i for i, v in enumerate(items)}
        self.index_to_item = np.array(items)
        user_index = {u: i for i, u in enumerate(users)}

        rows = weights_df["video_id"].map(self.item_index).to_numpy()
        cols = weights_df["user_id"].map(user_index).to_numpy()
        vals = weights_df["weight"].to_numpy(dtype=np.float32)
        R = sp.csr_matrix((vals, (rows, cols)), shape=(len(items), len(users)))

        # Cosine similarity between item rows.
        norms = np.sqrt(np.asarray(R.multiply(R).sum(axis=1))).ravel()
        Rn = normalize(R)
        sim = np.asarray((Rn @ Rn.T).todense())

        # Shrink toward zero where the co-occurrence support is thin.
        support = np.outer(norms, norms)
        sim = sim * support / (support + self.shrink)
        np.fill_diagonal(sim, 0.0)

        # Keep only the top k neighbours per item. A dense similarity matrix
        # does not survive a real catalogue - at 11k items this is already
        # 121M floats, and pruning is what keeps the artifact loadable.
        if self.k < sim.shape[1]:
            cut = np.argpartition(-sim, self.k, axis=1)[:, self.k:]
            np.put_along_axis(sim, cut, 0.0, axis=1)

        self.similarity = sp.csr_matrix(sim)
        return self

    def recommend(self, history, n: int = 200, weights=None, exclude_seen: bool = True):
        rows = [self.item_index[v] for v in history if v in self.item_index]
        if not rows:
            return []

        w = np.ones(len(rows))
        if weights:
            w = np.array([
                wt for v, wt in zip(history, weights) if v in self.item_index
            ], dtype=float)

        # Sum of neighbour similarities, weighted by how much the user liked
        # the item each neighbour came from.
        scores = np.asarray((self.similarity[rows].T @ w)).ravel()

        if exclude_seen:
            scores[rows] = -np.inf

        k = min(n, len(scores) - 1)
        top = np.argpartition(-scores, k)[:n]
        top = top[np.argsort(-scores[top])]
        return [
            (int(self.index_to_item[i]), float(scores[i]))
            for i in top if np.isfinite(scores[i]) and scores[i] > 0
        ]