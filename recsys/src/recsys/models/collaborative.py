"""Collaborative filtering over implicit feedback (ALS).

Watch data is implicit: no rating, only presence. So we model *confidence*
rather than preference, following Hu, Koren & Volinsky (2008):

    p_ui = 1 if the user interacted, else 0
    c_ui = 1 + alpha * r_ui        (r_ui = the engagement weight)

Missing entries are weak negatives, not unknowns. That is the whole trick, and
it is why plain SVD on a sparse matrix underperforms here.

Self-contained so the pipeline runs on numpy and scipy alone. Above roughly 1M
interactions, swap in the `implicit` library - same maths, Cython inner loop,
about two orders of magnitude faster.
"""

import numpy as np
import scipy.sparse as sp

from recsys.models.base import BaseRecommender


class CollaborativeRecommender(BaseRecommender):
    name = "collaborative"

    def __init__(self, factors: int = 64, regularization: float = 0.05,
                 iterations: int = 15, alpha: float = 40.0, seed: int = 42):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.seed = seed
        self.user_factors = None
        self.item_factors = None
        self.user_index: dict[int, int] = {}
        self.item_index: dict[int, int] = {}
        self.index_to_item: np.ndarray | None = None
        self._matrix: sp.csr_matrix | None = None

    def build_matrix(self, weights_df) -> sp.csr_matrix:
        users = weights_df["user_id"].unique()
        items = weights_df["video_id"].unique()
        self.user_index = {int(u): i for i, u in enumerate(users)}
        self.item_index = {int(v): i for i, v in enumerate(items)}
        self.index_to_item = np.array(items)

        rows = weights_df["user_id"].map(self.user_index).to_numpy()
        cols = weights_df["video_id"].map(self.item_index).to_numpy()
        vals = weights_df["weight"].to_numpy(dtype=np.float32)
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))

    def fit(self, weights_df) -> "CollaborativeRecommender":
        R = self.build_matrix(weights_df)
        self._matrix = R
        C = R * self.alpha          # confidence above the baseline of 1

        rng = np.random.default_rng(self.seed)
        n_users, n_items = R.shape
        self.user_factors = rng.normal(0, 0.01, (n_users, self.factors))
        self.item_factors = rng.normal(0, 0.01, (n_items, self.factors))

        Ct = C.T.tocsr()
        for it in range(self.iterations):
            self.user_factors = self._als_step(C, self.item_factors)
            self.item_factors = self._als_step(Ct, self.user_factors)
            print(f" [{it + 1}/{self.iterations}]", end="", flush=True)
        return self

    def _als_step(self, C: sp.csr_matrix, Y: np.ndarray) -> np.ndarray:
        """Solve one side with the other held fixed.

        The YtY precompute is what makes ALS tractable. Without it each row
        needs an O(n_items * f^2) product; with it, each row only touches the
        items it actually interacted with.
        """
        f = self.factors
        YtY = Y.T @ Y
        reg = self.regularization * np.eye(f)
        X = np.zeros((C.shape[0], f))

        for u in range(C.shape[0]):
            start, end = C.indptr[u], C.indptr[u + 1]
            idx = C.indices[start:end]
            if len(idx) == 0:
                continue
            c = C.data[start:end]
            Yu = Y[idx]
            # (YtY + Yu^T diag(c) Yu + reg) x = Yu^T (c + 1)
            A = YtY + (Yu.T * c) @ Yu + reg
            b = Yu.T @ (c + 1.0)
            X[u] = np.linalg.solve(A, b)
        return X

    def recommend(self, user_id: int, n: int = 200, filter_seen: bool = True):
        u = self.user_index.get(int(user_id))
        if u is None:
            return []

        scores = self.item_factors @ self.user_factors[u]

        if filter_seen and self._matrix is not None:
            seen = self._matrix.indices[self._matrix.indptr[u]:self._matrix.indptr[u + 1]]
            scores[seen] = -np.inf

        k = min(n, len(scores) - 1)
        top = np.argpartition(-scores, k)[:n]
        top = top[np.argsort(-scores[top])]
        return [
            (int(self.index_to_item[i]), float(scores[i]))
            for i in top if np.isfinite(scores[i])
        ]

    def get_user_factors(self, user_id: int):
        """Exposed so a reranker can use latent factors as features."""
        i = self.user_index.get(int(user_id))
        return self.user_factors[i] if i is not None else np.zeros(self.factors)

    def get_item_factors(self, video_id: int):
        i = self.item_index.get(int(video_id))
        return self.item_factors[i] if i is not None else np.zeros(self.factors)