"""Runs every model over the test window and writes a comparison table."""
"""Runs a recommender over the evaluation window and reports metrics.

The design choice worth noticing: `recommend_fn` is a plain callable taking a
user id and returning an ordered list of item ids. That single interface lets
you evaluate one retriever, a blend of several, or the full pipeline through
exactly the same code path - which is what makes the ablation table honest.
Anything that can rank items for a user can be measured here.
"""

import time

import numpy as np
import pandas as pd

from recsys.evaluation import metrics as m


class Evaluator:
    def __init__(
        self,
        k_values: list[int] | None = None,
        max_k: int | None = None,
        item_vectors=None,
        id_to_row: dict | None = None,
    ):
        self.k_values = sorted(k_values or [5, 10, 20])
        # Ask each model for the largest k we intend to measure, once, then
        # slice. Calling recommend() separately per k would be the same work
        # repeated three times.
        self.max_k = max_k or max(self.k_values)
        self.item_vectors = item_vectors
        self.id_to_row = id_to_row or {}

    def evaluate(
        self,
        recommend_fn,
        ground_truth: dict[int, set[int]],
        n_items: int,
        popularity: dict[int, int] | None = None,
        n_users_total: int | None = None,
        max_users: int | None = None,
        seed: int = 42,
    ) -> dict:
        """Average every metric across users in `ground_truth`.

        max_users: subsample for speed during development. Metrics on 500 users
        are stable enough to compare models; use the full set for final numbers.
        """
        users = list(ground_truth.keys())
        if max_users is not None and max_users < len(users):
            rng = np.random.default_rng(seed)
            users = list(rng.choice(users, size=max_users, replace=False))

        per_user = {self._key(name, k): [] for k in self.k_values
                    for name in ("precision", "recall", "ndcg", "map", "hit_rate")}
        per_user["mrr"] = []
        per_user["novelty"] = []
        per_user["diversity"] = []

        all_recommendations = []
        skipped = 0
        start = time.time()

        for user_id in users:
            relevant = ground_truth[user_id]
            if not relevant:
                skipped += 1
                continue

            recommended = recommend_fn(user_id)
            if not recommended:
                # A model that returns nothing still gets scored - zeroes, not
                # exclusion. Dropping these users would let a model that fails
                # on half the population post the same average as one that
                # works for everybody.
                recommended = []

            recommended = list(recommended)[:self.max_k]
            all_recommendations.append(recommended)

            for k in self.k_values:
                per_user[self._key("precision", k)].append(m.precision_at_k(recommended, relevant, k))
                per_user[self._key("recall", k)].append(m.recall_at_k(recommended, relevant, k))
                per_user[self._key("ndcg", k)].append(m.ndcg_at_k(recommended, relevant, k))
                per_user[self._key("map", k)].append(m.average_precision_at_k(recommended, relevant, k))
                per_user[self._key("hit_rate", k)].append(m.hit_rate_at_k(recommended, relevant, k))

            per_user["mrr"].append(m.reciprocal_rank(recommended, relevant))

            if popularity is not None and n_users_total:
                per_user["novelty"].append(m.novelty(recommended, popularity, n_users_total))

            if self.item_vectors is not None:
                per_user["diversity"].append(
                    m.intra_list_diversity(recommended, self.item_vectors, self.id_to_row)
                )

        results = {
            name: float(np.mean(values)) if values else 0.0
            for name, values in per_user.items()
        }

        # Catalogue-level metrics need every list at once, not per-user means.
        results["coverage"] = m.catalogue_coverage(all_recommendations, n_items)
        results["gini"] = m.gini_coefficient(all_recommendations)
        results["users_evaluated"] = len(all_recommendations)
        results["users_skipped"] = skipped
        results["seconds"] = round(time.time() - start, 1)

        return results

    def compare(
        self,
        models: dict,
        ground_truth: dict[int, set[int]],
        n_items: int,
        **kwargs,
    ) -> pd.DataFrame:
        """Evaluate several models and return one table, best NDCG first.

        Always include a popularity baseline in `models`. A hybrid that cannot
        beat time-decayed popularity is not working, and without the baseline
        sitting in the same table you have no way to notice.
        """
        rows = {}
        for name, fn in models.items():
            print(f"  evaluating {name} ...", end="", flush=True)
            rows[name] = self.evaluate(fn, ground_truth, n_items, **kwargs)
            print(f" done ({rows[name]['seconds']}s)")

        df = pd.DataFrame(rows).T
        sort_key = self._key("ndcg", self.k_values[1] if len(self.k_values) > 1 else self.k_values[0])
        if sort_key in df.columns:
            df = df.sort_values(sort_key, ascending=False)
        return df

    @staticmethod
    def _key(name: str, k: int) -> str:
        return f"{name}@{k}"


def format_table(df: pd.DataFrame, k: int = 10) -> str:
    """Trim the full results frame to the columns worth putting in a paper."""
    columns = [
        f"precision@{k}", f"recall@{k}", f"ndcg@{k}",
        f"map@{k}", f"hit_rate@{k}", "mrr",
        "coverage", "gini", "novelty", "diversity",
    ]
    present = [c for c in columns if c in df.columns]
    return df[present].round(4).to_string(float_format=lambda x: f"{x:>9.4f}")


def popularity_counts(interactions_df) -> tuple[dict[int, int], int]:
    """Interaction counts per item, for the novelty metric.

    Build this from the TRAINING window only. Using the full dataset would let
    an item's future popularity influence how novel it looked at the time it
    was recommended - a small leak, but a leak.
    """
    engaged = interactions_df[interactions_df["event_type"] != "impression"]
    counts = engaged.groupby("video_id").size().to_dict()
    return counts, engaged["user_id"].nunique()