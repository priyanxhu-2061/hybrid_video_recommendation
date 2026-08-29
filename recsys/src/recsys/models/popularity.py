"""Trending and popularity baselines.

Two jobs: cold-start fallback, and the baseline every other model must beat.

If your hybrid cannot outperform time-decayed popularity, the hybrid is not
working - check the split for leakage before touching hyperparameters. A large
number of published "our model beats popularity by 40%" results come from
random splits that leak the future.
"""

import numpy as np

from recsys.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    name = "trending"

    def __init__(self, half_life_hours: float = 8760.0):
        """half_life_hours: 8760 is one year. Match it to your data - on
        MovieLens there is no 'trending this week', so a short half-life would
        just surface whatever happened to be rated last."""
        self.half_life_hours = half_life_hours
        self.ranked: list[tuple[int, float]] = []

    def fit(self, interactions_df) -> "PopularityRecommender":
        df = interactions_df[interactions_df["event_type"] != "impression"].copy()
        age_h = (df["created_at"].max() - df["created_at"]).dt.total_seconds() / 3600
        df["decayed"] = np.exp(-np.log(2) * age_h / self.half_life_hours)

        scores = df.groupby("video_id")["decayed"].sum().sort_values(ascending=False)
        self.ranked = [(int(v), float(s)) for v, s in scores.items()]
        return self

    def recommend(self, *args, n: int = 200, exclude: set[int] | None = None, **kwargs):
        """Identical for every user - that is the point of a baseline. The
        *args swallows a user_id or history so this matches the interface of
        the other retrievers and can be dropped into the same evaluator."""
        exclude = exclude or set()
        return [(v, s) for v, s in self.ranked if v not in exclude][:n]

    def top(self, n: int = 50, exclude: set[int] | None = None):
        return self.recommend(n=n, exclude=exclude)