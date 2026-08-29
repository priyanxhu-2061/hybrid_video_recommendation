"""LightGBM LambdaMART reranker - stage 2 of the two-stage design.

Trained on logged impressions grouped by (user, session): each group is one
displayed list, labels come from what was clicked and how long it was watched.

Two things to get right or the model learns nonsense:

1. Log impressions, not just clicks. Without negatives from what was shown and
   ignored, every training row is a positive.
2. Correct for position bias. Items at rank 1 get clicked because they are at
   rank 1. Either add position as a training-only feature and zero it at
   inference, or reweight by an inverse-propensity estimate.
"""


class Reranker:
    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self.model = None

    def fit(self, X, y, group, X_valid=None, y_valid=None, group_valid=None):
        raise NotImplementedError

    def rank(self, features, top_k: int) -> list[dict]:
        raise NotImplementedError

    def feature_importance(self):
        """Check this every retrain - a feature dominating usually means leakage."""
        raise NotImplementedError
