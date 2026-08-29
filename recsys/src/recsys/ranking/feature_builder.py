"""Builds the feature matrix the reranker scores. This file is where most of the
accuracy actually comes from.

Feature groups:

user      history length, mean completion ratio, category distribution, active
          hours, days since signup, ALS latent factors
item      age, duration, category, tag count, time-decayed popularity, CTR,
          mean completion ratio, ALS item factors
pair      content cosine similarity, ALS dot product, item-kNN score, category
          match with user's top categories, creator already watched
context   hour of day, weekday, device, session depth, position of previous click
source    which retrievers proposed it, and at what rank in each

The source features matter: knowing that both content AND collaborative proposed
an item is itself strong evidence, and it lets the ranker learn the blend weights
that `hybrid/blender.py` otherwise hardcodes.
"""


class FeatureBuilder:
    def build(self, user_id: int, pool: list[dict], history: list[int], context: dict | None = None):
        raise NotImplementedError
