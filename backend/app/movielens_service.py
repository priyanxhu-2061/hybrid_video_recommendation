"""Serves recommendations directly from MovieLens artifacts.

No database. The trained artifacts already contain everything needed: the four
retrievers, the item metadata, and - once we rebuild it here - each user's
history from the training window.

That last part is the compromise worth naming. A real backend reads history from
a table that updates as users watch things. This reads it from the frozen
training data, so a user's history never changes and new interactions do not
influence future recommendations. Fine for demonstrating the model; not a
production design.
"""

import pandas as pd

from recsys.features.interaction_weights import compute_weights
from recsys.ingestion.movielens import load
from recsys.ingestion.splitters import split
from recsys.ingestion.validators import clean_interactions
from recsys.serving.predictor import Predictor
from recsys.utils.io import load_config, resolve


class MovieLensService:
    def __init__(self):
        self.predictor: Predictor | None = None
        self.history: dict[int, tuple[list[int], list[float]]] = {}
        self.videos: pd.DataFrame | None = None
        self.ready = False

    def load(self, config_path: str = "config/default.yaml") -> None:
        config = load_config(resolve(config_path))

        self.predictor = Predictor.load(
            resolve(config["artifacts"]["dir"]),
            version=config["artifacts"].get("version", "latest"),
            candidates_per_source=config["hybrid"].get("candidates_per_source", 200),
        )

        data_config = config["data"]
        interactions, videos = load(
            resolve(data_config["raw_dir"]),
            sample_users=data_config.get("sample_users"),
        )
        interactions = clean_interactions(
            interactions,
            min_per_user=data_config.get("min_interactions_per_user", 5),
            min_per_item=data_config.get("min_interactions_per_item", 3),
        )
        interactions = compute_weights(
            interactions,
            half_life_days=config["features"].get("interaction_half_life_days", 365),
        )

        # Only the training window. Using everything would mean recommending
        # from data the model was never fitted on, and any evaluation done
        # against this API would be measuring a leak.
        train, _, _ = split(interactions, data_config["split"])
        self.history = self._build_history(train)

        self.videos = videos.set_index("video_id")
        self.ready = True

    @staticmethod
    def _build_history(train: pd.DataFrame, max_per_user: int = 200) -> dict:
        df = train[train["weight"] > 0].sort_values("created_at", ascending=False)
        out = {}
        for user_id, group in df.groupby("user_id", sort=False):
            head = group.head(max_per_user)
            out[int(user_id)] = (head["video_id"].tolist(), head["weight"].tolist())
        return out

    # ------------------------------------------------------------- queries

    def feed(self, user_id: int, top_k: int = 20, diversify: bool = True) -> dict:
        history, weights = self.history.get(user_id, ([], []))
        result = self.predictor.recommend(
            user_id, history=history, weights=weights,
            top_k=top_k, diversify=diversify,
        )
        result["history_size"] = len(history)
        return result

    def similar(self, video_id: int, top_k: int = 10) -> dict:
        return self.predictor.similar(video_id, top_k=top_k)

    def user_history(self, user_id: int, limit: int = 20) -> list[dict]:
        """What the model knows about this user - useful for eyeballing whether
        a feed makes sense. A feed full of horror for someone whose history is
        all comedies is a bug you can only see by looking at both."""
        ids, weights = self.history.get(user_id, ([], []))
        return [
            {"video_id": v, "weight": round(w, 3), **self.video(v)}
            for v, w in list(zip(ids, weights))[:limit]
        ]

    def video(self, video_id: int) -> dict:
        if self.videos is None or video_id not in self.videos.index:
            return {}
        row = self.videos.loc[video_id]
        return {
            "title": row["title"],
            "category": row["category"],
            "tags": list(row["tags"])[:8],
        }

    def known_users(self, limit: int = 20) -> list[int]:
        return sorted(self.history.keys())[:limit]

    def stats(self) -> dict:
        return {
            "model_version": self.predictor.version if self.predictor else None,
            "users": len(self.history),
            "videos": 0 if self.videos is None else len(self.videos),
        }


service = MovieLensService()