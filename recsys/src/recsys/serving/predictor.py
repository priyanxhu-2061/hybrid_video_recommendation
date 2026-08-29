"""The one module the backend imports. Wraps loaded artifacts behind a stable
interface so retraining never changes API code."""
"""The one module the backend imports.

Wraps loaded artifacts behind a stable interface: retrieve from every source,
blend, diversify, return. Retraining swaps the artifacts underneath and this
signature never changes, so API code never needs touching.

Stateless with respect to user data. History arrives as an argument - the caller
(FastAPI in production, the training pipeline during evaluation) is responsible
for fetching it. That keeps the predictor free of database concerns and means
there is never a question about whether its view of a user is stale.
"""

from pathlib import Path

from recsys.hybrid.blender import HybridBlender
from recsys.hybrid.diversifier import Diversifier
from recsys.serving.artifacts import load_artifacts

COLD_START_THRESHOLD = 5


class Predictor:
    def __init__(
        self,
        content=None,
        collaborative=None,
        item_knn=None,
        trending=None,
        item_meta: dict | None = None,
        blender: HybridBlender | None = None,
        diversifier: Diversifier | None = None,
        candidates_per_source: int = 200,
        version: str = "unversioned",
    ):
        self.content = content
        self.collaborative = collaborative
        self.item_knn = item_knn
        self.trending = trending
        self.item_meta = item_meta or {}
        self.blender = blender or HybridBlender(strategy="rank_fusion")
        self.diversifier = diversifier or Diversifier()
        self.candidates_per_source = candidates_per_source
        self._version = version

    # ------------------------------------------------------------- loading

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        version: str = "latest",
        blender: HybridBlender | None = None,
        diversifier: Diversifier | None = None,
        candidates_per_source: int = 200,
    ) -> "Predictor":
        loaded = load_artifacts(artifact_dir, version)

        # Blend weights are a serving-time knob, not a trained parameter. Read
        # them from the saved config so a retrain can change them, but let the
        # caller override without retraining.
        if blender is None:
            config = (loaded.get("_metadata") or {}).get("config") or {}
            hybrid_config = config.get("hybrid", {})
            blender = HybridBlender(
                strategy=hybrid_config.get("strategy", "rank_fusion"),
                weights=hybrid_config.get("weights"),
                k=hybrid_config.get("rank_fusion_k", 60),
            )

        return cls(
            content=loaded.get("content"),
            collaborative=loaded.get("collaborative"),
            item_knn=loaded.get("item_knn"),
            trending=loaded.get("trending"),
            item_meta=loaded.get("item_meta"),
            blender=blender,
            diversifier=diversifier,
            candidates_per_source=candidates_per_source,
            version=loaded.get("_version", "unversioned"),
        )

    @property
    def version(self) -> str:
        return self._version

    # ---------------------------------------------------------- prediction

    def recommend(
        self,
        user_id: int,
        history: list[int] | None = None,
        weights: list[float] | None = None,
        top_k: int = 20,
        exclude: set[int] | None = None,
        diversify: bool = True,
    ) -> dict:
        """Full pipeline for one user.

        history: video ids the user has engaged with, most recent first.
        weights: matching engagement strengths. Passing these matters - an
                 unweighted profile treats a 1-star rating and a 5-star rating
                 as equally representative of what the user wants.

        Returns {"items": [...], "strategy": str, "model_version": str}.
        """
        history = list(history or [])
        exclude = set(exclude or []) | set(history)

        if len(history) < COLD_START_THRESHOLD:
            return self._cold_start(user_id, top_k, exclude)

        pool = self.blender.merge(self._retrieve(user_id, history, weights))
        if not pool:
            return self._cold_start(user_id, top_k, exclude)

        pool = [p for p in pool if p["video_id"] not in exclude]

        if diversify:
            # Hand the diversifier a deep pool. Passing only top_k leaves it
            # nothing to substitute in, and MMR degenerates to a reordering of
            # items it was already forced to keep.
            items = self.diversifier.apply(
                pool[:top_k * 5],
                item_vectors=self._item_vectors(),
                id_to_row=self._id_to_row(),
                item_meta=self.item_meta,
                top_k=top_k,
            )
        else:
            items = pool[:top_k]

        return {
            "items": [self._decorate(i) for i in items],
            "strategy": "hybrid",
            "model_version": self._version,
        }

    def recommend_ids(self, user_id, history=None, weights=None, top_k=20, **kwargs) -> list[int]:
        """Just the ranked ids. This is the shape Evaluator.recommend_fn wants."""
        result = self.recommend(user_id, history, weights, top_k, **kwargs)
        return [i["video_id"] for i in result["items"]]

    def similar(self, video_id: int, top_k: int = 10) -> dict:
        """Content-only similarity. Works with no user and no history, which is
        why the watch-page rail and logged-out browsing both use it."""
        if self.content is None:
            return {"items": [], "strategy": "unavailable", "model_version": self._version}

        pairs = self.content.similar_items(video_id, n=top_k)
        items = [{"video_id": v, "score": s, "sources": ["content"]} for v, s in pairs]
        return {
            "items": [self._decorate(i) for i in items],
            "strategy": "content_similarity",
            "model_version": self._version,
        }

    # ------------------------------------------------------------ internals

    def _retrieve(self, user_id, history, weights) -> dict:
        """Ask every available source for candidates.

        Each is wrapped: one retriever raising should degrade the feed, not
        break it. A user missing from the CF matrix is normal, not exceptional.
        """
        n = self.candidates_per_source
        out = {}

        if self.content is not None:
            out["content"] = self._safe(
                lambda: self.content.recommend(history, n=n, weights=weights)
            )
        if self.collaborative is not None:
            out["collaborative"] = self._safe(
                lambda: self.collaborative.recommend(user_id, n=n)
            )
        if self.item_knn is not None:
            out["item_knn"] = self._safe(
                lambda: self.item_knn.recommend(history, n=n, weights=weights)
            )
        if self.trending is not None:
            out["trending"] = self._safe(
                lambda: self.trending.top(n=max(n // 4, 10))
            )

        return {k: v for k, v in out.items() if v}

    @staticmethod
    def _safe(fn) -> list:
        try:
            return fn() or []
        except Exception:
            return []

    def _cold_start(self, user_id, top_k, exclude) -> dict:
        """No usable history: fall back to trending.

        Note for MovieLens work - every user in that dataset has at least 20
        ratings, so this path is never exercised by the benchmark. To evaluate
        it you have to simulate cold start by truncating histories.
        """
        if self.trending is None:
            return {"items": [], "strategy": "unavailable", "model_version": self._version}

        pairs = self.trending.top(n=top_k, exclude=exclude)
        items = [{"video_id": v, "score": s, "sources": ["trending"]} for v, s in pairs]
        return {
            "items": [self._decorate(i) for i in items],
            "strategy": "cold_start",
            "model_version": self._version,
        }

    def _decorate(self, item: dict) -> dict:
        """Attach title and category, and a short reason string.

        The reason is not decoration. 'Because you watched...' is what makes a
        feed feel intentional rather than arbitrary, and having the source list
        in the response is what lets you debug a bad feed without rerunning the
        model.
        """
        meta = self.item_meta.get(item["video_id"], {})
        sources = item.get("sources", [])
        return {
            **item,
            "title": meta.get("title"),
            "category": meta.get("category"),
            "explanation": self._explain(sources),
        }

    @staticmethod
    def _explain(sources: list[str]) -> str:
        if not sources:
            return ""
        if len(sources) > 1:
            return "Matches your taste and what similar viewers watch"
        return {
            "content": "Similar to what you have watched",
            "collaborative": "Popular with viewers like you",
            "item_knn": "People who watched your favourites watched this",
            "trending": "Trending now",
        }.get(sources[0], "")

    def _item_vectors(self):
        return getattr(self.content, "item_vectors", None)

    def _id_to_row(self) -> dict:
        return getattr(self.content, "id_to_row", {}) or {}