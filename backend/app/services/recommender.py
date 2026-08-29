"""Online orchestration layer.

Deliberately thin: it wires together candidate sources and the ranker that were
trained offline in `recsys/`. All learning lives there; this file only decides
which strategy applies to this request.
"""

from app.core.config import settings
from app.schemas.recommendation import RecommendationResponse
from app.services.model_registry import model_registry


class RecommenderService:
    def __init__(self, db):
        self.db = db

    async def recommend_for_user(self, user_id: int, top_k: int) -> RecommendationResponse:
        history = await self._load_history(user_id)

        # Cold start: too few interactions means CF has nothing to work with.
        if len(history) < settings.COLD_START_THRESHOLD:
            return await self._cold_start(user_id, top_k)

        n = settings.CANDIDATES_PER_SOURCE
        candidates = {
            "content": model_registry.content.recommend(history, n=n),
            "collaborative": model_registry.collaborative.recommend(user_id, n=n),
            "item_knn": model_registry.item_knn.recommend(history, n=n),
            "trending": model_registry.trending.top(n=n // 4),
        }

        pool = model_registry.hybrid.merge(candidates)
        features = model_registry.features.build(user_id, pool, history)
        ranked = model_registry.ranker.rank(features, top_k=top_k * 3)
        final = model_registry.diversifier.apply(ranked, top_k=top_k)

        return RecommendationResponse(
            user_id=user_id,
            model_version=model_registry.version,
            strategy="hybrid",
            items=final,
        )

    async def similar_to_video(self, video_id: int, top_k: int) -> RecommendationResponse:
        raise NotImplementedError

    async def _cold_start(self, user_id: int, top_k: int) -> RecommendationResponse:
        """Onboarding categories plus trending, handing over to CF as history grows."""
        raise NotImplementedError

    async def _load_history(self, user_id: int) -> list[int]:
        raise NotImplementedError
