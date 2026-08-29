from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.config import settings
from app.db.session import get_db
from app.models.user import User
from app.schemas.recommendation import RecommendationResponse
from app.services.recommender import RecommenderService

router = APIRouter()


@router.get("/feed", response_model=RecommendationResponse)
async def personalised_feed(
    top_k: int = Query(default=settings.FINAL_TOP_K, le=100),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Main entry point: retrieve -> merge -> rerank -> diversify."""
    return await RecommenderService(db).recommend_for_user(user.id, top_k=top_k)


@router.get("/similar/{video_id}", response_model=RecommendationResponse)
async def similar_videos(
    video_id: int,
    top_k: int = Query(default=10, le=50),
    db: AsyncSession = Depends(get_db),
):
    """Content-based only - works logged out and drives the watch-page rail."""
    return await RecommenderService(db).similar_to_video(video_id, top_k=top_k)
