from fastapi import APIRouter

from app.api.v1.endpoints import auth, feedback, interactions, recommendations, videos

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(videos.router, prefix="/videos", tags=["videos"])
api_router.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
api_router.include_router(interactions.router, prefix="/interactions", tags=["interactions"])
api_router.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
