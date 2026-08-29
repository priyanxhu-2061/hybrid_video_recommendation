from pydantic import BaseModel, Field


class RecommendedVideo(BaseModel):
    video_id: int
    title: str
    thumbnail_url: str | None = None
    score: float
    # Which retriever(s) proposed it - drives the "because you watched..." line
    # in the UI and makes ranking debuggable.
    sources: list[str] = Field(default_factory=list)
    explanation: str | None = None


class RecommendationResponse(BaseModel):
    user_id: int
    model_version: str
    strategy: str  # hybrid | cold_start | popularity_fallback
    items: list[RecommendedVideo]
