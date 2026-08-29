from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Interaction(Base):
    """Every implicit and explicit signal lands here. This table is the training set."""

    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), index=True)

    event_type: Mapped[str] = mapped_column(String(32))  # impression, view, like, skip, complete
    watch_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    completion_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    rating: Mapped[float | None] = mapped_column(Float)

    session_id: Mapped[str | None] = mapped_column(String(64), index=True)
    source: Mapped[str | None] = mapped_column(String(32))  # which retriever surfaced it
    position: Mapped[int | None] = mapped_column(Integer)   # rank shown at - needed to debias
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
