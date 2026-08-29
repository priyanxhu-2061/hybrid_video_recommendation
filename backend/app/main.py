"""Minimal API over the trained MovieLens recommender.

    cd backend
    uvicorn app.main:app --reload

Then open http://localhost:8000/docs for the interactive interface.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.movielens_service import service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loading takes about a minute - artifacts plus a full pass over the
    # ratings file. Do it once at boot, not per request.
    print("loading model and history ...")
    service.load()
    print(f"ready: {service.stats()}")
    yield


app = FastAPI(
    title="Hybrid Video Recommender (MovieLens)",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok" if service.ready else "loading", **service.stats()}


@app.get("/api/v1/users", tags=["users"])
async def users(limit: int = Query(20, le=200)):
    """Some valid user ids to try. Without this you would be guessing which of
    138,000 MovieLens ids happen to be in the sampled subset."""
    return {"user_ids": service.known_users(limit)}


@app.get("/api/v1/users/{user_id}/history", tags=["users"])
async def history(user_id: int, limit: int = Query(20, le=100)):
    items = service.user_history(user_id, limit)
    if not items:
        raise HTTPException(404, f"No history for user {user_id}. Try /api/v1/users.")
    return {"user_id": user_id, "items": items}


@app.get("/api/v1/recommendations/feed", tags=["recommendations"])
async def feed(
    user_id: int,
    top_k: int = Query(20, le=100),
    diversify: bool = True,
):
    """The main endpoint. `diversify=false` shows what MMR is actually doing -
    run it both ways on the same user and compare."""
    if not service.ready:
        raise HTTPException(503, "Model still loading")
    result = service.feed(user_id, top_k=top_k, diversify=diversify)
    result["items"] = [{**i, **service.video(i["video_id"])} for i in result["items"]]
    return result


@app.get("/api/v1/recommendations/similar/{video_id}", tags=["recommendations"])
async def similar(video_id: int, top_k: int = Query(10, le=50)):
    """Content-based only. No user, no history - works for anyone."""
    if not service.ready:
        raise HTTPException(503, "Model still loading")
    result = service.similar(video_id, top_k=top_k)
    if not result["items"]:
        raise HTTPException(404, f"Video {video_id} not in the content model")
    result["items"] = [{**i, **service.video(i["video_id"])} for i in result["items"]]
    return result


@app.get("/api/v1/videos/{video_id}", tags=["videos"])
async def video(video_id: int):
    meta = service.video(video_id)
    if not meta:
        raise HTTPException(404, f"Video {video_id} not found")
    return {"video_id": video_id, **meta}