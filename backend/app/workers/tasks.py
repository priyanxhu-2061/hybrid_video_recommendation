"""Celery tasks.

- nightly_retrain: rebuild CF factors and the reranker from the interactions table
- refresh_trending: recompute popularity windows every 15 minutes
- rebuild_ann_index: embed new videos and update the FAISS index
"""
