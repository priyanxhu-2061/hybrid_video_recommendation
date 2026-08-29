"""Precompute top-N for every active user and push to Redis.

Worth doing once traffic grows: the feed becomes a cache read, and the online
path only handles new users and real-time reranking.
"""
