"""FAISS index over item embeddings.

Brute-force cosine is fine up to ~50k videos. Past that, build an IVF-Flat or
HNSW index so retrieval stays under a few milliseconds.
"""
