# recsys - offline ML package

Installed as a package (`pip install -e .`) so the backend can import the serving
side without copying files. Nothing here is imported by the API at request time
except `recsys.serving`.

## Flow

```
data/raw            interactions.csv, videos.csv (or a DB dump)
   |  ingestion/    load, validate, deduplicate
   v
data/interim        cleaned events
   |  features/     text vectors, user profiles, item stats
   v
data/processed      train / valid / test splits (time-based, never random)
   |  models/       content, collaborative, item-kNN fitted separately
   |  candidates/   FAISS index built from item embeddings
   |  ranking/      LightGBM reranker trained on merged candidates
   v
artifacts/          versioned .pkl / .index / .txt - what the API loads
```

## Why time-based splits

Random splits leak the future into training and make offline scores look great
while the live feed disappoints. Split on timestamp: train on everything before
T, validate on the next window, test on the window after that.

## Metrics that matter

Precision@k and Recall@k for relevance, NDCG@k for ordering, and then the ones
people forget: catalogue coverage, intra-list diversity, and novelty. A hybrid
that only optimises NDCG collapses into recommending the same 50 popular videos.
