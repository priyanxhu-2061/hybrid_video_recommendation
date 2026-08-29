# Architecture

## Request path (online)

```
browser -> GET /api/v1/recommendations/feed
             |
             v
        RecommenderService
             |
   +---------+---------+---------+
   |         |         |         |
content    ALS      item-kNN  trending      <- each returns ~200 candidates
   |         |         |         |
   +---------+----+----+---------+
                  v
            HybridBlender          merge, dedupe, keep source provenance
                  v
            FeatureBuilder         user x item x context features
                  v
              Reranker             LightGBM LambdaMART, top_k * 3
                  v
             Diversifier           MMR + creator/category caps -> top_k
                  v
             JSON response
```

Target budget: retrieval 30ms, feature build 20ms, rank 15ms, diversify 5ms.
Cache the whole result in Redis for 5 minutes keyed by user and model version.

## Training path (offline, nightly)

interactions table -> ingestion -> weighting -> temporal split -> fit retrievers
-> build FAISS index -> generate candidates -> train reranker -> evaluate ->
write `artifacts/<timestamp>/` -> repoint `latest` -> API reload.

## Why the split

`recsys/` can be run, tuned, and evaluated with no database and no server. The
API imports one module (`recsys.serving.predictor`) and never touches training
code. That means you can iterate on the model in notebooks while the app keeps
running against the last good artifact.
