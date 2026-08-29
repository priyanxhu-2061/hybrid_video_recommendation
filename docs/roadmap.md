# Build order

1. Schema, auth, video CRUD, seed data. No ML yet.
2. Popularity feed end to end, browser to database. Proves the pipes.
3. Interaction logging including impressions. Do this before any modelling -
   everything downstream depends on the data you start collecting now.
4. Content-based retriever + /similar. First real recommendations.
5. ALS collaborative filtering once you have enough interactions.
6. HybridBlender with RRF. Evaluate against baselines.
7. LightGBM reranker, trained on logged impressions.
8. Diversity and cold-start handling.
9. Nightly retrain, artifact versioning, hot reload.

Steps 1-3 have no ML in them and are still the ones that decide whether the
system works.
