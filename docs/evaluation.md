# Evaluation protocol

1. Temporal split. Train on everything before T, validate on the following week,
   test on the week after that.
2. Baselines first: random, most-popular, time-decayed popularity. Report them.
3. Then each single model: content only, CF only, item-kNN only.
4. Then hybrid without the reranker, then the full system.
5. Report precision@k, recall@k, NDCG@k, coverage, novelty, intra-list diversity.

The ablation table is the result. It shows what each component contributes, and
it is what turns "we built a hybrid" into evidence that the hybrid was worth it.

## Online, once deployed
CTR, mean completion ratio, session length, and share of the catalogue served.
Offline NDCG and online engagement diverge more often than you would expect.
