"""Parameter sweep. Runs training once, then re-evaluates across settings.

Fitting the models is the expensive part and does not depend on blend weights
or lambda, so we fit once and vary only what happens downstream. Eight
configurations in roughly the time one full run used to take.

    python -m recsys.pipelines.sweep --config config/default.yaml
"""

import argparse

import pandas as pd

from recsys.evaluation.evaluator import Evaluator, popularity_counts
from recsys.features.interaction_weights import compute_weights, to_user_item_weights
from recsys.hybrid.blender import HybridBlender
from recsys.hybrid.diversifier import Diversifier
from recsys.ingestion.movielens import load
from recsys.ingestion.splitters import build_ground_truth, split
from recsys.ingestion.validators import clean_interactions
from recsys.models.collaborative import CollaborativeRecommender
from recsys.models.content_based import ContentBasedRecommender
from recsys.models.item_knn import ItemKNNRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.pipelines.train import build_history_index, build_item_meta, step
from recsys.serving.predictor import Predictor
from recsys.utils.io import load_config, resolve
from recsys.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--content-weights", nargs="+", type=float,
                        default=[1.0, 2.0, 3.0, 5.0])
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[0.75, 0.85, 0.95, 1.0])
    args = parser.parse_args()

    config = load_config(resolve(args.config))
    set_seed(config.get("seed", 42))
    data_config = config["data"]

    with step("load + fit"):
        raw_dir = resolve(data_config["raw_dir"])
        interactions, videos = load(raw_dir, sample_users=data_config.get("sample_users"))
        interactions = clean_interactions(
            interactions,
            min_per_user=data_config.get("min_interactions_per_user", 5),
            min_per_item=data_config.get("min_interactions_per_item", 3),
        )
        videos = videos[videos["video_id"].isin(interactions["video_id"].unique())]
        interactions = compute_weights(
            interactions,
            half_life_days=config["features"].get("interaction_half_life_days", 365),
        )
        train, valid, _ = split(interactions, data_config["split"])
        train_weights = to_user_item_weights(train)

        content = ContentBasedRecommender(
            max_features=config["features"].get("max_features", 20000),
            ngram_range=tuple(config["features"].get("ngram_range", [1, 2])),
        ).fit(videos)
        cf_config = config["models"]["collaborative"]
        collaborative = CollaborativeRecommender(
            factors=cf_config["factors"],
            regularization=cf_config["regularization"],
            iterations=cf_config["iterations"],
            alpha=cf_config["alpha"],
            seed=config.get("seed", 42),
        ).fit(train_weights)
        knn_config = config["models"]["item_knn"]
        item_knn = ItemKNNRecommender(k=knn_config["k"], shrink=knn_config["shrink"]).fit(train_weights)
        trending = PopularityRecommender(
            half_life_hours=config["models"].get("popularity", {}).get("half_life_hours", 8760)
        ).fit(train)

    history_index = build_history_index(train)
    item_meta = build_item_meta(videos)

    eval_config = config["evaluation"]
    ground_truth = build_ground_truth(
        valid, min_completion=eval_config.get("min_completion_for_relevance", 0.7)
    )
    ground_truth = {u: r for u, r in ground_truth.items() if u in history_index}
    print(f"evaluating on {len(ground_truth):,} users\n")

    popularity, n_users_total = popularity_counts(train)
    evaluator = Evaluator(
        k_values=eval_config["k_values"],
        item_vectors=content.item_vectors,
        id_to_row=content.id_to_row,
    )
    max_k = max(eval_config["k_values"])
    base_weights = dict(config["hybrid"].get("weights", {}))
    diversity_config = config["diversity"]

    def run(content_weight, lambda_):
        weights = {**base_weights, "content": content_weight}
        predictor = Predictor(
            content=content, collaborative=collaborative, item_knn=item_knn,
            trending=trending, item_meta=item_meta,
            blender=HybridBlender(
                strategy=config["hybrid"]["strategy"],
                weights=weights,
                k=config["hybrid"].get("rank_fusion_k", 60),
            ),
            diversifier=Diversifier(
                lambda_=lambda_,
                max_per_creator=diversity_config["max_per_creator"],
                max_per_category=diversity_config["max_per_category"],
            ),
            candidates_per_source=config["hybrid"].get("candidates_per_source", 200),
        )

        def fn(user_id):
            history, w = history_index.get(user_id, ([], []))
            return predictor.recommend_ids(user_id, history=history, weights=w, top_k=max_k)

        return evaluator.evaluate(
            fn, ground_truth, n_items=len(videos),
            popularity=popularity, n_users_total=n_users_total,
            max_users=eval_config.get("max_eval_users"),
        )

    rows = {}
    for cw in args.content_weights:
        label = f"content_w={cw}"
        print(f"  {label} ...", end="", flush=True)
        rows[label] = run(cw, diversity_config["lambda"])
        print(" done")

    for lam in args.lambdas:
        label = f"lambda={lam}"
        print(f"  {label} ...", end="", flush=True)
        rows[label] = run(base_weights.get("content", 1.0), lam)
        print(" done")

    columns = ["ndcg@10", "precision@10", "recall@10", "coverage", "gini", "novelty", "diversity"]
    table = pd.DataFrame(rows).T[columns]

    print("\n" + "=" * 90)
    print(table.round(4).to_string(float_format=lambda x: f"{x:>9.4f}"))
    print("=" * 90)
    print("\nThe accuracy-diversity trade-off is the ndcg@10 column against")
    print("coverage. Rising content weight should cost NDCG and buy coverage;")
    print("where that curve bends is the number worth reporting.")


if __name__ == "__main__":
    main()