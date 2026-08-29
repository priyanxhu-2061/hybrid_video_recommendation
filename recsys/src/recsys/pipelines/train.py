"""End-to-end training.

    python -m recsys.pipelines.train --config config/default.yaml

Loads MovieLens, cleans it, weights the interactions, splits temporally, fits
every retriever, evaluates all of them against a popularity baseline, and writes
versioned artifacts.

Evaluation happens BEFORE saving, and the comparison table prints regardless.
If the hybrid loses to popularity you want that on screen now, not discovered
after it is deployed.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.evaluation.evaluator import Evaluator, format_table, popularity_counts
from recsys.features.interaction_weights import compute_weights, to_user_item_weights
from recsys.hybrid.blender import HybridBlender
from recsys.hybrid.diversifier import Diversifier
from recsys.ingestion.movielens import load, load_genome
from recsys.ingestion.splitters import build_ground_truth, split
from recsys.ingestion.validators import clean_interactions, report
from recsys.models.collaborative import CollaborativeRecommender
from recsys.models.content_based import ContentBasedRecommender
from recsys.models.item_knn import ItemKNNRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.serving.artifacts import save_artifacts
from recsys.serving.predictor import Predictor
from recsys.utils.io import load_config, resolve
from recsys.utils.seed import set_seed
from recsys.models.genome_content import GenomeContentRecommender

def step(label: str):
    """Timing wrapper. Knowing which stage is slow is most of what you need to
    know when a run takes twenty minutes and you want it to take five."""
    class _Timer:
        def __enter__(self):
            print(f"[{label}] ...", end="", flush=True)
            self.start = time.time()
            return self
        def __exit__(self, *exc):
            print(f" {time.time() - self.start:.1f}s")
    return _Timer()


def build_history_index(train_df: pd.DataFrame, max_per_user: int = 200) -> dict:
    """user_id -> (video_ids, weights), most recent first.

    Built once from the training window. At serving time this comes from the
    database instead, but during evaluation the training window IS the user's
    known history - anything later is what we are trying to predict.
    """
    df = train_df[train_df["weight"] > 0].sort_values("created_at", ascending=False)
    out = {}
    for user_id, group in df.groupby("user_id", sort=False):
        head = group.head(max_per_user)
        out[user_id] = (head["video_id"].tolist(), head["weight"].tolist())
    return out


def build_item_meta(videos_df: pd.DataFrame) -> dict:
    """video_id -> title/category/creator, for explanations and the caps."""
    return {
        int(row.video_id): {
            "title": row.title,
            "category": row.category,
            "creator_id": row.creator_id,
        }
        for row in videos_df.itertuples()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--sample-users", type=int, default=None,
                        help="override data.sample_users")
    parser.add_argument("--no-save", action="store_true",
                        help="evaluate only, write no artifacts")
    args = parser.parse_args()

    config = load_config(resolve(args.config))
    set_seed(config.get("seed", 42))

    data_config = config["data"]
    sample_users = args.sample_users or data_config.get("sample_users")

    # ------------------------------------------------------------ load
    with step("load"):
        raw_dir = resolve(data_config["raw_dir"])
        interactions, videos = load(raw_dir, sample_users=sample_users)

    with step("clean"):
        interactions = clean_interactions(
            interactions,
            min_per_user=data_config.get("min_interactions_per_user", 5),
            min_per_item=data_config.get("min_interactions_per_item", 3),
        )
        videos = videos[videos["video_id"].isin(interactions["video_id"].unique())]

    summary = report(interactions)
    print("  ", summary)

    # ------------------------------------------------------------ weight
    with step("weight"):
        interactions = compute_weights(
            interactions,
            half_life_days=config["features"].get("interaction_half_life_days", 365),
        )

    # ------------------------------------------------------------ split
    with step("split"):
        train, valid, test = split(interactions, data_config["split"])
    print(f"   train {len(train):,} | valid {len(valid):,} | test {len(test):,}")

    if valid.empty:
        raise SystemExit(
            "Validation window is empty. Increase data.split.valid_days - on a "
            "dataset spanning years, a window of days catches nobody."
        )

    with step("aggregate"):
        train_weights = to_user_item_weights(train)
    print(f"   {len(train_weights):,} user-item pairs")

    # ------------------------------------------------------------ fit
    with step("fit content"):
        content = ContentBasedRecommender(
            max_features=config["features"].get("max_features", 20000),
            ngram_range=tuple(config["features"].get("ngram_range", [1, 2])),
        ).fit(videos)

    genome = None
    if config["features"].get("use_genome", False):
        with step("load genome"):
            genome = load_genome(raw_dir)
        if genome is not None:
                print(f"   genome available: {genome[1].shape[0]:,} films")
                content_genome = GenomeContentRecommender(genome).fit()
        else:
            content_genome = None
            
    with step("fit collaborative"):
        cf_config = config["models"]["collaborative"]
        collaborative = CollaborativeRecommender(
            factors=cf_config["factors"],
            regularization=cf_config["regularization"],
            iterations=cf_config["iterations"],
            alpha=cf_config["alpha"],
            seed=config.get("seed", 42),
        ).fit(train_weights)

    with step("fit item_knn"):
        knn_config = config["models"]["item_knn"]
        item_knn = ItemKNNRecommender(
            k=knn_config["k"], shrink=knn_config["shrink"]
        ).fit(train_weights)

    with step("fit popularity"):
        trending = PopularityRecommender(
            half_life_hours=config["models"].get("popularity", {}).get("half_life_hours", 8760)
        ).fit(train)

    # ------------------------------------------------------------ evaluate
    history_index = build_history_index(train)
    item_meta = build_item_meta(videos)

    eval_config = config["evaluation"]
    ground_truth = build_ground_truth(
        valid, min_completion=eval_config.get("min_completion_for_relevance", 0.7)
    )
    # Only users we have history for can be scored - a model given no input has
    # nothing to be right or wrong about.
    ground_truth = {u: r for u, r in ground_truth.items() if u in history_index}
    print(f"\nevaluating on {len(ground_truth):,} users")

    if not ground_truth:
        raise SystemExit(
            "No evaluable users. Either the validation window is too narrow or "
            "min_completion_for_relevance is too strict."
        )

    hybrid_config = config["hybrid"]
    blender = HybridBlender(
        strategy=hybrid_config["strategy"],
        weights=hybrid_config.get("weights"),
        k=hybrid_config.get("rank_fusion_k", 60),
    )
    diversity_config = config["diversity"]
    diversifier = Diversifier(
        lambda_=diversity_config["lambda"],
        max_per_creator=diversity_config["max_per_creator"],
        max_per_category=diversity_config["max_per_category"],
    )

    predictor = Predictor(
        content=content,
        collaborative=collaborative,
        item_knn=item_knn,
        trending=trending,
        item_meta=item_meta,
        blender=blender,
        diversifier=diversifier,
        candidates_per_source=hybrid_config.get("candidates_per_source", 200),
        version="in-training",
    )

    max_k = max(eval_config["k_values"])

    def single(model, use_history: bool):
        """Wrap one retriever as a recommend_fn for the evaluator."""
        def fn(user_id):
            history, weights = history_index.get(user_id, ([], []))
            try:
                if use_history:
                    pairs = model.recommend(history, n=max_k, weights=weights)
                else:
                    pairs = model.recommend(user_id, n=max_k)
            except Exception:
                return []
            return [v for v, _ in pairs]
        return fn

    def popularity_fn(user_id):
        history, _ = history_index.get(user_id, ([], []))
        return [v for v, _ in trending.top(n=max_k, exclude=set(history))]

    def hybrid_fn(diversify: bool):
        def fn(user_id):
            history, weights = history_index.get(user_id, ([], []))
            return predictor.recommend_ids(
                user_id, history=history, weights=weights,
                top_k=max_k, diversify=diversify,
            )
        return fn

    popularity, n_users_total = popularity_counts(train)

    evaluator = Evaluator(
        k_values=eval_config["k_values"],
        item_vectors=content.item_vectors,
        id_to_row=content.id_to_row,
    )

    models_to_compare = {
        "popularity (baseline)": popularity_fn,
        "content (tfidf)": single(content, use_history=True),
        "collaborative": single(collaborative, use_history=False),
        "item_knn": single(item_knn, use_history=True),
        "hybrid (no diversity)": hybrid_fn(diversify=False),
        "hybrid + mmr": hybrid_fn(diversify=True),
    }
    if content_genome is not None:
        models_to_compare["content (genome)"] = single(content_genome, use_history=True)

    print()
    table = evaluator.compare(
        models_to_compare,
        ground_truth,
        n_items=len(videos),
        popularity=popularity,
        n_users_total=n_users_total,
        max_users=eval_config.get("max_eval_users"),
    )

    print("\n" + "=" * 100)
    print(format_table(table, k=10))
    print("=" * 100)

    baseline = table.loc["popularity (baseline)", "ndcg@10"]
    best_name = table.index[0]
    best = table.loc[best_name, "ndcg@10"]
    print(f"\nbest: {best_name}  ndcg@10 {best:.4f}  "
          f"vs popularity {baseline:.4f}  ({best / baseline:.2f}x)"
          if baseline else f"\nbest: {best_name}")

    if best <= baseline:
        print("\nNothing beat popularity. Before tuning anything, check the split "
              "for leakage and confirm the CF matrix is not near-empty.")

    # ------------------------------------------------------------ save
    if args.no_save:
        print("\n--no-save: artifacts not written")
        return

    with step("save"):
        out_dir = save_artifacts(
            {
                "content": content,
                "collaborative": collaborative,
                "item_knn": item_knn,
                "trending": trending,
                "item_meta": item_meta,
            },
            resolve(config["artifacts"]["dir"]),
            config=config,
            metrics=table.loc[best_name].to_dict(),
            data_summary={**summary, "sample_users": sample_users,
                          "train_rows": len(train), "valid_rows": len(valid)},
        )
    print(f"   {out_dir}")


if __name__ == "__main__":
    main()