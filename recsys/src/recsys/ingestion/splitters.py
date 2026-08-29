"""Temporal splitting.

Random splits leak the future into training: the model sees interaction t+1
while predicting t, offline metrics look excellent, and the live system
disappoints. Split on time, always.
"""

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    valid_days: int = 180,
    test_days: int = 180,
    time_col: str = "created_at",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train | valid | test, cut at fixed day offsets back from the last event."""
    end = df[time_col].max()
    test_start = end - pd.Timedelta(days=test_days)
    valid_start = test_start - pd.Timedelta(days=valid_days)

    train = df[df[time_col] < valid_start]
    valid = df[(df[time_col] >= valid_start) & (df[time_col] < test_start)]
    test = df[df[time_col] >= test_start]

    # A user or item unseen in train cannot be scored by collaborative
    # filtering - it has no factors. Keep them out of the eval sets so the
    # metrics measure ranking quality rather than cold-start coverage, which is
    # a different question and deserves its own experiment.
    known_users = set(train["user_id"])
    known_items = set(train["video_id"])
    valid = valid[valid["user_id"].isin(known_users) & valid["video_id"].isin(known_items)]
    test = test[test["user_id"].isin(known_users) & test["video_id"].isin(known_items)]

    return (
        train.reset_index(drop=True),
        valid.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def leave_last_n_split(
    df: pd.DataFrame,
    n_valid: int = 5,
    n_test: int = 5,
    min_train: int = 5,
    time_col: str = "created_at",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user chronological holdout: last n_test events to test, the n_valid
    before them to valid, everything earlier to train.

    Why this over a global date cut on MovieLens: activity there is bursty.
    Most users rate a batch of films in one sitting and never return, so a
    global window at the end of the timeline catches almost nobody - our first
    run left 90 evaluable users out of 5,000.

    This still never trains on a user's future. The guarantee is per-user
    rather than global: for any user, everything in train precedes everything
    in valid, which precedes everything in test. What it gives up is the global
    guarantee - user A's test events may predate user B's training events. That
    is a real weakness worth naming in a write-up, and it is the trade the
    MovieLens literature almost universally makes.

    min_train: users with fewer than this many events left over after the
    holdout are dropped entirely. A user with two training events teaches the
    model nothing and adds noise to the metrics.
    """
    df = df.sort_values(["user_id", time_col])

    # Rank within each user, counting backwards from their most recent event.
    # Rank 0 is the newest, so ranks [0, n_test) are the test items.
    reverse_rank = df.groupby("user_id").cumcount(ascending=False)

    sizes = df.groupby("user_id")["video_id"].transform("size")
    eligible = sizes >= (n_valid + n_test + min_train)

    test = df[eligible & (reverse_rank < n_test)]
    valid = df[eligible & (reverse_rank >= n_test) & (reverse_rank < n_test + n_valid)]
    train = df[~eligible | (reverse_rank >= n_test + n_valid)]

    known_users = set(train["user_id"])
    known_items = set(train["video_id"])
    valid = valid[valid["user_id"].isin(known_users) & valid["video_id"].isin(known_items)]
    test = test[test["user_id"].isin(known_users) & test["video_id"].isin(known_items)]

    return (
        train.reset_index(drop=True),
        valid.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def split(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dispatch on config, so the strategy is a config change not a code change."""
    strategy = config.get("strategy", "temporal")

    if strategy == "temporal":
        return temporal_split(
            df,
            valid_days=config.get("valid_days", 180),
            test_days=config.get("test_days", 180),
        )
    if strategy in ("leave_last_n", "leave_n_out"):
        return leave_last_n_split(
            df,
            n_valid=config.get("n_valid", 5),
            n_test=config.get("n_test", 5),
            min_train=config.get("min_train", 5),
        )
    raise ValueError(
        f"Unknown split strategy '{strategy}'. Use 'temporal' or 'leave_last_n'."
    )


def build_ground_truth(df: pd.DataFrame, min_completion: float = 0.7) -> dict[int, set[int]]:
    """What counts as 'relevant' in the eval window.

    An interaction is not an endorsement. On MovieLens, completion_ratio is
    derived from the star rating, so 0.7 corresponds to roughly 3.5 stars -
    requiring real approval rather than mere presence. Loosen this and every
    model's scores rise together while telling you less.
    """
    relevant = df[
        (df["event_type"].isin(["like", "complete"]))
        & (df["completion_ratio"] >= min_completion)
    ]
    return relevant.groupby("user_id")["video_id"].apply(set).to_dict()