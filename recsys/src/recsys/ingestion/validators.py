"""Schema and sanity checks - catch bad data before it becomes a bad model."""

import pandas as pd


def clean_interactions(
    df: pd.DataFrame,
    min_per_user: int = 5,
    min_per_item: int = 3,
) -> pd.DataFrame:
    """Drop impossible rows, then iteratively prune sparse users and items.

    The pruning must iterate. Removing sparse users can push items below the
    item threshold, and removing those items can push more users below the user
    threshold. A single pass leaves the matrix sparser than you asked for, and
    collaborative filtering is very sensitive to that.
    """
    df = df.copy()
    df = df[df["watch_seconds"] >= 0]
    df["completion_ratio"] = df["completion_ratio"].clip(0.0, 1.0)
    df = df.drop_duplicates(subset=["user_id", "video_id", "created_at"])

    while True:
        before = len(df)
        engaged = df[df["event_type"] != "impression"]

        ok_users = engaged.groupby("user_id").size()
        ok_users = ok_users[ok_users >= min_per_user].index
        ok_items = engaged.groupby("video_id").size()
        ok_items = ok_items[ok_items >= min_per_item].index

        df = df[df["user_id"].isin(ok_users) & df["video_id"].isin(ok_items)]
        if len(df) == before or df.empty:
            break

    return df.reset_index(drop=True)


def report(df: pd.DataFrame) -> dict:
    """Headline numbers. Density is the one to watch - below about 0.1% you
    should expect collaborative filtering to struggle."""
    engaged = df[df["event_type"] != "impression"]
    n_users = df["user_id"].nunique()
    n_items = df["video_id"].nunique()
    density = len(engaged) / (n_users * n_items) if n_users and n_items else 0.0
    return {
        "rows": len(df),
        "users": n_users,
        "items": n_items,
        "density": round(density, 5),
        "from": str(df["created_at"].min()),
        "to": str(df["created_at"].max()),
    }