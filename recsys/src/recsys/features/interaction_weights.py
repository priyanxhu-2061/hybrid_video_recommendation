"""Turns raw events into a single implicit-feedback strength.

Getting this wrong quietly caps the ceiling of every model downstream. The most
common mistake is counting every interaction as a positive: a 1-star rating and
a 5-star rating arrive looking identical, and collaborative filtering learns
that being watched at all is the same as being liked.
"""

import numpy as np
import pandas as pd

EVENT_BASE = {
    "impression": 0.0,   # shown, not engaged - a negative, kept for the ranker
    "skip": -0.5,        # explicit rejection
    "view": 0.2,
    "complete": 1.0,
    "like": 1.5,
}


def compute_weights(df: pd.DataFrame, half_life_days: float = 365.0) -> pd.DataFrame:
    """Adds a `weight` column: engagement strength, decayed by recency.

    half_life_days: how fast old interactions fade. Match it to the span of
    your data - a 30-day half-life on 19 years of MovieLens ratings would
    weight everything before the final months to essentially zero.
    """
    df = df.copy()
    base = df["event_type"].map(EVENT_BASE).fillna(0.0)

    # Completion carries more signal than the event label alone.
    weight = base + 1.2 * df["completion_ratio"].fillna(0.0)

    # Recency decay: last year's taste beats taste from a decade ago.
    age_days = (df["created_at"].max() - df["created_at"]).dt.total_seconds() / 86400
    decay = np.exp(-np.log(2) * age_days / half_life_days)

    # Clipped at zero: negatives have done their job by scoring low, and a
    # negative confidence weight would break the ALS solve.
    df["weight"] = (weight * decay).clip(lower=0.0)
    return df


def to_user_item_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated interactions into one weight per (user, item)."""
    agg = (
        df[df["weight"] > 0]
        .groupby(["user_id", "video_id"], as_index=False)["weight"]
        .sum()
    )
    # Repeat engagement is a positive signal, but sublinearly so - the tenth
    # rewatch says far less than the second.
    agg["weight"] = np.log1p(agg["weight"])
    return agg