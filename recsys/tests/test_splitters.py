"""Splitter correctness, with leakage as the property under test.

A leaking split does not fail loudly. It produces excellent offline metrics and
a system that disappoints in deployment, so these assertions exist to make the
failure loud.
"""

import pandas as pd
import pytest

from recsys.ingestion.splitters import (
    build_ground_truth,
    leave_last_n_split,
    split,
    temporal_split,
)


def make_frame(n_users=10, n_per_user=20):
    """Synthetic interactions with a strictly increasing timestamp per user."""
    rows = []
    base = pd.Timestamp("2020-01-01")
    for u in range(n_users):
        for i in range(n_per_user):
            rows.append({
                "user_id": u,
                "video_id": (u * 7 + i) % 30,
                "event_type": "like" if i % 3 == 0 else "complete",
                "completion_ratio": 0.9 if i % 3 == 0 else 0.5,
                "created_at": base + pd.Timedelta(days=i * 10),
            })
    return pd.DataFrame(rows)


def test_leave_last_n_never_leaks_within_a_user():
    """The core guarantee: no user's training data postdates their test data."""
    df = make_frame()
    train, valid, test = leave_last_n_split(df, n_valid=5, n_test=5, min_train=5)

    for user in test["user_id"].unique():
        user_train = train[train["user_id"] == user]["created_at"]
        user_test = test[test["user_id"] == user]["created_at"]
        if len(user_train) and len(user_test):
            assert user_train.max() < user_test.min()


def test_leave_last_n_orders_valid_between_train_and_test():
    df = make_frame()
    train, valid, test = leave_last_n_split(df, n_valid=5, n_test=5, min_train=5)

    for user in valid["user_id"].unique():
        user_train = train[train["user_id"] == user]["created_at"]
        user_valid = valid[valid["user_id"] == user]["created_at"]
        user_test = test[test["user_id"] == user]["created_at"]
        if len(user_train) and len(user_valid):
            assert user_train.max() < user_valid.min()
        if len(user_valid) and len(user_test):
            assert user_valid.max() < user_test.min()


def test_leave_last_n_holds_out_the_requested_count():
    df = make_frame(n_users=5, n_per_user=20)
    _, valid, test = leave_last_n_split(df, n_valid=5, n_test=5, min_train=5)
    # Held-out rows may be filtered if their item is unseen in train, so the
    # count is an upper bound rather than an equality.
    assert len(test) <= 5 * 5
    assert len(valid) <= 5 * 5


def test_users_below_the_minimum_stay_entirely_in_train():
    """A user with too little history teaches nothing and adds noise."""
    df = make_frame(n_users=3, n_per_user=6)   # 6 < 5 + 5 + 5
    train, valid, test = leave_last_n_split(df, n_valid=5, n_test=5, min_train=5)
    assert len(train) == len(df)
    assert valid.empty and test.empty


def test_evaluation_sets_contain_no_unseen_users_or_items():
    """Collaborative filtering has no representation for these, so scoring them
    would measure cold-start coverage rather than ranking quality."""
    df = make_frame()
    train, valid, test = leave_last_n_split(df)

    known_users = set(train["user_id"])
    known_items = set(train["video_id"])
    for part in (valid, test):
        assert set(part["user_id"]) <= known_users
        assert set(part["video_id"]) <= known_items


def test_temporal_split_is_globally_ordered():
    """The stronger guarantee: no test row predates any train row, across users."""
    df = make_frame()
    train, valid, test = temporal_split(df, valid_days=40, test_days=40)
    if len(train) and len(test):
        assert train["created_at"].max() < test["created_at"].min()


def test_splits_do_not_overlap():
    df = make_frame()
    train, valid, test = leave_last_n_split(df)

    def keys(part):
        return set(zip(part["user_id"], part["video_id"], part["created_at"]))

    assert not (keys(train) & keys(test))
    assert not (keys(train) & keys(valid))
    assert not (keys(valid) & keys(test))


def test_split_dispatches_on_config():
    df = make_frame()
    a = split(df, {"strategy": "leave_last_n", "n_valid": 5, "n_test": 5})
    b = leave_last_n_split(df, n_valid=5, n_test=5)
    assert len(a[0]) == len(b[0])


def test_split_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        split(make_frame(), {"strategy": "random"})


def test_ground_truth_requires_genuine_approval():
    """An interaction is not an endorsement.

    Only items above the completion threshold count as relevant, otherwise the
    metric rewards predicting exposure rather than approval.
    """
    df = make_frame(n_users=2, n_per_user=9)
    truth = build_ground_truth(df, min_completion=0.7)

    for user, items in truth.items():
        user_rows = df[(df["user_id"] == user) & (df["video_id"].isin(items))]
        assert (user_rows["completion_ratio"] >= 0.7).all()


def test_ground_truth_threshold_is_restrictive():
    df = make_frame(n_users=2, n_per_user=9)
    strict = build_ground_truth(df, min_completion=0.7)
    loose = build_ground_truth(df, min_completion=0.0)

    strict_total = sum(len(v) for v in strict.values())
    loose_total = sum(len(v) for v in loose.values())
    assert strict_total < loose_total