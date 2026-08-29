"""MovieLens adapter (20M and 32M).

MovieLens gives us explicit star ratings. The rest of this pipeline is built for
implicit watch signals, so this module translates between the two. Doing the
translation here - in one place, at the edge - means every model downstream stays
dataset-agnostic. Swap in real watch logs later and nothing else changes.

Expected files, in either naming convention:
    ratings.csv / rating.csv   userId, movieId, rating, timestamp
    movies.csv  / movie.csv    movieId, title, genres
    tags.csv    / tag.csv      userId, movieId, tag, timestamp

20M only, optional:
    genome_scores.csv          movieId, tagId, relevance
    genome_tags.csv            tagId, tag
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Ratings run 0.5 to 5.0 in half-star steps. Where we cut them into
# like / complete / skip is a modelling choice, not a data fact:
#
#   >= 4.0  a clear endorsement            -> like
#   >= 3.0  watched, mildly positive       -> complete
#   <  3.0  actively disliked              -> skip  (a real negative)
#
# The 3.0 boundary is the one that matters. Treating every rating as a positive
# because "they bothered to rate it" is the standard beginner mistake with
# MovieLens - it throws away the only genuine negative signal in the dataset.
LIKE_THRESHOLD = 4.0
POSITIVE_THRESHOLD = 3.0

RATING_MIN, RATING_MAX = 0.5, 5.0


def _find(raw_dir: Path, *candidates: str) -> Path:
    """Resolve a file that ships under different names depending on the source.

    GroupLens uses plural filenames (ratings.csv); the Kaggle mirror uses
    singular (rating.csv). Same contents, same columns - only the name differs,
    so there is no reason to make the caller care which one they downloaded.
    """
    for name in candidates:
        path = raw_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"None of {candidates} found in {raw_dir}. "
        "Download MovieLens from https://grouplens.org/datasets/movielens/ "
        "or Kaggle and unzip it there."
    )


def _rating_to_event(rating: float) -> str:
    if rating >= LIKE_THRESHOLD:
        return "like"
    if rating >= POSITIVE_THRESHOLD:
        return "complete"
    return "skip"


def _rating_to_completion(rating):
    """Map the star scale onto [0, 1] to stand in for completion ratio.

    This is a proxy, and worth naming as one in your write-up: a 5-star rating
    is not literally 'watched to the end'. But it preserves the ordering and the
    relative spacing, which is all the weighting function actually uses.
    """
    return (rating - RATING_MIN) / (RATING_MAX - RATING_MIN)


def load_ratings(
    raw_dir: str | Path,
    sample_users: int | None = None,
    min_rating_count: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Load the ratings file and reshape it into the interactions schema.

    sample_users: cap the number of users. Use this. The full 20M/32M rows will
    run out of patience (and possibly RAM) on a laptop, and every architectural
    decision you need to make is visible at 20,000 users. Move to the full set
    once the pipeline is correct.
    """
    raw_dir = Path(raw_dir)
    path = _find(raw_dir, "ratings.csv", "rating.csv")

    # Explicit dtypes matter here. Left to itself pandas picks int64/float64 for
    # everything and this frame balloons past 2GB.
    df = pd.read_csv(
        path,
        usecols=["userId", "movieId", "rating", "timestamp"],
        dtype={
            "userId": np.int32,
            "movieId": np.int32,
            "rating": np.float32,
        },
    )

    # The Kaggle 20M upload stores timestamps as datetime strings; GroupLens
    # ships raw epoch seconds. Detect rather than assume.
        # GroupLens ships raw epoch seconds; the Kaggle mirror ships datetime
    # strings. Ask pandas rather than numpy - on pandas 3.x a text column is
    # StringDtype, which np.issubdtype cannot interpret at all.
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        df["created_at"] = pd.to_datetime(df["timestamp"], unit="s")
    else:
        df["created_at"] = pd.to_datetime(df["timestamp"], format="mixed")
        
    if sample_users is not None:
        rng = np.random.default_rng(seed)
        users = df["userId"].unique()
        if sample_users < len(users):
            # Sample whole users, never random rows. Slicing rows would leave
            # every user with a shredded history, and collaborative filtering
            # would look far worse than it actually is.
            keep = rng.choice(users, size=sample_users, replace=False)
            df = df[df["userId"].isin(keep)]

    if min_rating_count > 0:
        counts = df.groupby("userId")["movieId"].transform("size")
        df = df[counts >= min_rating_count]

    ratings = df["rating"].to_numpy()

    out = pd.DataFrame({
        "user_id": df["userId"].to_numpy(),
        "video_id": df["movieId"].to_numpy(),
        "event_type": [_rating_to_event(r) for r in ratings],
        "watch_seconds": 0.0,              # unavailable in MovieLens
        "completion_ratio": _rating_to_completion(ratings),
        "rating": ratings,
        "session_id": None,                # no session concept in this dataset
        "source": None,                    # no impression logging (see phase 4)
        "position": None,
        "created_at": df["created_at"].to_numpy(),
    })

    # The raw file is ordered by userId then movieId, NOT by time. The temporal
    # splitter depends on chronological order, so sort here rather than trusting
    # the file.
    return out.sort_values("created_at").reset_index(drop=True)


def load_movies(raw_dir: str | Path, max_tags_per_movie: int = 12) -> pd.DataFrame:
    """Load the movies file (+ tags if present) into the videos schema.

    MovieLens has no plot descriptions, so the content model has to work from
    title, genres, and community tags. The tags are what make content-based
    viable at all here - genres alone give you about 20 distinct vectors.
    """
    raw_dir = Path(raw_dir)
    movies = pd.read_csv(
        _find(raw_dir, "movies.csv", "movie.csv"),
        dtype={"movieId": np.int32},
    )

    # Titles arrive as "Toy Story (1995)". Split the year out - it is a useful
    # feature on its own, and leaving it inline pollutes the TF-IDF vocabulary
    # with four-digit tokens.
    extracted = movies["title"].str.extract(r"^(.*?)\s*\((\d{4})\)\s*$")
    movies["clean_title"] = extracted[0].fillna(movies["title"])
    movies["year"] = pd.to_numeric(extracted[1], errors="coerce")

    movies["genre_list"] = (
        movies["genres"]
        .fillna("")
        .apply(lambda g: [x for x in g.split("|") if x and x != "(no genres listed)"])
    )

    tag_map = _aggregate_tags(raw_dir, max_tags_per_movie)
    movies["user_tags"] = movies["movieId"].map(tag_map).apply(
        lambda t: t if isinstance(t, list) else []
    )

    return pd.DataFrame({
        "video_id": movies["movieId"].to_numpy(),
        "title": movies["clean_title"].to_numpy(),
        # The 'description' the content model reads. Genres carry the coarse
        # signal, tags the fine-grained one.
        "description": [
            " ".join(g) + " " + " ".join(t)
            for g, t in zip(movies["genre_list"], movies["user_tags"])
        ],
        "tags": movies["genre_list"] + movies["user_tags"],
        "category": movies["genre_list"].apply(lambda g: g[0] if g else "Unknown"),
        "duration_seconds": 0,             # unavailable
        # No creator field in MovieLens. Genre stands in so the diversifier's
        # per-creator cap has something to work with - imperfect, but it stops
        # one genre owning the whole feed.
        "creator_id": movies["genre_list"].apply(lambda g: g[0] if g else "Unknown"),
        "published_at": pd.to_datetime(movies["year"], format="%Y", errors="coerce"),
    })


def _aggregate_tags(raw_dir: Path, top_n: int) -> dict[int, list[str]]:
    """Most-applied tags per movie.

    Frequency-ranked, not deduplicated-arbitrary: a tag twelve people applied
    describes the film, a tag one person applied describes that person.
    """
    try:
        path = _find(raw_dir, "tags.csv", "tag.csv")
    except FileNotFoundError:
        return {}

    tags = pd.read_csv(path, dtype={"movieId": np.int32}, usecols=["movieId", "tag"])
    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
    tags = tags[tags["tag"].str.len().between(2, 40)]

    counts = tags.groupby(["movieId", "tag"]).size().reset_index(name="n")
    counts = counts.sort_values(["movieId", "n"], ascending=[True, False])
    top = counts.groupby("movieId").head(top_n)

    return top.groupby("movieId")["tag"].apply(list).to_dict()


def load_genome(raw_dir: str | Path):
    """Tag genome: a dense item embedding, free, in the 20M dataset.

    Instead of "twelve people called this film dystopian", the genome gives a
    relevance score in [0, 1] for each of 1,128 tags against ~10,000 movies.
    That is already an embedding matrix - no TF-IDF, no training - and it is
    usually a stronger content signal than raw tags.

    Returns (movie_ids, matrix, tag_names), or None if the files are absent
    (the 32M dataset ships without them).
    """
    raw_dir = Path(raw_dir)
    try:
        scores_path = _find(raw_dir, "genome_scores.csv", "genome-scores.csv")
        tags_path = _find(raw_dir, "genome_tags.csv", "genome-tags.csv")
    except FileNotFoundError:
        return None

    scores = pd.read_csv(
        scores_path,
        dtype={"movieId": np.int32, "tagId": np.int32, "relevance": np.float32},
    )
    tag_names = pd.read_csv(tags_path).sort_values("tagId")["tag"].tolist()

    # Long to wide: one row per movie, one column per tag.
    wide = scores.pivot(index="movieId", columns="tagId", values="relevance")
    wide = wide.sort_index(axis=1).fillna(0.0)

    return wide.index.to_numpy(), wide.to_numpy(dtype=np.float32), tag_names


def load(raw_dir: str | Path, sample_users: int | None = None):
    """One call for the pipeline: returns (interactions, videos).

    Videos are filtered to those that actually appear in the ratings. The full
    catalogue includes thousands of films nobody in your sample has rated, and
    carrying them inflates catalogue-coverage denominators so every model looks
    like it covers 2% of the catalogue.
    """
    interactions = load_ratings(raw_dir, sample_users=sample_users)
    videos = load_movies(raw_dir)
    videos = videos[videos["video_id"].isin(interactions["video_id"].unique())]
    return interactions, videos.reset_index(drop=True)