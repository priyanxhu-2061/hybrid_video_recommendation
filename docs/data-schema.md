# Data schema

## users
id, email, hashed_password, display_name, preferred_categories[], created_at

## videos
id, external_id, title, description, tags[], category, duration_seconds,
thumbnail_url, published_at, view_count, like_count

## interactions
id, user_id, video_id, event_type, watch_seconds, completion_ratio, rating,
session_id, source, position, created_at

`source` and `position` exist for training, not for the product. `source` records
which retriever surfaced the item; `position` records where it sat in the list.
Without both, you cannot correct for position bias and the reranker will just
learn to reproduce the current ranking.

## Indexes worth having early
- interactions (user_id, created_at DESC) - history lookups
- interactions (video_id) - item statistics
- interactions (session_id) - grouping for ranking training
- videos (category), videos (published_at DESC)
