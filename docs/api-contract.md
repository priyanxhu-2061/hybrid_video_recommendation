# API contract

GET  /api/v1/recommendations/feed?top_k=20      -> RecommendationResponse
GET  /api/v1/recommendations/similar/{id}       -> RecommendationResponse
POST /api/v1/interactions                       -> 202 Accepted
POST /api/v1/feedback                           -> 202 Accepted
GET  /api/v1/videos/{id}                        -> Video
GET  /api/v1/videos?q=&category=&page=          -> paginated
POST /api/v1/auth/register | /login | /refresh

`RecommendationResponse.items[].sources` is what lets the UI say why an item
appeared, and what lets you debug a bad feed without rerunning the model.
