from recsys.serving.predictor import Predictor


class FakeContent:
    item_vectors = None
    id_to_row = {}
    def recommend(self, history, n=200, weights=None):
        return [(901, 0.9), (902, 0.8), (903, 0.7)]
    def similar_items(self, video_id, n=10):
        return [(902, 0.95), (903, 0.90)]


class FakeCF:
    def recommend(self, user_id, n=200):
        return [(902, 5.0), (904, 4.0)]


class FakeBroken:
    def recommend(self, history, n=200, weights=None):
        raise RuntimeError("this retriever is down")


class FakeTrending:
    def top(self, n=50, exclude=None):
        exclude = exclude or set()
        return [(v, 1.0) for v in (905, 906, 907) if v not in exclude][:n]


item_meta = {
    901: {"title": "Alpha",   "category": "Action", "creator_id": "A"},
    902: {"title": "Beta",    "category": "Drama",  "creator_id": "B"},
    903: {"title": "Gamma",   "category": "Action", "creator_id": "A"},
    904: {"title": "Delta",   "category": "Comedy", "creator_id": "C"},
    905: {"title": "Epsilon", "category": "Drama",  "creator_id": "D"},
    906: {"title": "Zeta",    "category": "Comedy", "creator_id": "E"},
    907: {"title": "Eta",     "category": "Action", "creator_id": "F"},
}

p = Predictor(
    content=FakeContent(),
    collaborative=FakeCF(),
    item_knn=FakeBroken(),
    trending=FakeTrending(),
    item_meta=item_meta,
    version="test-1",
)

history = [1, 2, 3, 4, 5, 6]

print("--- normal user ---")
out = p.recommend(user_id=7, history=history, top_k=4)
print("strategy:", out["strategy"], "| version:", out["model_version"])
for i in out["items"]:
    print(" ", i["video_id"], i["title"], i["sources"], "|", i["explanation"])
    
print()
print("--- cold start (2 items of history) ---")
out = p.recommend(user_id=8, history=[1, 2], top_k=3)
print("strategy:", out["strategy"])
print(" ", [i["video_id"] for i in out["items"]])

print()
print("--- history excluded ---")
out = p.recommend(user_id=7, history=history + [901], top_k=4)
print(" ", [i["video_id"] for i in out["items"]], " 901 must be absent")

print()
print("--- similar ---")
out = p.similar(901, top_k=2)
print("strategy:", out["strategy"], [i["video_id"] for i in out["items"]])

print()
print("--- recommend_ids ---")
print(" ", p.recommend_ids(7, history, top_k=4))