"""Loads trained artifacts from recsys/artifacts and holds them in memory.

Keeping this separate means a retrain is a file swap plus a reload call - the API
never imports training code.
"""


class ModelRegistry:
    version: str = "unloaded"

    def load(self, artifact_dir: str, version: str = "latest") -> None:
        # joblib.load / faiss.read_index / lightgbm.Booster(model_file=...)
        ...

    def unload(self) -> None:
        ...

    def reload(self, artifact_dir: str, version: str) -> None:
        """Hot swap after a scheduled retrain - no restart, no dropped requests."""
        ...


model_registry = ModelRegistry()
