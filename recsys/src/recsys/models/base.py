"""Shared retriever interface, so pipelines and the predictor can treat every
candidate source interchangeably."""

from abc import ABC, abstractmethod

import joblib


class BaseRecommender(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, *args, **kwargs): ...

    @abstractmethod
    def recommend(self, *args, n: int = 200, **kwargs) -> list[tuple[int, float]]: ...

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)