from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.utils.io import read_json, write_json


@dataclass
class MinMaxNormalizer:
    min_value: float = 0.0
    max_value: float = 1.0
    eps: float = 1e-8

    def fit(self, values: np.ndarray) -> "MinMaxNormalizer":
        self.min_value = float(np.min(values))
        self.max_value = float(np.max(values))
        return self

    def transform(self, array: np.ndarray) -> np.ndarray:
        scale = self.max_value - self.min_value
        return ((array - self.min_value) / (scale + self.eps)).astype(np.float32)

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        scale = self.max_value - self.min_value
        return (array * (scale + self.eps) + self.min_value).astype(np.float32)

    def save(self, path: str | Path) -> None:
        write_json(path, asdict(self))

    @classmethod
    def load(cls, path: str | Path) -> "MinMaxNormalizer":
        return cls(**read_json(path))
