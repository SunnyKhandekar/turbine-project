from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


@dataclass(slots=True)
class LocalModelArtifact:
    asset_id: str
    feature_names: list[str]
    contamination: float
    sample_count: int
    scaler: MinMaxScaler
    model: IsolationForest
    score_threshold: float
    train_score_mean: float
    train_score_std: float
    model_path: Path | None = None
    metadata: dict[str, float | int | str] = field(default_factory=dict)

    def score(self, x: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(x)
        return self.model.decision_function(transformed)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.score(x) < self.score_threshold).astype(int)


@dataclass(slots=True)
class FederatedIsolationForestEnsemble:
    feature_names: list[str]
    client_models: list[LocalModelArtifact]
    weights: np.ndarray
    global_threshold: float

    def score(self, x: np.ndarray) -> np.ndarray:
        scores = np.column_stack([artifact.score(x) for artifact in self.client_models])
        return np.average(scores, axis=1, weights=self.weights)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.score(x) < self.global_threshold).astype(int)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


def load_joblib(path: Path):
    return joblib.load(path)
