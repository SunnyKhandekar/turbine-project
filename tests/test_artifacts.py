import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from turbine_project.models.artifacts import FederatedIsolationForestEnsemble, LocalModelArtifact


def _artifact() -> LocalModelArtifact:
    x = np.array([[0.0], [0.1], [0.2], [2.5]])
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    model = IsolationForest(contamination=0.25, random_state=42, n_estimators=20)
    model.fit(x_scaled)
    scores = model.decision_function(x_scaled)
    return LocalModelArtifact(
        asset_id="a",
        feature_names=["power_2_avg"],
        contamination=0.25,
        sample_count=4,
        scaler=scaler,
        model=model,
        score_threshold=float(np.quantile(scores, 0.25)),
        train_score_mean=float(scores.mean()),
        train_score_std=float(scores.std(ddof=0)),
    )


def test_federated_ensemble_scores_rows() -> None:
    artifact = _artifact()
    ensemble = FederatedIsolationForestEnsemble(
        feature_names=["power_2_avg"],
        client_models=[artifact],
        weights=np.array([1.0]),
        global_threshold=artifact.score_threshold,
    )
    scores = ensemble.score(np.array([[0.0], [3.0]]))
    assert scores.shape == (2,)
