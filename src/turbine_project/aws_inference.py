from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import boto3
import joblib
import numpy as np

from .models.artifacts import FederatedIsolationForestEnsemble

LOGGER = logging.getLogger(__name__)
_MODEL_CACHE: FederatedIsolationForestEnsemble | None = None


def load_global_model(model_path: str | Path) -> FederatedIsolationForestEnsemble:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = joblib.load(Path(model_path))
    return _MODEL_CACHE


def lambda_handler(event: dict, context: object | None = None) -> dict:
    model_path = os.environ.get("MODEL_PATH", "models/global/federated_isolation_forest.joblib")
    topic_arn = os.environ.get("SNS_TOPIC_ARN", "")
    model = load_global_model(model_path)
    records = event.get("records", [])
    if not records:
        return {"statusCode": 400, "body": json.dumps({"error": "No records supplied"})}

    x = np.array([[record[feature] for feature in model.feature_names] for record in records], dtype=float)
    scores = model.score(x)
    predictions = model.predict(x)
    anomalies = []
    for record, score, prediction in zip(records, scores, predictions, strict=False):
        payload = {
            "asset_id": record.get("asset_id"),
            "timestamp": record.get("time_stamp"),
            "score": float(score),
            "prediction": int(prediction),
        }
        if prediction == 1:
            anomalies.append(payload)

    if anomalies and topic_arn:
        boto3.client("sns").publish(
            TopicArn=topic_arn,
            Subject="Wind turbine anomaly detected",
            Message=json.dumps({"anomalies": anomalies}),
        )

    return {"statusCode": 200, "body": json.dumps({"anomalies": anomalies, "records": len(records)})}
