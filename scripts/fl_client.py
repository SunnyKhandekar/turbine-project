# scripts/fl_client.py

import flwr as fl
import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
import joblib
import os

# Load local chunk (e.g., chunk_0, chunk_1, etc.)
client_id = int(sys.argv[1])
chunk_path = f"data/chunk_{client_id}.parquet"

print(f"📦 Loading data from: {chunk_path}")
df = pd.read_parquet(chunk_path)

# Feature selection
features = ["power_2_avg"]
df = df.dropna(subset=features)
X = df[features].values

# Local model
class IFClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = IsolationForest(n_estimators=100, contamination=0.05)
        self.X = X

    def get_parameters(self, config=None):
        return [np.array([])]  # Dummy return; IsolationForest doesn't use weights directly

    def fit(self, parameters, config):
        print(f"🔧 Training on chunk_{client_id} with {len(self.X)} rows...")
        self.model.fit(self.X)
        joblib.dump(self.model, f"models/final_model_client{client_id}.joblib")
        print(f"✅ Model saved for chunk_{client_id}")
        return [np.array([])], len(self.X), {}

    def evaluate(self, parameters, config):
        preds = self.model.predict(self.X)
        y_true = np.zeros_like(preds)
        acc = (preds == y_true).mean()
        return float(acc), len(self.X), {"accuracy": float(acc)}

print("🚀 Connecting to FL server...")
fl.client.start_numpy_client(server_address="52.90.82.208:8081", client=IFClient())
