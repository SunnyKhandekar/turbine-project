# scripts/fl_server.py

import flwr as fl
from sklearn.ensemble import IsolationForest
import numpy as np

# Dummy global model parameters (initialization)
def get_initial_parameters():
    model = IsolationForest(n_estimators=100, contamination=0.05)
    model.fit(np.random.rand(100, 1))  # Dummy fit to initialize
    return model

class FedServer(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
        )

print("🚀 Starting FL server...")
fl.server.start_server(
    server_address="0.0.0.0:8081",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=FedServer(),
)
