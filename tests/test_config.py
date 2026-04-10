from pathlib import Path

from turbine_project.config import load_config


def test_load_config_uses_absolute_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  csv_path: sample.csv
  processed_dir: data/processed
  reports_dir: outputs/reports
  chunk_size: 100
  feature_columns: [power_2_avg]
  timestamp_column: time_stamp
  asset_id_column: asset_id
  split_column: train_test
  label_column: status_type_id
  split_train_value: train
  split_test_value: prediction
  healthy_label: 0
  anomaly_labels: [3, 4, 5]
  low_quantile: 0.01
  high_quantile: 0.99
  event_gap_minutes: 30
  min_required_columns: [time_stamp, asset_id, train_test, status_type_id, power_2_avg]
training:
  contamination: 0.05
  random_state: 42
  n_estimators: 100
  max_samples: auto
  model_dir: models
  synthetic_eval_multiplier: 3.0
  synthetic_eval_seed: 123
  calibration_fraction: 0.2
  threshold_grid_size: 21
  aggregation_target_estimators: 200
federated:
  rounds: 1
  min_clients: 1
  server_address: 127.0.0.1:8081
  global_model_path: models/global/model.joblib
inference:
  predictions_dir: outputs/predictions
  plots_dir: outputs/plots
  metrics_dir: outputs/metrics
  lambda_model_path: models/global/model.joblib
  sns_topic_arn: ''
  dashboard_dir: outputs/dashboard
  monitoring_dir: outputs/monitoring
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.dataset.chunk_size == 100
    assert config.dataset.csv_path.is_absolute()
