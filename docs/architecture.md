# Architecture

## System Architecture

```mermaid
flowchart LR
    A["Raw SCADA CSV<br/>12.8 GB dataset"] --> B["Chunked preprocessing<br/>schema checks, imputation, clipping"]
    B --> C["Per-turbine Parquet store<br/>asset_id partitioned"]
    C --> D["Local turbine training<br/>multivariate IsolationForest"]
    D --> E["Threshold calibration<br/>healthy vs fault-like states"]
    E --> F["FedAvg-style aggregation<br/>tree-weighted global ensemble"]
    F --> G["Inference and scoring<br/>per row and per event"]
    G --> H["Predictions and plots"]
    G --> I["Monitoring and drift outputs"]
    G --> J["Executive dashboard"]
    F --> K["Cloud-ready deployment scaffolding"]
```

## Training and Evaluation Flow

```mermaid
flowchart TD
    A["Processed turbine data"] --> B["Split by train_test"]
    B --> C["Healthy-state training subset<br/>status_type_id = 0"]
    B --> D["Prediction / holdout subset"]
    C --> E["Scale features"]
    E --> F["Train local IsolationForest"]
    F --> G["Calibrate threshold on operational labels"]
    G --> H["Save local model artifact"]
    H --> I["Aggregate client models"]
    I --> J["Global ensemble model"]
    J --> K["Score holdout data"]
    K --> L["Row-level metrics"]
    K --> M["Fault-event metrics"]
    K --> N["Turbine-level metrics"]
```

## Local Productization Stack

```mermaid
flowchart LR
    A["CLI pipeline"] --> B["JSON metrics"]
    A --> C["Predictions parquet"]
    A --> D["Plots"]
    A --> E["Monitoring drift reports"]
    B --> F["Streamlit dashboard"]
    D --> F
    E --> F
    A --> G["Docker compose"]
    A --> H["Local CI PowerShell script"]
    I["AWS-ready inference handler"] --> J["Lambda / SNS future deployment"]
```

## Design Rationale

- `IsolationForest` was kept as the core model because it is the most consistent algorithmic choice across the project document and is easier to explain in a resource-constrained industrial setting.
- The aggregation layer is intentionally described as **FedAvg-style** rather than literal FedAvg for tree parameters, which keeps the architecture honest and technically defensible.
- Event-level evaluation was added because managers care more about catching meaningful fault periods than individual anomalous rows.
- The dashboard and drift reporting were added to make the project look and behave like an operational system rather than a one-off academic experiment.
