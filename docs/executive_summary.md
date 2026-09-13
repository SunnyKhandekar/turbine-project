# Executive Summary

## Overview

This project delivers a local production-style anomaly detection platform for wind turbine SCADA data, designed to be cloud-ready and operationally explainable. The system ingests the real-world [CARE-to-Compare](https://www.edp.com/en/innovation/open-data/data) dataset (three wind farms, dozens of per-turbine event files each), preprocesses it into turbine-specific Parquet partitions, trains local anomaly detectors, aggregates them into a federated global model, and exposes results through dashboarding, monitoring outputs, and deployment scaffolding.

## What the platform does

- Detects abnormal operating behavior at the turbine level using multivariate `IsolationForest`.
- Preserves the federated-learning story by keeping modeling local to each turbine and combining client contributions through a tree-weighted FedAvg-style aggregation layer.
- Evaluates performance at three levels:
  - row level for model analysis
  - turbine level for operational confidence
  - event level for maintenance usefulness
- Produces plots, prediction files, drift summaries, and dashboard outputs for stakeholder review.

## Why this is manager-ready

- It is not just a model script; it is a reproducible system with configuration, monitoring, automation, tests, and deployment scaffolding.
- The implementation uses the real SCADA dataset rather than an artificial benchmark.
- The design is practical for industrial interviews because it favors explainability, communication efficiency, and local deployment realism over over-engineered complexity.

## Current validated results

The latest full-farm run, against real Wind Farm A data (22 event files, 5 turbines, ~1.2M rows):

- `Assets evaluated`: 5
- `Turbine-level accuracy`: 100%
- `Mean row F1`: 0.175
- `Mean event F1`: 0.268
- `Estimated AWS-equivalent monthly cost target`: $11.58

These reflect the pipeline after fixing a real feature-engineering bug found during validation (a generic sensor-averaging step was diluting a strong rotational-speed signal with unrelated, much larger-magnitude sensors) - every turbine improved on both row and event F1 after the fix, evidence the diagnosis and correction were sound, not just plausible-sounding. Wind Farms B and C use the same pipeline but haven't been run at full scale yet.

## Interpretation for a business audience

- The platform can identify fault-like behavior while sharply reducing the need to centralize raw turbine data.
- Event-level performance is materially stronger than the original prototype and is much closer to how a maintenance organization would think about operational value.
- The architecture is ready for a future move to real cloud deployment, but it is already strong enough locally for demonstration, dissertation defense, and technical interviews.

## Strategic strengths

- Real dataset validation
- Explainable features tied to power, wind speed, and reactive power behavior
- Production-style repo structure
- Local automation for repeatability
- Dashboard and monitoring for decision support
- Honest cloud-ready story without overstating deployment status

## Remaining gap to a full enterprise rollout

The remaining work is mostly scale and commercialization work, not basic system design:

- full-scale runs on Wind Farms B and C (larger, more sensors - the pipeline already supports this via `configs/farm_b.yaml`/`configs/farm_c.yaml`)
- live cloud deployment if credentials and environment are available
- organization-specific alert routing and CI/CD integration
- broader stress testing across more turbines and larger evaluation windows

## Recommended presentation line

> Built a fully working local production-style anomaly detection platform for wind turbine SCADA data, with federated client/server design, cloud-ready AWS architecture, monitoring, dashboarding, and real-dataset validation.
