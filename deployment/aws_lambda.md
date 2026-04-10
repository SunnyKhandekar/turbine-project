# AWS Lambda Deployment Notes

1. Package `src/turbine_project/aws_inference.py` and the global model artifact located at `models/global/federated_isolation_forest.joblib`.
2. Set environment variables:
   - `MODEL_PATH`
   - `SNS_TOPIC_ARN` (optional)
3. Configure the Lambda handler as `turbine_project.aws_inference.lambda_handler`.
4. Use the event schema:

```json
{
  "records": [
    {
      "asset_id": 34,
      "time_stamp": "2024-09-10T00:00:00.000Z",
      "power_2_avg": 0.2705
    }
  ]
}
```

5. The handler returns anomaly scores and publishes SNS alerts when the optional topic ARN is configured.
