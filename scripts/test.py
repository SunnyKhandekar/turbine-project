# scripts/test.py

import pandas as pd
import numpy as np
import os
import glob
import joblib

print("=== TESTING STARTED ===\n")

# Load the model
model_path = "models/final_model.joblib"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ ERROR: Model not found at {model_path}")

model = joblib.load(model_path)
print(f"✅ Loaded model from {model_path}")

# Define feature used (same as in training)
features_used = ['power_2_avg']
print(f"🔧 Feature used for testing: {features_used}\n")

# Create output folder if needed
os.makedirs("outputs", exist_ok=True)

# Process each chunk
chunk_files = sorted(glob.glob("data/chunk_*.parquet"))

if not chunk_files:
    raise FileNotFoundError("❌ ERROR: No chunk files found in 'data/' folder.")

for chunk_file in chunk_files:
    chunk_name = os.path.basename(chunk_file)
    print(f"🔍 Processing {chunk_name}...")

    data = pd.read_parquet(chunk_file)

    # Handle missing values in the feature
    if data[features_used].isnull().values.any():
        print(f"⚠️ Warning: Found NaNs in {chunk_name}. Dropping rows with NaNs in {features_used}.")
        data = data.dropna(subset=features_used)

    # Predict anomalies
    scores = model.decision_function(data[features_used])
    preds = model.predict(data[features_used])  # -1 for anomaly, 1 for normal

    # Convert to readable format
    data['anomaly_score'] = scores
    data['anomaly'] = np.where(preds == -1, 1, 0)

    # Save predictions
    output_csv = f"outputs/predictions_{chunk_name.replace('.parquet', '.csv')}"
    data[['time_stamp', 'asset_id', 'anomaly_score', 'anomaly']].to_csv(output_csv, index=False)

    print(f"✅ Saved predictions to {output_csv}")

print("\n🎉 TESTING COMPLETE! Check the 'outputs' folder for predictions.")
