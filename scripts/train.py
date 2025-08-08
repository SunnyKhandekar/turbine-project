import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === CONFIGURATION ===
DATA_PATH = 'data/chunk_0.parquet'
MODEL_PATH = 'models/final_model.joblib'
CONTAMINATION = 0.05
FEATURES = ['power_2_avg']

# === TRAINING START ===
print("=== TRAINING STARTED ===\n")

# Load data
if not os.path.exists(DATA_PATH):
    print(f"❌ ERROR: {DATA_PATH} not found!")
    exit(1)

df = pd.read_parquet(DATA_PATH)
print(f"✅ Loaded {df.shape[0]} rows from {DATA_PATH}")

# Show features selected
print(f"\n🔧 Features selected for training: {FEATURES}")

# Check for NaNs
nan_counts = df[FEATURES].isna().sum()
if nan_counts.sum() > 0:
    print(f"\n⚠️ WARNING: Found NaNs in features:")
    print(nan_counts[nan_counts > 0])
    print("\nPlease handle missing data before training.")
    exit(1)
else:
    print("✅ No NaNs in training data.\n")

# Prepare data
X = df[FEATURES].values

# Train Isolation Forest
model = IsolationForest(contamination=CONTAMINATION, random_state=42)
model.fit(X)
print(f"✅ Isolation Forest trained successfully on {len(FEATURES)} feature(s).")

# Save the model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
dump(model, MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH} (contamination={int(CONTAMINATION*100)}%)\n")

# === VISUALIZATION ===
print("📊 Generating anomaly score graph...")

# Compute anomaly scores
scores = model.decision_function(X)
preds = model.predict(X)  # -1 = anomaly, 1 = normal

# Plot decision function scores
plt.figure(figsize=(10, 5))
sns.histplot(scores, bins=50, kde=True, color='skyblue')
plt.axvline(np.percentile(scores, CONTAMINATION * 100), color='red', linestyle='--', label='Anomaly Threshold')
plt.title("Isolation Forest Anomaly Scores")
plt.xlabel("Decision Function Score")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig('outputs/anomaly_scores.png')
print("✅ Saved 'outputs/anomaly_scores.png'\n")

# === CONFUSION MATRIX ===
print("🧮 Generating confusion matrix (for inspection)...")

# Since Isolation Forest is unsupervised, use synthetic labels (for visualization only)
# All data is treated as normal (label=1), anomalies are detected by model
true_labels = np.ones_like(preds)  # Pretend all points are normal
pred_labels = np.where(preds == 1, 1, 0)  # 1=normal, 0=anomaly

cm = confusion_matrix(true_labels, pred_labels, labels=[1, 0])

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Anomaly'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Inspection Only)")
plt.savefig('outputs/confusion_matrix.png')
print("✅ Saved 'outputs/confusion_matrix.png'\n")

print("=== TRAINING COMPLETE ===")