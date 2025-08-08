# scripts/preprocess.py

import pandas as pd
import os

print("=== STARTING DATA PREPROCESSING ===")

# 1️⃣ Verify file
csv_file = 'turbine_data.csv'
if not os.path.exists(csv_file):
    print(f"❌ ERROR: File {csv_file} not found.")
    exit()

file_size_mb = os.path.getsize(csv_file) / (1024 * 1024)
print(f"\n✅ File found: {csv_file}")
print(f"📦 File size: {file_size_mb:.1f} MB")

# 2️⃣ Test reading columns
try:
    cols = pd.read_csv(csv_file, nrows=0).columns.tolist()
    print(f"\n🔍 Columns detected ({len(cols)} total):")
    print(cols)
except Exception as e:
    print("\n❌ ERROR reading CSV headers:", str(e))
    exit()

# 3️⃣ Process data in chunks with NaN logging
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

print("\n⚙️ Processing chunks (50,000 rows per chunk)...\n")

try:
    for i, chunk in enumerate(pd.read_csv(csv_file, chunksize=50000)):
        
        # NaN check
        nan_counts = chunk.isnull().sum()
        nan_counts = nan_counts[nan_counts > 0]  # Only non-zero NaNs

        if not nan_counts.empty:
            print(f"🟡 Chunk {i}: NaN values found in {len(nan_counts)} features.")
            print(nan_counts)
        else:
            print(f"✅ Chunk {i}: No NaNs found in this chunk.")

        # Save to Parquet
        out_file = os.path.join(output_dir, f"chunk_{i}.parquet")
        chunk.to_parquet(out_file)
        print(f"💾 Saved {out_file} ({len(chunk)} rows)")

    print("\n🎉 ALL DONE! Check the 'data' folder for processed chunks.")

except Exception as e:
    print("\n❌ ERROR during processing:", str(e))
