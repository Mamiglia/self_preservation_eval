import json
import pandas as pd
import os

# Load the .eval file (JSONL format)
data = []
log_dir = os.path.join(os.path.dirname(__file__), "logs")
eval_files = [f for f in os.listdir(log_dir) if f.endswith(".eval")]
if not eval_files:
    raise FileNotFoundError(f"No .eval files found in {log_dir}")
file_path = os.path.join(log_dir, eval_files[0])
print(f"Using eval file: {file_path}")

with open(file_path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"Skipping invalid line: {line[:50]}...")
            continue

# Convert to DataFrame
df = pd.DataFrame(data)

# If categories are inside metadata
if "metadata" in df.columns:
    df["category"] = df["metadata"].apply(lambda x: x.get("category") if isinstance(x, dict) else None)

# Check category distribution
print(df["category"].value_counts())

# Split by category and save
for cat, group in df.groupby("category"):
    group.to_json(f"eval_{cat}.jsonl", orient="records", lines=True)
    group.to_csv(f"eval_{cat}.csv", index=False)

print("Done! Files saved per category.")