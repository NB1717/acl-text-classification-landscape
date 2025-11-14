import pandas as pd
from sklearn.model_selection import train_test_split
import os

# === Input data path ===
input_path = "../data/FinEntity/processed/finentity_taskB_full.csv"
output_dir = "../data/FinEntity/processed"

print(f"Loading Task B dataset from: {input_path}")
df = pd.read_csv(input_path)
print("Total rows:", len(df))

# Removing missing values (NaN)
df = df.dropna(subset=["sentence", "entity_text", "label"])
print("After dropping NaNs:", len(df))

# === Data splitting ===
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
dev_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

print(f"\nSizes:\nTrain: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_df)}")

# === Saving outputs ===
train_path = os.path.join(output_dir, "finentity_taskB_train.csv")
dev_path = os.path.join(output_dir, "finentity_taskB_dev.csv")
test_path = os.path.join(output_dir, "finentity_taskB_test.csv")

train_df.to_csv(train_path, index=False)
dev_df.to_csv(dev_path, index=False)
test_df.to_csv(test_path, index=False)

print("\n✅ Saved all splits to:", output_dir)

# === Label distribution in each split ===
for name, subset in [("Train", train_df), ("Dev", dev_df), ("Test", test_df)]:
    print(f"\n{name} label distribution:")
    print(subset["label"].value_counts())
