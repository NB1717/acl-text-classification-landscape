import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1) Input path
input_path = os.path.join("..", "data", "FinEntity", "processed", "finentity_taskA_full.csv")
print("Loading Task A dataset from:", input_path)

df = pd.read_csv(input_path)
print("Total sentences:", len(df))

# 2) Splitting into Train / Temp (Temp will later be divided into Dev and Test)
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["sentence_label"], random_state=42)

# 3) Splitting Temp into Dev and Test
dev_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["sentence_label"], random_state=42)

# 4) Printing statistics
print("\nSizes:")
print(f"Train: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_df)}")

print("\nLabel distribution per split:")
for name, part in [("Train", train_df), ("Dev", dev_df), ("Test", test_df)]:
    print(f"\n{name}:\n", part["sentence_label"].value_counts())

# 5) Output path
output_dir = os.path.join("..", "data", "FinEntity", "processed")

train_df.to_csv(os.path.join(output_dir, "finentity_taskA_train.csv"), index=False)
dev_df.to_csv(os.path.join(output_dir, "finentity_taskA_dev.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "finentity_taskA_test.csv"), index=False)

print("\n✅ Saved all splits to:", output_dir)
