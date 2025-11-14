import pandas as pd

# Path to the test file
df = pd.read_csv("../data/FinEntity/processed/finentity_taskB_test.csv")

# Removing incomplete rows
df = df.dropna(subset=["sentence", "entity_text", "label"])

# Selecting 30 samples from each class
samples = []
for label in ["Positive", "Negative", "Neutral"]:
    subset = df[df["label"] == label].sample(n=30, random_state=42)
    samples.append(subset)

# Combining and shuffling
sampled_df = pd.concat(samples).sample(frac=1, random_state=42).reset_index(drop=True)

# Saving to the output directory
output_path = "../sorties/taskB_manual_samples.csv"
sampled_df.to_csv(output_path, index=False)

print(f"✅ Saved 90 balanced samples (30 per class) to: {output_path}\n")
print(sampled_df[["sentence", "entity_text", "label"]].head(10))
