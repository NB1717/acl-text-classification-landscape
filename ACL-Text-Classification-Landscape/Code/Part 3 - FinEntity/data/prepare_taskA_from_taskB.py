import os
import pandas as pd

# 1) Input path: entity-level file
input_path = os.path.join("..", "data", "FinEntity", "processed", "finentity_taskB_full.csv")
print("Loading entity-level data from:", input_path)

df = pd.read_csv(input_path)
print("Entity-level shape:", df.shape)
print(df.head())

# 2) Checking required columns
required_cols = ["sentence_id", "sentence", "entity_text", "label"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# 3) Removing problematic rows
df = df.dropna(subset=["sentence_id", "sentence", "label"])
print("After dropping NaNs:", df.shape)

# 4) Function to convert multiple entity labels into a single sentence label
def infer_sentence_label(labels):
    """
    labels: مجموعه‌ای از برچسب‌های موجودیت برای یک جمله
    خروجی: برچسب جمله (Positive / Negative / Neutral) یا None اگر مبهم باشد.
    """
    unique = set(labels)

    has_pos = "Positive" in unique
    has_neg = "Negative" in unique
    has_neu = "Neutral" in unique

    # Neutral only
    if has_neu and not has_pos and not has_neg:
        return "Neutral"
    # Positive (may also include neutral)
    if has_pos and not has_neg:
        return "Positive"
    # Negative (may also include neutral)
    if has_neg and not has_pos:
        return "Negative"
    # If both positive and negative are present → ambiguous sentence, remove it
    return None

# 5) Grouping by sentence_id
grouped = df.groupby("sentence_id")

rows = []
dropped_mixed = 0

for sent_id, group in grouped:
    sentence_text = group["sentence"].iloc[0]
    labels = group["label"].tolist()

    sentence_label = infer_sentence_label(labels)
    if sentence_label is None:
        dropped_mixed += 1
        continue  # Ambiguous sentences are not used for Task A


    rows.append({
        "sentence_id": sent_id,
        "sentence": sentence_text,
        "sentence_label": sentence_label,
    })

taskA_df = pd.DataFrame(rows)
print("\nSentence-level DataFrame (Task A) shape:", taskA_df.shape)
print(taskA_df.head())
print(f"Dropped sentences with mixed positive/negative sentiment: {dropped_mixed}")

# 6) Mapping text labels to numeric IDs
label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
taskA_df["label_id"] = taskA_df["sentence_label"].map(label_map)

# 7) Creating the model input column (for classic models and BERT)
# For Task A, the sentence itself is sufficient
taskA_df["text"] = taskA_df["sentence"].astype(str)

# 8) Saving the output
output_dir = os.path.join("..", "data", "FinEntity", "processed")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "finentity_taskA_full.csv")
taskA_df.to_csv(output_path, index=False, encoding="utf-8")

print("\nSaved Task A sentence-level dataset to:", output_path)

# 9) Printing the label distribution for the report
print("\nLabel distribution (sentence_label):")
print(taskA_df["sentence_label"].value_counts())
