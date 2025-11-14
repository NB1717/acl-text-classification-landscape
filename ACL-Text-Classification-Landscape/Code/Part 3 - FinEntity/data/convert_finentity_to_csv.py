import json
import os
import pandas as pd

# Path to the JSON file from the perspective of the code/ directory
input_path = os.path.join("..", "data", "FinEntity", "data", "FinEntity.json")

print("Loading JSON from:", input_path)
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Number of sentences in JSON:", len(data))

rows = []

for sent_id, item in enumerate(data):
    sentence = item.get("content", "")
    annotations = item.get("annotations", [])

    # If for any reason a sentence has no annotation, we can skip it
    # or add it as an empty row. For now, we only create a row if an annotation exists.
    for ann in annotations:
        rows.append({
            "sentence_id": sent_id,
            "sentence": sentence,
            "entity_text": ann.get("value", ""),
            "label": ann.get("label", ""),
            "tag": ann.get("tag", ""),
            "start": ann.get("start", None),
            "end": ann.get("end", None),
        })


df = pd.DataFrame(rows)
print("Number of entity-level rows:", len(df))
print(df.head())


output_dir = os.path.join("..", "data", "FinEntity", "processed")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "finentity_taskB_full.csv")
df.to_csv(output_path, index=False, encoding="utf-8")

print("\nSaved CSV to:", output_path)
