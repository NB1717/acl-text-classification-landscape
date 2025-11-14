import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import pandas as pd
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------
#  Paths & constants
# ------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "FinEntity", "processed", "finentity_taskB_test.csv")
LOGREG_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "logreg_taskB.joblib")
TFIDF_PATH = os.path.join(BASE_DIR, "..", "models", "tfidf_taskB.joblib")
FINBERT_CHECKPOINT_PATH = os.path.join(BASE_DIR, "..", "models", "finbert_taskB_entity_aware.pt")
FINBERT_METRICS_PATH = os.path.join(BASE_DIR, "..", "sorties", "taskB_finbert_results.json")

FIG_DIR = os.path.join(BASE_DIR, "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
FIG_PATH = os.path.join(FIG_DIR, "figure_taskB_confusion_matrices.png")

# ------------------------
# 1) Load test split & build entity-aware texts
# ------------------------
print(f"Loading test data from: {DATA_PATH}")
df_test = pd.read_csv(DATA_PATH)
df_test = df_test.dropna(subset=["sentence", "entity_text", "label"])

def make_entity_aware(row):
    return f"{row['sentence']} [ENTITY] {row['entity_text']} [/ENTITY]"

texts_test = df_test.apply(make_entity_aware, axis=1).tolist()
y_true = df_test["label"].tolist()

# Label order (we will enforce this everywhere)
LABEL_ORDER = ["Negative", "Neutral", "Positive"]

# ------------------------
# 2) Logistic Regression predictions
# ------------------------
print("Loading Logistic Regression + TF-IDF...")
tfidf = joblib.load(TFIDF_PATH)
logreg = joblib.load(LOGREG_MODEL_PATH)

X_test = tfidf.transform(texts_test)
y_pred_logreg = logreg.predict(X_test)

# ------------------------
# 3) FinBERT predictions
# ------------------------
print("Loading FinBERT model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load label mapping from metrics to be 100% consistent
with open(FINBERT_METRICS_PATH, "r", encoding="utf-8") as f:
    finbert_metrics = json.load(f)

label_mapping = finbert_metrics.get("label_mapping", {"Negative": 0, "Neutral": 1, "Positive": 2})
# Build index -> label in the correct order
idx_to_label = {v: k for k, v in label_mapping.items()}
num_labels = len(label_mapping)

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/FinBERT")
model = AutoModelForSequenceClassification.from_pretrained(
    "ProsusAI/FinBERT",
    num_labels=num_labels,
)
state_dict = torch.load(FINBERT_CHECKPOINT_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

class TaskBDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = LABEL_ORDER.index(self.labels[idx])
        return item

test_dataset = TaskBDataset(texts_test, labels=y_true)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

all_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)

# Map indices back to label names using the mapping from training
y_pred_finbert = [idx_to_label[idx] for idx in all_preds]

# ------------------------
# 4) Confusion matrices
# ------------------------
cm_logreg = confusion_matrix(y_true, y_pred_logreg, labels=LABEL_ORDER)
cm_finbert = confusion_matrix(y_true, y_pred_finbert, labels=LABEL_ORDER)

print("LogReg confusion matrix:\n", cm_logreg)
print("FinBERT confusion matrix:\n", cm_finbert)

# ------------------------
# 5) Plot side-by-side
# ------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

def plot_cm(ax, cm, title):
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(LABEL_ORDER)))
    ax.set_yticks(np.arange(len(LABEL_ORDER)))
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_yticklabels(LABEL_ORDER)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=9,
            )

plot_cm(axes[0], cm_logreg, "LogReg (TF–IDF)")
plot_cm(axes[1], cm_finbert, "FinBERT (entity-aware)")

fig.suptitle("FinEntity Task B – Confusion matrices on test set", fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig(FIG_PATH, dpi=300)
print(f"✅ Saved confusion matrices figure to: {FIG_PATH}")
