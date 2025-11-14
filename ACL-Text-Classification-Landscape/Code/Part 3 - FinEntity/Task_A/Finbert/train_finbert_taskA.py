# -*- coding: utf-8 -*-
"""
Fine-tuning FinBERT on FinEntity Task A (sentence-level sentiment)
نسخهٔ بهبودیافته با:
- class weights برای مقابله با نامتوازنی کلاس‌ها
- batch_size=32
- 6 epoch
- early stopping بر اساس F1 روی dev
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# ===============================
# 1. General settings and paths
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

BASE_DIR = os.path.join("..", "data", "FinEntity", "processed")
TRAIN_PATH = os.path.join(BASE_DIR, "finentity_taskA_train.csv")
DEV_PATH   = os.path.join(BASE_DIR, "finentity_taskA_dev.csv")
TEST_PATH  = os.path.join(BASE_DIR, "finentity_taskA_test.csv")

label2id = {"Negative": 0, "Neutral": 1, "Positive": 2}
id2label = {v: k for k, v in label2id.items()}

# ===============================
# 2. Custom Dataset
# ===============================
class FinEntityDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["sentence"].tolist()
        self.labels = df["sentence_label"].map(label2id).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ===============================
# 3. Loading data
# ===============================
train_df = pd.read_csv(TRAIN_PATH)
dev_df   = pd.read_csv(DEV_PATH)
test_df  = pd.read_csv(TEST_PATH)

print("Train size:", len(train_df), "| Dev size:", len(dev_df), "| Test size:", len(test_df))

model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
).to(device)

train_data = FinEntityDataset(train_df, tokenizer)
dev_data   = FinEntityDataset(dev_df, tokenizer)
test_data  = FinEntityDataset(test_df, tokenizer)

# Larger batch_size for more stable training
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
dev_loader   = DataLoader(dev_data,   batch_size=64)
test_loader  = DataLoader(test_data,  batch_size=64)

# ===============================
# 4. Class weights for class imbalance
# ===============================
y_train = train_df["sentence_label"].map(label2id).values
classes = np.array([0, 1, 2])   # Must be a NumPy array
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
print("Class weights:", class_weights)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

# ===============================
# 5. Optimizer و Scheduler
# ===============================
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 6
num_training_steps = num_epochs * len(train_loader)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# ===============================
# 6. Evaluation function on a DataLoader
# ===============================
def evaluate(dataloader, split_name):
    model.eval()
    preds, labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            logits = outputs.logits
            loss = criterion(logits, batch["labels"])
            total_loss += loss.item()

            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    print(f"\n=== {split_name.upper()} ===")
    print(f"Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} | F1-macro: {f1:.4f}")
    print(classification_report(labels, preds, target_names=[id2label[i] for i in range(3)]))

    return avg_loss, acc, f1

# ===============================
# 7. Training loop with early stopping
# ===============================
best_dev_f1 = 0.0
best_state_dict = None
patience = 2
no_improve_epochs = 0

for epoch in range(num_epochs):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    total_loss = 0.0

    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        logits = outputs.logits
        loss = criterion(logits, batch["labels"])

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} finished | Avg train loss: {avg_train_loss:.4f}")

    # Evaluating on the dev set for early stopping
    dev_loss, dev_acc, dev_f1 = evaluate(dev_loader, "dev")

    if dev_f1 > best_dev_f1:
        best_dev_f1 = dev_f1
        best_state_dict = model.state_dict()
        no_improve_epochs = 0
        print(f"✅ New best dev F1: {best_dev_f1:.4f} (model checkpoint updated)")
    else:
        no_improve_epochs += 1
        print(f"Dev F1 did not improve. Patience counter = {no_improve_epochs}")
        if no_improve_epochs >= patience:
            print("⏹ Early stopping triggered.")
            break

# ===============================
# 8. Loading the best model and performing final evaluation
# ===============================
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
    print(f"\nLoaded best model with dev F1 = {best_dev_f1:.4f}")

print("\nFinal evaluation on TRAIN, DEV, and TEST with best model:\n")
evaluate(train_loader, "train")
evaluate(dev_loader, "dev")
evaluate(test_loader, "test")
