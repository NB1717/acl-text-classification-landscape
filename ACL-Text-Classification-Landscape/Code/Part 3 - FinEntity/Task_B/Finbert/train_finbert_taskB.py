import os
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
)


# -----------------------------
# 1) Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# -----------------------------
# 2) Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "FinEntity" / "processed"
MODELS_DIR = BASE_DIR.parent / "models"
SORTIES_DIR = BASE_DIR.parent / "sorties"

MODELS_DIR.mkdir(exist_ok=True, parents=True)
SORTIES_DIR.mkdir(exist_ok=True, parents=True)


# -----------------------------
# 3) Label mapping
# -----------------------------
LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# -----------------------------
# 4) Dataset class
# -----------------------------
class FinEntityTaskBDataset(Dataset):
    """
    Entity-level dataset for Task B.
    Each row represents (sentence, entity_text, label).
    The input text is formatted in an entity-aware way:
        "sentence [ENTITY] entity_text [/ENTITY]"
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Map string labels to IDs
        self.labels = self.df["label"].map(LABEL2ID).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sentence = str(row["sentence"])
        entity_text = str(row["entity_text"])

        # Entity-aware formatting
        formatted_text = f"{sentence} [ENTITY] {entity_text} [/ENTITY]"

        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item


# -----------------------------
# 5) Load data
# -----------------------------
def load_splits():
    train_path = DATA_DIR / "finentity_taskB_train.csv"
    dev_path = DATA_DIR / "finentity_taskB_dev.csv"
    test_path = DATA_DIR / "finentity_taskB_test.csv"

    print(f"Loading train split from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Train size before dropna: {len(train_df)}")
    train_df = train_df.dropna(subset=["sentence", "entity_text", "label"])
    print(f"Train size after dropna:  {len(train_df)}\n")

    print(f"Loading dev split from:   {dev_path}")
    dev_df = pd.read_csv(dev_path)
    print(f"Dev size before dropna:   {len(dev_df)}")
    dev_df = dev_df.dropna(subset=["sentence", "entity_text", "label"])
    print(f"Dev size after dropna:    {len(dev_df)}\n")

    print(f"Loading test split from:  {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Test size before dropna:  {len(test_df)}")
    test_df = test_df.dropna(subset=["sentence", "entity_text", "label"])
    print(f"Test size after dropna:   {len(test_df)}\n")

    return train_df, dev_df, test_df


# -----------------------------
# 6) Evaluation helper
# -----------------------------
def evaluate(model, dataloader, device, split_name="DEV"):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    losses = []

    loss_fn = CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits

            losses.append(loss.item())

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = float(np.mean(losses))
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    # ROC–AUC (macro, OVR) if possible
    try:
        y_true_oh = np.eye(len(LABEL2ID))[all_labels]
        macro_roc_auc = roc_auc_score(
            y_true_oh, np.array(all_probs), multi_class="ovr"
        )
    except Exception:
        macro_roc_auc = float("nan")

    print(f"\n=== {split_name.upper()} ===")
    print(f"Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} | F1-macro: {macro_f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=[ID2LABEL[i] for i in range(len(LABEL2ID))]))

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_roc_auc": macro_roc_auc,
        "y_true": all_labels,
        "y_pred": all_preds,
    }


# -----------------------------
# 7) Main training loop
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_df, dev_df, test_df = load_splits()

    # Tokenizer & model
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    # Datasets
    max_length = 128
    train_dataset = FinEntityTaskBDataset(train_df, tokenizer, max_length=max_length)
    dev_dataset = FinEntityTaskBDataset(dev_df, tokenizer, max_length=max_length)
    test_dataset = FinEntityTaskBDataset(test_df, tokenizer, max_length=max_length)

    # Dataloaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Class weights (balanced)
    train_labels_ids = train_df["label"].map(LABEL2ID).values
    classes = np.array(sorted(LABEL2ID.values()))
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_labels_ids,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Class weights:", class_weights)

    loss_fn = CrossEntropyLoss(weight=class_weights_tensor)

    # Optimizer & scheduler
    learning_rate = 2e-5
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_epochs = 5
    num_training_steps = num_epochs * len(train_loader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    # Early stopping settings
    best_dev_f1 = -1.0
    best_state_dict = None
    patience = 2
    epochs_no_improve = 0

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []

        print(f"\nEpoch {epoch}/{num_epochs}")
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # Replace default loss with class-weighted loss (for safety)
            logits = outputs.logits
            weighted_loss = loss_fn(logits, labels)

            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_losses.append(weighted_loss.item())

        avg_train_loss = float(np.mean(epoch_losses))
        print(f"Epoch {epoch} finished | Avg Train Loss: {avg_train_loss:.4f}")

        # Evaluate on dev
        dev_metrics = evaluate(model, dev_loader, device, split_name="DEV")
        dev_f1 = dev_metrics["macro_f1"]

        # Early stopping check
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
            print(f"✨ New best DEV macro F1: {best_dev_f1:.4f} (model checkpoint updated)")
        else:
            epochs_no_improve += 1
            print(f"No improvement on DEV F1 for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Load best model (according to dev F1)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Final evaluation on all splits
    print("\n===== FINAL EVALUATION WITH BEST CHECKPOINT =====")
    train_metrics = evaluate(model, train_loader, device, split_name="TRAIN")
    dev_metrics = evaluate(model, dev_loader, device, split_name="DEV")
    test_metrics = evaluate(model, test_loader, device, split_name="TEST")

    # -------------------------
    # Save model & results
    # -------------------------
    model_path = MODELS_DIR / "finbert_taskB_entity_aware.pt"
    print(f"\n✅ Saving best model to: {model_path}")
    torch.save(model.state_dict(), model_path)

    results = {
        "train": {
            "accuracy": train_metrics["accuracy"],
            "macro_f1": train_metrics["macro_f1"],
            "macro_roc_auc": train_metrics["macro_roc_auc"],
        },
        "dev": {
            "accuracy": dev_metrics["accuracy"],
            "macro_f1": dev_metrics["macro_f1"],
            "macro_roc_auc": dev_metrics["macro_roc_auc"],
        },
        "test": {
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "macro_roc_auc": test_metrics["macro_roc_auc"],
        },
        "label_mapping": LABEL2ID,
    }

    results_path = SORTIES_DIR / "taskB_finbert_results.json"
    print(f"✅ Saving metrics to: {results_path}")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
