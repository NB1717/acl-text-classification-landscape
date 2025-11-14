import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score
)
import joblib

# ---------- Setting paths ----------
DATA_DIR = "../data/FinEntity/processed"
OUTPUT_MODEL_DIR = "../models"
OUTPUT_RESULTS_DIR = "../sorties"

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)


def load_split(split_name: str) -> pd.DataFrame:
    """
    Load one split of Task B dataset: train / dev / test.
    """
    path = os.path.join(DATA_DIR, f"finentity_taskB_{split_name}.csv")
    print(f"Loading {split_name} split from: {path}")
    df = pd.read_csv(path)
    # Ensuring there are no NaN values in the main columns
    df = df.dropna(subset=["sentence", "entity_text", "label"])
    print(f"{split_name.capitalize()} size after dropna: {len(df)}")
    return df


def build_input_text(df: pd.DataFrame) -> pd.Series:
    """
    For each sample, combines the sentence and entity in an entity-aware manner.
    Example:

    '... sentence ... [ENTITY] Apple [/ENTITY]'
    """
    return df.apply(
        lambda r: f"{r['sentence']} [ENTITY] {r['entity_text']} [/ENTITY]",
        axis=1
    )


def evaluate_split(model, X, y_true, split_name: str):
    """
    Calculating Accuracy, Macro-F1, classification report, and AUC

    """
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    print(f"\n=== {split_name.upper()} ===")
    print(f"Accuracy: {acc:.4f} | F1-macro: {f1_macro:.4f}")
    print(classification_report(y_true, y_pred))

    # Multiclass AUC (only if all classes are present)
    auc_macro = None
    try:
        y_proba = model.predict_proba(X)
        auc_macro = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        print(f"Macro ROC-AUC: {auc_macro:.4f}")
    except Exception as e:
        print(f"Could not compute ROC-AUC for {split_name}: {e}")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "roc_auc_macro": auc_macro
    }


if __name__ == "__main__":
    # ---------- 1) Loading data ----------
    train_df = load_split("train")
    dev_df = load_split("dev")
    test_df = load_split("test")

    # ---------- 2) Building entity-aware input ----------
    X_train_text = build_input_text(train_df)
    X_dev_text = build_input_text(dev_df)
    X_test_text = build_input_text(test_df)

    y_train = train_df["label"]
    y_dev = dev_df["label"]
    y_test = test_df["label"]

    # ---------- 3) TF–IDF vector ----------
    print("\nBuilding TF–IDF representations...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode",
        min_df=2,
        max_df=0.95
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_dev = vectorizer.transform(X_dev_text)
    X_test = vectorizer.transform(X_test_text)

    print("TF–IDF shapes:", X_train.shape, X_dev.shape, X_test.shape)

    # ---------- 4) Training Logistic Regression ----------
    print("\nTraining Logistic Regression (Task B, entity-level)...")
    logreg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   # To handle class imbalance
        solver="liblinear",        # Stable for small multiclass datasets
        multi_class="ovr"
    )
    logreg.fit(X_train, y_train)

    # ---------- 5) Evaluation on the three splits ----------
    results = {}
    results["train"] = evaluate_split(logreg, X_train, y_train, "train")
    results["dev"] = evaluate_split(logreg, X_dev, y_dev, "dev")
    results["test"] = evaluate_split(logreg, X_test, y_test, "test")

    # ---------- 6) Saving the model and TF–IDF ----------
    model_path = os.path.join(OUTPUT_MODEL_DIR, "logreg_taskB.joblib")
    vec_path = os.path.join(OUTPUT_MODEL_DIR, "tfidf_taskB.joblib")
    joblib.dump(logreg, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"\n✅ Saved model to: {model_path}")
    print(f"✅ Saved TF–IDF vectorizer to: {vec_path}")

    # ---------- 7) Saving results as JSON ----------
    results_path = os.path.join(OUTPUT_RESULTS_DIR, "taskB_logreg_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Saved metrics to: {results_path}")
