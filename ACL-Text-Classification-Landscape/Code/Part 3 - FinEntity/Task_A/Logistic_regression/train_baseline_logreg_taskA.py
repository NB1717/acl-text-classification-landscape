import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

# === Data paths ===
base_dir = os.path.join("..", "data", "FinEntity", "processed")
train_path = os.path.join(base_dir, "finentity_taskA_train.csv")
dev_path = os.path.join(base_dir, "finentity_taskA_dev.csv")
test_path = os.path.join(base_dir, "finentity_taskA_test.csv")

print("Loading data...")
train_df = pd.read_csv(train_path)
dev_df = pd.read_csv(dev_path)
test_df = pd.read_csv(test_path)

# === Converting text to TF-IDF vectors ===
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train = vectorizer.fit_transform(train_df["sentence"])
X_dev = vectorizer.transform(dev_df["sentence"])
X_test = vectorizer.transform(test_df["sentence"])

y_train = train_df["sentence_label"]
y_dev = dev_df["sentence_label"]
y_test = test_df["sentence_label"]

# === Logistic Regression model ===
model = LogisticRegression(max_iter=1000, solver="lbfgs")
model.fit(X_train, y_train)

# === Evaluation ===
for split, X, y in [("TRAIN", X_train, y_train), ("DEV", X_dev, y_dev), ("TEST", X_test, y_test)]:
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")
    print(f"\n=== {split} ===")
    print(f"Accuracy: {acc:.4f} | F1-macro: {f1:.4f}")

print("\nDetailed classification report on TEST set:")
print(classification_report(y_test, model.predict(X_test)))
