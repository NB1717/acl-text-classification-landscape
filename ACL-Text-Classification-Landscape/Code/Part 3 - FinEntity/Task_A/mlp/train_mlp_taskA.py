import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# === Data paths ===
base_dir = os.path.join("..", "data", "FinEntity", "processed")
train_path = os.path.join(base_dir, "finentity_taskA_train.csv")
dev_path = os.path.join(base_dir, "finentity_taskA_dev.csv")
test_path = os.path.join(base_dir, "finentity_taskA_test.csv")

print("Loading data...")
train_df = pd.read_csv(train_path)
dev_df = pd.read_csv(dev_path)
test_df = pd.read_csv(test_path)

# === Using the sentence text for Task A ===
X_train_text = train_df["sentence"]
X_dev_text = dev_df["sentence"]
X_test_text = test_df["sentence"]

y_train = train_df["sentence_label"]
y_dev = dev_df["sentence_label"]
y_test = test_df["sentence_label"]

# === Converting text to TF-IDF (like Logistic Regression for fair comparison) ===
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_text)
X_dev = vectorizer.transform(X_dev_text)
X_test = vectorizer.transform(X_test_text)

print("TF-IDF shapes:", X_train.shape, X_dev.shape, X_test.shape)

# === Defining the MLP model (Multilayer Perceptron) ===
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    batch_size=64,
    max_iter=20,
    random_state=42,
    verbose=True
)

print("\nTraining MLP...")
mlp.fit(X_train, y_train)

# === Helper function for evaluation ===
def evaluate(split_name, X, y_true):
    y_pred = mlp.predict(X)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n=== {split_name} ===")
    print(f"Accuracy: {acc:.4f} | F1-macro: {f1:.4f}")
    return y_pred

# === Evaluation on Train / Dev / Test ===
y_train_pred = evaluate("TRAIN", X_train, y_train)
y_dev_pred = evaluate("DEV", X_dev, y_dev)
y_test_pred = evaluate("TEST", X_test, y_test)

print("\nDetailed classification report on TEST set:")
print(classification_report(y_test, y_test_pred))
