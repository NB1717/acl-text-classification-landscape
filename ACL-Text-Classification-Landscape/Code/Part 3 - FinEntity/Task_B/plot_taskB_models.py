import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Path to the outputs directory
BASE_DIR = os.path.join("..", "sorties")

# ---------- 1) Reading JSON files ----------
logreg_path = os.path.join(BASE_DIR, "taskB_logreg_results.json")
finbert_path = os.path.join(BASE_DIR, "taskB_finbert_results.json")

with open(logreg_path, "r", encoding="utf-8") as f:
    logreg_metrics = json.load(f)

with open(finbert_path, "r", encoding="utf-8") as f:
    finbert_metrics = json.load(f)

# ---------- 2) Extracting numbers from the test split ----------
logreg_acc = logreg_metrics["test"]["accuracy"]
# Support for two different F1 key names
logreg_f1 = logreg_metrics["test"].get("macro_f1", logreg_metrics["test"].get("f1_macro"))

finbert_acc = finbert_metrics["test"]["accuracy"]
finbert_f1  = finbert_metrics["test"]["macro_f1"]

print("LogReg - acc:", logreg_acc, "| macro F1:", logreg_f1)
print("FinBERT - acc:", finbert_acc, "| macro F1:", finbert_f1)

# ---------- 3) Building chart data ----------
models   = ["LogReg (TF–IDF)", "FinBERT"]
accuracy = [logreg_acc, finbert_acc]
macro_f1 = [logreg_f1, finbert_f1]

x = np.arange(len(models))
width = 0.35

# ---------- 4) Plotting the chart ----------
fig, ax = plt.subplots(figsize=(6, 4))

bars1 = ax.bar(x - width/2, accuracy, width, label="Accuracy", color="#7EA6E0")
bars2 = ax.bar(x + width/2, macro_f1, width, label="Macro F1", color="#F6B26B")

ax.set_ylim(0.6, 1.0)
ax.set_ylabel("Score")
ax.set_title("FinEntity Task B – Entity-level sentiment (test set)")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=10)
ax.legend(loc="lower right")

def annotate_bars(bars):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 0.01,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

annotate_bars(bars1)
annotate_bars(bars2)

plt.tight_layout()

# ---------- 5) Saving the figure ----------
fig_dir = os.path.join("..", "figures")
os.makedirs(fig_dir, exist_ok=True)

out_path = os.path.join(fig_dir, "figure_taskB_model_comparison.png")
plt.savefig(out_path, dpi=300)
print(f"✅ Figure saved to: {out_path}")
