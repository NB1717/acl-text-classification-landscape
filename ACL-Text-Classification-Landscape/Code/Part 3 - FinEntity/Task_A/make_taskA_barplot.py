import os
import matplotlib.pyplot as plt

# Model names
models = ["LogReg", "MLP", "FinBERT"]

# True values on the test set (
accuracy = [0.6761, 0.7113, 0.7465]
macro_f1 = [0.5747, 0.6608, 0.7227]

x = range(len(models))
width = 0.35  

fig, ax = plt.subplots()


ax.bar([i - width/2 for i in x], accuracy, width, label="Accuracy")


ax.bar([i + width/2 for i in x], macro_f1, width, label="Macro F1")

ax.set_xticks(list(x))
ax.set_xticklabels(models)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score")
ax.set_title("FinEntity Task A: Model Performance")
ax.legend()

plt.tight_layout()


out_dir = os.path.join("..", "figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "taskA_model_comparison.png")
plt.savefig(out_path, dpi=300)

print("Saved figure to:", out_path)
