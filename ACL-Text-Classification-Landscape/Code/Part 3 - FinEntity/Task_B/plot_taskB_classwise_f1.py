import numpy as np
import matplotlib.pyplot as plt

# Classes (entity-level sentiments)
classes = ["Negative", "Neutral", "Positive"]

# Class-by-class F1-score on the test set (from terminal output)
logreg_f1  = [0.67, 0.84, 0.69]  # Logistic Regression (TF–IDF, entity-aware)
finbert_f1 = [0.80, 0.89, 0.82]  # FinBERT (entity-aware)

x = np.arange(len(classes))
width = 0.35  

fig, ax = plt.subplots(figsize=(6.5, 4))

# Columns

rects1 = ax.bar(x - width/2, logreg_f1, width, label="LogReg (TF–IDF)")
rects2 = ax.bar(x + width/2, finbert_f1, width, label="FinBERT")

# Labels and visual appearance settings
ax.set_ylabel("F1-score")
ax.set_xlabel("Sentiment class")
ax.set_title("FinEntity Task B – Class-wise F1 by model (test set)")
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0.5, 1.0)  # To make the differences more visible
ax.legend(loc="lower right")

# Writing the F1 value on each bar
def autolabel(rects):
    for r in rects:
        height = r.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(r.get_x() + r.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()


plt.savefig("../figures/figure_taskB_classwise_f1.png", dpi=300)


plt.close()