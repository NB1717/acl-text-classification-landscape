# ACL Text Classification Landscape

This repository contains a three-part project that maps the landscape of text classification research in the ACL Anthology and benchmarks modern models on two specialized datasets: **EDOS 2023** (online sexism detection) and **FinEntity 2023** (financial entity-level sentiment analysis).

---

## 📌 Part 1 — Cartography of ACL Text Classification Papers

This section provides a large-scale analysis of **12,814** classification papers from the ACL Anthology.  
Structured metadata (title, authors, year) was extracted directly, and semantic fields such as **task type, domain, benchmark status, and evaluation metrics** were automatically inferred.

### Key Findings
- A sharp growth in classification research after **2015**, coinciding with neural and transformer models.
- Post-2018 growth exceeds **1,500 papers per year**.
- Benchmark introductions steadily increase after 2015, with a notable peak between **2019–2023** (e.g., GLUE, SuperGLUE, EDOS, FinEntity).
- **Task type distributions** across all papers:
  - **Multi-class:** 45.8%  
  - **Binary:** 29.7%  
  - **Multi-label:** 24.5%
- Frequent domains: sentiment analysis, NER, stance detection, toxic content detection.
- Multi-label research remains comparatively under-explored.

---

## 📌 Part 2 — EDOS 2023 (Explainable Detection of Online Sexism)

### **Task A — Binary Sexism Detection**
Models evaluated:
- Logistic Regression
- MLP
- DeBERTa-v3-base

**Results**
| Model              | Accuracy | F1-Macro |
|--------------------|----------|----------|
| Logistic Regression| 0.80     | 0.75     |
| MLP                | 0.81     | 0.71     |
| DeBERTa-v3-base    | 0.88     | 0.84     |

**Summary**
DeBERTa-v3-base achieves the highest performance.  
Classical models show strong bias toward the “Not Sexist” class and struggle with subtle sexist expressions.

---

### **Task B — Multi-Class Sexism Classification**
Models evaluated:
- Random Forest
- SVM
- TextCNN
- HateBERT

**Results**
| Model        | F1-Macro |
|--------------|----------|
| Random Forest| 0.3025    |
| SVM          | 0.3460  |
| TextCNN      | 0.3145    |
| HateBERT     | 0.5480     |

**Summary**
HateBERT delivers the most balanced performance across classes.  
Classical models (RF, SVM, TextCNN) consistently overpredict the dominant “none” class.

---

## 📌 Part 3 — FinEntity 2023 (Financial Entity Sentiment)

The FinEntity benchmark contains **979** sentences and **2,131** entity mentions.

### **Task A — Sentence-Level Sentiment Classification**
Models:
- Logistic Regression (TF-IDF)
- MLP (TF-IDF)
- FinBERT (fine-tuned)

**Results**
| Model                    | Accuracy | F1-Macro |
|--------------------------|----------|----------|
| Logistic Regression      | 0.68     | 0.57     |
| MLP                      | 0.71     | 0.66     |
| FinBERT (fine-tuned)     | 0.75     | 0.72     |

**Summary**
FinBERT provides the best overall performance and improves minority-class detection using class-weighted loss and early stopping.

---

### **Task B — Entity-Level Sentiment Classification**
Models:
- Logistic Regression (entity-marked + TF-IDF)
- FinBERT (entity-aware fine-tuned)
- GPT-4o (zero-shot)

**Results**
| Model               | Accuracy | F1-Macro |
|---------------------|----------|----------|
| Logistic Regression | 0.77     | 0.73     |
| FinBERT             | 0.85     | 0.84     |
| GPT-4o (zero-shot)  | 0.87     | 0.82     |

**Summary**
FinBERT significantly outperforms classical models by handling contextual cues and polarity distinctions.  
GPT-4o shows strong zero-shot performance, capturing sentiment distinctions without fine-tuning.

---

## ✔️ Conclusion

Across all tasks:
- Domain-specific transformers (**DeBERTa-v3-base**, **FinBERT**) provide the highest performance.
- Classical models serve as basic baselines but struggle with minority classes and nuanced language.
- The large-scale cartography highlights a rapidly growing trend in classification research and a need for more balanced, domain-aware benchmarks.

---

## 🔧 Running the Code

Each module includes its own training script and `requirements.txt`.

Example:
```bash
pip install -r requirements.txt
python train_model.py

