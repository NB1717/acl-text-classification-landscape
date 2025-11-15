# ACL Text Classification Landscape

This repository contains a three-part project analyzing the landscape of text classification research in the ACL Anthology and benchmarking modern models on two public datasets: **EDOS 2023** (online sexism detection) and **FinEntity 2023** (financial sentiment classification).

---

## 📌 Project Structure

- **Part 1 – Cartography**  
  Analysis of ACL text classification papers with **<200 citations**, including task type, domain, evaluation metrics, and venue trends. Scripts generate plots and tables from `all_classification_refined.csv`.

- **Part 2 – EDOS**  
  Benchmark on online sexism detection:  
  - **Task A:** Binary sexism detection — Models: Logistic Regression, MLP, DeBERTa  
  - **Task B:** Multi-class sexism categorization — Models: SVM, Random Forest, TextCNN, HateBERT  

- **Part 3 – FinEntity**  
  Benchmark on financial sentiment classification:  
  - **Task A:** Sentence-level sentiment — Models: Logistic Regression, MLP, FinBERT  
  - **Task B:** Entity-level sentiment — Models: Logistic Regression (entity-marked), FinBERT  
  Preprocessing scripts (splits, conversion, sampling) are found in the `data/` directory.

---

## 📚 Datasets

- ACL Anthology metadata (public)  
- EDOS 2023 — Explainable Detection of Online Sexism  
- FinEntity 2023 — Financial entity-level sentiment  

All datasets used are publicly available under their original licenses.

---

## ✔️ Summary

This project provides:
- a focused analysis of low-citation ACL text classification papers,
- reproducible baselines for EDOS and FinEntity,
- comparisons between classical ML and transformer-based models.

Useful for research, benchmarking experiments, and academic NLP work.

---

## 🔧 Running the Code

Each module includes its own training scripts and `requirements.txt`.

Example:

```bash
pip install -r requirements.txt
python train_model.py
```
Evaluation reports and plots are generated inside each module’s directory.
