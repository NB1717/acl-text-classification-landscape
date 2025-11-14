# Support Vector Machine (SVM) Model for Multi-Class Classification

This script trains an SVM model for multi-class classification of sexist content into categories using the aggregated EDOS dataset.

## Task B: Multi-Class Classification

The model classifies text into 5 categories:
1. **none** - Not sexist content
2. **1. threats, plans to harm and incitement** - Threatening content or incitement to harm
3. **2. derogation** - Derogatory attacks on women
4. **3. animosity** - Expressions of animosity toward women
5. **4. prejudiced discussions** - Prejudiced discussions supporting discrimination

## Features

- **TF-IDF Vectorization**: Extracts features using TF-IDF with configurable n-grams
- **Multi-Class SVM**: Uses Support Vector Machine with RBF kernel (supports multi-class natively)
- **Class Balancing**: Uses balanced class weights to handle imbalanced data
- **Comprehensive Evaluation**: Reports macro and weighted metrics for multi-class classification
- **Visualizations**: Generates confusion matrices and F1 score plots
- **Model Persistence**: Saves trained model, vectorizer, and label encoder

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the script from the svm directory:

```bash
python svm.py
```

The script will:
1. Load the aggregated EDOS dataset from `../../data/edos_labelled_aggregated.csv`
2. Map label categories to numeric labels
3. Split data into train/dev/test sets using the existing split column
4. Extract TF-IDF features from the text
5. Train an SVM classifier with balanced class weights
6. Evaluate on all splits and generate metrics
7. Create visualizations (confusion matrices, F1 plots, class distribution)
8. Save the trained model, vectorizer, and label encoder to `models/` directory

## Configuration

You can modify these parameters in the `main()` function:

### Feature Extraction
- `MAX_FEATURES`: Maximum number of TF-IDF features (default: 5000)
- `NGRAM_RANGE`: N-gram range for feature extraction (default: (1, 2))

### SVM Parameters
- `C`: Regularization parameter (default: 1.0)
  - Higher values = less regularization, more complex decision boundary
  - Lower values = more regularization, simpler decision boundary
- `KERNEL`: Kernel type (default: 'rbf')
  - 'linear': Linear kernel (faster, good for high-dimensional data)
  - 'rbf': Radial Basis Function (default, good for non-linear patterns)
  - 'poly': Polynomial kernel
  - 'sigmoid': Sigmoid kernel
- `GAMMA`: Kernel coefficient (default: 'scale')
  - 'scale': Uses 1 / (n_features * X.var())
  - 'auto': Uses 1 / n_features
  - float: Custom gamma value
- `CLASS_WEIGHT`: Class balancing (default: 'balanced')
  - 'balanced': Automatically adjusts weights inversely proportional to class frequency

## Output

The script generates:
- **Console output**: Classification reports and metrics for all splits
- **Plots**: Saved to `outputs/` directory:
  - Confusion matrices (train, dev, test sets) with counts and percentages
  - F1 score bar plots showing per-class performance across splits
  - Class distribution plots showing label distribution in each split
- **Models**: Saved model, vectorizer, and label encoder in `models/` directory

## Model Performance Metrics

The model reports:
- **Accuracy**: Overall classification accuracy
- **Macro F1**: Unweighted mean of per-class F1 scores (treats all classes equally)
- **Weighted F1**: Weighted mean of per-class F1 scores (accounts for class imbalance)
- **Macro Precision/Recall**: Unweighted mean across classes
- **Weighted Precision/Recall**: Weighted mean across classes
- **Per-class F1**: Individual F1 score for each category

## Multi-Class Classification Details

SVM uses One-vs-One (OVO) strategy for multi-class classification:
- Creates binary classifiers for each pair of classes
- Makes predictions based on majority voting
- Works well for imbalanced datasets when using class weights

## Tips for Better Performance

1. **Adjust Regularization**: 
   - If overfitting: decrease `C` (try 0.1 or 0.5)
   - If underfitting: increase `C` (try 5.0 or 10.0)

2. **Try Different Kernels**:
   - Linear kernel is faster and works well for high-dimensional sparse data
   - RBF kernel (default) captures non-linear patterns better

3. **Feature Engineering**:
   - Experiment with `MAX_FEATURES` (try 3000, 5000, 10000)
   - Try different n-gram ranges: (1,1) for unigrams, (1,3) for trigrams

4. **Handle Class Imbalance**:
   - Keep `CLASS_WEIGHT='balanced'` for imbalanced datasets
   - Monitor per-class F1 scores to identify which classes need more attention

## Loading Saved Model

To load the trained model for inference:

```python
import pickle

# Load model components
with open('models/svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/tfidf_vectorizer_svm.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Use for predictions
text = "Your text here"
X = vectorizer.transform([text])
prediction = model.predict(X)
class_name = label_encoder.inverse_transform(prediction)[0]
```

## Comparison with Binary Classification

- **Binary (Task A)**: Classifies as "sexist" vs "not sexist"
- **Multi-Class (Task B)**: Classifies into specific categories of sexism
- Multi-class is more challenging due to:
  - More classes (5 vs 2)
  - Imbalanced class distribution
  - Need for finer-grained discrimination

