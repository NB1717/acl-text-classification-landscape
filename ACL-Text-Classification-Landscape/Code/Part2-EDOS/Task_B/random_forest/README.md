# Random Forest Model for Multi-Class Classification

This script trains a Random Forest model for multi-class classification of sexist content into categories using the aggregated EDOS dataset.

## Task B: Multi-Class Classification

The model classifies text into 5 categories:
1. **none** - Not sexist content
2. **1. threats, plans to harm and incitement** - Threatening content or incitement to harm
3. **2. derogation** - Derogatory attacks on women
4. **3. animosity** - Expressions of animosity toward women
5. **4. prejudiced discussions** - Prejudiced discussions supporting discrimination

## Features

- **TF-IDF Vectorization**: Extracts features using TF-IDF with configurable n-grams
- **Ensemble Method**: Random Forest combines multiple decision trees for robust predictions
- **Class Balancing**: Uses balanced class weights to handle imbalanced data
- **Feature Importance**: Analyzes and visualizes which features are most important
- **Comprehensive Evaluation**: Reports macro and weighted metrics for multi-class classification
- **Visualizations**: Generates confusion matrices, F1 score plots, feature importance, and class distribution
- **Model Persistence**: Saves trained model, vectorizer, and label encoder

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the script from the random_forest directory:

```bash
python rf.py
```

The script will:
1. Load the aggregated EDOS dataset from `../../data/edos_labelled_aggregated.csv`
2. Map label categories to numeric labels
3. Split data into train/dev/test sets using the existing split column
4. Extract TF-IDF features from the text
5. Train a Random Forest classifier with balanced class weights
6. Evaluate on all splits and generate metrics
7. Create visualizations (confusion matrices, F1 plots, feature importance, class distribution)
8. Save the trained model, vectorizer, and label encoder to `models/` directory

## Configuration

You can modify these parameters in the `main()` function:

### Feature Extraction
- `MAX_FEATURES`: Maximum number of TF-IDF features (default: 5000)
- `NGRAM_RANGE`: N-gram range for feature extraction (default: (1, 2))

### Random Forest Parameters
- `N_ESTIMATORS`: Number of trees in the forest (default: 100)
  - More trees = better performance but slower training
  - Try values: 50, 100, 200, 500
- `MAX_DEPTH`: Maximum depth of trees (default: None = unlimited)
  - `None`: Unlimited depth (can overfit)
  - Integer: Limits depth (e.g., 10, 20, 30)
  - Lower values = less overfitting, simpler model
- `MIN_SAMPLES_SPLIT`: Minimum samples required to split a node (default: 2)
  - Higher values = more regularization (e.g., 5, 10)
- `MIN_SAMPLES_LEAF`: Minimum samples required at a leaf node (default: 1)
  - Higher values = more regularization (e.g., 2, 5)
- `CLASS_WEIGHT`: Class balancing (default: 'balanced')
  - 'balanced': Automatically adjusts weights inversely proportional to class frequency

## Output

The script generates:
- **Console output**: Classification reports and metrics for all splits
- **Plots**: Saved to `outputs/` directory:
  - Confusion matrices (train, dev, test sets) with counts and percentages
  - F1 score bar plots showing per-class performance across splits
  - Feature importance plot showing top 30 most important features
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

## Random Forest Advantages

Compared to SVM, Random Forest:
- **Less prone to overfitting**: Ensemble averaging reduces variance
- **Handles non-linearity**: Automatically captures complex patterns
- **Feature importance**: Provides interpretability
- **Robust to outliers**: Less sensitive than SVM
- **No kernel selection**: Doesn't require choosing kernel functions
- **Parallelizable**: Can train trees in parallel (n_jobs=-1)

## Tips for Better Performance

1. **Adjust Tree Depth**:
   - If overfitting: reduce `MAX_DEPTH` (try 20, 30)
   - If underfitting: increase or set to `None`

2. **Increase Number of Trees**:
   - More trees generally improve performance (diminishing returns after ~200)
   - Trade-off: more trees = longer training time

3. **Regularization**:
   - Increase `MIN_SAMPLES_SPLIT` (e.g., 5) to reduce overfitting
   - Increase `MIN_SAMPLES_LEAF` (e.g., 2) to reduce overfitting

4. **Feature Engineering**:
   - Experiment with `MAX_FEATURES` (try 3000, 5000, 10000)
   - Try different n-gram ranges: (1,1) for unigrams, (1,3) for trigrams

5. **Handle Class Imbalance**:
   - Keep `CLASS_WEIGHT='balanced'` for imbalanced datasets
   - Monitor per-class F1 scores to identify which classes need attention

## Loading Saved Model

To load the trained model for inference:

```python
import pickle

# Load model components
with open('models/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/tfidf_vectorizer_rf.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('models/label_encoder_rf.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Use for predictions
text = "Your text here"
X = vectorizer.transform([text]).toarray()  # Convert to dense for RF
prediction = model.predict(X)
class_name = label_encoder.inverse_transform(prediction)[0]
```

## Memory Considerations

Random Forest converts sparse TF-IDF matrices to dense arrays:
- **Memory usage**: Higher than SVM (stores full feature matrix)
- **Training time**: Can be slow with many trees and features
- **Recommendation**: If memory issues, reduce `MAX_FEATURES` or `N_ESTIMATORS`

## Comparison with SVM

**Random Forest advantages:**
- Better generalization (less overfitting)
- Feature importance insights
- Handles class imbalance well
- No kernel selection needed

**SVM advantages:**
- Works with sparse matrices (memory efficient)
- Can be faster for very large feature sets
- RBF kernel can capture complex non-linear patterns

## Example Configurations

### Fast Training (Less Overfitting)
```python
N_ESTIMATORS = 50
MAX_DEPTH = 20
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 2
```

### High Performance (More Trees)
```python
N_ESTIMATORS = 200
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
```

### Balanced (Recommended)
```python
N_ESTIMATORS = 100
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2
```

