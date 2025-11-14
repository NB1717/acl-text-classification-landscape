# Logistic Regression Model for EDOS Binary Classification

This script trains a logistic regression model for binary classification of sexist vs. not sexist text using the aggregated EDOS dataset.

## Features

- **TF-IDF Vectorization**: Extracts features using TF-IDF with configurable n-grams
- **Class Balancing**: Uses balanced class weights to handle imbalanced data
- **Comprehensive Evaluation**: Reports accuracy, precision, recall, F1-score, and AUC-ROC
- **Visualizations**: Generates ROC curves and confusion matrices
- **Feature Analysis**: Identifies top features for both classes
- **Model Persistence**: Saves trained model and vectorizer for future use

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the script from the logistic regression directory:

```bash
python lr.py
```

The script will:
1. Load the aggregated EDOS dataset from `../data/edos_labelled_aggregated.csv`
2. Split data into train/dev/test sets using the existing split column
3. Extract TF-IDF features from the text
4. Train a logistic regression model with balanced class weights
5. Evaluate on all splits and generate metrics
6. Create visualizations (ROC curves and confusion matrices)
7. Display top features for each class
8. Save the trained model and vectorizer to `models/` directory

## Output

The script generates:
- **Console output**: Classification reports and metrics for all splits
- **Plots**: ROC curves and confusion matrices saved to `outputs/` directory
- **Models**: Saved model and vectorizer in `models/` directory

## Configuration

You can modify these parameters in the `main()` function:

- `MAX_FEATURES`: Maximum number of TF-IDF features (default: 5000)
- `NGRAM_RANGE`: N-gram range for feature extraction (default: (1, 2))
- `C`: Regularization strength (default: 1.0)
- `MAX_ITER`: Maximum iterations for model training (default: 1000)

## Model Performance

The model will display performance metrics on:
- **Train set**: Training data performance
- **Dev set**: Development/validation set performance
- **Test set**: Test set performance (final evaluation)

Metrics include:
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC (Area Under the ROC Curve)

