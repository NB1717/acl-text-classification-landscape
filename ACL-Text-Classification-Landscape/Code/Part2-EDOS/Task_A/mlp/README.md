# Multi-Layer Perceptron (MLP) Model for EDOS Binary Classification

This script trains a Multi-Layer Perceptron (neural network) for binary classification of sexist vs. not sexist text using the aggregated EDOS dataset.

## Features

- **TF-IDF Vectorization**: Extracts features using TF-IDF with configurable n-grams
- **Deep Neural Network**: Multi-layer perceptron with configurable architecture
- **Early Stopping**: Prevents overfitting with validation-based early stopping
- **Comprehensive Evaluation**: Reports accuracy, precision, recall, F1-score, and AUC-ROC
- **Visualizations**: Generates ROC curves, confusion matrices, F1 box plots, and training loss curves
- **Model Persistence**: Saves trained model and vectorizer for future use

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the script from the MLP directory:

```bash
python mlp.py
```

The script will:
1. Load the aggregated EDOS dataset from `../data/edos_labelled_aggregated.csv`
2. Split data into train/dev/test sets using the existing split column
3. Extract TF-IDF features from the text
4. Train an MLP classifier with the specified architecture
5. Evaluate on all splits and generate metrics
6. Create visualizations (ROC curves, confusion matrices, F1 plots, loss curves)
7. Save the trained model and vectorizer to `models/` directory

## Configuration

You can modify these parameters in the `main()` function:

### Feature Extraction
- `MAX_FEATURES`: Maximum number of TF-IDF features (default: 5000)
- `NGRAM_RANGE`: N-gram range for feature extraction (default: (1, 2))

### Network Architecture
- `HIDDEN_LAYER_SIZES`: Tuple specifying number of neurons in each hidden layer (default: (128, 64))
  - Example: `(100,)` for single hidden layer with 100 neurons
  - Example: `(256, 128, 64)` for three hidden layers
- `ACTIVATION`: Activation function - 'identity', 'logistic', 'tanh', 'relu' (default: 'relu')
- `SOLVER`: Optimization algorithm - 'lbfgs', 'sgd', 'adam' (default: 'adam')
  - 'adam' is recommended for large datasets
  - 'lbfgs' is faster for small datasets but uses more memory

### Training Parameters
- `ALPHA`: L2 regularization parameter (default: 0.0001)
- `LEARNING_RATE_INIT`: Initial learning rate (default: 0.001)
- `MAX_ITER`: Maximum iterations for training (default: 500)
- `EARLY_STOPPING`: Enable early stopping to prevent overfitting (default: True)
- `VALIDATION_FRACTION`: Fraction of training data to use for validation (default: 0.1)
- `N_ITER_NO_CHANGE`: Number of iterations with no improvement before stopping (default: 10)

## Architecture Examples

### Simple Network
```python
HIDDEN_LAYER_SIZES = (100,)
ACTIVATION = 'relu'
SOLVER = 'adam'
```

### Deep Network
```python
HIDDEN_LAYER_SIZES = (256, 128, 64)
ACTIVATION = 'relu'
SOLVER = 'adam'
ALPHA = 0.0001  # Regularization to prevent overfitting
```

### Wide Network
```python
HIDDEN_LAYER_SIZES = (512,)
ACTIVATION = 'tanh'
SOLVER = 'lbfgs'  # Good for smaller datasets
```

## Output

The script generates:
- **Console output**: Classification reports and metrics for all splits
- **Plots**: Saved to `outputs/` directory:
  - ROC curves (dev and test sets)
  - Confusion matrices (train, dev, test sets)
  - F1 box plots and bar plots
  - Training loss curve (if early stopping is enabled)
- **Models**: Saved model and vectorizer in `models/` directory

## Model Performance

The model will display performance metrics on:
- **Train set**: Training data performance
- **Dev set**: Development/validation set performance
- **Test set**: Test set performance (final evaluation)

Metrics include:
- Accuracy
- Precision
- Recall
- F1-Score (binary and macro)
- Per-class F1 scores
- AUC-ROC (Area Under the ROC Curve)

## Memory Considerations

- The script converts sparse TF-IDF matrices to dense arrays for the MLP
- For very large datasets, you may need to reduce `MAX_FEATURES` or use a smaller network
- If memory issues occur, consider using `solver='lbfgs'` or reducing hidden layer sizes

## Tips for Better Performance

1. **Tune Architecture**: Experiment with different hidden layer sizes
2. **Regularization**: Adjust `ALPHA` to control overfitting
3. **Early Stopping**: Keep enabled to prevent overfitting
4. **Learning Rate**: Adjust `LEARNING_RATE_INIT` if training is unstable
5. **Feature Engineering**: Experiment with different `MAX_FEATURES` and `NGRAM_RANGE`

## Comparison with Logistic Regression

The MLP should generally perform better than logistic regression on complex patterns, but:
- Takes longer to train
- Requires more memory
- More hyperparameters to tune
- May overfit more easily (use early stopping and regularization)

