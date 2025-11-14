# TextCNN Model for Multi-Class Classification

This script trains a TextCNN (Convolutional Neural Network for text) model for multi-class classification of sexist content into categories using the aggregated EDOS dataset.

## Task B: Multi-Class Classification

The model classifies text into 5 categories:
1. **none** - Not sexist content
2. **1. threats, plans to harm and incitement** - Threatening content or incitement to harm
3. **2. derogation** - Derogatory attacks on women
4. **3. animosity** - Expressions of animosity toward women
5. **4. prejudiced discussions** - Prejudiced discussions supporting discrimination

## Features

- **CNN Architecture**: Uses convolutional filters to capture n-gram patterns (3, 4, 5-grams)
- **Embedding Layer**: Learns word embeddings during training
- **Multi-Filter Design**: Multiple filter sizes capture different patterns simultaneously
- **Comprehensive Evaluation**: Reports macro and weighted metrics for multi-class classification
- **Early Stopping**: Prevents overfitting with validation-based early stopping
- **Visualizations**: Generates confusion matrices, F1 plots, training history, and class distribution
- **Model Persistence**: Saves trained model, vocabulary, and label encoder

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

**Note**: For GPU support, ensure you have the appropriate PyTorch version for your CUDA setup. Visit [PyTorch website](https://pytorch.org/) for installation instructions.

## Usage

Run the script from the textcnn directory:

```bash
python textcnn.py
```

The script will:
1. Load the aggregated EDOS dataset from `../../data/edos_labelled_aggregated.csv`
2. Map label categories to numeric labels
3. Build vocabulary from training data
4. Split data into train/dev/test sets using the existing split column
5. Train TextCNN model with early stopping
6. Evaluate on all splits and generate metrics
7. Create visualizations (confusion matrices, F1 plots, training history)
8. Save the trained model, vocabulary, and label encoder to `models/` directory

## Model Architecture

TextCNN consists of:
1. **Embedding Layer**: Maps token indices to dense vectors (embed_dim=128)
2. **Convolutional Layers**: Multiple 1D convolutions with filter sizes [3, 4, 5]
   - Each filter size captures different n-gram patterns
   - 100 filters per size
3. **Max Pooling**: Extracts most important features from each filter
4. **Concatenation**: Combines features from all filter sizes
5. **Dropout**: Regularization (dropout=0.5)
6. **Fully Connected Layer**: Final classification layer

## Configuration

You can modify these parameters in the `main()` function:

### Model Architecture
- `MAX_LENGTH`: Maximum sequence length (default: 256)
- `VOCAB_SIZE`: Maximum vocabulary size (default: 10000)
- `MIN_FREQ`: Minimum token frequency to include in vocab (default: 2)
- `EMBED_DIM`: Embedding dimension (default: 128)
- `NUM_FILTERS`: Number of filters per filter size (default: 100)
- `FILTER_SIZES`: List of filter sizes [3, 4, 5] for n-grams
- `DROPOUT`: Dropout rate (default: 0.5)

### Training Parameters
- `BATCH_SIZE`: Batch size (default: 32)
- `NUM_EPOCHS`: Maximum number of epochs (default: 10)
- `LEARNING_RATE`: Learning rate (default: 0.001)
- `WEIGHT_DECAY`: L2 regularization (default: 0.0001)

## Output

The script generates:
- **Console output**: Training progress and classification reports for all splits
- **Plots**: Saved to `outputs/` directory:
  - Confusion matrices (train, dev, test sets) with counts and percentages
  - F1 score bar plots showing per-class performance across splits
  - Training history (loss and accuracy curves)
  - Class distribution plots
- **Models**: Saved model, vocabulary, and label encoder in `models/` directory

## Model Performance Metrics

The model reports:
- **Accuracy**: Overall classification accuracy
- **Macro F1**: Unweighted mean of per-class F1 scores (treats all classes equally)
- **Weighted F1**: Weighted mean of per-class F1 scores (accounts for class imbalance)
- **Macro Precision/Recall**: Unweighted mean across classes
- **Weighted Precision/Recall**: Weighted mean across classes
- **Per-class F1**: Individual F1 score for each category

## TextCNN Advantages

- **Captures Local Patterns**: Convolutional filters detect important word sequences
- **Efficient**: Faster than RNNs, can be parallelized well
- **Multi-scale Features**: Different filter sizes capture 3, 4, 5-gram patterns simultaneously
- **Less Overfitting**: Dropout and early stopping help generalize

## Tips for Better Performance

1. **Adjust Filter Sizes**:
   ```python
   FILTER_SIZES = [2, 3, 4, 5]  # Add 2-grams for shorter patterns
   ```

2. **Increase Model Capacity**:
   - Increase `NUM_FILTERS` (try 150, 200)
   - Increase `EMBED_DIM` (try 256)

3. **Regularization**:
   - Increase `DROPOUT` if overfitting (try 0.6, 0.7)
   - Increase `WEIGHT_DECAY` (try 0.001)

4. **Sequence Length**:
   - Increase `MAX_LENGTH` if texts are long (if memory allows)
   - Decrease if memory constrained

5. **Vocabulary**:
   - Increase `VOCAB_SIZE` to capture more words (try 20000)
   - Adjust `MIN_FREQ` to filter rare tokens

## Loading Saved Model

To load the trained model for inference:

```python
import torch
import pickle

from textcnn import TextCNN  # Import the model class

# Model configuration (must match training config)
vocab_size = 10000
embed_dim = 128
num_filters = 100
filter_sizes = [3, 4, 5]
num_classes = 5
dropout = 0.5

# Load vocabulary and encoder
with open('models/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

with open('models/label_encoder_textcnn.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Create and load model
model = TextCNN(vocab_size, embed_dim, num_filters, filter_sizes, num_classes, dropout)
model.load_state_dict(torch.load('models/textcnn_model.pth'))
model.eval()

# Use for predictions
text = "Your text here"
# Tokenize and convert to indices (same as in TextDataset)
# ... preprocessing code ...
# prediction = model(text_tensor)
```

## Memory Considerations

- TextCNN processes sequences and can be memory intensive
- Reduce `BATCH_SIZE` if you encounter out-of-memory errors (try 16, 8)
- Reduce `MAX_LENGTH` if needed (try 128)
- GPU recommended for faster training

## Comparison with Other Models

**TextCNN advantages:**
- Captures local n-gram patterns effectively
- Faster training than RNNs/LSTMs
- Good for short to medium length texts
- Less prone to overfitting than deep RNNs

**TextCNN limitations:**
- Doesn't capture long-range dependencies as well as RNNs/Transformers
- Fixed filter sizes may not capture all relevant patterns
- Less interpretable than simpler models

## Expected Performance

TextCNN typically:
- Performs better than TF-IDF + Linear models for complex patterns
- May outperform SVM/RF for certain text classification tasks
- Generally better than simple MLPs
- Usually worse than large transformers (DeBERTa, BERT) but much faster

## GPU Support

The model automatically uses GPU if available. To force CPU:

```python
device = torch.device('cpu')
```

Or set in code before training starts.

