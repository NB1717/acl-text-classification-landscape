# DeBERTa-v3-large Model for EDOS Binary Classification

This script trains a DeBERTa-v3-large transformer model for binary classification of sexist vs. not sexist text using the aggregated EDOS dataset.

## Features

- **State-of-the-art Transformer**: Uses microsoft/deberta-v3-large (184M parameters)
- **Comprehensive Evaluation**: Reports accuracy, precision, recall, F1-score, and AUC-ROC
- **Early Stopping**: Prevents overfitting with validation-based early stopping
- **Visualizations**: Generates ROC curves, confusion matrices, and F1 plots
- **Model Persistence**: Saves trained model and tokenizer for future use
- **GPU Support**: Automatically uses GPU if available with mixed precision training

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

**Note**: For GPU support, ensure you have the appropriate PyTorch version for your CUDA setup. Visit [PyTorch website](https://pytorch.org/) for installation instructions.

## Usage

Run the script from the Deberta directory:

```bash
python deberta.py
```

The script will:
1. Load the aggregated EDOS dataset from `../data/edos_labelled_aggregated.csv`
2. Split data into train/dev/test sets using the existing split column
3. Load DeBERTa-v3-large model and tokenizer
4. Train the model with early stopping
5. Evaluate on all splits and generate metrics
6. Create visualizations (ROC curves, confusion matrices, F1 plots)
7. Save the trained model and tokenizer to `models/deberta/` directory

## Configuration

You can modify these parameters in the `main()` function:

### Model Configuration
- `MODEL_NAME`: HuggingFace model identifier (default: "microsoft/deberta-v3-large")
- `MAX_LENGTH`: Maximum sequence length (default: 512)
- `BATCH_SIZE`: Training batch size (default: 8, adjust based on GPU memory)

### Training Parameters
- `NUM_EPOCHS`: Number of training epochs (default: 3)
- `LEARNING_RATE`: Learning rate (default: 2e-5)
- `WEIGHT_DECAY`: L2 regularization (default: 0.01)
- `WARMUP_STEPS`: Number of warmup steps for learning rate scheduler (default: 500)

## GPU Memory Considerations

DeBERTa-v3-large is a large model (~184M parameters). GPU memory requirements:
- **Minimum**: 16GB GPU memory (batch size 4-8)
- **Recommended**: 24GB+ GPU memory (batch size 8-16)

If you encounter out-of-memory errors:
1. Reduce `BATCH_SIZE` (try 4 or even 2)
2. Reduce `MAX_LENGTH` (try 256 or 128)
3. Use gradient accumulation (add to TrainingArguments)
4. Use CPU (slower but works if no GPU available)

## Model Architecture

- **Model**: DeBERTa-v3-large (microsoft/deberta-v3-large)
- **Parameters**: ~184M
- **Input**: Text sequences (max 512 tokens)
- **Output**: Binary classification (Sexist / Not Sexist)

## Output

The script generates:
- **Console output**: Classification reports and metrics for all splits
- **Plots**: Saved to `outputs/` directory:
  - ROC curves (dev and test sets)
  - Confusion matrices (train, dev, test sets)
  - F1 bar plots
- **Models**: Saved model and tokenizer in `models/deberta/` directory

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

## Comparison with Other Models

DeBERTa-v3-large typically outperforms:
- Logistic Regression (TF-IDF features)
- MLP (TF-IDF features)

Expected improvements:
- **5-15%** better accuracy
- **10-20%** better F1 for minority class
- Better handling of context and nuance

## Tips for Better Performance

1. **Fine-tuning**: Increase `NUM_EPOCHS` if underfitting (try 5-10)
2. **Learning Rate**: Experiment with different rates (1e-5 to 5e-5)
3. **Batch Size**: Larger batches if GPU memory allows
4. **Sequence Length**: Increase `MAX_LENGTH` if texts are long (if memory allows)

## Loading Saved Model

To load the trained model for inference:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "models/deberta"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Use for predictions
```

## Notes

- Training time: ~30-60 minutes on GPU (depends on GPU and data size)
- CPU training is possible but very slow (several hours)
- Early stopping will stop training if validation metrics don't improve
- Model saves checkpoints after each epoch

