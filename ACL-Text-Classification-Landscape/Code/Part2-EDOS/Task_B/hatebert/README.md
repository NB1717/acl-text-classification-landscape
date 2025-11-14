# HateBERT Model for Multi-Class Classification

This script trains a HateBERT (GroNLP/hateBERT) transformer model for multi-class classification of sexist content into categories using the aggregated EDOS dataset.

## Task B: Multi-Class Classification

The model classifies text into 5 categories:
1. **none** - Not sexist content
2. **1. threats, plans to harm and incitement** - Threatening content or incitement to harm
3. **2. derogation** - Derogatory attacks on women
4. **3. animosity** - Expressions of animosity toward women
5. **4. prejudiced discussions** - Prejudiced discussions supporting discrimination

## Features

- **Specialized Model**: Uses GroNLP/hateBERT, pre-trained specifically for hate speech detection
- **Multi-Class Classification**: Handles 5 categories instead of binary
- **Comprehensive Evaluation**: Reports macro and weighted metrics for multi-class classification
- **Early Stopping**: Prevents overfitting with validation-based early stopping
- **Visualizations**: Generates confusion matrices, F1 plots, and class distribution
- **Model Persistence**: Saves trained model and tokenizer for future use
- **GPU Support**: Automatically uses GPU/MPS if available with mixed precision training

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

**Note**: For GPU support, ensure you have the appropriate PyTorch version for your CUDA setup. Visit [PyTorch website](https://pytorch.org/) for installation instructions.

## Usage

Run the script from the hatebert directory:

```bash
python hatebert.py
```

The script will:
1. Load the aggregated EDOS dataset from `../../data/edos_labelled_aggregated.csv`
2. Map label categories to numeric labels (0-4)
3. Split data into train/dev/test sets using the existing split column
4. Load HateBERT model with 5 output classes
5. Train the model with early stopping
6. Evaluate on all splits and generate metrics
7. Create visualizations (confusion matrices, F1 plots, class distribution)
8. Save the trained model and tokenizer to `models/hatebert/` directory

## Configuration

You can modify these parameters in the `main()` function:

### Model Configuration
- `MODEL_NAME`: HuggingFace model identifier (default: "GroNLP/hateBERT")
- `MAX_LENGTH`: Maximum sequence length (default: 512)
- `BATCH_SIZE`: Training batch size (default: 8, adjust based on GPU memory)
- `GRADIENT_ACCUMULATION_STEPS`: Steps for gradient accumulation (default: 2)
  - Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

### Training Parameters
- `NUM_EPOCHS`: Number of training epochs (default: 3)
- `LEARNING_RATE`: Learning rate (default: 2e-5)
- `WEIGHT_DECAY`: L2 regularization (default: 0.01)
- `WARMUP_STEPS`: Number of warmup steps for learning rate scheduler (default: 500)

## GPU Memory Considerations

HateBERT (~110M parameters):
- **Minimum**: 8GB GPU memory (batch size 8)
- **Recommended**: 12GB+ GPU memory (batch size 16)

If you encounter out-of-memory errors:
1. Reduce `BATCH_SIZE` (try 4 or even 2)
2. Increase `GRADIENT_ACCUMULATION_STEPS` to maintain effective batch size
3. Reduce `MAX_LENGTH` (try 256 or 128)
4. Use CPU (slower but works if no GPU available)

## Model Architecture

**HateBERT (GroNLP/hateBERT)**
- Based on BERT architecture
- Pre-trained specifically on hate speech detection tasks
- Parameters: ~110M
- Advantages: 
  - Faster training than larger models
  - Lower memory requirements
  - Specialized for hate speech/sexism detection
  - May perform better for this specific task due to pre-training

**Model Details:**
- **Input**: Text sequences (max 512 tokens)
- **Output**: Multi-class classification (5 classes)

## Output

The script generates:
- **Console output**: Classification reports and metrics for all splits
- **Plots**: Saved to `outputs/` directory:
  - Confusion matrices (train, dev, test sets) with counts and percentages
  - F1 bar plots showing per-class performance across splits
  - Class distribution plots
- **Models**: Saved model and tokenizer in `models/hatebert/` directory

## Model Performance Metrics

The model reports:
- **Accuracy**: Overall classification accuracy
- **Macro F1**: Unweighted mean of per-class F1 scores (treats all classes equally)
- **Weighted F1**: Weighted mean of per-class F1 scores (accounts for class imbalance)
- **Macro Precision/Recall**: Unweighted mean across classes
- **Weighted Precision/Recall**: Weighted mean across classes
- **Per-class F1**: Individual F1 score for each category

## Why HateBERT?

HateBERT is specifically designed for hate speech detection:
- **Domain-specific pre-training**: Pre-trained on hate speech datasets
- **Better domain understanding**: May capture hate speech patterns better than general models
- **Efficiency**: Smaller and faster than DeBERTa-large
- **Task alignment**: Pre-training aligns with the sexism detection task

## Comparison with Other Models

HateBERT typically:
- **Better than** TF-IDF-based models (Logistic Regression, SVM, Random Forest)
- **Similar or better than** TextCNN for this task
- **Potentially better than** DeBERTa-base for hate speech due to specialized pre-training
- **Faster than** DeBERTa-large with similar or better performance

Expected improvements:
- **10-15%** better macro F1 than traditional ML models
- **15-20%** better F1 for minority classes
- **Faster training** than larger transformer models

## Tips for Better Performance

1. **Fine-tuning**: Increase `NUM_EPOCHS` if underfitting (try 5-10)
2. **Learning Rate**: Experiment with different rates (1e-5 to 5e-5)
3. **Batch Size**: Larger batches if GPU memory allows (increase gradient accumulation if needed)
4. **Sequence Length**: Increase `MAX_LENGTH` if texts are long (if memory allows)
5. **Class Imbalance**: The model uses CrossEntropyLoss which handles class imbalance through training

## Loading Saved Model

To load the trained model for inference:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "models/hatebert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Use for predictions
# predictions = model(**tokenizer(text, return_tensors="pt"))
```

## Comparison with Binary Classification (Task A)

**Multi-class (Task B) is more challenging:**
- 5 classes instead of 2
- More fine-grained discrimination required
- Class imbalance more pronounced
- Lower overall metrics expected (especially macro F1)

**Expected performance difference:**
- Binary accuracy: ~85-90%
- Multi-class accuracy: ~70-80%
- Binary macro F1: ~0.75-0.85
- Multi-class macro F1: ~0.40-0.60 (due to minority classes)

## Notes

- **Training time**: ~30-60 minutes on GPU (depends on GPU and data size)
- CPU training is possible but very slow (several hours)
- Early stopping will stop training if validation metrics don't improve
- Model saves checkpoints after each epoch
- Best model is automatically loaded at the end
- HateBERT's specialized pre-training may give it an advantage for this task

## Troubleshooting

### Out of Memory Errors
- Reduce `BATCH_SIZE` to 4 or 2
- Increase `GRADIENT_ACCUMULATION_STEPS` to maintain effective batch size
- Reduce `MAX_LENGTH` to 256

### Python 3.13 Issues
If you encounter compatibility errors, consider using Python 3.11 or 3.12 instead.

