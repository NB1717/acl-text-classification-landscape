"""
DeBERTa-v3-large Model for Binary Classification: Sexist vs Not Sexist
Using aggregated EDOS dataset
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# Set device - handle MPS (Apple Silicon), CUDA, or CPU
# Option to force CPU if MPS runs out of memory (set to True if you get MPS OOM errors)
# NOTE: If you still get OOM errors even with batch_size=1, set this to True to use CPU
FORCE_CPU = False  # Set to True to use CPU instead of MPS/CUDA

if FORCE_CPU:
    device = torch.device('cpu')
elif torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")


def load_data(data_path):
    """Load the aggregated EDOS dataset"""
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    print(f"\nLabel distribution:")
    print(df['label_sexist'].value_counts())
    print(f"\nSplit distribution:")
    print(df['split'].value_counts())
    return df


def prepare_labels(df):
    """Convert text labels to binary (1 for sexist, 0 for not sexist)"""
    df['label_binary'] = (df['label_sexist'] == 'sexist').astype(int)
    return df


def split_data(df):
    """
    Split data into train/val/test sets
    Uses the existing split column
    """
    train_df = df[df['split'] == 'train'].copy()
    dev_df = df[df['split'] == 'dev'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    print(f"\nData splits:")
    print(f"Train: {len(train_df)} samples")
    print(f"Dev: {len(dev_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    return train_df, dev_df, test_df


class EDOSDataset(Dataset):
    """Dataset class for EDOS data"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    macro_f1 = (f1_per_class[0] + f1_per_class[1]) / 2
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_f1': macro_f1,
        'f1_not_sexist': f1_per_class[0],
        'f1_sexist': f1_per_class[1],
        'recall_sexist': recall_per_class[1]
    }


def train_model(model, train_dataset, eval_dataset, output_dir, 
                num_epochs=3, batch_size=16, learning_rate=2e-5,
                weight_decay=0.01, warmup_steps=500, gradient_accumulation_steps=1):
    """Train DeBERTa model"""
    print(f"\n{'='*60}")
    print(f"Training DeBERTa-v3-large")
    print(f"{'='*60}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Output directory: {output_dir}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=100,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        greater_is_better=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available (MPS doesn't support fp16 yet)
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,
        report_to='none',  # Disable wandb/tensorboard
        # MPS-specific optimizations
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Clear MPS cache before training if using MPS
    if device.type == 'mps':
        torch.mps.empty_cache()
        print("Cleared MPS cache before training")
    
    # Train
    trainer.train()
    
    # Clear MPS cache after training
    if device.type == 'mps':
        torch.mps.empty_cache()
    
    # Save final model
    trainer.save_model()
    
    return trainer


def evaluate_model(model, tokenizer, dataset, device, batch_size=16):
    """Evaluate model and return predictions"""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            
            # Clear MPS cache periodically to free memory
            if device.type == 'mps' and len(all_predictions) % (batch_size * 10) == 0:
                torch.mps.empty_cache()
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def print_evaluation_report(y_true, y_pred, y_pred_proba, split_name=""):
    """Print detailed evaluation report"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro_f1 = (f1_per_class[0] + f1_per_class[1]) / 2
    
    try:
        auc_score = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc_score = None
    
    print(f"\n{'='*60}")
    print(f"Results on {split_name} set:")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"\nPer-class F1-Scores:")
    print(f"  Not Sexist: {f1_per_class[0]:.4f}")
    print(f"  Sexist:     {f1_per_class[1]:.4f}")
    print(f"\nPer-class Recall:")
    print(f"  Not Sexist: {recall_per_class[0]:.4f}")
    print(f"  Sexist:     {recall_per_class[1]:.4f}")
    if auc_score:
        print(f"AUC-ROC:   {auc_score:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                  Not Sexist  Sexist")
    print(f"Actual Not Sexist     {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"      Sexist          {cm[1,0]:5d}    {cm[1,1]:5d}")
    print(f"\n  False Negatives (missed sexist): {cm[1,0]}")
    print(f"  False Positives (false alarms):   {cm[0,1]}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Not Sexist', 'Sexist']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_f1': macro_f1,
        'f1_per_class': f1_per_class,
        'recall_per_class': recall_per_class,
        'precision_per_class': precision_per_class,
        'auc': auc_score,
        'confusion_matrix': cm,
        'false_negatives': cm[1, 0],
        'false_positives': cm[0, 1]
    }


def plot_roc_curve(y_true, y_pred_proba, split_name="", save_path=None):
    """Plot ROC curve"""
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random classifier', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {split_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3, linestyle='--')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        plt.close()
    except Exception as e:
        print(f"Could not plot ROC curve: {e}")


def plot_confusion_matrix(y_true, y_pred, split_name="", save_path=None):
    """Plot confusion matrix with percentages"""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Absolute values
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Not Sexist', 'Sexist'],
                yticklabels=['Not Sexist', 'Sexist'])
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('Actual', fontsize=12)
    ax1.set_title(f'Confusion Matrix (Counts) - {split_name}', fontsize=12, fontweight='bold')
    
    # Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2,
                xticklabels=['Not Sexist', 'Sexist'],
                yticklabels=['Not Sexist', 'Sexist'])
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('Actual', fontsize=12)
    ax2.set_title(f'Confusion Matrix (Percentages) - {split_name}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_f1_boxplot(all_results, save_path=None):
    """Plot F1 scores per class across splits"""
    splits = []
    classes = []
    f1_scores = []
    
    class_names = ['Not Sexist', 'Sexist']
    
    for split_name, results in all_results.items():
        if 'f1_per_class' in results:
            f1_per_class = results['f1_per_class']
            for class_idx, f1_score in enumerate(f1_per_class):
                splits.append(split_name.capitalize())
                classes.append(class_names[class_idx])
                f1_scores.append(f1_score)
    
    df_plot = pd.DataFrame({
        'Split': splits,
        'Class': classes,
        'F1-Score': f1_scores
    })
    
    # Bar plot
    plt.figure(figsize=(10, 6))
    df_pivot = df_plot.pivot(index='Split', columns='Class', values='F1-Score')
    ax = df_pivot.plot(kind='bar', width=0.8, color=['#66b3ff', '#ff6666'], figsize=(10, 6))
    plt.title('F1-Score by Class and Split', fontsize=14, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=12)
    plt.xlabel('Data Split', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Class', title_fontsize=11, fontsize=10)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim([0, 1.1])
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
    
    # Add macro F1 scores
    for i, (split_name, results) in enumerate(all_results.items()):
        if 'macro_f1' in results:
            macro_f1 = results['macro_f1']
            plt.text(i, 1.05, f'Macro F1: {macro_f1:.3f}', 
                    ha='center', fontsize=9, style='italic', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"F1 bar plot saved to {save_path}")
    plt.close()


def main():
    """Main function to run the complete pipeline"""
    # Configuration
    script_dir = Path(__file__).parent
    possible_paths = [
        script_dir.parent / "data" / "edos_labelled_aggregated.csv",
        script_dir / ".." / "data" / "edos_labelled_aggregated.csv",
        Path("../data/edos_labelled_aggregated.csv"),
        Path("data/edos_labelled_aggregated.csv")
    ]
    
    DATA_PATH = None
    for path in possible_paths:
        if path.exists():
            DATA_PATH = str(path.resolve())
            break
    
    if DATA_PATH is None:
        raise FileNotFoundError(
            f"Could not find edos_labelled_aggregated.csv. Tried: {possible_paths}"
        )
    
    # Model configuration
    # NOTE: If you continue getting OOM errors, consider using a smaller model:
    MODEL_NAME = "microsoft/deberta-v3-base"  # Much smaller, ~184M params vs 435M
    # MODEL_NAME = "microsoft/deberta-v3-large"
    # Reduce sequence length for MPS to save memory (512 -> 256 saves significant memory)
    MAX_LENGTH = 256 if device.type == 'mps' else 512
    # Reduce batch size if using MPS (Apple Silicon) to avoid OOM errors
    # For MPS: batch_size=1-2 is necessary for large models
    # For CUDA: batch_size=8-16 may work
    # For CPU: batch_size=8-16 is usually fine
    BATCH_SIZE = 1 if device.type == 'mps' else 8  # Very small batch for MPS
    GRADIENT_ACCUMULATION_STEPS = 8 if device.type == 'mps' else 1  # Compensate with gradient accumulation
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 500
    
    OUTPUT_DIR = "outputs"
    MODEL_DIR = "models/deberta"
    
    # Create directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load and prepare data
    df = load_data(DATA_PATH)
    df = prepare_labels(df)
    
    # Split data
    train_df, dev_df, test_df = split_data(df)
    
    # Load tokenizer and model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
    
    # Clear cache before moving model to device
    if device.type == 'mps':
        torch.mps.empty_cache()
    
    model.to(device)
    
    # Clear cache after moving model
    if device.type == 'mps':
        torch.mps.empty_cache()
        print("Model loaded to MPS. Memory cache cleared.")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = EDOSDataset(train_df['text'], train_df['label_binary'], tokenizer, MAX_LENGTH)
    dev_dataset = EDOSDataset(dev_df['text'], dev_df['label_binary'], tokenizer, MAX_LENGTH)
    test_dataset = EDOSDataset(test_df['text'], test_df['label_binary'], tokenizer, MAX_LENGTH)
    
    # Train model
    trainer = train_model(
        model,
        train_dataset,
        dev_dataset,
        MODEL_DIR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS
    )
    
    # Load best model
    best_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    best_model.to(device)
    
    # Evaluate on all splits
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    train_pred, train_true, train_proba = evaluate_model(
        best_model, tokenizer, train_dataset, device, BATCH_SIZE
    )
    train_results = print_evaluation_report(train_true, train_pred, train_proba, "Train")
    
    dev_pred, dev_true, dev_proba = evaluate_model(
        best_model, tokenizer, dev_dataset, device, BATCH_SIZE
    )
    dev_results = print_evaluation_report(dev_true, dev_pred, dev_proba, "Dev")
    
    test_pred, test_true, test_proba = evaluate_model(
        best_model, tokenizer, test_dataset, device, BATCH_SIZE
    )
    test_results = print_evaluation_report(test_true, test_pred, test_proba, "Test")
    
    # Plot visualizations
    plot_roc_curve(dev_true, dev_proba, "Dev", 
                   save_path=os.path.join(OUTPUT_DIR, "roc_curve_dev.png"))
    plot_roc_curve(test_true, test_proba, "Test",
                   save_path=os.path.join(OUTPUT_DIR, "roc_curve_test.png"))
    
    plot_confusion_matrix(train_true, train_pred, "Train",
                         save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_train.png"))
    plot_confusion_matrix(dev_true, dev_pred, "Dev",
                         save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_dev.png"))
    plot_confusion_matrix(test_true, test_pred, "Test",
                         save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_test.png"))
    
    # Plot F1 box plot
    all_results = {
        'train': train_results,
        'dev': dev_results,
        'test': test_results
    }
    plot_f1_boxplot(all_results, save_path=os.path.join(OUTPUT_DIR, "f1_boxplot.png"))
    
    # Save tokenizer
    tokenizer.save_pretrained(MODEL_DIR)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:    {test_results['accuracy']:.4f}")
    print(f"  Precision:   {test_results['precision']:.4f}")
    print(f"  Recall:      {test_results['recall']:.4f}")
    print(f"  F1-Score:    {test_results['f1']:.4f}")
    print(f"  Macro F1:    {test_results['macro_f1']:.4f}")
    print(f"  F1 (Not Sexist): {test_results['f1_per_class'][0]:.4f}")
    print(f"  F1 (Sexist):     {test_results['f1_per_class'][1]:.4f}")
    print(f"\nFalse Negatives (missed sexist): {test_results['false_negatives']}")
    print(f"False Positives (false alarms):   {test_results['false_positives']}")
    print(f"Sexist Recall:                    {test_results['recall_per_class'][1]:.4f}")
    if test_results['auc']:
        print(f"  AUC-ROC:     {test_results['auc']:.4f}")
    print("\nMacro F1 across splits:")
    print(f"  Train: {train_results['macro_f1']:.4f}")
    print(f"  Dev:   {dev_results['macro_f1']:.4f}")
    print(f"  Test:  {test_results['macro_f1']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

