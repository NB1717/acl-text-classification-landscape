"""
HateBERT Model for Multi-Class Classification
Task B: Classify sexist content into categories (none, threats, derogation, animosity, prejudiced discussions)
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


# Set device
FORCE_CPU = False

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
    print(f"\nLabel category distribution:")
    print(df['label_category'].value_counts())
    print(f"\nSplit distribution:")
    print(df['split'].value_counts())
    return df


def prepare_labels(df):
    """Prepare multi-class labels from label_category"""
    category_mapping = {
        'none': 0,
        '1. threats, plans to harm and incitement': 1,
        '2. derogation': 2,
        '3. animosity': 3,
        '4. prejudiced discussions': 4
    }
    
    df['label_category_encoded'] = df['label_category'].map(category_mapping)
    
    if df['label_category_encoded'].isna().any():
        unmapped = df[df['label_category_encoded'].isna()]['label_category'].unique()
        print(f"Warning: Unmapped categories found: {unmapped}")
        df = df.dropna(subset=['label_category_encoded'])
    
    df['label_category_encoded'] = df['label_category_encoded'].astype(int)
    
    print(f"\nEncoded label distribution:")
    print(df['label_category_encoded'].value_counts().sort_index())
    
    return df


def split_data(df):
    """Split data into train/val/test sets"""
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
    """Compute metrics for multi-class evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    # Macro and weighted metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    }


def train_model(model, train_dataset, eval_dataset, output_dir, 
                num_epochs=3, batch_size=8, learning_rate=2e-5,
                weight_decay=0.01, warmup_steps=500, gradient_accumulation_steps=1,
                model_name=""):
    """Train HateBERT model"""
    print(f"\n{'='*60}")
    print(f"Training {model_name} for Multi-Class Classification")
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
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4 if torch.cuda.is_available() else 0,
        report_to='none',
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
    
    if device.type == 'mps':
        torch.mps.empty_cache()
        print("Cleared MPS cache before training")
    
    trainer.train()
    
    if device.type == 'mps':
        torch.mps.empty_cache()
    
    trainer.save_model()
    
    return trainer


def evaluate_model(model, tokenizer, dataset, device, class_names, batch_size=8):
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
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def print_evaluation_report(y_true, y_pred, y_pred_proba, class_names, split_name=""):
    """Print detailed evaluation report for multi-class"""
    accuracy = accuracy_score(y_true, y_pred)
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    print(f"\n{'='*60}")
    print(f"Results on {split_name} set:")
    print(f"{'='*60}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall:    {recall_macro:.4f}")
    print(f"Macro F1:        {f1_macro:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall:     {recall_weighted:.4f}")
    print(f"Weighted F1:         {f1_weighted:.4f}")
    
    print(f"\nPer-class F1-Scores:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:35s}: {f1_per_class[i]:.4f} (support: {support_per_class[i]:5d})")
    
    print(f"\nPer-class Recall:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:35s}: {recall_per_class[i]:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print("                  ", end="")
    for i in range(len(class_names)):
        print(f"Class{i}", end=" ")
    print()
    for i in range(len(class_names)):
        short_name = class_names[i][:15] if len(class_names[i]) > 15 else class_names[i]
        print(f"Actual {short_name:15s}", end=" ")
        for j in range(len(class_names)):
            print(f"{cm[i,j]:5d}", end=" ")
        print()
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'recall_per_class': recall_per_class,
        'precision_per_class': precision_per_class,
        'support_per_class': support_per_class,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(y_true, y_pred, class_names, split_name="", save_path=None):
    """Plot confusion matrix with percentages"""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    short_names = []
    for name in class_names:
        if len(name) > 20:
            short_names.append(name[:17] + "...")
        else:
            short_names.append(name)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=short_names, yticklabels=short_names)
    ax1.set_xlabel('Predicted', fontsize=11)
    ax1.set_ylabel('Actual', fontsize=11)
    ax1.set_title(f'Confusion Matrix (Counts) - {split_name}', fontsize=12, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2,
                xticklabels=short_names, yticklabels=short_names)
    ax2.set_xlabel('Predicted', fontsize=11)
    ax2.set_ylabel('Actual', fontsize=11)
    ax2.set_title(f'Confusion Matrix (Percentages) - {split_name}', fontsize=12, fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax2.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_f1_scores(all_results, class_names, save_path=None):
    """Plot F1 scores per class across different splits"""
    splits = []
    classes = []
    f1_scores = []
    
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
    
    plt.figure(figsize=(14, 7))
    df_pivot = df_plot.pivot(index='Split', columns='Class', values='F1-Score')
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    ax = df_pivot.plot(kind='bar', width=0.8, color=colors, figsize=(14, 7))
    
    plt.title('F1-Score by Class and Split', fontsize=14, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=12)
    plt.xlabel('Data Split', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Class', title_fontsize=10, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim([0, 1.1])
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=7)
    
    for i, (split_name, results) in enumerate(all_results.items()):
        if 'f1_macro' in results:
            macro_f1 = results['f1_macro']
            plt.text(i, 1.05, f'Macro F1: {macro_f1:.3f}', 
                    ha='center', fontsize=9, style='italic', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"F1 bar plot saved to {save_path}")
    plt.close()


def plot_class_distribution(y_train, y_dev, y_test, class_names, save_path=None):
    """Plot distribution of classes across splits"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    splits = [('Train', y_train), ('Dev', y_dev), ('Test', y_test)]
    
    for idx, (split_name, y) in enumerate(splits):
        if isinstance(y, pd.Series):
            y = y.values
        unique, counts = np.unique(y, return_counts=True)
        axes[idx].bar(range(len(unique)), counts, color=plt.cm.Set3(np.linspace(0, 1, len(unique))))
        axes[idx].set_xticks(range(len(unique)))
        axes[idx].set_xticklabels([class_names[i] if i < len(class_names) else str(i) for i in unique], 
                                 rotation=45, ha='right', fontsize=8)
        axes[idx].set_ylabel('Count', fontsize=10)
        axes[idx].set_title(f'{split_name} Set', fontsize=12, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
        
        for i, (u, c) in enumerate(zip(unique, counts)):
            axes[idx].text(i, c, str(c), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    plt.close()


def main():
    """Main function to run the complete pipeline"""
    # Configuration
    script_dir = Path(__file__).parent
    possible_paths = [
        script_dir.parent.parent / "data" / "edos_labelled_aggregated.csv",
        script_dir / ".." / ".." / "data" / "edos_labelled_aggregated.csv",
        Path("../../data/edos_labelled_aggregated.csv"),
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
    MODEL_NAME = "GroNLP/hateBERT"
    MAX_LENGTH = 512
    BATCH_SIZE = 8  # HateBERT is smaller (~110M params)
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 8 * 2 = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 500
    NUM_CLASSES = 5  # Multi-class: 5 categories
    
    OUTPUT_DIR = "outputs"
    # Update model directory name based on model being used
    model_short_name = MODEL_NAME.split("/")[-1].lower().replace("-", "_")
    MODEL_DIR = f"models/{model_short_name}"
    
    # Class names
    CLASS_NAMES = [
        'none',
        '1. threats, plans to harm and incitement',
        '2. derogation',
        '3. animosity',
        '4. prejudiced discussions'
    ]
    
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
        num_labels=NUM_CLASSES  # 5 classes for multi-class classification
    )
    model.to(device)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = EDOSDataset(train_df['text'], train_df['label_category_encoded'], tokenizer, MAX_LENGTH)
    dev_dataset = EDOSDataset(dev_df['text'], dev_df['label_category_encoded'], tokenizer, MAX_LENGTH)
    test_dataset = EDOSDataset(test_df['text'], test_df['label_category_encoded'], tokenizer, MAX_LENGTH)
    
    # Plot class distribution
    plot_class_distribution(train_df['label_category_encoded'], 
                           dev_df['label_category_encoded'],
                           test_df['label_category_encoded'],
                           CLASS_NAMES,
                           save_path=os.path.join(OUTPUT_DIR, "class_distribution.png"))
    
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
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        model_name=MODEL_NAME
    )
    
    # Load best model
    best_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    best_model.to(device)
    
    # Evaluate on all splits
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    train_pred, train_true, train_proba = evaluate_model(
        best_model, tokenizer, train_dataset, device, CLASS_NAMES, BATCH_SIZE
    )
    train_results = print_evaluation_report(train_true, train_pred, train_proba, CLASS_NAMES, "Train")
    
    dev_pred, dev_true, dev_proba = evaluate_model(
        best_model, tokenizer, dev_dataset, device, CLASS_NAMES, BATCH_SIZE
    )
    dev_results = print_evaluation_report(dev_true, dev_pred, dev_proba, CLASS_NAMES, "Dev")
    
    test_pred, test_true, test_proba = evaluate_model(
        best_model, tokenizer, test_dataset, device, CLASS_NAMES, BATCH_SIZE
    )
    test_results = print_evaluation_report(test_true, test_pred, test_proba, CLASS_NAMES, "Test")
    
    # Plot visualizations
    plot_confusion_matrix(train_true, train_pred, CLASS_NAMES, "Train",
                         save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_train.png"))
    plot_confusion_matrix(dev_true, dev_pred, CLASS_NAMES, "Dev",
                         save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_dev.png"))
    plot_confusion_matrix(test_true, test_pred, CLASS_NAMES, "Test",
                         save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_test.png"))
    
    # Plot F1 scores
    all_results = {
        'train': train_results,
        'dev': dev_results,
        'test': test_results
    }
    plot_f1_scores(all_results, CLASS_NAMES, save_path=os.path.join(OUTPUT_DIR, "f1_scores.png"))
    
    # Save tokenizer
    tokenizer.save_pretrained(MODEL_DIR)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Task: Multi-Class Classification (5 classes)")
    print(f"Model directory: {MODEL_DIR}")
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:         {test_results['accuracy']:.4f}")
    print(f"  Macro F1:         {test_results['f1_macro']:.4f}")
    print(f"  Weighted F1:      {test_results['f1_weighted']:.4f}")
    print(f"  Macro Precision:  {test_results['precision_macro']:.4f}")
    print(f"  Macro Recall:     {test_results['recall_macro']:.4f}")
    print(f"\nPer-class F1-Scores:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name:35s}: {test_results['f1_per_class'][i]:.4f}")
    print("\nMacro F1 across splits:")
    print(f"  Train: {train_results['f1_macro']:.4f}")
    print(f"  Dev:   {dev_results['f1_macro']:.4f}")
    print(f"  Test:  {test_results['f1_macro']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

