"""
Performance Comparison Plots for Task_B Models
Creates comparison charts for Random Forest, SVM, TextCNN, and HateBERT models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import re
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    roc_auc_score
)


# Class names for Task B
CLASS_NAMES = [
    'none',
    '1. threats, plans to harm and incitement',
    '2. derogation',
    '3. animosity',
    '4. prejudiced discussions'
]

CLASS_NAMES_SHORT = [
    'none',
    'threats',
    'derogation',
    'animosity',
    'prejudiced'
]


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
        df = df.dropna(subset=['label_category_encoded'])
    
    df['label_category_encoded'] = df['label_category_encoded'].astype(int)
    return df


def evaluate_rf_model(model_path, vectorizer_path, label_encoder_path, data_path):
    """Load and evaluate Random Forest model"""
    print(f"Loading Random Forest model...")
    
    # Load data
    df = pd.read_csv(data_path)
    df = prepare_labels(df)
    test_df = df[df['split'] == 'test'].copy()
    
    # Load model, vectorizer, and label encoder
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Extract features and evaluate
    X_test = vectorizer.transform(test_df['text'])
    X_test = X_test.toarray()  # RF needs dense arrays
    y_test = test_df['label_category_encoded'].values
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    macro_f1 = np.mean(f1_per_class)
    
    # Multi-class AUC-ROC (macro-averaged one-vs-rest)
    try:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    except ValueError:
        auc = None
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy, 
        'macro_f1': macro_f1, 
        'f1_per_class': f1_per_class,
        'auc': auc, 
        'confusion_matrix': cm, 
        'y_true': y_test, 
        'y_pred': y_pred
    }


def evaluate_svm_model(model_path, vectorizer_path, label_encoder_path, data_path):
    """Load and evaluate SVM model"""
    print(f"Loading SVM model...")
    
    # Load data
    df = pd.read_csv(data_path)
    df = prepare_labels(df)
    test_df = df[df['split'] == 'test'].copy()
    
    # Load model, vectorizer, and label encoder
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Extract features and evaluate
    X_test = vectorizer.transform(test_df['text'])
    y_test = test_df['label_category_encoded'].values
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    macro_f1 = np.mean(f1_per_class)
    
    # Multi-class AUC-ROC (macro-averaged one-vs-rest)
    try:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    except ValueError:
        auc = None
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy, 
        'macro_f1': macro_f1, 
        'f1_per_class': f1_per_class,
        'auc': auc, 
        'confusion_matrix': cm, 
        'y_true': y_test, 
        'y_pred': y_pred
    }


def evaluate_textcnn_model(model_path, vocab_path, label_encoder_path, data_path):
    """Load and evaluate TextCNN model"""
    print(f"Loading TextCNN model...")
    
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    
    # Load data
    df = pd.read_csv(data_path)
    df = prepare_labels(df)
    test_df = df[df['split'] == 'test'].copy()
    
    # Load vocab and label encoder
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Define TextDataset class (simplified version)
    class TextDataset(Dataset):
        def __init__(self, texts, labels, vocab, max_length=256):
            self.texts = texts
            self.labels = labels
            self.vocab = vocab
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts.iloc[idx])
            label = self.labels.iloc[idx]
            
            tokens = text.lower().split()
            indices = [vocab.get(token, vocab.get('<UNK>', 0)) for token in tokens]
            
            if len(indices) > self.max_length:
                indices = indices[:self.max_length]
            else:
                indices = indices + [vocab.get('<PAD>', 0)] * (self.max_length - len(indices))
            
            return {
                'text': torch.tensor(indices, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.long)
            }
    
    # Define TextCNN model (simplified)
    class TextCNN(nn.Module):
        def __init__(self, vocab_size, embed_dim=128, num_filters=100, 
                     filter_sizes=[3, 4, 5], num_classes=5, dropout=0.5):
            super(TextCNN, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.convs = nn.ModuleList([
                nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, 
                         kernel_size=fs, padding=1)
                for fs in filter_sizes
            ])
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
        def forward(self, x):
            x = self.embedding(x)
            x = x.permute(0, 2, 1)
            conv_outputs = []
            for conv in self.convs:
                conv_out = torch.relu(conv(x))
                pooled = torch.max_pool1d(conv_out, kernel_size=conv_out.size(2)).squeeze(2)
                conv_outputs.append(pooled)
            x = torch.cat(conv_outputs, dim=1)
            x = self.dropout(x)
            x = self.fc(x)
            return x
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = len(vocab)
    model = TextCNN(vocab_size=vocab_size, num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    test_dataset = TextDataset(test_df['text'], test_df['label_category_encoded'], vocab, max_length=256)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(texts)
            probs = torch.softmax(outputs, dim=-1)
            preds = torch.argmax(outputs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    y_pred = np.array(all_preds)
    y_test = np.array(all_labels)
    y_pred_proba = np.array(all_probs)
    
    accuracy = accuracy_score(y_test, y_pred)
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    macro_f1 = np.mean(f1_per_class)
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    except ValueError:
        auc = None
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy, 
        'macro_f1': macro_f1, 
        'f1_per_class': f1_per_class,
        'auc': auc, 
        'confusion_matrix': cm, 
        'y_true': y_test, 
        'y_pred': y_pred
    }


def evaluate_hatebert_model(model_dir, data_path):
    """Load and evaluate HateBERT model to get AUC-ROC"""
    print(f"Loading HateBERT model for evaluation...")
    
    # Load data
    df = pd.read_csv(data_path)
    df = prepare_labels(df)
    test_df = df[df['split'] == 'test'].copy()
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    # Define dataset class
    class EDOSDataset(Dataset):
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
    
    # Create dataset and dataloader
    test_dataset = EDOSDataset(test_df['text'], test_df['label_category_encoded'], tokenizer, max_length=512)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    y_pred = np.array(all_preds)
    y_test = np.array(all_labels)
    y_pred_proba = np.array(all_probs)
    
    accuracy = accuracy_score(y_test, y_pred)
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    macro_f1 = np.mean(f1_per_class)
    
    # Multi-class AUC-ROC (macro-averaged one-vs-rest)
    try:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    except ValueError:
        auc = None
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'f1_per_class': f1_per_class,
        'auc': auc,
        'confusion_matrix': cm,
        'y_true': y_test,
        'y_pred': y_pred
    }


def load_hatebert_results(results_file):
    """Parse HateBERT results from results.txt file"""
    print(f"Loading HateBERT results from {results_file}...")
    
    with open(results_file, 'r') as f:
        content = f.read()
    
    # Extract test set performance
    test_section = re.search(r'Results on Test set:.*?Accuracy:\s+([\d.]+).*?Macro F1:\s+([\d.]+)', 
                             content, re.DOTALL)
    
    if test_section:
        accuracy = float(test_section.group(1))
        macro_f1 = float(test_section.group(2))
        return {'accuracy': accuracy, 'macro_f1': macro_f1, 'auc': None}  # AUC will be computed separately
    
    raise ValueError("Could not parse HateBERT results file")


def get_hatebert_confusion_matrix(results_file):
    """Extract HateBERT confusion matrix from results.txt"""
    print(f"Loading HateBERT confusion matrix...")
    
    with open(results_file, 'r') as f:
        content = f.read()
    
    # Find confusion matrix section
    cm_match = re.search(r'Confusion Matrix:.*?Actual none\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+).*?Actual 1\. threats.*?(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+).*?Actual 2\. derogation.*?(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+).*?Actual 3\. animosity.*?(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+).*?Actual 4\. prejudiced.*?(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', 
                         content, re.DOTALL)
    
    if cm_match:
        cm = np.array([
            [int(cm_match.group(1)), int(cm_match.group(2)), int(cm_match.group(3)), int(cm_match.group(4)), int(cm_match.group(5))],
            [int(cm_match.group(6)), int(cm_match.group(7)), int(cm_match.group(8)), int(cm_match.group(9)), int(cm_match.group(10))],
            [int(cm_match.group(11)), int(cm_match.group(12)), int(cm_match.group(13)), int(cm_match.group(14)), int(cm_match.group(15))],
            [int(cm_match.group(16)), int(cm_match.group(17)), int(cm_match.group(18)), int(cm_match.group(19)), int(cm_match.group(20))],
            [int(cm_match.group(21)), int(cm_match.group(22)), int(cm_match.group(23)), int(cm_match.group(24)), int(cm_match.group(25))]
        ])
        return cm
    
    return None


def get_hatebert_f1_scores(results_file):
    """Extract per-class F1 scores from HateBERT results"""
    with open(results_file, 'r') as f:
        content = f.read()
    
    # Extract per-class F1 scores from test set
    f1_pattern = r'Per-class F1-Scores:.*?none\s+:\s+([\d.]+).*?1\. threats.*?:\s+([\d.]+).*?2\. derogation.*?:\s+([\d.]+).*?3\. animosity.*?:\s+([\d.]+).*?4\. prejudiced.*?:\s+([\d.]+)'
    f1_match = re.search(f1_pattern, content, re.DOTALL)
    
    if f1_match:
        f1_scores = [
            float(f1_match.group(1)),  # none
            float(f1_match.group(2)),  # threats
            float(f1_match.group(3)),  # derogation
            float(f1_match.group(4)),  # animosity
            float(f1_match.group(5))   # prejudiced
        ]
        return np.array(f1_scores)
    
    return None


def create_confusion_matrices_plot(confusion_matrices_dict, save_path=None):
    """Create side-by-side confusion matrix plot for all models"""
    models = list(confusion_matrices_dict.keys())
    num_models = len(models)
    
    fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5))
    
    if num_models == 1:
        axes = [axes]
    
    for idx, (model_name, cm) in enumerate(confusion_matrices_dict.items()):
        ax = axes[idx]
        
        # Normalize confusion matrix to percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create custom annotations with count and percentage
        annot = []
        for i in range(len(CLASS_NAMES_SHORT)):
            row = []
            for j in range(len(CLASS_NAMES_SHORT)):
                row.append(f'{int(cm[i, j])}\n({cm_percent[i, j]:.1f}%)')
            annot.append(row)
        
        # Create heatmap
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax,
                   xticklabels=CLASS_NAMES_SHORT, yticklabels=CLASS_NAMES_SHORT,
                   cbar_kws={'label': 'Count'}, vmin=0, vmax=cm.max())
        
        ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=10, fontweight='bold')
        ax.set_title(f'{model_name}\n(Test Set)', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrices plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_combined_bar_chart(results_dict, save_path=None):
    """Create a single bar chart with grouped bars for Accuracy, F1-Macro, and AUC-ROC"""
    models = list(results_dict.keys())
    accuracies = [results_dict[model]['accuracy'] for model in models]
    macro_f1s = [results_dict[model]['macro_f1'] for model in models]
    aucs = [results_dict[model].get('auc', None) for model in models]
    valid_aucs = [auc if auc is not None else 0 for auc in aucs]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, accuracies, width, label='Accuracy', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, macro_f1s, width, label='F1-Macro', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + width, valid_aucs, width, label='AUC-ROC', 
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison (Task B)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nCombined bar chart saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_classwise_f1_plot(results_dict, save_path=None):
    """Create bar chart showing F1 scores per class for each model"""
    models = list(results_dict.keys())
    num_classes = len(CLASS_NAMES_SHORT)
    
    x = np.arange(num_classes)
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, model in enumerate(models):
        f1_scores = results_dict[model].get('f1_per_class', [])
        if len(f1_scores) == num_classes:
            offset = (idx - len(models)/2 + 0.5) * width
            bars = ax.bar(x + offset, f1_scores, width, label=model, 
                         color=colors[idx % len(colors)], alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_title('Class-wise F1 Scores by Model (Task B)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES_SHORT, fontsize=10, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nClass-wise F1 scores plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def load_textcnn_training_history(history_path):
    """Load TextCNN training history from JSON file"""
    print(f"Loading TextCNN training history from {history_path}...")
    
    if not os.path.exists(history_path):
        return None
    
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        return {
            'epochs': np.array(history['epochs']),
            'train_losses': np.array(history['train_losses']),
            'dev_losses': np.array(history['dev_losses']),
            'model_name': 'TextCNN'
        }
    except Exception as e:
        print(f"⚠ Error loading TextCNN training history: {e}")
        return None


def load_hatebert_training_loss(trainer_state_path):
    """Load HateBERT training loss from trainer_state.json"""
    print(f"Loading HateBERT training loss from {trainer_state_path}...")
    
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)
    
    train_epochs = []
    train_losses = []
    eval_epochs = []
    eval_losses = []
    
    # Extract training loss and eval loss
    for entry in trainer_state.get('log_history', []):
        if 'loss' in entry and 'eval_loss' not in entry:
            # Training loss entry
            train_epochs.append(entry['epoch'])
            train_losses.append(entry['loss'])
        elif 'eval_loss' in entry:
            # Eval loss entry
            eval_epochs.append(entry['epoch'])
            eval_losses.append(entry['eval_loss'])
    
    if train_epochs and train_losses:
        result = {
            'train_epochs': np.array(train_epochs),
            'train_losses': np.array(train_losses),
            'model_name': 'HateBERT'
        }
        
        if eval_epochs and eval_losses:
            result['eval_epochs'] = np.array(eval_epochs)
            result['eval_losses'] = np.array(eval_losses)
        
        return result
    
    return None


def create_learning_curves_plot(textcnn_data, hatebert_data, save_path=None):
    """Create learning curves comparing TextCNN and HateBERT"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot HateBERT if available
    if hatebert_data:
        # Plot training loss
        ax.plot(hatebert_data['train_epochs'], hatebert_data['train_losses'], 
               linewidth=2, color='#3498db', marker='o', markersize=4, 
               label='HateBERT (Train)', alpha=0.8)
        
        # Plot dev loss if available
        if hatebert_data.get('eval_losses') is not None and len(hatebert_data['eval_losses']) > 0:
            ax.plot(hatebert_data['eval_epochs'], hatebert_data['eval_losses'], 
                   linewidth=2, color='#e74c3c', marker='s', markersize=5, 
                   label='HateBERT (Dev)', alpha=0.8)
    
    # Plot TextCNN if available
    if textcnn_data:
        ax.plot(textcnn_data['epochs'], textcnn_data['train_losses'], 
               linewidth=2, color='#2ecc71', marker='o', markersize=4, 
               label='TextCNN (Train)', alpha=0.8)
        ax.plot(textcnn_data['epochs'], textcnn_data['dev_losses'], 
               linewidth=2, color='#f39c12', marker='s', markersize=5, 
               label='TextCNN (Dev)', alpha=0.8)
    else:
        print("⚠ TextCNN training history not available (training history is only plotted, not saved as data)")
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation Loss Comparison (TextCNN vs. HateBERT)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nLearning curves plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main function to create comparison charts"""
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    
    # Find data file
    data_paths = [
        base_dir / "data" / "edos_labelled_aggregated.csv",
        script_dir / ".." / "data" / "edos_labelled_aggregated.csv",
        Path("../data/edos_labelled_aggregated.csv"),
        Path("data/edos_labelled_aggregated.csv")
    ]
    
    DATA_PATH = None
    for path in data_paths:
        if path.exists():
            DATA_PATH = str(path.resolve())
            break
    
    if DATA_PATH is None:
        raise FileNotFoundError(f"Could not find edos_labelled_aggregated.csv")
    
    print("="*60)
    print("Collecting Performance Metrics for All Task B Models")
    print("="*60)
    
    results = {}
    confusion_matrices = {}
    f1_per_class_dict = {}
    
    # Get Random Forest results
    try:
        rf_model_path = script_dir / "random_forest" / "models" / "random_forest_model.pkl"
        rf_vectorizer_path = script_dir / "random_forest" / "models" / "tfidf_vectorizer_rf.pkl"
        rf_label_encoder_path = script_dir / "random_forest" / "models" / "label_encoder_rf.pkl"
        
        if all(p.exists() for p in [rf_model_path, rf_vectorizer_path, rf_label_encoder_path]):
            rf_results = evaluate_rf_model(
                str(rf_model_path), str(rf_vectorizer_path), str(rf_label_encoder_path), DATA_PATH
            )
            results['Random Forest'] = {k: v for k, v in rf_results.items() if k not in ['confusion_matrix', 'y_true', 'y_pred']}
            confusion_matrices['Random Forest'] = rf_results['confusion_matrix']
            f1_per_class_dict['Random Forest'] = rf_results['f1_per_class']
            auc_str = f", AUC={rf_results['auc']:.4f}" if rf_results.get('auc') else ""
            print(f"✓ Random Forest: Accuracy={rf_results['accuracy']:.4f}, "
                  f"F1-Macro={rf_results['macro_f1']:.4f}{auc_str}")
        else:
            print("⚠ Random Forest model not found, skipping...")
    except Exception as e:
        print(f"⚠ Error loading Random Forest: {e}")
        import traceback
        traceback.print_exc()
    
    # Get SVM results
    try:
        svm_model_path = script_dir / "svm" / "models" / "svm_model.pkl"
        svm_vectorizer_path = script_dir / "svm" / "models" / "tfidf_vectorizer_svm.pkl"
        svm_label_encoder_path = script_dir / "svm" / "models" / "label_encoder.pkl"
        
        if all(p.exists() for p in [svm_model_path, svm_vectorizer_path, svm_label_encoder_path]):
            svm_results = evaluate_svm_model(
                str(svm_model_path), str(svm_vectorizer_path), str(svm_label_encoder_path), DATA_PATH
            )
            results['SVM'] = {k: v for k, v in svm_results.items() if k not in ['confusion_matrix', 'y_true', 'y_pred']}
            confusion_matrices['SVM'] = svm_results['confusion_matrix']
            f1_per_class_dict['SVM'] = svm_results['f1_per_class']
            auc_str = f", AUC={svm_results['auc']:.4f}" if svm_results.get('auc') else ""
            print(f"✓ SVM: Accuracy={svm_results['accuracy']:.4f}, "
                  f"F1-Macro={svm_results['macro_f1']:.4f}{auc_str}")
        else:
            print("⚠ SVM model not found, skipping...")
    except Exception as e:
        print(f"⚠ Error loading SVM: {e}")
        import traceback
        traceback.print_exc()
    
    # Get TextCNN results
    try:
        textcnn_model_path = script_dir / "textcnn" / "models" / "textcnn_best_model.pth"
        if not textcnn_model_path.exists():
            textcnn_model_path = script_dir / "textcnn" / "models" / "textcnn_model.pth"
        textcnn_vocab_path = script_dir / "textcnn" / "models" / "vocab.pkl"
        textcnn_label_encoder_path = script_dir / "textcnn" / "models" / "label_encoder_textcnn.pkl"
        
        if all(p.exists() for p in [textcnn_model_path, textcnn_vocab_path, textcnn_label_encoder_path]):
            textcnn_results = evaluate_textcnn_model(
                str(textcnn_model_path), str(textcnn_vocab_path), str(textcnn_label_encoder_path), DATA_PATH
            )
            results['TextCNN'] = {k: v for k, v in textcnn_results.items() if k not in ['confusion_matrix', 'y_true', 'y_pred']}
            confusion_matrices['TextCNN'] = textcnn_results['confusion_matrix']
            f1_per_class_dict['TextCNN'] = textcnn_results['f1_per_class']
            auc_str = f", AUC={textcnn_results['auc']:.4f}" if textcnn_results.get('auc') else ""
            print(f"✓ TextCNN: Accuracy={textcnn_results['accuracy']:.4f}, "
                  f"F1-Macro={textcnn_results['macro_f1']:.4f}{auc_str}")
        else:
            print("⚠ TextCNN model not found, skipping...")
    except Exception as e:
        print(f"⚠ Error loading TextCNN: {e}")
        import traceback
        traceback.print_exc()
    
    # Get HateBERT results
    try:
        hatebert_results_path = script_dir / "hatebert" / "outputs" / "results.txt"
        hatebert_model_dir = script_dir / "hatebert" / "models" / "hatebert"
        
        if hatebert_results_path.exists() and hatebert_model_dir.exists():
            # Load basic results from file
            hatebert_results = load_hatebert_results(str(hatebert_results_path))
            
            # Evaluate model to get AUC-ROC and other metrics
            try:
                hatebert_eval = evaluate_hatebert_model(str(hatebert_model_dir), DATA_PATH)
                # Update with computed metrics
                hatebert_results['auc'] = hatebert_eval['auc']
                hatebert_results['f1_per_class'] = hatebert_eval['f1_per_class']
                
                # Use computed confusion matrix
                confusion_matrices['HateBERT'] = hatebert_eval['confusion_matrix']
                f1_per_class_dict['HateBERT'] = hatebert_eval['f1_per_class']
                
                auc_str = f", AUC={hatebert_eval['auc']:.4f}" if hatebert_eval.get('auc') else ""
                print(f"✓ HateBERT: Accuracy={hatebert_results['accuracy']:.4f}, "
                      f"F1-Macro={hatebert_results['macro_f1']:.4f}{auc_str}")
            except Exception as e:
                print(f"⚠ Could not evaluate HateBERT model for AUC: {e}")
                # Fall back to results file only
                hatebert_cm = get_hatebert_confusion_matrix(str(hatebert_results_path))
                if hatebert_cm is not None:
                    confusion_matrices['HateBERT'] = hatebert_cm
                
                hatebert_f1 = get_hatebert_f1_scores(str(hatebert_results_path))
                if hatebert_f1 is not None:
                    f1_per_class_dict['HateBERT'] = hatebert_f1
                
                print(f"✓ HateBERT: Accuracy={hatebert_results['accuracy']:.4f}, "
                      f"F1-Macro={hatebert_results['macro_f1']:.4f} (AUC not computed)")
            
            results['HateBERT'] = hatebert_results
        else:
            print("⚠ HateBERT results file or model not found, skipping...")
    except Exception as e:
        print(f"⚠ Error loading HateBERT results: {e}")
        import traceback
        traceback.print_exc()
    
    if not results:
        print("\n❌ No results found! Please ensure models are trained and saved.")
        return
    
    print("\n" + "="*60)
    print("Creating Comparison Charts")
    print("="*60)
    
    # Create output directory
    output_dir = script_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Create confusion matrices plot
    if confusion_matrices:
        create_confusion_matrices_plot(
            confusion_matrices,
            save_path=str(output_dir / "confusion_matrices_comparison.png")
        )
    
    # Create combined performance bar chart
    if results:
        create_combined_bar_chart(
            results,
            save_path=str(output_dir / "performance_comparison_combined.png")
        )
    
    # Create class-wise F1 scores plot
    if f1_per_class_dict:
        # Add f1_per_class to results dict for plotting
        for model in results:
            if model in f1_per_class_dict:
                results[model]['f1_per_class'] = f1_per_class_dict[model]
        
        create_classwise_f1_plot(
            results,
            save_path=str(output_dir / "classwise_f1_scores.png")
        )
    
    # Load training curves
    print("\n" + "="*60)
    print("Loading Training Curves")
    print("="*60)
    
    textcnn_training_data = None
    hatebert_training_data = None
    
    # Load TextCNN training history
    try:
        textcnn_history_path = script_dir / "textcnn" / "outputs" / "training_history.json"
        if textcnn_history_path.exists():
            textcnn_training_data = load_textcnn_training_history(str(textcnn_history_path))
            if textcnn_training_data:
                print(f"✓ TextCNN: Loaded {len(textcnn_training_data['epochs'])} epochs of training history")
        else:
            print("⚠ TextCNN training history file not found (would need to re-run training to generate it)")
    except Exception as e:
        print(f"⚠ Error loading TextCNN training data: {e}")
    
    # Load HateBERT training loss
    try:
        hatebert_model_dir = script_dir / "hatebert" / "models" / "hatebert"
        checkpoint_dirs = sorted([d for d in hatebert_model_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint')])
        
        if checkpoint_dirs:
            latest_checkpoint = checkpoint_dirs[-1]
            trainer_state_path = latest_checkpoint / "trainer_state.json"
            if trainer_state_path.exists():
                hatebert_training_data = load_hatebert_training_loss(str(trainer_state_path))
                if hatebert_training_data:
                    print(f"✓ HateBERT: Loaded {len(hatebert_training_data['train_losses'])} training loss values")
        else:
            print("⚠ HateBERT checkpoints not found, skipping training curves...")
    except Exception as e:
        print(f"⚠ Error loading HateBERT training data: {e}")
    
    # Note: TextCNN training history would need to be saved during training
    # For now, we'll create the plot with just HateBERT if available
    
    # Create learning curves plot
    if textcnn_training_data or hatebert_training_data:
        create_learning_curves_plot(
            textcnn_training_data,
            hatebert_training_data,
            save_path=str(output_dir / "learning_curves.png")
        )
    else:
        print("⚠ No training curve data available")
    
    # Print summary table
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    print(f"{'Model':<20} {'Accuracy':<12} {'F1-Macro':<12} {'AUC-ROC':<12}")
    print("-" * 60)
    for model, metrics in results.items():
        auc_val = metrics.get('auc', None)
        auc_str = f"{auc_val:.4f}" if auc_val is not None else "N/A"
        print(f"{model:<20} {metrics['accuracy']:<12.4f} {metrics['macro_f1']:<12.4f} {auc_str:<12}")
    print("="*60)


if __name__ == "__main__":
    main()

