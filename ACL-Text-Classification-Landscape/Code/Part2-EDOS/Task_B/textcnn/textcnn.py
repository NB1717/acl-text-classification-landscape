"""
TextCNN (Convolutional Neural Network) Model for Multi-Class Classification
Task B: Classify sexist content into categories (none, threats, derogation, animosity, prejudiced discussions)
Using aggregated EDOS dataset
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class TextDataset(Dataset):
    """Dataset class for text data"""
    def __init__(self, texts, labels, vocab, max_length=512):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        # Tokenize and convert to indices
        tokens = self.tokenize(text)
        indices = [self.vocab.get(token, self.vocab.get('<UNK>', 0)) for token in tokens]
        
        # Pad or truncate to max_length
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices = indices + [self.vocab.get('<PAD>', 0)] * (self.max_length - len(indices))
        
        return {
            'text': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def tokenize(self, text):
        """Simple tokenization: lowercase and split on whitespace"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()


class TextCNN(nn.Module):
    """
    TextCNN model for text classification
    Uses multiple convolutional filters of different sizes
    """
    def __init__(self, vocab_size, embed_dim=128, num_filters=100, 
                 filter_sizes=[3, 4, 5], num_classes=5, dropout=0.5):
        super(TextCNN, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Convolutional layers for different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, 
                     kernel_size=fs, padding=1)
            for fs in filter_sizes
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        # Embedding: (batch_size, seq_length, embed_dim)
        x = self.embedding(x)
        
        # Conv1d expects (batch_size, embed_dim, seq_length)
        x = x.permute(0, 2, 1)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))
            # Max pooling over time: (batch_size, num_filters)
            pooled = torch.max_pool1d(conv_out, kernel_size=conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # Concatenate all filter outputs
        x = torch.cat(conv_outputs, dim=1)
        
        # Dropout
        x = self.dropout(x)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x


def build_vocab(texts, min_freq=2, max_vocab_size=10000):
    """Build vocabulary from texts"""
    print("Building vocabulary...")
    
    # Tokenize all texts
    all_tokens = []
    for text in texts:
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        all_tokens.extend(tokens)
    
    # Count tokens
    token_counts = Counter(all_tokens)
    
    # Build vocab with special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    # Add tokens that meet frequency threshold
    for token, count in token_counts.most_common(max_vocab_size - 2):
        if count >= min_freq:
            vocab[token] = len(vocab)
            if len(vocab) >= max_vocab_size:
                break
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Most common tokens: {list(vocab.keys())[:20]}")
    
    return vocab


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


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        texts = batch['text'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate_model(model, dataloader, criterion, device, class_names, split_name=""):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    print(f"\n{'='*60}")
    print(f"Results on {split_name} set:")
    print(f"{'='*60}")
    print(f"Loss:            {avg_loss:.4f}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision:  {precision_macro:.4f}")
    print(f"Macro Recall:    {recall_macro:.4f}")
    print(f"Macro F1:        {f1_macro:.4f}")
    print(f"Weighted F1:     {f1_weighted:.4f}")
    
    print(f"\nPer-class F1-Scores:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:35s}: {f1_per_class[i]:.4f} (support: {support_per_class[i]:5d})")
    
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'support_per_class': support_per_class,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }
    
    return results


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


def plot_training_history(train_losses, train_accs, dev_losses, dev_accs, save_path=None):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, dev_losses, 'r-', label='Dev Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3, linestyle='--')
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, dev_accs, 'r-', label='Dev Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
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


def save_model(model, vocab, label_encoder, save_dir="models"):
    """Save trained model, vocabulary, and label encoder"""
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, "textcnn_model.pth")
    vocab_path = os.path.join(save_dir, "vocab.pkl")
    encoder_path = os.path.join(save_dir, "label_encoder_textcnn.pkl")
    
    torch.save(model.state_dict(), model_path)
    
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\nModel saved to {model_path}")
    print(f"Vocabulary saved to {vocab_path}")
    print(f"Label encoder saved to {encoder_path}")


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
    
    # TextCNN Configuration
    MAX_LENGTH = 256  # Maximum sequence length
    VOCAB_SIZE = 10000  # Maximum vocabulary size
    MIN_FREQ = 2  # Minimum token frequency
    EMBED_DIM = 128  # Embedding dimension
    NUM_FILTERS = 100  # Number of filters per filter size
    FILTER_SIZES = [3, 4, 5]  # Convolution filter sizes (n-grams)
    NUM_CLASSES = 5
    DROPOUT = 0.5
    
    # Training Configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    
    OUTPUT_DIR = "outputs"
    
    # Class names
    CLASS_NAMES = [
        'none',
        '1. threats, plans to harm and incitement',
        '2. derogation',
        '3. animosity',
        '4. prejudiced discussions'
    ]
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and prepare data
    df = load_data(DATA_PATH)
    df = prepare_labels(df)
    
    # Split data
    train_df, dev_df, test_df = split_data(df)
    
    # Build vocabulary from training data
    vocab = build_vocab(train_df['text'], min_freq=MIN_FREQ, max_vocab_size=VOCAB_SIZE)
    vocab_size = len(vocab)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TextDataset(train_df['text'], train_df['label_category_encoded'], vocab, MAX_LENGTH)
    dev_dataset = TextDataset(dev_df['text'], dev_df['label_category_encoded'], vocab, MAX_LENGTH)
    test_dataset = TextDataset(test_df['text'], test_df['label_category_encoded'], vocab, MAX_LENGTH)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['label_category_encoded'].values)
    
    # Plot class distribution
    plot_class_distribution(train_df['label_category_encoded'], 
                           dev_df['label_category_encoded'],
                           test_df['label_category_encoded'],
                           CLASS_NAMES,
                           save_path=os.path.join(OUTPUT_DIR, "class_distribution.png"))
    
    # Create model
    print(f"\nCreating TextCNN model...")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dim: {EMBED_DIM}")
    print(f"  Filter sizes: {FILTER_SIZES}")
    print(f"  Num filters: {NUM_FILTERS}")
    print(f"  Max length: {MAX_LENGTH}")
    
    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_filters=NUM_FILTERS,
        filter_sizes=FILTER_SIZES,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    
    train_losses = []
    train_accs = []
    dev_losses = []
    dev_accs = []
    best_dev_f1 = 0
    patience = 3
    no_improve = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        # Evaluate on dev
        dev_results = evaluate_model(model, dev_loader, criterion, device, CLASS_NAMES, "Dev")
        dev_losses.append(dev_results['loss'])
        dev_accs.append(dev_results['accuracy'])
        
        # Early stopping check
        if dev_results['f1_macro'] > best_dev_f1:
            best_dev_f1 = dev_results['f1_macro']
            no_improve = 0
            # Save best model
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), os.path.join('models', 'textcnn_best_model.pth'))
            print(f"✓ New best model! Dev Macro F1: {best_dev_f1:.4f}")
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            # Load best model
            model.load_state_dict(torch.load(os.path.join('models', 'textcnn_best_model.pth')))
            break
    
    # Plot training history
    plot_training_history(train_losses, train_accs, dev_losses, dev_accs,
                         save_path=os.path.join(OUTPUT_DIR, "training_history.png"))
    
    # Final evaluation on all splits
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    
    train_results = evaluate_model(model, train_loader, criterion, device, CLASS_NAMES, "Train")
    dev_results = evaluate_model(model, dev_loader, criterion, device, CLASS_NAMES, "Dev")
    test_results = evaluate_model(model, test_loader, criterion, device, CLASS_NAMES, "Test")
    
    # Plot confusion matrices
    plot_confusion_matrix(train_results['labels'], train_results['predictions'], CLASS_NAMES, "Train",
                         save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_train.png"))
    plot_confusion_matrix(dev_results['labels'], dev_results['predictions'], CLASS_NAMES, "Dev",
                         save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_dev.png"))
    plot_confusion_matrix(test_results['labels'], test_results['predictions'], CLASS_NAMES, "Test",
                         save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_test.png"))
    
    # Plot F1 scores
    all_results = {
        'train': train_results,
        'dev': dev_results,
        'test': test_results
    }
    plot_f1_scores(all_results, CLASS_NAMES, save_path=os.path.join(OUTPUT_DIR, "f1_scores.png"))
    
    # Save model
    save_model(model, vocab, label_encoder, save_dir="models")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Test Set Performance:")
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

