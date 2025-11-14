"""
Support Vector Machine (SVM) Model for Multi-Class Classification
Task B: Classify sexist content into categories (none, threats, derogation, animosity, prejudiced discussions)
Using aggregated EDOS dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


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
    # Map category names to numeric labels
    category_mapping = {
        'none': 0,
        '1. threats, plans to harm and incitement': 1,
        '2. derogation': 2,
        '3. animosity': 3,
        '4. prejudiced discussions': 4
    }
    
    df['label_category_encoded'] = df['label_category'].map(category_mapping)
    
    # Check for any unmapped categories
    if df['label_category_encoded'].isna().any():
        unmapped = df[df['label_category_encoded'].isna()]['label_category'].unique()
        print(f"Warning: Unmapped categories found: {unmapped}")
        df = df.dropna(subset=['label_category_encoded'])
    
    df['label_category_encoded'] = df['label_category_encoded'].astype(int)
    
    print(f"\nEncoded label distribution:")
    print(df['label_category_encoded'].value_counts().sort_index())
    
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


def extract_features(train_texts, dev_texts, test_texts, max_features=5000, ngram_range=(1, 2)):
    """
    Extract TF-IDF features from text
    """
    print(f"\nExtracting TF-IDF features (max_features={max_features}, ngram_range={ngram_range})...")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        lowercase=True,
        strip_accents='unicode',
        min_df=2,
        max_df=0.95
    )
    
    # Fit on training data only
    X_train = vectorizer.fit_transform(train_texts)
    X_dev = vectorizer.transform(dev_texts)
    X_test = vectorizer.transform(test_texts)
    
    print(f"Feature matrix shape - Train: {X_train.shape}, Dev: {X_dev.shape}, Test: {X_test.shape}")
    
    return X_train, X_dev, X_test, vectorizer


def train_model(X_train, y_train, C=1.0, kernel='rbf', gamma='scale', 
                class_weight='balanced', custom_class_weight=None, random_state=42):
    """
    Train SVM classifier for multi-class classification
    """
    print(f"\nTraining SVM classifier...")
    print(f"  C (regularization): {C}")
    print(f"  Kernel: {kernel}")
    print(f"  Gamma: {gamma}")
    print(f"  Class weight: {class_weight}")
    
    # Use OneVsRest for multi-class (or SVC supports multi-class directly)
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        class_weight=class_weight,
        random_state=random_state,
        probability=True,  # Enable probability estimates for ROC curves
        decision_function_shape='ovo'  # One-vs-One for multi-class
    )
    
    model.fit(X_train, y_train)
    print("Training complete!")
    
    return model


def evaluate_model(model, X, y, class_names, split_name="", return_predictions=False):
    """
    Evaluate model and return metrics for multi-class classification
    """
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    accuracy = accuracy_score(y, y_pred)
    
    # Calculate metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y, y_pred, average=None, zero_division=0
    )
    
    print(f"\n{'='*60}")
    print(f"Results on {split_name} set:")
    print(f"{'='*60}")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall:     {recall_macro:.4f}")
    print(f"Macro F1:         {f1_macro:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall:     {recall_weighted:.4f}")
    print(f"Weighted F1:         {f1_weighted:.4f}")
    
    print(f"\nPer-class F1-Scores:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:35s}: {f1_per_class[i]:.4f} (support: {support_per_class[i]:5d})")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y, y_pred, target_names=class_names))
    
    results = {
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
        'confusion_matrix': cm
    }
    
    if return_predictions:
        results['predictions'] = y_pred
        results['probabilities'] = y_pred_proba
    
    return results


def plot_confusion_matrix(y_true, y_pred, class_names, split_name="", save_path=None):
    """Plot confusion matrix with percentages"""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Shorten class names for display
    short_names = []
    for name in class_names:
        if len(name) > 20:
            short_names.append(name[:17] + "...")
        else:
            short_names.append(name)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Absolute values
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=short_names, yticklabels=short_names)
    ax1.set_xlabel('Predicted', fontsize=11)
    ax1.set_ylabel('Actual', fontsize=11)
    ax1.set_title(f'Confusion Matrix (Counts) - {split_name}', fontsize=12, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    # Percentages
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
    
    # Bar plot
    plt.figure(figsize=(14, 7))
    df_pivot = df_plot.pivot(index='Split', columns='Class', values='F1-Score')
    
    # Create color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    ax = df_pivot.plot(kind='bar', width=0.8, color=colors, figsize=(14, 7))
    
    plt.title('F1-Score by Class and Split', fontsize=14, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=12)
    plt.xlabel('Data Split', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Class', title_fontsize=10, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim([0, 1.1])
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=7)
    
    # Add macro F1 scores
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
        unique, counts = np.unique(y, return_counts=True)
        axes[idx].bar(range(len(unique)), counts, color=plt.cm.Set3(np.linspace(0, 1, len(unique))))
        axes[idx].set_xticks(range(len(unique)))
        axes[idx].set_xticklabels([class_names[i] if i < len(class_names) else str(i) for i in unique], 
                                 rotation=45, ha='right', fontsize=8)
        axes[idx].set_ylabel('Count', fontsize=10)
        axes[idx].set_title(f'{split_name} Set', fontsize=12, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for i, (u, c) in enumerate(zip(unique, counts)):
            axes[idx].text(i, c, str(c), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    plt.close()


def save_model(model, vectorizer, label_encoder, save_dir="models"):
    """Save trained model, vectorizer, and label encoder"""
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, "svm_model.pkl")
    vectorizer_path = os.path.join(save_dir, "tfidf_vectorizer_svm.pkl")
    encoder_path = os.path.join(save_dir, "label_encoder.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\nModel saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")
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
    
    # SVM Configuration
    MAX_FEATURES = 5000
    NGRAM_RANGE = (1, 2)
    C = 1.0  # Regularization parameter
    KERNEL = 'rbf'  # 'linear', 'poly', 'rbf', 'sigmoid'
    GAMMA = 'scale'  # 'scale', 'auto', or float value
    CLASS_WEIGHT = 'balanced'  # Handle class imbalance
    
    OUTPUT_DIR = "outputs"
    
    # Class names for display
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
    
    # Extract features
    X_train, X_dev, X_test, vectorizer = extract_features(
        train_df['text'],
        dev_df['text'],
        test_df['text'],
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE
    )
    
    y_train = train_df['label_category_encoded'].values
    y_dev = dev_df['label_category_encoded'].values
    y_test = test_df['label_category_encoded'].values
    
    # Create label encoder for reference
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    
    # Plot class distribution
    plot_class_distribution(y_train, y_dev, y_test, CLASS_NAMES,
                            save_path=os.path.join(OUTPUT_DIR, "class_distribution.png"))
    
    # Train model
    model = train_model(
        X_train, y_train,
        C=C,
        kernel=KERNEL,
        gamma=GAMMA,
        class_weight=CLASS_WEIGHT
    )
    
    # Evaluate on all splits
    train_results = evaluate_model(model, X_train, y_train, CLASS_NAMES, "Train", return_predictions=True)
    dev_results = evaluate_model(model, X_dev, y_dev, CLASS_NAMES, "Dev", return_predictions=True)
    test_results = evaluate_model(model, X_test, y_test, CLASS_NAMES, "Test", return_predictions=True)
    
    # Plot confusion matrices
    plot_confusion_matrix(y_train, train_results['predictions'], CLASS_NAMES, "Train",
                         save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_train.png"))
    plot_confusion_matrix(y_dev, dev_results['predictions'], CLASS_NAMES, "Dev",
                         save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_dev.png"))
    plot_confusion_matrix(y_test, test_results['predictions'], CLASS_NAMES, "Test",
                         save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_test.png"))
    
    # Plot F1 scores
    all_results = {
        'train': train_results,
        'dev': dev_results,
        'test': test_results
    }
    plot_f1_scores(all_results, CLASS_NAMES, save_path=os.path.join(OUTPUT_DIR, "f1_scores.png"))
    
    # Save model
    save_model(model, vectorizer, label_encoder, save_dir="models")
    
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

