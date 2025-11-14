"""
Multi-Layer Perceptron (MLP) Model for Binary Classification: Sexist vs Not Sexist
Using aggregated EDOS dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
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
import pickle
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


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


def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into train/val/test sets
    Uses the existing split column if available, otherwise creates new splits
    """
    # Use existing splits from the dataset
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
    
    # Convert sparse matrices to dense for MLP (if memory allows)
    # If memory is an issue, you can keep sparse matrices
    print("Converting sparse matrices to dense arrays for MLP...")
    X_train = X_train.toarray()
    X_dev = X_dev.toarray()
    X_test = X_test.toarray()
    
    print(f"Feature matrix shape - Train: {X_train.shape}, Dev: {X_dev.shape}, Test: {X_test.shape}")
    
    return X_train, X_dev, X_test, vectorizer


def train_model(X_train, y_train, hidden_layer_sizes=(100,), activation='relu', 
                solver='adam', alpha=0.0001, learning_rate='constant', 
                learning_rate_init=0.001, max_iter=500, random_state=42, 
                early_stopping=False, validation_fraction=0.1, n_iter_no_change=10):
    """
    Train MLP classifier
    """
    print(f"\nTraining MLP classifier...")
    print(f"  Architecture: {hidden_layer_sizes}")
    print(f"  Activation: {activation}")
    print(f"  Solver: {solver}")
    print(f"  Alpha (L2 regularization): {alpha}")
    print(f"  Learning rate: {learning_rate_init}")
    print(f"  Max iterations: {max_iter}")
    if early_stopping:
        print(f"  Early stopping: Enabled (patience={n_iter_no_change})")
    
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        batch_size='auto',
        shuffle=True,
        verbose=False,
        warm_start=False
    )
    
    model.fit(X_train, y_train)
    
    print(f"Training complete!")
    print(f"  Final loss: {model.loss_:.6f}")
    print(f"  Iterations: {model.n_iter_}")
    
    return model


def evaluate_model(model, X, y, split_name="", return_predictions=False):
    """
    Evaluate model and return metrics
    """
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, average='binary', zero_division=0
    )
    
    # Calculate per-class metrics and macro F1
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y, y_pred, average=None, zero_division=0
    )
    macro_f1 = (f1_per_class[0] + f1_per_class[1]) / 2
    # Store support_per_class in results for JSON export
    
    try:
        auc_score = roc_auc_score(y, y_pred_proba)
    except ValueError:
        auc_score = None
    
    print(f"\n{'='*60}")
    print(f"Results on {split_name} set:")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score (Binary):  {f1:.4f}")
    print(f"Macro F1-Score:     {macro_f1:.4f}")
    print(f"\nPer-class F1-Scores:")
    print(f"  Not Sexist: {f1_per_class[0]:.4f}")
    print(f"  Sexist:     {f1_per_class[1]:.4f}")
    if auc_score:
        print(f"AUC-ROC:   {auc_score:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print(f"\nDetailed Classification Report:")
    print(classification_report(y, y_pred, target_names=['Not Sexist', 'Sexist']))
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_f1': macro_f1,
        'f1_per_class': f1_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'support_per_class': support_per_class,
        'auc': auc_score,
        'confusion_matrix': confusion_matrix(y, y_pred)
    }
    
    if return_predictions:
        results['predictions'] = y_pred
        results['probabilities'] = y_pred_proba
    
    return results


def plot_roc_curve(y_true, y_pred_proba, split_name="", save_path=None):
    """Plot ROC curve"""
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {split_name}')
        plt.legend()
        plt.grid(alpha=0.3)
        
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
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title(f'Confusion Matrix (Counts) - {split_name}')
    
    # Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2,
                xticklabels=['Not Sexist', 'Sexist'],
                yticklabels=['Not Sexist', 'Sexist'])
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title(f'Confusion Matrix (Percentages) - {split_name}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_f1_boxplot(all_results, save_path=None):
    """
    Plot box plots of F1 scores per class across different splits
    all_results: dict with keys like 'train', 'dev', 'test' containing results dicts
    """
    # Prepare data for box plot
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
    
    # Create DataFrame for easier plotting
    df_plot = pd.DataFrame({
        'Split': splits,
        'Class': classes,
        'F1-Score': f1_scores
    })
    
    # Create box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_plot, x='Split', y='F1-Score', hue='Class', palette='Set2')
    plt.title('F1-Score Distribution by Class and Split', fontsize=14, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=12)
    plt.xlabel('Data Split', fontsize=12)
    plt.legend(title='Class', title_fontsize=11, fontsize=10)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim([0, 1.1])
    
    # Add macro F1 scores as text
    for split_name, results in all_results.items():
        if 'macro_f1' in results:
            macro_f1 = results['macro_f1']
            split_idx = list(all_results.keys()).index(split_name)
            plt.text(split_idx, 1.05, f'Macro F1: {macro_f1:.3f}', 
                    ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"F1 box plot saved to {save_path}")
    plt.close()
    
    # Also create a bar plot showing F1 scores per class per split (alternative visualization)
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
        bar_plot_path = save_path.replace('.png', '_barplot.png')
        plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
        print(f"F1 bar plot saved to {save_path.replace('.png', '_barplot.png')}")
    plt.close()


def plot_training_loss(model, save_path=None):
    """Plot training loss curve if available"""
    if hasattr(model, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_curve_, linewidth=2)
        plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training loss curve saved to {save_path}")
        plt.close()
    else:
        print("Loss curve not available (early stopping may not be enabled)")


def save_metrics(all_results, class_names, save_path=None):
    """Save all metrics to a JSON file (for binary classification)"""
    if save_path is None:
        save_path = "outputs/metrics.json"
    
    # Convert numpy arrays to lists for JSON serialization
    metrics_dict = {}
    
    for split_name, results in all_results.items():
        metrics_dict[split_name] = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1': float(results['f1']),
            'macro_f1': float(results['macro_f1']),
            'auc': float(results['auc']) if results['auc'] is not None else None,
            'per_class_metrics': {}
        }
        
        # Add per-class metrics (binary: 0=Not Sexist, 1=Sexist)
        for i, class_name in enumerate(class_names):
            metrics_dict[split_name]['per_class_metrics'][class_name] = {
                'precision': float(results['precision_per_class'][i]),
                'recall': float(results['recall_per_class'][i]),
                'f1': float(results['f1_per_class'][i]),
                'support': int(results.get('support_per_class', [0, 0])[i])
            }
        
        # Add confusion matrix (convert numpy array to list)
        if 'confusion_matrix' in results:
            metrics_dict[split_name]['confusion_matrix'] = results['confusion_matrix'].tolist()
    
    # Save to JSON file
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"\nMetrics saved to {save_path}")


def save_model(model, vectorizer, save_dir="models"):
    """Save trained model and vectorizer"""
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, "mlp_model.pkl")
    vectorizer_path = os.path.join(save_dir, "tfidf_vectorizer_mlp.pkl")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"\nModel saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")


def main():
    """Main function to run the complete pipeline"""
    # Configuration
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    # Try multiple possible data paths
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
    
    # MLP Configuration - Regularized to reduce overfitting
    MAX_FEATURES = 5000  # Reduced from 10000: fewer features reduce overfitting risk
    NGRAM_RANGE = (1, 2)  # Keep bigrams for context
    
    # Network architecture: Slightly smaller to reduce overfitting
    HIDDEN_LAYER_SIZES = (100, 50) 
    ACTIVATION = 'relu'  # 'identity', 'logistic', 'tanh', 'relu'
    SOLVER = 'adam'  # 'lbfgs', 'sgd', 'adam'
    ALPHA = 0.01  # Increased from 0.0001: stronger L2 regularization
    LEARNING_RATE = 'constant'  # 'constant', 'invscaling', 'adaptive'
    LEARNING_RATE_INIT = 0.0005  # Reduced from 0.001: slower, more stable learning
    MAX_ITER = 500
    EARLY_STOPPING = True
    VALIDATION_FRACTION = 0.15  # Increased from 0.1: more validation data for early stopping
    N_ITER_NO_CHANGE = 15  # Increased from 10: more patience, better generalization
    
    OUTPUT_DIR = "outputs"
    
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
    
    y_train = train_df['label_binary'].values
    y_dev = dev_df['label_binary'].values
    y_test = test_df['label_binary'].values
    
    # Train model
    model = train_model(
        X_train, y_train,
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation=ACTIVATION,
        solver=SOLVER,
        alpha=ALPHA,
        learning_rate=LEARNING_RATE,
        learning_rate_init=LEARNING_RATE_INIT,
        max_iter=MAX_ITER,
        early_stopping=EARLY_STOPPING,
        validation_fraction=VALIDATION_FRACTION,
        n_iter_no_change=N_ITER_NO_CHANGE
    )
    
    # Evaluate on all splits
    train_results = evaluate_model(model, X_train, y_train, "Train", return_predictions=True)
    dev_results = evaluate_model(model, X_dev, y_dev, "Dev", return_predictions=True)
    test_results = evaluate_model(model, X_test, y_test, "Test", return_predictions=True)
    
    # Plot ROC curves
    plot_roc_curve(y_dev, dev_results['probabilities'], "Dev", 
                   save_path=os.path.join(OUTPUT_DIR, "roc_curve_dev.png"))
    plot_roc_curve(y_test, test_results['probabilities'], "Test",
                   save_path=os.path.join(OUTPUT_DIR, "roc_curve_test.png"))
    
    # Plot confusion matrices for all splits
    plot_confusion_matrix(y_train, train_results['predictions'], "Train",
                         save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_train.png"))
    plot_confusion_matrix(y_dev, dev_results['predictions'], "Dev",
                         save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_dev.png"))
    plot_confusion_matrix(y_test, test_results['predictions'], "Test",
                         save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_test.png"))
    
    # Plot F1 box plots
    all_results = {
        'train': train_results,
        'dev': dev_results,
        'test': test_results
    }
    plot_f1_boxplot(all_results, save_path=os.path.join(OUTPUT_DIR, "f1_boxplot.png"))
    
    # Save metrics to JSON
    class_names = ['Not Sexist', 'Sexist']
    save_metrics(all_results, class_names, save_path=os.path.join(OUTPUT_DIR, "metrics.json"))
    
    # Plot training loss curve
    plot_training_loss(model, save_path=os.path.join(OUTPUT_DIR, "training_loss.png"))
    
    # Save model
    save_model(model, vectorizer, save_dir="models")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Test Set Performance:")
    print(f"  Accuracy:    {test_results['accuracy']:.4f}")
    print(f"  Precision:   {test_results['precision']:.4f}")
    print(f"  Recall:      {test_results['recall']:.4f}")
    print(f"  F1-Score:    {test_results['f1']:.4f}")
    print(f"  Macro F1:    {test_results['macro_f1']:.4f}")
    print(f"  F1 (Not Sexist): {test_results['f1_per_class'][0]:.4f}")
    print(f"  F1 (Sexist):     {test_results['f1_per_class'][1]:.4f}")
    if test_results['auc']:
        print(f"  AUC-ROC:     {test_results['auc']:.4f}")
    print("\nMacro F1 across splits:")
    print(f"  Train: {train_results['macro_f1']:.4f}")
    print(f"  Dev:   {dev_results['macro_f1']:.4f}")
    print(f"  Test:  {test_results['macro_f1']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

