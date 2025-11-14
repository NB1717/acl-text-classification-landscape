"""
Performance Comparison Bar Chart for Task_A Models
Creates bar charts comparing Accuracy, F1-Macro, and AUC-ROC across all three models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import re
from pathlib import Path


def load_deberta_results(results_file):
    """Parse DeBERTa results from results.txt file"""
    print(f"Loading DeBERTa results from {results_file}...")
    
    with open(results_file, 'r') as f:
        content = f.read()
    
    # Extract test set performance with AUC
    test_section = re.search(r'Results on Test set:.*?Accuracy:\s+([\d.]+).*?Macro F1:\s+([\d.]+).*?AUC-ROC:\s+([\d.]+)', 
                             content, re.DOTALL)
    
    if test_section:
        accuracy = float(test_section.group(1))
        macro_f1 = float(test_section.group(2))
        auc = float(test_section.group(3))
        return {'accuracy': accuracy, 'macro_f1': macro_f1, 'auc': auc}
    else:
        # Fallback: try to find in summary section
        summary_match = re.search(r'Test Set Performance:.*?Accuracy:\s+([\d.]+).*?Macro F1:\s+([\d.]+).*?AUC-ROC:\s+([\d.]+)', 
                                  content, re.DOTALL)
        if summary_match:
            accuracy = float(summary_match.group(1))
            macro_f1 = float(summary_match.group(2))
            auc = float(summary_match.group(3))
            return {'accuracy': accuracy, 'macro_f1': macro_f1, 'auc': auc}
    
    raise ValueError("Could not parse DeBERTa results file")


def evaluate_lr_model(model_path, vectorizer_path, data_path):
    """Load and evaluate Logistic Regression model"""
    print(f"Loading Logistic Regression model...")
    
    # Load data
    df = pd.read_csv(data_path)
    df['label_binary'] = (df['label_sexist'] == 'sexist').astype(int)
    test_df = df[df['split'] == 'test'].copy()
    
    # Load model and vectorizer
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Extract features and evaluate
    X_test = vectorizer.transform(test_df['text'])
    y_test = test_df['label_binary'].values
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    macro_f1 = (f1_per_class[0] + f1_per_class[1]) / 2
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = None
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'auc': auc, 'confusion_matrix': cm, 'y_true': y_test, 'y_pred': y_pred}


def evaluate_mlp_model(model_path, vectorizer_path, data_path):
    """Load and evaluate MLP model"""
    print(f"Loading MLP model...")
    
    # Load data
    df = pd.read_csv(data_path)
    df['label_binary'] = (df['label_sexist'] == 'sexist').astype(int)
    test_df = df[df['split'] == 'test'].copy()
    
    # Load model and vectorizer
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Extract features and evaluate
    X_test = vectorizer.transform(test_df['text'])
    X_test = X_test.toarray()  # MLP needs dense arrays
    y_test = test_df['label_binary'].values
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    macro_f1 = (f1_per_class[0] + f1_per_class[1]) / 2
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = None
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'auc': auc, 'confusion_matrix': cm, 'y_true': y_test, 'y_pred': y_pred}


def evaluate_mlp_model_with_cm(model_path, vectorizer_path, data_path):
    """Load and evaluate MLP model, returning confusion matrix"""
    print(f"Loading MLP model...")
    
    # Load data
    df = pd.read_csv(data_path)
    df['label_binary'] = (df['label_sexist'] == 'sexist').astype(int)
    test_df = df[df['split'] == 'test'].copy()
    
    # Load model and vectorizer
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Extract features and evaluate
    X_test = vectorizer.transform(test_df['text'])
    X_test = X_test.toarray()  # MLP needs dense arrays
    y_test = test_df['label_binary'].values
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    macro_f1 = (f1_per_class[0] + f1_per_class[1]) / 2
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = None
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'auc': auc, 'confusion_matrix': cm, 'y_true': y_test, 'y_pred': y_pred}


def get_deberta_confusion_matrix(results_file, data_path):
    """Extract DeBERTa confusion matrix from results.txt"""
    print(f"Loading DeBERTa confusion matrix...")
    
    # Try to parse from results file first
    with open(results_file, 'r') as f:
        content = f.read()
    
    # Look for confusion matrix in results - format:
    # Actual Not Sexist      2806      224
    #       Sexist            245      725
    cm_match = re.search(r'Actual Not Sexist\s+(\d+)\s+(\d+).*?Sexist\s+(\d+)\s+(\d+)', content, re.DOTALL)
    if cm_match:
        tn = int(cm_match.group(1))  # Not Sexist -> Not Sexist
        fp = int(cm_match.group(2))  # Not Sexist -> Sexist
        fn = int(cm_match.group(3))  # Sexist -> Not Sexist
        tp = int(cm_match.group(4))  # Sexist -> Sexist
        cm = np.array([[tn, fp], [fn, tp]])
        return cm
    
    return None


def create_confusion_matrices_plot(confusion_matrices_dict, save_path=None):
    """
    Create side-by-side confusion matrix plot for all models
    
    Args:
        confusion_matrices_dict: Dictionary with model names as keys and confusion matrices as values
        save_path: Path to save the figure
    """
    models = list(confusion_matrices_dict.keys())
    num_models = len(models)
    
    fig, axes = plt.subplots(1, num_models, figsize=(5*num_models, 5))
    
    # If only one model, make axes iterable
    if num_models == 1:
        axes = [axes]
    
    class_names = ['Not Sexist', 'Sexist']
    
    for idx, (model_name, cm) in enumerate(confusion_matrices_dict.items()):
        ax = axes[idx]
        
        # Normalize confusion matrix to percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create custom annotations with count and percentage
        annot = []
        for i in range(len(class_names)):
            row = []
            for j in range(len(class_names)):
                row.append(f'{int(cm[i, j])}\n({cm_percent[i, j]:.1f}%)')
            annot.append(row)
        
        # Create heatmap with custom annotations
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax,
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'}, vmin=0, vmax=cm.max())
        
        ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}\n(Test Set)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrices plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def load_mlp_training_loss(model_path):
    """Load MLP model and extract training loss curve"""
    print(f"Loading MLP training loss...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    if hasattr(model, 'loss_curve_'):
        # MLP loss_curve_ is per iteration, convert to approximate epochs
        # Assuming early stopping validation_fraction=0.15, we can estimate
        # For MLP, iterations might not directly map to epochs, so we'll use iterations
        iterations = np.arange(1, len(model.loss_curve_) + 1)
        losses = model.loss_curve_
        return {'iterations': iterations, 'losses': losses, 'model_name': 'MLP'}
    else:
        return None


def load_deberta_training_loss(trainer_state_path):
    """Load DeBERTa training loss from trainer_state.json"""
    print(f"Loading DeBERTa training loss from {trainer_state_path}...")
    
    import json
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)
    
    epochs = []
    losses = []
    
    # Extract training loss (not eval_loss)
    for entry in trainer_state.get('log_history', []):
        if 'loss' in entry and 'eval_loss' not in entry:
            epochs.append(entry['epoch'])
            losses.append(entry['loss'])
    
    if epochs and losses:
        return {'epochs': np.array(epochs), 'losses': np.array(losses), 'model_name': 'DeBERTa-v3-base'}
    else:
        return None


def load_deberta_validation_metrics(trainer_state_path):
    """Load DeBERTa validation/test loss and accuracy from trainer_state.json"""
    print(f"Loading DeBERTa validation metrics from {trainer_state_path}...")
    
    import json
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)
    
    epochs = []
    eval_losses = []
    eval_accuracies = []
    
    # Extract validation metrics (eval_loss and eval_accuracy)
    for entry in trainer_state.get('log_history', []):
        if 'eval_loss' in entry:
            epochs.append(entry['epoch'])
            eval_losses.append(entry['eval_loss'])
            # eval_accuracy should always be present when eval_loss is present
            eval_accuracies.append(entry.get('eval_accuracy', None))
    
    result = {}
    if epochs and eval_losses:
        result['epochs'] = np.array(epochs)
        result['losses'] = np.array(eval_losses)
        # Filter out None values but keep corresponding epochs
        valid_indices = [i for i, acc in enumerate(eval_accuracies) if acc is not None]
        if valid_indices:
            result['accuracies'] = np.array([eval_accuracies[i] for i in valid_indices])
            result['accuracy_epochs'] = np.array([epochs[i] for i in valid_indices])
        else:
            result['accuracies'] = np.array([])
            result['accuracy_epochs'] = np.array([])
        result['model_name'] = 'DeBERTa-v3-base'
        return result
    else:
        return None


def create_training_curves_plot(mlp_data, deberta_data, save_path=None):
    """
    Create combined training loss curves for MLP and DeBERTa in the same figure
    
    Args:
        mlp_data: Dict with 'iterations', 'losses', 'model_name'
        deberta_data: Dict with 'epochs', 'losses', 'model_name'
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Normalize MLP iterations to 0-1 range for comparison (or use separate y-axis)
    # We'll use a dual-axis approach or normalize both to show relative progress
    
    # Plot MLP training curve
    if mlp_data:
        # Normalize iterations to 0-1 range for visual comparison
        mlp_normalized = mlp_data['iterations'] / mlp_data['iterations'].max()
        ax.plot(mlp_normalized, mlp_data['losses'], 
                linewidth=2, color='#e74c3c', marker='o', markersize=4, 
                label=f'{mlp_data["model_name"]} (Iterations)', alpha=0.8)
    
    # Plot DeBERTa training curve
    if deberta_data:
        # Normalize epochs to 0-1 range for visual comparison
        deberta_normalized = deberta_data['epochs'] / deberta_data['epochs'].max()
        ax.plot(deberta_normalized, deberta_data['losses'], 
                linewidth=2, color='#3498db', marker='o', markersize=4, 
                label=f'{deberta_data["model_name"]} (Epochs)', alpha=0.8)
    
    ax.set_xlabel('Normalized Training Progress (0-1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining curves plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_validation_loss_plot(deberta_data, save_path=None):
    """
    Create validation/test loss plot for DeBERTa
    
    Args:
        deberta_data: Dict with 'epochs', 'losses', 'model_name'
        save_path: Path to save the figure
    """
    if not deberta_data:
        print("⚠ No validation loss data available")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(deberta_data['epochs'], deberta_data['losses'], 
            linewidth=2, color='#3498db', marker='o', markersize=6)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_title(f'{deberta_data["model_name"]} Validation Loss', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nValidation loss plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_validation_accuracy_plot(deberta_data, save_path=None):
    """
    Create validation/test accuracy plot for DeBERTa
    
    Args:
        deberta_data: Dict with 'accuracy_epochs' or 'epochs', 'accuracies', 'model_name'
        save_path: Path to save the figure
    """
    if not deberta_data or len(deberta_data.get('accuracies', [])) == 0:
        print("⚠ No validation accuracy data available")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Use accuracy_epochs if available, otherwise use epochs
    accuracy_epochs = deberta_data.get('accuracy_epochs', deberta_data.get('epochs', []))
    
    ax.plot(accuracy_epochs, deberta_data['accuracies'], 
            linewidth=2, color='#2ecc71', marker='o', markersize=6)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{deberta_data["model_name"]} Validation Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nValidation accuracy plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_comparison_bar_chart(results_dict, save_path=None):
    """
    Create bar chart comparing Accuracy, F1-Macro, and AUC-ROC for all models
    
    Args:
        results_dict: Dictionary with model names as keys and dicts with 'accuracy', 'macro_f1', and 'auc' as values
        save_path: Path to save the figure
    """
    models = list(results_dict.keys())
    accuracies = [results_dict[model]['accuracy'] for model in models]
    macro_f1s = [results_dict[model]['macro_f1'] for model in models]
    aucs = [results_dict[model].get('auc', None) for model in models]
    
    # Filter out None AUC values if any
    valid_aucs = [auc if auc is not None else 0 for auc in aucs]
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Set up color palette
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green
    
    # Plot Accuracy
    bars1 = ax1.bar(models, accuracies, color=colors[:len(models)], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    ax1.tick_params(axis='x', rotation=15)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot F1-Macro
    bars2 = ax2.bar(models, macro_f1s, color=colors[:len(models)], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('F1-Macro Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1-Macro Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1.0])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.tick_params(axis='x', rotation=15)
    
    # Add value labels on bars
    for bar, f1 in zip(bars2, macro_f1s):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot AUC-ROC
    bars3 = ax3.bar(models, valid_aucs, color=colors[:len(models)], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax3.set_title('AUC-ROC Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 1.0])
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)
    ax3.tick_params(axis='x', rotation=15)
    
    # Add value labels on bars
    for bar, auc in zip(bars3, valid_aucs):
        if auc > 0:  # Only label if AUC is valid
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nBar chart saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_combined_bar_chart(results_dict, save_path=None):
    """
    Create a single bar chart with grouped bars for Accuracy, F1-Macro, and AUC-ROC
    
    Args:
        results_dict: Dictionary with model names as keys and dicts with 'accuracy', 'macro_f1', and 'auc' as values
        save_path: Path to save the figure
    """
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
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
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
            if height > 0:  # Only label if value is valid
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nCombined bar chart saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main function to create comparison charts"""
    # Get script directory and set up paths
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
    print("Collecting Performance Metrics for All Models")
    print("="*60)
    
    results = {}
    confusion_matrices = {}
    
    # Get Logistic Regression results
    try:
        lr_model_path = script_dir / "logistic_regression" / "models" / "logistic_regression_model.pkl"
        lr_vectorizer_path = script_dir / "logistic_regression" / "models" / "tfidf_vectorizer.pkl"
        
        if lr_model_path.exists() and lr_vectorizer_path.exists():
            lr_results = evaluate_lr_model(
                str(lr_model_path), str(lr_vectorizer_path), DATA_PATH
            )
            results['Logistic Regression'] = {k: v for k, v in lr_results.items() if k not in ['confusion_matrix', 'y_true', 'y_pred']}
            confusion_matrices['Logistic Regression'] = lr_results['confusion_matrix']
            auc_str = f", AUC={lr_results['auc']:.4f}" if lr_results.get('auc') else ""
            print(f"✓ Logistic Regression: Accuracy={lr_results['accuracy']:.4f}, "
                  f"F1-Macro={lr_results['macro_f1']:.4f}{auc_str}")
        else:
            print("⚠ Logistic Regression model not found, skipping...")
    except Exception as e:
        print(f"⚠ Error loading Logistic Regression: {e}")
    
    # Get MLP results
    try:
        mlp_model_path = script_dir / "mlp" / "models" / "mlp_model.pkl"
        mlp_vectorizer_path = script_dir / "mlp" / "models" / "tfidf_vectorizer_mlp.pkl"
        
        if mlp_model_path.exists() and mlp_vectorizer_path.exists():
            mlp_results = evaluate_mlp_model_with_cm(
                str(mlp_model_path), str(mlp_vectorizer_path), DATA_PATH
            )
            results['MLP'] = {k: v for k, v in mlp_results.items() if k not in ['confusion_matrix', 'y_true', 'y_pred']}
            confusion_matrices['MLP'] = mlp_results['confusion_matrix']
            auc_str = f", AUC={mlp_results['auc']:.4f}" if mlp_results.get('auc') else ""
            print(f"✓ MLP: Accuracy={mlp_results['accuracy']:.4f}, "
                  f"F1-Macro={mlp_results['macro_f1']:.4f}{auc_str}")
        else:
            print("⚠ MLP model not found, skipping...")
    except Exception as e:
        print(f"⚠ Error loading MLP: {e}")
    
    # Get DeBERTa results
    try:
        deberta_results_path = script_dir / "deberta" / "results.txt"
        if deberta_results_path.exists():
            deberta_results = load_deberta_results(str(deberta_results_path))
            results['DeBERTa-v3-base'] = deberta_results
            
            # Try to extract confusion matrix from results file
            deberta_cm = get_deberta_confusion_matrix(str(deberta_results_path), DATA_PATH)
            if deberta_cm is not None:
                confusion_matrices['DeBERTa-v3-base'] = deberta_cm
            
            auc_str = f", AUC={deberta_results['auc']:.4f}" if deberta_results.get('auc') else ""
            print(f"✓ DeBERTa-v3-base: Accuracy={deberta_results['accuracy']:.4f}, "
                  f"F1-Macro={deberta_results['macro_f1']:.4f}{auc_str}")
        else:
            print("⚠ DeBERTa results file not found, skipping...")
    except Exception as e:
        print(f"⚠ Error loading DeBERTa results: {e}")
    
    if not results:
        print("\n❌ No results found! Please ensure models are trained and saved.")
        return
    
    print("\n" + "="*60)
    print("Creating Comparison Charts")
    print("="*60)
    
    # Create output directory
    output_dir = script_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Create side-by-side bar charts
    create_comparison_bar_chart(
        results, 
        save_path=str(output_dir / "performance_comparison.png")
    )
    
    # Create combined grouped bar chart
    create_combined_bar_chart(
        results,
        save_path=str(output_dir / "performance_comparison_combined.png")
    )
    
    # Create confusion matrices plot
    if confusion_matrices:
        create_confusion_matrices_plot(
            confusion_matrices,
            save_path=str(output_dir / "confusion_matrices_comparison.png")
        )
    
    # Create training curves plot
    print("\n" + "="*60)
    print("Loading Training Curves")
    print("="*60)
    
    mlp_training_data = None
    deberta_training_data = None
    
    # Load MLP training loss
    try:
        mlp_model_path = script_dir / "mlp" / "models" / "mlp_model.pkl"
        if mlp_model_path.exists():
            mlp_training_data = load_mlp_training_loss(str(mlp_model_path))
            if mlp_training_data:
                print(f"✓ MLP: Loaded {len(mlp_training_data['losses'])} loss values")
        else:
            print("⚠ MLP model not found, skipping training curve...")
    except Exception as e:
        print(f"⚠ Error loading MLP training loss: {e}")
    
    # Load DeBERTa training loss and validation metrics
    deberta_validation_data = None
    try:
        # Try to find the latest checkpoint
        deberta_model_dir = script_dir / "deberta" / "models" / "deberta"
        checkpoint_dirs = sorted([d for d in deberta_model_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint')])
        
        if checkpoint_dirs:
            latest_checkpoint = checkpoint_dirs[-1]
            trainer_state_path = latest_checkpoint / "trainer_state.json"
            if trainer_state_path.exists():
                deberta_training_data = load_deberta_training_loss(str(trainer_state_path))
                if deberta_training_data:
                    print(f"✓ DeBERTa: Loaded {len(deberta_training_data['losses'])} training loss values across {deberta_training_data['epochs'][-1]:.2f} epochs")
                
                # Also load validation metrics
                deberta_validation_data = load_deberta_validation_metrics(str(trainer_state_path))
                if deberta_validation_data:
                    print(f"✓ DeBERTa: Loaded {len(deberta_validation_data['losses'])} validation metrics across {len(deberta_validation_data['epochs'])} evaluation points")
            else:
                print("⚠ DeBERTa trainer_state.json not found, skipping training curves...")
        else:
            print("⚠ DeBERTa checkpoints not found, skipping training curves...")
    except Exception as e:
        print(f"⚠ Error loading DeBERTa training data: {e}")
    
    # Create training curves plot if we have data
    if mlp_training_data or deberta_training_data:
        create_training_curves_plot(
            mlp_training_data,
            deberta_training_data,
            save_path=str(output_dir / "training_curves.png")
        )
    else:
        print("⚠ No training curve data available")
    
    # Create validation loss plot
    if deberta_validation_data:
        create_validation_loss_plot(
            deberta_validation_data,
            save_path=str(output_dir / "validation_loss.png")
        )
    
    # Create validation accuracy plot
    if deberta_validation_data:
        create_validation_accuracy_plot(
            deberta_validation_data,
            save_path=str(output_dir / "validation_accuracy.png")
        )
    
    # Print summary table
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    print(f"{'Model':<25} {'Accuracy':<12} {'F1-Macro':<12} {'AUC-ROC':<12}")
    print("-" * 75)
    for model, metrics in results.items():
        auc_val = metrics.get('auc', None)
        auc_str = f"{auc_val:.4f}" if auc_val is not None else "N/A"
        print(f"{model:<25} {metrics['accuracy']:<12.4f} {metrics['macro_f1']:<12.4f} {auc_str:<12}")
    print("="*75)


if __name__ == "__main__":
    main()

