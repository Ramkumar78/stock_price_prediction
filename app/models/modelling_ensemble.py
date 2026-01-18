"""
Ensemble Model Training for Stock Price Prediction

This script creates a soft voting ensemble combining XGBoost, LightGBM, and CatBoost
classifiers to predict whether SPY will go up or down over the next 3 days.

Ensemble Strategy:
-----------------
- Uses the top 40 features selected from permutation importance analysis
- Combines predictions from all three models using soft voting (probability averaging)
- Each model uses its optimized hyperparameters from individual training
- Evaluates ensemble performance against individual models

Models in Ensemble:
------------------
1. XGBoost: Level-wise tree growth with aggressive regularization
2. LightGBM: Leaf-wise tree growth with default parameters
3. CatBoost: Ordered boosting with symmetric trees

Expected Benefit:
----------------
Ensemble models typically reduce variance and improve generalization by combining
diverse base learners. Each algorithm has different biases and may capture
different patterns in the data.

Output Files (saved to output/models/ensemble/):
----------------------------------------------
- ensemble_model.pkl - Trained ensemble model
- metrics.json - Performance metrics
- confusion_matrix.png - Confusion matrix visualization
- model_comparison.png - Individual vs ensemble performance

Usage:
------
    python modelling_ensemble.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import joblib
from app.core.ensemble import SoftVotingEnsemble

# Configuration
INPUT_FILE = 'output/spy_features_full.csv'
OUTPUT_DIR = 'output/models/ensemble'
TRAIN_END_DATE = '2024-12-31'
TEST_START_DATE = '2025-01-01'
RANDOM_STATE = 42

# Load selected features from individual models
SELECTED_FEATURES_FILE = 'output/models/lightgbm/selected_features.txt'

# Model hyperparameters (from individual training)
XGBOOST_PARAMS = {
    'max_depth': 3,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'reg_alpha': 1.0,
    'reg_lambda': 3.0,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'eval_metric': 'auc',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'use_label_encoder': False
}

LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1
}

CATBOOST_PARAMS = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'depth': 6,
    'iterations': 1000,
    'learning_rate': 0.03,
    'random_seed': RANDOM_STATE,
    'verbose': False,
    'thread_count': -1
}


def load_selected_features(filepath):
    """Load the selected features from file."""
    print(f"Loading selected features from {filepath}...")
    with open(filepath, 'r') as f:
        features = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(features)} selected features")
    return features


def load_and_prepare_data(filepath, selected_features):
    """
    Load features and target from CSV file.
    
    Returns:
        X (pd.DataFrame): Features (selected subset)
        y (pd.Series): Target variable
        dates (pd.DatetimeIndex): Date index
    """
    print("\nLoading data...")
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Normalize index to date-only format
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Separate features and target
    y = df['target']
    X = df[selected_features]  # Use only selected features
    
    print(f"Using {X.shape[1]} selected features")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, df.index


def split_train_test(X, y, dates, train_end_date, test_start_date):
    """Split data into train and test sets based on date."""
    print(f"\nSplitting data...")
    print(f"Train: up to {train_end_date}")
    print(f"Test: from {test_start_date}")
    
    train_mask = dates <= train_end_date
    test_mask = dates >= test_start_date
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"\nTrain set: {len(X_train)} samples ({dates[train_mask][0]} to {dates[train_mask][-1]})")
    print(f"Test set: {len(X_test)} samples ({dates[test_mask][0]} to {dates[test_mask][-1]})")
    print(f"Train target distribution: {y_train.value_counts().to_dict()}")
    print(f"Test target distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def handle_missing_values(X_train, X_test, y_train, y_test):
    """Handle NaN values in features."""
    print("\nHandling missing values...")
    print(f"Original train shape: {X_train.shape}")
    print(f"Original test shape: {X_test.shape}")
    
    # Find first date where ALL features are populated
    train_valid_mask = ~X_train.isna().any(axis=1)
    
    if train_valid_mask.any():
        first_valid_train_idx = train_valid_mask.idxmax()
        print(f"First fully-populated date in train: {first_valid_train_idx}")
        
        X_train_clean = X_train.loc[first_valid_train_idx:]
        y_train_clean = y_train.loc[first_valid_train_idx:]
        
        X_train_clean = X_train_clean.dropna()
        y_train_clean = y_train_clean.loc[X_train_clean.index]
    else:
        raise ValueError("No fully-populated rows found in training data!")
    
    X_test_clean = X_test.dropna()
    y_test_clean = y_test.loc[X_test_clean.index]
    
    print(f"\nAfter filtering:")
    print(f"  Train: {len(X_train_clean)} samples")
    print(f"  Test: {len(X_test_clean)} samples")
    
    assert X_train_clean.isna().sum().sum() == 0, "Train data still has NaN!"
    assert X_test_clean.isna().sum().sum() == 0, "Test data still has NaN!"
    print("  ✓ Verified: No NaN values remaining")
    
    return X_train_clean, X_test_clean, y_train_clean, y_test_clean




def create_ensemble_model(xgb_params, lgb_params, cat_params):
    """
    Create a soft voting ensemble of XGBoost, LightGBM, and CatBoost.
    
    Uses custom implementation to avoid sklearn compatibility issues.
    """
    print("\n" + "="*60)
    print("CREATING ENSEMBLE MODEL")
    print("="*60)
    
    # Initialize individual models
    xgb_model = xgb.XGBClassifier(**xgb_params)
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    cat_model = CatBoostClassifier(**cat_params)
    
    # Create custom voting ensemble
    ensemble = SoftVotingEnsemble([
        ('XGBoost', xgb_model),
        ('LightGBM', lgb_model),
        ('CatBoost', cat_model)
    ])
    
    print("\nEnsemble composition:")
    print("  1. XGBoost (level-wise, regularized)")
    print("  2. LightGBM (leaf-wise, default)")
    print("  3. CatBoost (ordered boosting)")
    print("  Voting: Soft (probability averaging)")
    
    return ensemble


def train_ensemble(ensemble, X_train, y_train):
    """Train the ensemble model."""
    print("\n" + "="*60)
    print("TRAINING ENSEMBLE MODEL")
    print("="*60)
    
    ensemble.fit(X_train, y_train)
    
    return ensemble


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """Evaluate model performance."""
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} EVALUATION")
    print(f"{'='*60}")
    
    # Make predictions
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    y_train_pred = (y_train_pred_proba > 0.5).astype(int)
    y_test_pred = (y_test_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'train': {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1': f1_score(y_train, y_train_pred),
            'roc_auc': roc_auc_score(y_train, y_train_pred_proba)
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_pred_proba)
        }
    }
    
    # Print results
    print("\nTRAIN SET METRICS:")
    for metric, value in metrics['train'].items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print("\nTEST SET METRICS:")
    for metric, value in metrics['test'].items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Confusion matrix
    print("\nTEST CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Classification report
    print("\nTEST CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_test_pred, target_names=['Down', 'Up']))
    
    return metrics, y_test_pred, y_test_pred_proba, cm


def plot_confusion_matrix(cm, output_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'])
    plt.title('Ensemble Model - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved confusion matrix to {plot_path}")
    plt.close()


def compare_models(ensemble_metrics, output_dir):
    """
    Compare ensemble performance with individual models.
    Load metrics from individual model files and create comparison plot.
    """
    print("\n" + "="*60)
    print("COMPARING ENSEMBLE WITH INDIVIDUAL MODELS")
    print("="*60)
    
    # Load individual model metrics
    models_data = {
        'XGBoost': None,
        'LightGBM': None,
        'CatBoost': None,
        'Ensemble': ensemble_metrics
    }
    
    # Try to load individual model metrics
    for model_name in ['xgboost', 'lightgbm', 'catboost']:
        metrics_file = f'output/models/{model_name}/metrics.json'
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                models_data[model_name.capitalize()] = json.load(f)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics_to_plot = ['accuracy', 'roc_auc', 'f1']
    model_names = []
    train_scores = {m: [] for m in metrics_to_plot}
    test_scores = {m: [] for m in metrics_to_plot}
    
    for model_name, metrics in models_data.items():
        if metrics is not None:
            model_names.append(model_name)
            for metric in metrics_to_plot:
                train_scores[metric].append(metrics['train'][metric])
                test_scores[metric].append(metrics['test'][metric])
    
    # Plot train metrics
    x = np.arange(len(model_names))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        axes[0].bar(x + i*width, train_scores[metric], width, 
                   label=metric.upper().replace('_', ' '))
    
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Train Set Performance')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(model_names, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot test metrics
    for i, metric in enumerate(metrics_to_plot):
        axes[1].bar(x + i*width, test_scores[metric], width, 
                   label=metric.upper().replace('_', ' '))
    
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Test Set Performance')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(model_names, rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {plot_path}")
    plt.close()
    
    # Print comparison table
    print("\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    print(f"\n{'Model':<15} {'Test AUC':<12} {'Test Acc':<12} {'Test F1':<12}")
    print("-" * 60)
    for model_name in model_names:
        metrics = models_data[model_name]
        print(f"{model_name:<15} {metrics['test']['roc_auc']:<12.4f} "
              f"{metrics['test']['accuracy']:<12.4f} {metrics['test']['f1']:<12.4f}")


def save_model_and_results(ensemble, metrics, output_dir):
    """Save ensemble model and metrics."""
    print("\n" + "="*60)
    print("SAVING ENSEMBLE MODEL")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save ensemble model
    model_path = os.path.join(output_dir, 'ensemble_model.pkl')
    joblib.dump(ensemble, model_path)
    print(f"Saved ensemble model to {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


def main():
    """Main execution function."""
    print("="*60)
    print("ENSEMBLE MODEL TRAINING")
    print("="*60)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load selected features
    selected_features = load_selected_features(SELECTED_FEATURES_FILE)
    
    # 2. Load data with selected features
    X, y, dates = load_and_prepare_data(INPUT_FILE, selected_features)
    
    # 3. Split train/test
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, dates, TRAIN_END_DATE, TEST_START_DATE
    )
    
    # 4. Handle missing values
    X_train_clean, X_test_clean, y_train_clean, y_test_clean = handle_missing_values(
        X_train, X_test, y_train, y_test
    )
    
    # 5. Create ensemble model
    ensemble = create_ensemble_model(XGBOOST_PARAMS, LIGHTGBM_PARAMS, CATBOOST_PARAMS)
    
    # 6. Train ensemble
    ensemble = train_ensemble(ensemble, X_train_clean, y_train_clean)
    
    # 7. Evaluate ensemble
    metrics, y_pred, y_pred_proba, cm = evaluate_model(
        ensemble, X_train_clean, y_train_clean,
        X_test_clean, y_test_clean,
        model_name="Ensemble"
    )
    
    # 8. Plot confusion matrix
    plot_confusion_matrix(cm, OUTPUT_DIR)
    
    # 9. Compare with individual models
    compare_models(metrics, OUTPUT_DIR)
    
    # 10. Save model and results
    save_model_and_results(ensemble, metrics, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "="*60)
    print("ENSEMBLE TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📊 ENSEMBLE PERFORMANCE:")
    print(f"  Train ROC AUC: {metrics['train']['roc_auc']:.4f}")
    print(f"  Test ROC AUC:  {metrics['test']['roc_auc']:.4f}")
    print(f"  Test Accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"  Test F1 Score: {metrics['test']['f1']:.4f}")
    
    print(f"\nFiles saved to: {OUTPUT_DIR}/")
    print("  - ensemble_model.pkl")
    print("  - metrics.json")
    print("  - confusion_matrix.png")
    print("  - model_comparison.png")


if __name__ == "__main__":
    main()
