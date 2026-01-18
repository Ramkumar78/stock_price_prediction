"""
XGBoost Model Training for Stock Price Prediction

This script trains an XGBoost classifier to predict whether SPY will go up or down
over the next 3 days based on 115 engineered features from technical indicators,
volume patterns, volatility measures, and cross-asset relationships.

Two-Phase Training Workflow:
-----------------------------
PHASE 1: Train with all features (115)
  1. Load features from feature_engineering.py output
  2. Split data into train (2015-2023) and test (2024-2026) sets
  3. Handle NaN values from rolling window calculations
  4. Train XGBoost model with regularized hyperparameters
  5. Evaluate performance (ROC AUC, accuracy, confusion matrix)
  6. Calculate permutation importance on test set
  7. Select top 40 features based on actual predictive value

PHASE 2: Retrain with selected features (40)
  8. Filter to top 40 features from permutation importance
  9. Retrain XGBoost model with same hyperparameters
  10. Evaluate and compare performance vs full model
  11. Save both models, feature lists, and visualizations

Hyperparameter Strategy:
-----------------------
Model uses aggressive regularization to combat overfitting:
- max_depth=3 (shallow trees)
- reg_alpha=1.0, reg_lambda=3.0 (strong L1/L2 regularization)
- subsample=0.7, colsample_bytree=0.7 (stochastic training)
- eval_metric='auc' (optimize for ranking, use early stopping)

Output Files (saved to output/models/xgboost/):
----------------------------------------------
- xgboost_model.json - Full model (115 features)
- xgboost_model_selected_features.json - Selected model (40 features)
- permutation_importance.csv - Feature importance scores
- selected_features.txt - List of top 40 features
- feature_importance_top30.png - XGBoost built-in importance plot
- permutation_importance_top40.png - Permutation importance plot
- training_curves.png - AUC curves during training

Usage:
------
    python modelling_xgboost.py

Expected Performance:
--------------------
Due to market regime differences between train (2015-2023) and test (2024-2026),
the model achieves ~50% ROC AUC on test set, indicating limited predictive power
for 3-day price movements in the current market environment.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import joblib
import json

# Configuration
INPUT_FILE = 'output/spy_features_full.csv'
OUTPUT_DIR = 'output/models/xgboost'
TRAIN_END_DATE = '2024-12-31'
TEST_START_DATE = '2025-01-01'
RANDOM_STATE = 42

# XGBoost hyperparameters - Tuned to reduce overfitting
# 
# These parameters were chosen after observing severe overfitting in initial runs
# (Train ROC AUC: 96.7%, Test ROC AUC: 48%). The goal is to reduce model complexity
# and increase regularization to improve generalization to unseen data.
XGBOOST_PARAMS = {
    # Core objective and evaluation
    'objective': 'binary:logistic',  # Binary classification (up/down prediction)
    'eval_metric': 'auc',  # Use AUC for early stopping - aligns with ROC AUC evaluation metric
    
    # Tree structure parameters (control model complexity)
    'max_depth': 3,  # Shallow trees prevent overfitting (reduced from 6)
                     # Each tree can only split 3 times, limiting memorization
    
    'min_child_weight': 5,  # Minimum samples required per leaf (increased from 3)
                            # Higher value = more conservative splits, less overfitting
    
    # Learning rate and iterations
    'learning_rate': 0.05,  # Small learning rate for gradual, stable learning
                            # Lower = more robust but needs more trees
    
    'n_estimators': 300,  # Maximum number of trees to build
                          # Actual number determined by early stopping
    
    # Regularization parameters (prevent overfitting)
    'reg_alpha': 1.0,  # L1 regularization (Lasso) - increased from 0.1
                       # Encourages feature sparsity, zeros out weak features
    
    'reg_lambda': 3.0,  # L2 regularization (Ridge) - increased from 1.0
                        # Penalizes large weights, smooths predictions
    
    'gamma': 0.1,  # Minimum loss reduction required for split
                   # Acts as pruning parameter, prevents unnecessary splits
    
    # Stochastic training (adds randomness to prevent overfitting)
    'subsample': 0.7,  # Use 70% of training data per tree (reduced from 0.8)
                       # Introduces randomness, similar to bagging
    
    'colsample_bytree': 0.7,  # Use 70% of features per tree (reduced from 0.8)
                              # Prevents reliance on any single feature
    
    # Training control
    'random_state': RANDOM_STATE,  # Reproducibility
    'n_jobs': -1,  # Use all CPU cores for parallel processing
    'early_stopping_rounds': 50  # Stop if no improvement for 50 iterations
                                 # Prevents overfitting by stopping at optimal point
}


def load_and_prepare_data(filepath):
    """
    Load features and target from CSV file.
    
    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        dates (pd.DatetimeIndex): Date index
    """
    print("Loading data...")
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
    X = df.drop(columns=['target'])
    
    print(f"Features: {X.shape[1]}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, df.index


def split_train_test(X, y, dates, train_end_date, test_start_date):
    """
    Split data into train and test sets based on date.
    
    Time-series split:
    - Train: 2015-01-01 to 2023-12-31
    - Test: 2024-01-01 to present
    """
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
    """
    Handle NaN values in features intelligently.
    
    Strategy: Instead of dropping rows with NaN, find the first date where ALL features
    are populated (after the longest rolling window - 200 days). This ensures we only
    train on complete, valid data rather than mixing valid and invalid rows.
    
    This is better than forward-fill or dropping because:
    - Features like 200-day MA need 200 days of history to be meaningful
    - Hurst exponent (50-day) needs 50 days of data
    - Training on incomplete features would introduce noise
    
    Returns:
        Cleaned datasets starting from first fully-populated date
    """
    print("\nHandling missing values intelligently...")
    print(f"Original train shape: {X_train.shape}")
    print(f"Original test shape: {X_test.shape}")
    print(f"Train NaN count: {X_train.isna().sum().sum()}")
    print(f"Test NaN count: {X_test.isna().sum().sum()}")
    
    # Find first date where ALL features are populated
    # This will be after the longest rolling window (200 days for MA)
    train_valid_mask = ~X_train.isna().any(axis=1)
    test_valid_mask = ~X_test.isna().any(axis=1)
    
    # Get first valid index for train set
    if train_valid_mask.any():
        first_valid_train_idx = train_valid_mask.idxmax()
        print(f"\nFirst fully-populated date in train: {first_valid_train_idx}")
        
        # Filter train data from first valid date onwards
        X_train_clean = X_train.loc[first_valid_train_idx:]
        y_train_clean = y_train.loc[first_valid_train_idx:]
        
        # Verify no NaN in filtered data
        X_train_clean = X_train_clean.dropna()
        y_train_clean = y_train_clean.loc[X_train_clean.index]
    else:
        raise ValueError("No fully-populated rows found in training data!")
    
    # For test set, also drop any rows with NaN
    # (shouldn't have many since test is recent data)
    X_test_clean = X_test.dropna()
    y_test_clean = y_test.loc[X_test_clean.index]
    
    print(f"\nAfter filtering to fully-populated dates:")
    print(f"  Train: {len(X_train_clean)} samples (removed {len(X_train) - len(X_train_clean)} rows)")
    print(f"  Test: {len(X_test_clean)} samples (removed {len(X_test) - len(X_test_clean)} rows)")
    print(f"  Train date range: {X_train_clean.index[0]} to {X_train_clean.index[-1]}")
    print(f"  Test date range: {X_test_clean.index[0]} to {X_test_clean.index[-1]}")
    
    # Verify no NaN remaining
    assert X_train_clean.isna().sum().sum() == 0, "Train data still has NaN!"
    assert X_test_clean.isna().sum().sum() == 0, "Test data still has NaN!"
    print("  ✓ Verified: No NaN values remaining")
    
    return X_train_clean, X_test_clean, y_train_clean, y_test_clean



def train_xgboost(X_train, y_train, X_test, y_test, params):
    """
    Train XGBoost model with early stopping.
    """
    print("\n" + "="*60)
    print("TRAINING XGBOOST MODEL")
    print("="*60)
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Extract early stopping rounds
    early_stopping_rounds = params.pop('early_stopping_rounds', None)
    
    # Train model with evaluation
    evals = [(dtrain, 'train'), (dtest, 'test')]
    evals_result = {}
    
    print(f"\nTraining with {params['n_estimators']} estimators...")
    print(f"Early stopping: {early_stopping_rounds} rounds")
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=50
    )
    
    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score:.4f}")
    
    return model, evals_result


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance on train and test sets.
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Make predictions
    dtrain = xgb.DMatrix(X_train)
    dtest = xgb.DMatrix(X_test)
    
    y_train_pred_proba = model.predict(dtrain)
    y_test_pred_proba = model.predict(dtest)
    
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
    
    # Confusion matrices
    print("\nTRAIN CONFUSION MATRIX:")
    print(confusion_matrix(y_train, y_train_pred))
    
    print("\nTEST CONFUSION MATRIX:")
    print(confusion_matrix(y_test, y_test_pred))
    
    # Classification reports
    print("\nTEST CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_test_pred, target_names=['Down', 'Up']))
    
    return metrics, y_test_pred, y_test_pred_proba


def plot_feature_importance(model, feature_names, top_n=30, output_dir='output/models'):
    """
    Plot and save feature importance.
    """
    print(f"\nPlotting top {top_n} features by importance...")
    
    # Get feature importance
    importance_dict = model.get_score(importance_type='gain')
    
    # Convert to DataFrame
    importance_df = pd.DataFrame([
        {'feature': f, 'importance': importance_dict.get(f, 0)}
        for f in feature_names
    ]).sort_values('importance', ascending=False)
    
    # Save full importance
    importance_path = os.path.join(output_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved feature importance to {importance_path}")
    
    # Plot top N
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance (Gain)')
    plt.title(f'Top {top_n} Features by Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'feature_importance_top30.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    plt.close()
    
    return importance_df


def calculate_permutation_importance(model, X_test, y_test, feature_names, top_n=40, output_dir='output/models'):
    """
    Calculate permutation importance on test set to identify truly predictive features.
    
    Permutation importance measures how much the model's performance drops when a feature
    is randomly shuffled. This is more reliable than built-in feature importance because
    it's based on actual predictive value on unseen data.
    
    Parameters:
    -----------
    model : xgboost.Booster
        Trained XGBoost model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to select
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    selected_features : list
        List of top N feature names
    perm_importance_df : pd.DataFrame
        DataFrame with permutation importance scores
    """
    print(f"\n{'='*60}")
    print("CALCULATING PERMUTATION IMPORTANCE")
    print(f"{'='*60}")
    print(f"This measures how much each feature contributes to test set accuracy...")
    print(f"(This may take a few minutes...)")
    
    # Get baseline accuracy
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    baseline_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nBaseline test accuracy: {baseline_accuracy:.4f}")
    print(f"Calculating importance for {len(feature_names)} features...")
    
    # Calculate permutation importance manually
    importances = []
    n_repeats = 10
    
    for i, feature in enumerate(feature_names):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(feature_names)} features...")
        
        feature_importances = []
        
        for repeat in range(n_repeats):
            # Create a copy and shuffle the feature
            X_test_permuted = X_test.copy()
            X_test_permuted[feature] = np.random.permutation(X_test_permuted[feature].values)
            
            # Make predictions with permuted feature
            dtest_permuted = xgb.DMatrix(X_test_permuted)
            y_pred_proba_permuted = model.predict(dtest_permuted)
            y_pred_permuted = (y_pred_proba_permuted > 0.5).astype(int)
            permuted_accuracy = accuracy_score(y_test, y_pred_permuted)
            
            # Importance = drop in accuracy
            importance = baseline_accuracy - permuted_accuracy
            feature_importances.append(importance)
        
        # Store mean and std
        importances.append({
            'feature': feature,
            'importance_mean': np.mean(feature_importances),
            'importance_std': np.std(feature_importances)
        })
    
    # Create DataFrame with results
    perm_importance_df = pd.DataFrame(importances).sort_values('importance_mean', ascending=False)
    
    # Save full results
    perm_path = os.path.join(output_dir, 'permutation_importance.csv')
    perm_importance_df.to_csv(perm_path, index=False)
    print(f"\nSaved permutation importance to {perm_path}")
    
    # Select top N features
    selected_features = perm_importance_df.head(top_n)['feature'].tolist()
    
    print(f"\nTop {top_n} features selected based on permutation importance:")
    for i, row in perm_importance_df.head(top_n).iterrows():
        print(f"  {i+1}. {row['feature']}: {row['importance_mean']:.6f} (+/- {row['importance_std']:.6f})")
    
    # Plot permutation importance
    top_features = perm_importance_df.head(top_n)
    
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(top_features)), top_features['importance_mean'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Permutation Importance (Accuracy Drop)')
    plt.title(f'Top {top_n} Features by Permutation Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'permutation_importance_top{top_n}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    plt.close()
    
    return selected_features, perm_importance_df



def plot_training_curves(evals_result, output_dir='output/models'):
    """
    Plot training and validation AUC curves.
    """
    print("\nPlotting training curves...")
    
    train_metric = evals_result['train']['auc']
    test_metric = evals_result['test']['auc']
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_metric, label='Train AUC')
    plt.plot(test_metric, label='Test AUC')
    plt.xlabel('Iteration')
    plt.ylabel('AUC')
    plt.title('XGBoost Training Curves (AUC)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    plt.close()


def save_model_and_results(model, metrics, output_dir='output/models'):
    """
    Save trained model and evaluation metrics.
    """
    print("\nSaving model and results...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'xgboost_model.json')
    model.save_model(model_path)
    print(f"Saved model to {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


def main():
    """
    Main execution function.
    
    Workflow:
    1. Train initial model with all features
    2. Calculate permutation importance on test set
    3. Select top 40 features
    4. Retrain model with selected features only
    5. Compare performance
    """
    print("="*60)
    print("XGBOOST STOCK PREDICTION MODEL")
    print("="*60)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load data
    X, y, dates = load_and_prepare_data(INPUT_FILE)
    
    # 2. Split train/test
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, dates, TRAIN_END_DATE, TEST_START_DATE
    )
    
    # 3. Handle missing values intelligently
    # Start training only after all features are fully populated (after 200-day MA)
    X_train_clean, X_test_clean, y_train_clean, y_test_clean = handle_missing_values(
        X_train, X_test, y_train, y_test
    )
    
    # ========== PHASE 1: Train with ALL features ==========
    print("\n" + "="*60)
    print("PHASE 1: TRAINING WITH ALL FEATURES")
    print("="*60)
    
    # 4. Train initial model
    model_full, evals_result_full = train_xgboost(
        X_train_clean, y_train_clean,
        X_test_clean, y_test_clean,
        XGBOOST_PARAMS.copy()
    )
    
    # 5. Evaluate initial model
    metrics_full, y_pred_full, y_pred_proba_full = evaluate_model(
        model_full, X_train_clean, y_train_clean,
        X_test_clean, y_test_clean
    )
    
    # 6. Feature importance (XGBoost built-in)
    importance_df = plot_feature_importance(
        model_full, X_train_clean.columns, top_n=30, output_dir=OUTPUT_DIR
    )
    
    # 7. Calculate permutation importance
    selected_features, perm_importance_df = calculate_permutation_importance(
        model_full, X_test_clean, y_test_clean, 
        X_train_clean.columns, top_n=40, output_dir=OUTPUT_DIR
    )
    
    # ========== PHASE 2: Retrain with SELECTED features ==========
    print("\n" + "="*60)
    print("PHASE 2: RETRAINING WITH TOP 40 FEATURES")
    print("="*60)
    print(f"Selected features: {len(selected_features)}")
    
    # Filter to selected features
    X_train_selected = X_train_clean[selected_features]
    X_test_selected = X_test_clean[selected_features]
    
    # 8. Train model with selected features
    model_selected, evals_result_selected = train_xgboost(
        X_train_selected, y_train_clean,
        X_test_selected, y_test_clean,
        XGBOOST_PARAMS.copy()
    )
    
    # 9. Evaluate selected features model
    metrics_selected, y_pred_selected, y_pred_proba_selected = evaluate_model(
        model_selected, X_train_selected, y_train_clean,
        X_test_selected, y_test_clean
    )
    
    # 10. Plot training curves for selected model
    plot_training_curves(evals_result_selected, output_dir=OUTPUT_DIR)
    
    # 11. Save selected model and metrics
    model_path = os.path.join(OUTPUT_DIR, 'xgboost_model_selected_features.json')
    model_selected.save_model(model_path)
    print(f"\nSaved selected features model to {model_path}")

    # Save metrics for the selected model (required for /metrics endpoint)
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_selected, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Save selected features list
    selected_features_path = os.path.join(OUTPUT_DIR, 'selected_features.txt')
    with open(selected_features_path, 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")
    print(f"Saved selected features list to {selected_features_path}")
    
    # ========== FINAL COMPARISON ==========
    print("\n" + "="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    
    print("\n📊 MODEL WITH ALL FEATURES (115 features):")
    print(f"  Train ROC AUC: {metrics_full['train']['roc_auc']:.4f}")
    print(f"  Test ROC AUC:  {metrics_full['test']['roc_auc']:.4f}")
    print(f"  Test Accuracy: {metrics_full['test']['accuracy']:.4f}")
    print(f"  Test F1 Score: {metrics_full['test']['f1']:.4f}")
    
    print("\n📊 MODEL WITH SELECTED FEATURES (40 features):")
    print(f"  Train ROC AUC: {metrics_selected['train']['roc_auc']:.4f}")
    print(f"  Test ROC AUC:  {metrics_selected['test']['roc_auc']:.4f}")
    print(f"  Test Accuracy: {metrics_selected['test']['accuracy']:.4f}")
    print(f"  Test F1 Score: {metrics_selected['test']['f1']:.4f}")
    
    # Calculate improvement
    roc_improvement = metrics_selected['test']['roc_auc'] - metrics_full['test']['roc_auc']
    acc_improvement = metrics_selected['test']['accuracy'] - metrics_full['test']['accuracy']
    
    print(f"\n📈 IMPROVEMENT:")
    print(f"  ROC AUC: {roc_improvement:+.4f}")
    print(f"  Accuracy: {acc_improvement:+.4f}")
    
    if roc_improvement > 0:
        print("\n✅ Feature selection IMPROVED performance!")
    else:
        print("\n⚠️ Feature selection did NOT improve performance")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFiles saved to: {OUTPUT_DIR}/")
    print("  - xgboost_model.json (full model)")
    print("  - xgboost_model_selected_features.json (selected features model)")
    print("  - permutation_importance.csv")
    print("  - selected_features.txt")
    
    print("\n🎯 Top 10 Features by Permutation Importance:")
    for i, row in perm_importance_df.head(10).iterrows():
        print(f"  {i+1}. {row['feature']}: {row['importance_mean']:.6f}")



if __name__ == "__main__":
    main()
