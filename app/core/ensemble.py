import numpy as np
import pandas as pd

class SoftVotingEnsemble:
    """
    Custom soft voting ensemble that works with XGBoost, LightGBM, and CatBoost.

    This avoids sklearn compatibility issues with CatBoost by implementing
    soft voting (probability averaging) manually.
    """

    def __init__(self, models):
        """
        Initialize ensemble with list of (name, model) tuples.

        Args:
            models: List of (name, model) tuples
        """
        self.models = models
        self.fitted_models = []

    def fit(self, X, y):
        """Train all models in the ensemble."""
        print(f"\nTraining {len(self.models)} models...")

        for name, model in self.models:
            print(f"  Training {name}...")
            model.fit(X, y)
            self.fitted_models.append((name, model))

        print("✓ All models trained!")
        return self

    def predict_proba(self, X):
        """
        Predict probabilities by averaging predictions from all models.

        Args:
            X: Features

        Returns:
            Array of shape (n_samples, 2) with averaged probabilities
        """
        if not self.fitted_models:
            raise ValueError("Ensemble not fitted yet!")

        # Collect predictions from all models
        all_probas = []
        for name, model in self.fitted_models:
            probas = model.predict_proba(X)
            all_probas.append(probas)

        # Average probabilities
        avg_probas = np.mean(all_probas, axis=0)
        return avg_probas

    def predict(self, X):
        """
        Predict class labels.

        Args:
            X: Features

        Returns:
            Array of predicted class labels
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)
