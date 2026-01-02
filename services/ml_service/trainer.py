"""
Model Trainer for Predictive Maintenance
Handles full ML training pipeline
"""

import pandas as pd
import joblib
from pathlib import Path
from typing import Dict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from config.settings import settings


class PredictiveMaintenanceTrainer:
    """
    Trainer class for Predictive Maintenance ML model
    """

    def __init__(self):
        self.model = None
        self.scaler = None

    def load_data(self, data_path: Path):
        """Load dataset from CSV"""
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found at {data_path}")

        df = pd.read_csv(data_path)

        X = df.drop(columns=["failure"])
        y = df["failure"]

        return X, y

    def preprocess(self, X_train, X_test):
        """Scale features"""
        self.scaler = StandardScaler()

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def train_model(self, X_train, y_train):
        """Train RandomForest model"""
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=settings.RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test) -> Dict:
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

        return metrics

    def save_artifacts(self, model_path: Path, scaler_path: Path):
        """Save trained model and scaler"""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def full_training_pipeline(
        self,
        data_path: Path,
        model_path: Path,
        scaler_path: Path
    ) -> Dict:
        """
        Complete training pipeline
        """
        print("▶ Loading training data...")
        X, y = self.load_data(data_path)

        print("▶ Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=settings.RANDOM_STATE,
            stratify=y
        )

        print("▶ Preprocessing data...")
        X_train_scaled, X_test_scaled = self.preprocess(X_train, X_test)

        print("▶ Training model...")
        self.train_model(X_train_scaled, y_train)

        print("▶ Evaluating model...")
        metrics = self.evaluate(X_test_scaled, y_test)

        print("▶ Saving model artifacts...")
        self.save_artifacts(model_path, scaler_path)

        print("✓ Training pipeline completed")

        return metrics
