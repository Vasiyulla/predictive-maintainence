"""
Model Trainer for Predictive Maintenance
Handles full ML training pipeline with multiple algorithms:
- Random Forest (Default)
- Support Vector Machine (SVM)
- LSTM (Deep Learning)

Also trains a separate RUL Regressor for time-to-failure predictions.
"""

import pandas as pd
import joblib
import numpy as np
import json
from pathlib import Path
from typing import Dict, Literal, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Try importing TensorFlow for LSTM; handle case where it's not installed
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠ TensorFlow not found. LSTM training will be unavailable.")

from config.settings import settings

class PredictiveMaintenanceTrainer:
    """Trainer class for Predictive Maintenance ML models"""

    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.scaler = None
        self.model_type = "random_forest"
        self.feature_names = ['temperature', 'vibration', 'pressure', 'rpm', 'energy_kwh']

    def load_data(self, data_path: Path):
        """Load dataset from CSV"""
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found at {data_path}")

        df = pd.read_csv(data_path)
        
        # Ensure all features exist
        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        X = df[self.feature_names]
        y_failure = df["failure"]
        y_rul = df["rul_days"] if "rul_days" in df.columns else np.zeros(len(df))
        
        return X, y_failure, y_rul

    def preprocess(self, X_train, X_test):
        """Scale features"""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def build_lstm_model(self, input_shape):
        """Build Keras LSTM Model"""
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_classifier(self, X_train, y_train, model_type: str):
        """Train the specific classification algorithm"""
        self.model_type = model_type.lower()
        print(f"▶ Training Classifier using: {self.model_type.upper()}")

        if self.model_type == "random_forest":
            self.classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=settings.RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1
            )
            self.classifier.fit(X_train, y_train)

        elif self.model_type == "svm":
            self.classifier = SVC(
                kernel='rbf',
                probability=True,  # Required for predict_proba
                random_state=settings.RANDOM_STATE,
                class_weight="balanced"
            )
            self.classifier.fit(X_train, y_train)

        elif self.model_type == "lstm":
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is required for LSTM but not installed.")
            
            # Reshape input for LSTM: [samples, time_steps, features]
            # We treat the current state as 1 time step
            X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            
            self.classifier = self.build_lstm_model((1, X_train.shape[1]))
            self.classifier.fit(
                X_train_reshaped, 
                y_train, 
                epochs=15, 
                batch_size=32, 
                verbose=1,
                validation_split=0.1
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train_regressor(self, X_train, y_train):
        """Train RUL Regressor (Always Random Forest for stability)"""
        print("▶ Training RUL Regressor (Random Forest)...")
        self.regressor = RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            random_state=settings.RANDOM_STATE,
            n_jobs=-1
        )
        self.regressor.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluate classifier performance"""
        if self.model_type == "lstm":
            X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            y_prob = self.classifier.predict(X_test_reshaped).flatten()
            y_pred = (y_prob > 0.5).astype(int)
        else:
            y_pred = self.classifier.predict(X_test)

        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "model_type": self.model_type
        }

    def save_artifacts(self, model_dir: Path):
        """Save all models and metadata"""
        model_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save Scaler
        joblib.dump(self.scaler, model_dir / "scaler.joblib")

        # 2. Save RUL Regressor
        joblib.dump(self.regressor, model_dir / "rul_regressor.joblib")

        # 3. Save Classifier (Format depends on type)
        if self.model_type == "lstm":
            # Save Keras model
            keras_path = model_dir / "classifier.keras"
            self.classifier.save(keras_path)
            # Remove old joblib if exists to avoid confusion
            if (model_dir / "classifier.joblib").exists():
                (model_dir / "classifier.joblib").unlink()
        else:
            # Save Sklearn model
            joblib.dump(self.classifier, model_dir / "classifier.joblib")
            # Remove old keras if exists
            if (model_dir / "classifier.keras").exists():
                (model_dir / "classifier.keras").unlink()

        # 4. Save Metadata
        metadata = {
            "model_type": self.model_type,
            "features": self.feature_names,
            "last_trained": str(pd.Timestamp.now()),
            "tensorflow_version": tf.__version__ if TF_AVAILABLE and self.model_type == "lstm" else None
        }
        
        with open(model_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ All artifacts saved to {model_dir}")

    def full_training_pipeline(
        self,
        data_path: Path,
        model_dir: Path,
        model_type: str = "random_forest"
    ) -> Dict:
        """Run the complete training workflow"""
        
        print(f"\n=== Starting Training Pipeline ({model_type}) ===")
        
        # 1. Load Data
        X, y_fail, y_rul = self.load_data(data_path)

        # 2. Split Data
        X_train, X_test, y_f_train, y_f_test, y_r_train, y_r_test = train_test_split(
            X, y_fail, y_rul, test_size=0.2, random_state=settings.RANDOM_STATE, stratify=y_fail
        )

        # 3. Scale Features
        X_train_scaled, X_test_scaled = self.preprocess(X_train, X_test)

        # 4. Train Models
        self.train_classifier(X_train_scaled, y_f_train, model_type)
        self.train_regressor(X_train_scaled, y_r_train)

        # 5. Evaluate
        metrics = self.evaluate(X_test_scaled, y_f_test)
        print(f"Metrics: {metrics}")

        # 6. Save
        self.save_artifacts(model_dir)

        return metrics

if __name__ == "__main__":
    # Test run
    trainer = PredictiveMaintenanceTrainer()
    # Mock paths for testing
    data_path = settings.DATA_PATH / "raw" / "sensor_data.csv"
    model_dir = settings.MODEL_PATH.parent
    
    if data_path.exists():
        trainer.full_training_pipeline(data_path, model_dir, model_type="random_forest")