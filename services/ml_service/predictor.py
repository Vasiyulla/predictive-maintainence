"""
ML Prediction Module
Loads trained models (Classifier & Regressor) and performs real-time predictions.

Features:
- Classification: Supports Random Forest, SVM, and LSTM (Keras) models.
- Regression: Predicts Remaining Useful Life (RUL).
- Anomaly Detection: Analyzes Energy Efficiency.
- Explainability: Generates 'Why' reports for predictions.
"""

import joblib
import numpy as np
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Conditionally import TensorFlow for LSTM support
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from config.settings import settings

class PredictiveMaintenancePredictor:
    """
    ML Predictor for predictive maintenance.
    Manages loading and inference for multiple model types.
    """
    
    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize the predictor and load models.
        
        Args:
            model_dir: Directory containing model artifacts. 
                       Defaults to parent of settings.MODEL_PATH.
        """
        self.classifier = None
        self.regressor = None
        self.scaler = None
        
        # Standard features used in training
        self.feature_names = ['temperature', 'vibration', 'pressure', 'rpm', 'energy_kwh']
        
        self.model_loaded = False
        self.model_metadata = {}
        
        # Set model directory
        self.model_dir = model_dir if model_dir else settings.MODEL_PATH.parent
        
        # Load models on initialization
        self.load_models()

    def load_models(self) -> bool:
        """
        Load Scaler, RUL Regressor, and the appropriate Classifier (RF/SVM/LSTM).
        """
        try:
            print(f"▶ Loading models from {self.model_dir}...")
            
            # 1. Load Metadata (Critical for determining model type)
            meta_path = self.model_dir / "model_metadata.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    self.model_metadata = json.load(f)
            else:
                print("⚠ Metadata not found. Assuming Random Forest default.")
                self.model_metadata = {"model_type": "random_forest"}

            model_type = self.model_metadata.get("model_type", "random_forest")

            # 2. Load Scaler
            scaler_path = self.model_dir / "scaler.joblib"
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found at {scaler_path}")
            self.scaler = joblib.load(scaler_path)

            # 3. Load RUL Regressor
            regressor_path = self.model_dir / "rul_regressor.joblib"
            if not regressor_path.exists():
                raise FileNotFoundError(f"RUL Regressor not found at {regressor_path}")
            self.regressor = joblib.load(regressor_path)

            # 4. Load Classifier based on Type
            if model_type == "lstm":
                if not TF_AVAILABLE:
                    raise ImportError("Model is LSTM but TensorFlow is not installed.")
                
                keras_path = self.model_dir / "classifier.keras"
                if not keras_path.exists():
                    raise FileNotFoundError(f"LSTM model not found at {keras_path}")
                
                self.classifier = tf.keras.models.load_model(keras_path)
                print("✓ Loaded LSTM Classifier (Keras)")
                
            else:
                # Random Forest or SVM
                classifier_path = self.model_dir / "classifier.joblib"
                if not classifier_path.exists():
                    raise FileNotFoundError(f"Classifier not found at {classifier_path}")
                
                self.classifier = joblib.load(classifier_path)
                print(f"✓ Loaded {model_type.upper()} Classifier (Sklearn)")

            self.model_loaded = True
            return True

        except Exception as e:
            print(f"✗ Error loading models: {e}")
            self.model_loaded = False
            return False

    def _calculate_energy(self, data: Dict[str, float]) -> float:
        """
        Estimate energy consumption if physical sensor is missing.
        Formula matches generator logic: E ~ (RPM * 0.05) + (Temp * 0.1)
        """
        return (data.get('rpm', 0) * 0.05) + (data.get('temperature', 0) * 0.1)

    def _prepare_input(self, sensor_data: Dict[str, float]) -> Tuple[np.ndarray, float]:
        """
        Validate input, calculate missing energy, and scale features.
        Returns: (Scaled Array, Raw Energy Value)
        """
        data = sensor_data.copy()
        
        # Auto-fill energy if missing
        if 'energy_kwh' not in data:
            data['energy_kwh'] = self._calculate_energy(data)
            
        # Ensure all features exist
        missing = [f for f in self.feature_names if f not in data]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
            
        # Create vector in correct order
        vector = [data[f] for f in self.feature_names]
        
        # Scale
        X = np.array([vector])
        X_scaled = self.scaler.transform(X)
        
        return X_scaled, data['energy_kwh']

    def predict_failure_probability(
        self, 
        sensor_data: Dict[str, float],
        return_details: bool = True
    ) -> Dict:
        """
        Main prediction pipeline.
        
        Steps:
        1. Preprocess & Scale Data
        2. Predict Failure Probability (Classifier)
        3. Predict Remaining Useful Life (Regressor)
        4. Analyze Energy Efficiency
        """
        if not self.model_loaded:
            return {"error": "Models not loaded. Please train the model first."}

        try:
            # 1. Prepare Input
            X_scaled, energy_val = self._prepare_input(sensor_data)
            model_type = self.model_metadata.get("model_type", "random_forest")

            # 2. Classifier Prediction
            if model_type == "lstm":
                # Reshape for LSTM: [samples, time_steps, features] -> (1, 1, 5)
                X_lstm = X_scaled.reshape((1, 1, X_scaled.shape[1]))
                # LSTM returns sigmoid probability directly [[0.85]]
                fail_prob = float(self.classifier.predict(X_lstm, verbose=0)[0][0])
            elif model_type == "svm":
                # SVM predict_proba returns [[prob_0, prob_1]]
                fail_prob = float(self.classifier.predict_proba(X_scaled)[0][1])
            else:
                # Random Forest
                fail_prob = float(self.classifier.predict_proba(X_scaled)[0][1])

            # 3. RUL Prediction (Always Regressor)
            rul_days = float(self.regressor.predict(X_scaled)[0])
            rul_days = max(0.0, rul_days) # Clamp to 0

            # 4. Energy Analysis
            # Baseline: Theoretical consumption for these settings
            baseline = self._calculate_energy(sensor_data)
            energy_status = "Normal"
            
            if energy_val > baseline * 1.25:
                energy_status = "Inefficient (High)"
            elif energy_val < baseline * 0.75:
                energy_status = "Abnormal (Low)"

            # 5. Risk Assessment
            if fail_prob < settings.LOW_RISK_THRESHOLD: # 0.3
                risk_level = "LOW"
            elif fail_prob < settings.MEDIUM_RISK_THRESHOLD: # 0.7
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"

            result = {
                "success": True,
                "failure_predicted": bool(fail_prob > 0.5),
                "failure_probability": round(fail_prob, 4),
                "rul_days": round(rul_days, 1),
                "energy_status": energy_status,
                "energy_kwh": round(energy_val, 2),
                "risk_level": risk_level,
                "timestamp": datetime.utcnow().isoformat()
            }

            if return_details:
                result["model_info"] = {
                    "type": model_type,
                    "features": self.feature_names
                }

            return result

        except Exception as e:
            print(f"Prediction Error: {e}")
            return {"success": False, "error": str(e)}

    def get_prediction_explanation(self, sensor_data: Dict[str, float]) -> Dict:
        """
        Generate Explainable AI (XAI) report.
        Identifies which sensors are contributing most to the risk.
        """
        # Get base prediction
        pred = self.predict_failure_probability(sensor_data)
        if not pred.get("success"):
            return pred

        contributors = []
        
        # Define simplistic normal operating thresholds for explanation
        # In a real app, these should come from statistical analysis of training data
        thresholds = {
            'temperature': {'limit': 80, 'msg': 'Overheating'},
            'vibration': {'limit': 6, 'msg': 'Excessive Vibration'},
            'pressure': {'limit': 130, 'msg': 'High Pressure'},
            'rpm': {'limit': 2800, 'msg': 'Overspeed'},
            'energy_kwh': {'limit': 150, 'msg': 'Power Spike'}
        }

        # Check sensors against thresholds
        for feature, val in sensor_data.items():
            if feature in thresholds:
                limit = thresholds[feature]['limit']
                if val > limit:
                    contributors.append({
                        "feature": feature,
                        "value": val,
                        "reason": f"{thresholds[feature]['msg']} (> {limit})"
                    })

        # Add Energy Context
        if pred['energy_status'] != "Normal":
            contributors.append({
                "feature": "energy_efficiency",
                "value": pred['energy_kwh'],
                "reason": pred['energy_status']
            })

        # Feature Importance (Only for Random Forest)
        feature_importance = {}
        model_type = self.model_metadata.get("model_type", "random_forest")
        
        if model_type == "random_forest" and hasattr(self.classifier, "feature_importances_"):
            importances = self.classifier.feature_importances_
            feature_importance = dict(zip(self.feature_names, [round(x, 3) for x in importances]))

        return {
            "prediction_summary": pred,
            "contributors": contributors,
            "feature_importance": feature_importance,
            "analysis_text": f"Machine risk is {pred['risk_level']} with {pred['rul_days']} days RUL. " 
                             f"Detected {len(contributors)} anomalies."
        }

    def batch_predict(self, sensor_data_list: List[Dict[str, float]]) -> List[Dict]:
        """Process a list of sensor data points efficiently."""
        results = []
        for i, data in enumerate(sensor_data_list):
            res = self.predict_failure_probability(data, return_details=False)
            res['index'] = i
            results.append(res)
        return results

    def get_model_info(self) -> Dict:
        """Return current model status."""
        return {
            "loaded": self.model_loaded,
            "directory": str(self.model_dir),
            "metadata": self.model_metadata
        }

if __name__ == "__main__":
    # Test Block
    predictor = PredictiveMaintenancePredictor()
    
    test_data = {
        "temperature": 82.5,
        "vibration": 4.1,
        "pressure": 105.0,
        "rpm": 2100.0
        # energy_kwh will be auto-calculated
    }
    
    print("\n--- Test Prediction ---")
    result = predictor.predict_failure_probability(test_data)
    print(json.dumps(result, indent=2))
    
    print("\n--- Test Explanation ---")
    explanation = predictor.get_prediction_explanation(test_data)
    print(json.dumps(explanation, indent=2))