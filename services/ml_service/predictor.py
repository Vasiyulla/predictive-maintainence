"""
ML Prediction Module
Loads trained model and performs real-time predictions for predictive maintenance

This module handles:
- Loading pre-trained ML model and scaler
- Real-time failure probability prediction
- Risk level classification
- Batch predictions
- Model information retrieval
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import json

from config.settings import settings


class PredictiveMaintenancePredictor:
    """
    ML Predictor for failure probability prediction
    
    Attributes:
        model: Trained RandomForest classifier
        scaler: Feature scaler (StandardScaler)
        feature_names: List of expected feature names
        model_path: Path to saved model file
        scaler_path: Path to saved scaler file
        model_loaded: Whether model is successfully loaded
        model_metadata: Metadata about the model
    """
    
    def __init__(
        self, 
        model_path: Optional[Path] = None, 
        scaler_path: Optional[Path] = None
    ):
        """
        Initialize the predictor and load model
        
        Args:
            model_path: Path to model file (default: from settings)
            scaler_path: Path to scaler file (default: from settings)
        """
        self.model = None
        self.scaler = None
        self.feature_names = ['temperature', 'vibration', 'pressure', 'rpm']
        self.model_loaded = False
        self.model_metadata = {}
        
        # Use default paths if not provided
        if model_path is None:
            model_path = settings.MODEL_PATH
        if scaler_path is None:
            scaler_path = settings.SCALER_PATH
        
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        # Load model on initialization
        try:
            self.load_model()
        except Exception as e:
            print(f"⚠ Warning: Could not load model on initialization: {e}")
            print("  Please train the model or check file paths.")
    
    def load_model(self) -> bool:
        """
        Load trained model and scaler from disk
        
        Returns:
            bool: True if successful, False otherwise
        
        Raises:
            FileNotFoundError: If model or scaler files don't exist
        """
        # Check if model file exists
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}\n"
                f"Please train the model first by running:\n"
                f"  python -m services.ml_service.trainer"
            )
        
        # Check if scaler file exists
        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler not found at {self.scaler_path}\n"
                f"Please train the model first by running:\n"
                f"  python -m services.ml_service.trainer"
            )
        
        try:
            # Load model
            self.model = joblib.load(self.model_path)
            print(f"✓ Model loaded from {self.model_path}")
            
            # Load scaler
            self.scaler = joblib.load(self.scaler_path)
            print(f"✓ Scaler loaded from {self.scaler_path}")
            
            # Extract model metadata
            self._extract_model_metadata()
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.model_loaded = False
            return False
    
    def _extract_model_metadata(self):
        """Extract metadata from loaded model"""
        try:
            if self.model is not None:
                self.model_metadata = {
                    'model_type': type(self.model).__name__,
                    'n_estimators': getattr(self.model, 'n_estimators', None),
                    'max_depth': getattr(self.model, 'max_depth', None),
                    'n_features': self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else len(self.feature_names),
                    'feature_names': self.feature_names,
                    'model_path': str(self.model_path),
                    'scaler_path': str(self.scaler_path),
                    'loaded_at': datetime.utcnow().isoformat()
                }
        except Exception as e:
            print(f"Warning: Could not extract model metadata: {e}")
            self.model_metadata = {}
    
    def _validate_input(self, sensor_data: Dict[str, float]) -> Dict[str, float]:
        """
        Validate and prepare input data
        
        Args:
            sensor_data: Dictionary with sensor readings
        
        Returns:
            Dict: Validated sensor data
        
        Raises:
            ValueError: If required features are missing or invalid
        """
        # Check all required features are present
        missing_features = [f for f in self.feature_names if f not in sensor_data]
        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}\n"
                f"Required features: {self.feature_names}"
            )
        
        # Validate data types and ranges
        validated_data = {}
        
        for feature in self.feature_names:
            value = sensor_data[feature]
            
            # Convert to float
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid value for {feature}: {value} (must be numeric)")
            
            # Check for NaN or Inf
            if np.isnan(value) or np.isinf(value):
                raise ValueError(f"Invalid value for {feature}: {value} (NaN or Inf)")
            
            # Range validation (basic sanity checks)
            if feature == 'temperature':
                if value < 0 or value > 200:
                    print(f"⚠ Warning: Temperature {value}°C is outside typical range (0-200)")
            elif feature == 'vibration':
                if value < 0 or value > 50:
                    print(f"⚠ Warning: Vibration {value} mm/s is outside typical range (0-50)")
            elif feature == 'pressure':
                if value < 0 or value > 500:
                    print(f"⚠ Warning: Pressure {value} bar is outside typical range (0-500)")
            elif feature == 'rpm':
                if value < 0 or value > 10000:
                    print(f"⚠ Warning: RPM {value} is outside typical range (0-10000)")
            
            validated_data[feature] = value
        
        return validated_data
    
    def predict_failure_probability(
        self, 
        sensor_data: Dict[str, float],
        return_details: bool = True
    ) -> Dict:
        """
        Predict failure probability from sensor readings
        
        Args:
            sensor_data: Dictionary with keys: temperature, vibration, pressure, rpm
            return_details: Whether to include detailed prediction info
        
        Returns:
            Dictionary containing:
                - failure_predicted: bool (will machine fail?)
                - failure_probability: float (0-1, probability of failure)
                - normal_probability: float (0-1, probability of normal operation)
                - risk_level: str (LOW/MEDIUM/HIGH)
                - sensor_data: dict (input sensor values)
                - prediction_time: str (timestamp)
                - confidence: str (prediction confidence level)
        
        Raises:
            RuntimeError: If model is not loaded
            ValueError: If input data is invalid
        
        Example:
            >>> predictor = PredictiveMaintenancePredictor()
            >>> result = predictor.predict_failure_probability({
            ...     'temperature': 85.0,
            ...     'vibration': 7.0,
            ...     'pressure': 120.0,
            ...     'rpm': 2400.0
            ... })
            >>> print(result['risk_level'])
            MEDIUM
        """
        # Check if model is loaded
        if not self.model_loaded or self.model is None:
            raise RuntimeError(
                "Model not loaded. Please load model first:\n"
                "  predictor.load_model()"
            )
        
        # Validate input
        validated_data = self._validate_input(sensor_data)
        
        # Prepare input array
        X = np.array([[
            validated_data['temperature'],
            validated_data['vibration'],
            validated_data['pressure'],
            validated_data['rpm']
        ]])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        # Get failure probability (probability of class 1)
        failure_probability = float(probability[1])
        normal_probability = float(probability[0])
        
        # Determine risk level
        if failure_probability < settings.LOW_RISK_THRESHOLD:
            risk_level = "LOW"
        elif failure_probability < settings.MEDIUM_RISK_THRESHOLD:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Calculate confidence (how certain is the prediction?)
        confidence_score = max(probability)  # Highest probability
        if confidence_score > 0.9:
            confidence = "VERY_HIGH"
        elif confidence_score > 0.75:
            confidence = "HIGH"
        elif confidence_score > 0.6:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Build result
        result = {
            "failure_predicted": bool(prediction),
            "failure_probability": failure_probability,
            "normal_probability": normal_probability,
            "risk_level": risk_level,
            "sensor_data": validated_data,
            "prediction_time": datetime.utcnow().isoformat(),
            "confidence": confidence
        }
        
        # Add detailed information if requested
        if return_details:
            result.update({
                "thresholds": {
                    "low_risk": settings.LOW_RISK_THRESHOLD,
                    "medium_risk": settings.MEDIUM_RISK_THRESHOLD,
                    "high_risk": settings.HIGH_RISK_THRESHOLD
                },
                "model_info": {
                    "type": self.model_metadata.get('model_type', 'Unknown'),
                    "features_used": self.feature_names
                }
            })
        
        return result
    
    def batch_predict(
        self, 
        sensor_data_list: List[Dict[str, float]],
        return_details: bool = False
    ) -> List[Dict]:
        """
        Predict for multiple sensor readings (batch prediction)
        
        Args:
            sensor_data_list: List of sensor data dictionaries
            return_details: Whether to include detailed info for each prediction
        
        Returns:
            List of prediction result dictionaries
        
        Example:
            >>> data_list = [
            ...     {'temperature': 60, 'vibration': 3, 'pressure': 100, 'rpm': 2000},
            ...     {'temperature': 85, 'vibration': 7, 'pressure': 120, 'rpm': 2400}
            ... ]
            >>> results = predictor.batch_predict(data_list)
            >>> len(results)
            2
        """
        predictions = []
        
        for i, sensor_data in enumerate(sensor_data_list):
            try:
                prediction = self.predict_failure_probability(
                    sensor_data, 
                    return_details=return_details
                )
                prediction['batch_index'] = i
                predictions.append(prediction)
            except Exception as e:
                # Include error in results
                predictions.append({
                    'batch_index': i,
                    'error': str(e),
                    'sensor_data': sensor_data
                })
        
        return predictions
    
    def predict_from_dataframe(
        self, 
        df: pd.DataFrame,
        return_dataframe: bool = True
    ) -> Union[pd.DataFrame, List[Dict]]:
        """
        Predict from pandas DataFrame
        
        Args:
            df: DataFrame with columns: temperature, vibration, pressure, rpm
            return_dataframe: If True, return DataFrame; if False, return list of dicts
        
        Returns:
            DataFrame or list with predictions
        
        Example:
            >>> df = pd.DataFrame({
            ...     'temperature': [60, 85],
            ...     'vibration': [3, 7],
            ...     'pressure': [100, 120],
            ...     'rpm': [2000, 2400]
            ... })
            >>> results_df = predictor.predict_from_dataframe(df)
        """
        # Validate DataFrame has required columns
        missing_cols = [col for col in self.feature_names if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")
        
        # Convert to list of dictionaries
        sensor_data_list = df[self.feature_names].to_dict('records')
        
        # Batch predict
        predictions = self.batch_predict(sensor_data_list, return_details=False)
        
        if return_dataframe:
            # Convert predictions to DataFrame
            pred_df = pd.DataFrame(predictions)
            # Combine with original data
            result_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
            return result_df
        else:
            return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the model
        
        Returns:
            Dictionary mapping feature names to importance scores
        
        Example:
            >>> importance = predictor.get_feature_importance()
            >>> print(importance['temperature'])
            0.35
        """
        if not self.model_loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        
        if not hasattr(self.model, 'feature_importances_'):
            return {feature: None for feature in self.feature_names}
        
        importances = self.model.feature_importances_
        return {
            feature: float(importance) 
            for feature, importance in zip(self.feature_names, importances)
        }
    
    def get_model_info(self) -> Dict:
        """
        Get comprehensive model information
        
        Returns:
            Dictionary with model metadata and status
        
        Example:
            >>> info = predictor.get_model_info()
            >>> print(info['model_loaded'])
            True
        """
        info = {
            "model_loaded": self.model_loaded,
            "model_path": str(self.model_path),
            "scaler_path": str(self.scaler_path),
            "model_exists": self.model_path.exists(),
            "scaler_exists": self.scaler_path.exists(),
            "features": self.feature_names,
            "n_features": len(self.feature_names)
        }
        
        # Add model metadata if loaded
        if self.model_loaded and self.model_metadata:
            info.update(self.model_metadata)
        
        # Add feature importance if available
        try:
            info['feature_importance'] = self.get_feature_importance()
        except:
            info['feature_importance'] = None
        
        return info
    
    def get_prediction_explanation(
        self, 
        sensor_data: Dict[str, float]
    ) -> Dict:
        """
        Get detailed explanation of prediction
        
        Args:
            sensor_data: Sensor readings
        
        Returns:
            Dictionary with prediction explanation
        
        Example:
            >>> explanation = predictor.get_prediction_explanation({
            ...     'temperature': 85, 'vibration': 7, 
            ...     'pressure': 120, 'rpm': 2400
            ... })
            >>> print(explanation['summary'])
        """
        # Get prediction
        prediction = self.predict_failure_probability(sensor_data)
        
        # Get feature importance
        try:
            feature_importance = self.get_feature_importance()
        except:
            feature_importance = None
        
        # Analyze which features are out of normal range
        abnormal_features = []
        
        if sensor_data['temperature'] > 85:
            abnormal_features.append({
                'feature': 'temperature',
                'value': sensor_data['temperature'],
                'status': 'High',
                'threshold': 85
            })
        
        if sensor_data['vibration'] > 6:
            abnormal_features.append({
                'feature': 'vibration',
                'value': sensor_data['vibration'],
                'status': 'High',
                'threshold': 6
            })
        
        if sensor_data['pressure'] > 130:
            abnormal_features.append({
                'feature': 'pressure',
                'value': sensor_data['pressure'],
                'status': 'High',
                'threshold': 130
            })
        
        if sensor_data['rpm'] > 2600:
            abnormal_features.append({
                'feature': 'rpm',
                'value': sensor_data['rpm'],
                'status': 'High',
                'threshold': 2600
            })
        
        # Generate summary
        risk_level = prediction['risk_level']
        prob = prediction['failure_probability']
        
        if risk_level == "HIGH":
            summary = (
                f"High failure risk detected ({prob:.1%} probability). "
                f"Machine is operating in dangerous conditions. "
                f"{len(abnormal_features)} sensor(s) showing abnormal readings."
            )
        elif risk_level == "MEDIUM":
            summary = (
                f"Moderate failure risk detected ({prob:.1%} probability). "
                f"Machine showing early warning signs. "
                f"{len(abnormal_features)} sensor(s) elevated."
            )
        else:
            summary = (
                f"Low failure risk ({prob:.1%} probability). "
                f"Machine operating within acceptable parameters."
            )
        
        return {
            "summary": summary,
            "prediction": prediction,
            "abnormal_features": abnormal_features,
            "feature_importance": feature_importance,
            "recommendation": self._get_recommendation(risk_level)
        }
    
    def _get_recommendation(self, risk_level: str) -> str:
        """Get recommendation based on risk level"""
        recommendations = {
            "LOW": "Continue normal operation. Maintain regular monitoring schedule.",
            "MEDIUM": "Schedule preventive maintenance within 24-48 hours. Increase monitoring frequency.",
            "HIGH": "Immediate action required. Consider emergency shutdown and inspection."
        }
        return recommendations.get(risk_level, "Unable to provide recommendation")
    
    def save_prediction_log(
        self, 
        prediction: Dict, 
        log_path: Optional[Path] = None
    ):
        """
        Save prediction to log file
        
        Args:
            prediction: Prediction result dictionary
            log_path: Path to log file (default: logs/predictions.jsonl)
        """
        if log_path is None:
            log_path = Path("logs/predictions.jsonl")
        
        # Ensure log directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Append prediction to log file (JSONL format)
        with open(log_path, 'a') as f:
            f.write(json.dumps(prediction) + '\n')
    
    def reload_model(self) -> bool:
        """
        Reload model from disk (useful if model has been retrained)
        
        Returns:
            bool: True if successful
        """
        print("Reloading model...")
        return self.load_model()


# ==============================================================================
# Testing and Demo Functions
# ==============================================================================

def demo_predictions():
    """Run demo predictions with sample data"""
    print("\n" + "="*70)
    print("PREDICTIVE MAINTENANCE PREDICTOR - DEMO")
    print("="*70)
    
    # Initialize predictor
    try:
        predictor = PredictiveMaintenancePredictor()
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return
    
    # Test cases
    test_cases = [
        {
            "name": "Normal Operation",
            "data": {"temperature": 60, "vibration": 3, "pressure": 100, "rpm": 2000}
        },
        {
            "name": "Slightly Elevated",
            "data": {"temperature": 75, "vibration": 4.5, "pressure": 110, "rpm": 2200}
        },
        {
            "name": "High Temperature",
            "data": {"temperature": 90, "vibration": 3, "pressure": 100, "rpm": 2000}
        },
        {
            "name": "High Vibration",
            "data": {"temperature": 60, "vibration": 8, "pressure": 100, "rpm": 2000}
        },
        {
            "name": "Critical - Multiple Anomalies",
            "data": {"temperature": 95, "vibration": 9, "pressure": 140, "rpm": 2800}
        }
    ]
    
    print("\nRunning test predictions...\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"{i}. {test['name']}")
        print(f"   Input: {test['data']}")
        
        try:
            result = predictor.predict_failure_probability(test['data'])
            print(f"   → Failure Probability: {result['failure_probability']:.2%}")
            print(f"   → Risk Level: {result['risk_level']}")
            print(f"   → Confidence: {result['confidence']}")
            print(f"   → Predicted Failure: {result['failure_predicted']}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
        
        print()
    
    # Model info
    print("="*70)
    print("MODEL INFORMATION")
    print("="*70)
    info = predictor.get_model_info()
    print(f"Model Type: {info.get('model_type', 'N/A')}")
    print(f"Features: {', '.join(info['features'])}")
    print(f"N Estimators: {info.get('n_estimators', 'N/A')}")
    print(f"Max Depth: {info.get('max_depth', 'N/A')}")
    
    # Feature importance
    if info.get('feature_importance'):
        print("\nFeature Importance:")
        for feature, importance in info['feature_importance'].items():
            if importance is not None:
                print(f"  {feature:12s}: {importance:.4f}")
    
    print("="*70)


if __name__ == "__main__":
    demo_predictions()