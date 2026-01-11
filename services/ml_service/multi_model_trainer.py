"""
Multi-Model Trainer
Supports RandomForest, SVM, and LSTM models
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from config.settings import settings

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not installed. LSTM training unavailable.")

class MultiModelTrainer:
    """Train models with RandomForest, SVM, or LSTM"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_type = None
        self.feature_names = ['temperature', 'vibration', 'pressure', 'rpm']
    
    def load_data(self, data_path: Path):
        """Load training data"""
        df = pd.read_csv(data_path)
        X = df[self.feature_names]
        y = df['failure']
        return X, y
    
    def train_random_forest(self, X_train, y_train, n_estimators=100, max_depth=10):
        """Train RandomForest model"""
        print("\n🌲 Training RandomForest...")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=settings.RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        self.model_type = 'RandomForest'
        print("✓ RandomForest training complete")
    
    def train_svm(self, X_train, y_train, kernel='rbf', C=1.0):
        """Train SVM model"""
        print("\n🎯 Training SVM...")
        self.model = SVC(
            kernel=kernel,
            C=C,
            probability=True,
            random_state=settings.RANDOM_STATE,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        self.model_type = 'SVM'
        print("✓ SVM training complete")
    
    def train_lstm(self, X_train, y_train, epochs=50, batch_size=32):
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for LSTM. Install: pip install tensorflow")
        
        print("\n🧠 Training LSTM...")
        
        # Reshape for LSTM: (samples, timesteps, features)
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(1, X_train.shape[1]), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train
        model.fit(
            X_train_lstm, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        self.model = model
        self.model_type = 'LSTM'
        print("✓ LSTM training complete")
    
    def train(self, data_path: Path, model_type='RandomForest', **kwargs):
        """Main training function"""
        print("="*60)
        print(f"TRAINING {model_type.upper()} MODEL")
        print("="*60)
        
        # Load data
        X, y = self.load_data(data_path)
        print(f"\n✓ Loaded {len(X)} samples")
        print(f"  - Normal: {(y == 0).sum()}")
        print(f"  - Failure: {(y == 1).sum()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=settings.TEST_SIZE,
            random_state=settings.RANDOM_STATE,
            stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train based on model type
        if model_type == 'RandomForest':
            self.train_random_forest(
                X_train_scaled, y_train,
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10)
            )
        elif model_type == 'SVM':
            self.train_svm(
                X_train_scaled, y_train,
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 1.0)
            )
        elif model_type == 'LSTM':
            self.train_lstm(
                X_train_scaled, y_train,
                epochs=kwargs.get('epochs', 50),
                batch_size=kwargs.get('batch_size', 32)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Evaluate
        self.evaluate(X_test_scaled, y_test, model_type)
        
        return self.model, self.scaler
    
    def evaluate(self, X_test, y_test, model_type):
        """Evaluate model"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        if model_type == 'LSTM':
            X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            y_pred_prob = self.model.predict(X_test_lstm)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        else:
            y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✓ Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Failure']))
    
    def save_model(self, model_path: Path, scaler_path: Path):
        """Save trained model"""
        if self.model_type == 'LSTM':
            self.model.save(str(model_path).replace('.pkl', '.h5'))
        else:
            joblib.dump(self.model, model_path)
        
        joblib.dump(self.scaler, scaler_path)
        
        # Save model info
        info = {
            'type': self.model_type,
            'features': self.feature_names
        }
        info_path = model_path.parent / 'model_info.json'
        import json
        with open(info_path, 'w') as f:
            json.dump(info, f)
        
        print(f"\n✓ Model saved to {model_path}")
        print(f"✓ Scaler saved to {scaler_path}")

if __name__ == "__main__":
    trainer = MultiModelTrainer()
    
    data_path = Path("data/raw/sensor_data.csv")
    model_path = Path("data/models/predictive_model.pkl")
    scaler_path = Path("data/models/scaler.pkl")
    
    # Train RandomForest
    trainer.train(data_path, model_type='RandomForest')
    trainer.save_model(model_path, scaler_path)