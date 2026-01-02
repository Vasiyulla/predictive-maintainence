"""
PredictXAI Configuration Settings
Centralized configuration management for all services
"""
import os
from pathlib import Path
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application Info
    APP_NAME: str = "PredictXAI"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"
    
    # API Gateway
    GATEWAY_HOST: str = "127.0.0.1"
    GATEWAY_PORT: int = 8000
    
    # Auth Service
    AUTH_SERVICE_HOST: str = "127.0.0.1"
    AUTH_SERVICE_PORT: int = 8001
    SECRET_KEY: str = "your-secret-key-change-in-production-min-32-chars"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours
    
    # ML Service
    ML_SERVICE_HOST: str = "127.0.0.1"
    ML_SERVICE_PORT: int = 8002
    MODEL_PATH: Path = BASE_DIR / "data" / "models" / "predictive_model.pkl"
    SCALER_PATH: Path = BASE_DIR / "data" / "models" / "scaler.pkl"
    
    # Agent Service
    AGENT_SERVICE_HOST: str = "127.0.0.1"
    AGENT_SERVICE_PORT: int = 8003
    
    # Frontend
    FRONTEND_HOST: str = "127.0.0.1"
    FRONTEND_PORT: int = 8501
    
    # Database
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/database/predictxai.db"
    
    # ML Configuration
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    N_ESTIMATORS: int = 100
    MAX_DEPTH: int = 10
    
    # Monitoring Thresholds
    TEMPERATURE_MIN: float = 20.0
    TEMPERATURE_MAX: float = 100.0
    VIBRATION_MIN: float = 0.0
    VIBRATION_MAX: float = 10.0
    PRESSURE_MIN: float = 50.0
    PRESSURE_MAX: float = 150.0
    RPM_MIN: float = 500.0
    RPM_MAX: float = 3000.0
    
    # Risk Thresholds
    LOW_RISK_THRESHOLD: float = 0.3
    MEDIUM_RISK_THRESHOLD: float = 0.6
    HIGH_RISK_THRESHOLD: float = 0.6
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path = BASE_DIR / "logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Ensure required directories exist
settings.LOG_DIR.mkdir(exist_ok=True)
(BASE_DIR / "data" / "models").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "database").mkdir(exist_ok=True)