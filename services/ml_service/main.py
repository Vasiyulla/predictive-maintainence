"""
ML Service API
Microservice for model training and prediction
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from pathlib import Path
import traceback

from config.settings import settings
from .predictor import PredictiveMaintenancePredictor
from .trainer import PredictiveMaintenanceTrainer
from .data_generator import generate_sensor_data

# Initialize FastAPI app
app = FastAPI(
    title=f"{settings.APP_NAME} - ML Service",
    version=settings.APP_VERSION,
    description="Machine Learning Service for Predictive Maintenance"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor: Optional[PredictiveMaintenancePredictor] = None

@app.on_event("startup")
async def startup_event():
    """Initialize ML predictor on startup"""
    global predictor
    try:
        predictor = PredictiveMaintenancePredictor()
        print("✓ ML Predictor initialized successfully")
    except FileNotFoundError as e:
        print(f"⚠ Warning: {e}")
        print("  Model not found. Please train the model using /train endpoint")
        predictor = None

# Pydantic models
class SensorData(BaseModel):
    """Schema for sensor input data"""
    temperature: float = Field(..., ge=0, le=150, description="Temperature in °C")
    vibration: float = Field(..., ge=0, le=20, description="Vibration in mm/s")
    pressure: float = Field(..., ge=0, le=300, description="Pressure in bar")
    rpm: float = Field(..., ge=0, le=5000, description="RPM")

    # ... [Imports] ...
class TrainingRequest(BaseModel):
    """Schema for training request"""
    n_samples: int = Field(default=10000, ge=1000, le=100000)
    regenerate_data: bool = Field(default=False)
    model_type: str = Field(default="random_forest", description="Algorithm: random_forest, svm, or lstm")

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    failure_predicted: bool
    failure_probability: float
    normal_probability: float
    risk_level: str
    sensor_data: Dict[str, float]

class TrainingRequest(BaseModel):
    """Schema for training request"""
    n_samples: int = Field(default=10000, ge=1000, le=100000)
    regenerate_data: bool = Field(default=False)

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "service": "ml_service",
        "status": "healthy",
        "version": settings.APP_VERSION,
        "model_loaded": predictor is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_failure(sensor_data: SensorData):
    """Predict failure probability from sensor readings"""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Please train the model first using /train endpoint"
        )
    
    try:
        result = predictor.predict_failure_probability(sensor_data.dict())
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch")
async def batch_predict(sensor_data_list: List[SensorData]):
    """Batch prediction for multiple sensor readings"""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Please train the model first"
        )
    
    try:
        data_dicts = [data.dict() for data in sensor_data_list]
        results = predictor.batch_predict(data_dicts)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    if predictor is None:
        return {
            "model_loaded": False,
            "message": "Model not loaded"
        }
    
    return predictor.get_model_info()

@app.post("/train")
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Train or retrain the ML model
    This is a long-running operation executed in the background
    """
    def training_task():
        global predictor
        try:
            # Paths
            base_dir = Path(__file__).parent.parent.parent
            data_path = base_dir / "data" / "raw" / "sensor_data.csv"
            model_path = settings.MODEL_PATH
            scaler_path = settings.SCALER_PATH
            
            # Generate data if requested or doesn't exist
            if request.regenerate_data or not data_path.exists():
                print(f"Generating {request.n_samples} training samples...")
                generate_sensor_data(n_samples=request.n_samples, save_path=data_path)
            
            # Train model
            trainer = PredictiveMaintenanceTrainer()
            metrics = trainer.full_training_pipeline(data_path, model_path, scaler_path)
            
            # Reload predictor
            predictor = PredictiveMaintenancePredictor()
            
            print("✓ Training completed successfully")
            return metrics
            
        except Exception as e:
            print(f"✗ Training failed: {str(e)}")
            traceback.print_exc()
            raise
    
    # Start training in background
    background_tasks.add_task(training_task)
    
    return {
        "status": "training_started",
        "message": "Model training started in background",
        "n_samples": request.n_samples
    }

# ... [Inside train_model endpoint] ...
@app.post("/train")
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """Train or retrain the ML model with selected algorithm"""
    def training_task():
        global predictor
        try:
            # ... [Path setup] ...
            
            # Train model with selected type
            trainer = PredictiveMaintenanceTrainer()
            metrics = trainer.full_training_pipeline(
                data_path, 
                model_path, 
                scaler_path, 
                model_type=request.model_type # Pass model type
            )
            
            # Reload predictor
            if predictor:
                predictor.reload_model()
            
            print("✓ Training completed successfully")
            return metrics
            
        except Exception as e:
            # ... [Error handling] ...
            pass
    
    background_tasks.add_task(training_task)
    return {
        "status": "training_started",
        "algorithm": request.model_type,
        "n_samples": request.n_samples
    }
@app.get("/training/status")
async def get_training_status():
    """Get current training status"""
    # In production, this would check a job queue
    # For now, just return model status
    return {
        "model_loaded": predictor is not None,
        "model_path": str(settings.MODEL_PATH),
        "model_exists": settings.MODEL_PATH.exists()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.ML_SERVICE_HOST,
        port=settings.ML_SERVICE_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )

