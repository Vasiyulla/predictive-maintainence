"""
API Gateway
Central routing service for all microservices
"""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, Optional
import httpx

from config.settings import settings

# Initialize FastAPI app
app = FastAPI(
    title=f"{settings.APP_NAME} - API Gateway",
    version=settings.APP_VERSION,
    description="Central API Gateway for PredictXAI Platform"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Service URLs
AUTH_SERVICE_URL = f"http://{settings.AUTH_SERVICE_HOST}:{settings.AUTH_SERVICE_PORT}"
ML_SERVICE_URL = f"http://{settings.ML_SERVICE_HOST}:{settings.ML_SERVICE_PORT}"
AGENT_SERVICE_URL = f"http://{settings.AGENT_SERVICE_HOST}:{settings.AGENT_SERVICE_PORT}"

# Pydantic Models
class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    role: str = "operator"

class UserLogin(BaseModel):
    username: str
    password: str

class SensorData(BaseModel):
    temperature: float
    vibration: float
    pressure: float
    rpm: float

# Helper Functions
async def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Verify JWT token with auth service"""
    if not credentials:
        return None
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{AUTH_SERVICE_URL}/verify",
                params={"token": credentials.credentials}
            )
            
            if response.status_code == 200:
                return response.json()
            return None
    except:
        return None

# Health Check
@app.get("/health")
async def health_check():
    """Gateway health check"""
    services_health = {}
    
    # Check all services
    services = {
        "auth": AUTH_SERVICE_URL,
        "ml": ML_SERVICE_URL,
        "agent": AGENT_SERVICE_URL
    }
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        for service_name, url in services.items():
            try:
                response = await client.get(f"{url}/health")
                services_health[service_name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
            except:
                services_health[service_name] = {
                    "status": "unavailable",
                    "response_time_ms": None
                }
    
    all_healthy = all(s["status"] == "healthy" for s in services_health.values())
    
    return {
        "service": "api_gateway",
        "status": "healthy" if all_healthy else "degraded",
        "version": settings.APP_VERSION,
        "services": services_health
    }

# Authentication Routes
@app.post("/api/auth/register")
async def register(user: UserRegister):
    """Register new user"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{AUTH_SERVICE_URL}/register",
                json=user.dict()
            )
            
            if response.status_code == 201:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=response.json().get("detail", "Registration failed")
                )
    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Auth service unavailable"
        )

@app.post("/api/auth/login")
async def login(credentials: UserLogin):
    """Login user"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{AUTH_SERVICE_URL}/login",
                json=credentials.dict()
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials"
                )
    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Auth service unavailable"
        )

# ML Service Routes
@app.post("/api/ml/predict")
async def predict(
    sensor_data: SensorData,
    user=Depends(verify_token)
):
    """Get ML prediction"""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ML_SERVICE_URL}/predict",
                json=sensor_data.dict()
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=response.json().get("detail", "Prediction failed")
                )
    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="ML service unavailable"
        )

@app.get("/api/ml/model/info")
async def get_model_info(user=Depends(verify_token)):
    """Get ML model information"""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ML_SERVICE_URL}/model/info")
            return response.json()
    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="ML service unavailable"
        )

@app.post("/api/ml/train")
async def train_model(
    n_samples: int = 10000,
    user=Depends(verify_token)
):
    """Trigger model training"""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ML_SERVICE_URL}/train",
                json={"n_samples": n_samples, "regenerate_data": True}
            )
            return response.json()
    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="ML service unavailable"
        )

# Agent Service Routes
@app.post("/api/agents/analyze")
async def analyze_machine(
    sensor_data: SensorData,
    user=Depends(verify_token)
):
    """Execute multi-agent analysis"""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{AGENT_SERVICE_URL}/analyze",
                json=sensor_data.dict()
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=response.json().get("detail", "Analysis failed")
                )
    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Agent service unavailable"
        )

@app.post("/api/agents/analyze/summary")
async def analyze_with_summary(
    sensor_data: SensorData,
    user=Depends(verify_token)
):
    """Execute analysis with executive summary"""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{AGENT_SERVICE_URL}/analyze/summary",
                json=sensor_data.dict()
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Analysis failed"
                )
    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Agent service unavailable"
        )

@app.get("/api/agents/status")
async def get_agents_status(user=Depends(verify_token)):
    """Get agent status"""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{AGENT_SERVICE_URL}/agents/status")
            return response.json()
    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Agent service unavailable"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.GATEWAY_HOST,
        port=settings.GATEWAY_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )