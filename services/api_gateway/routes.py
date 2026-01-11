"""
API Gateway Routes
Central routing layer for PredictXAI microservices
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel 
from typing import Optional, List
import httpx

from config.settings import settings

router = APIRouter()
security = HTTPBearer(auto_error=False)

# -----------------------------
# Service URLs
# -----------------------------
# AUTH_SERVICE_URL = f"http://{settings.AUTH_SERVICE_HOST}:{settings.AUTH_SERVICE_PORT}"
# With this for local development
AUTH_SERVICE_URL = f"http://127.0.0.1:{settings.AUTH_SERVICE_PORT}"
ML_SERVICE_URL = f"http://{settings.ML_SERVICE_HOST}:{settings.ML_SERVICE_PORT}"
AGENT_SERVICE_URL = f"http://{settings.AGENT_SERVICE_HOST}:{settings.AGENT_SERVICE_PORT}"


# -----------------------------
# Pydantic Models
# -----------------------------
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

class MachineCreate(BaseModel):
    name: str
    type: str
    location: Optional[str]
    features: List[str]
    settings: Optional[dict] = None

# -----------------------------
# Token Verification
# -----------------------------
async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
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
    except httpx.RequestError:
        return None


# -----------------------------
# Health Check
# -----------------------------
@router.get("/health")
async def health_check():
    services = {
        "auth": AUTH_SERVICE_URL,
        "ml": ML_SERVICE_URL,
        "agent": AGENT_SERVICE_URL
    }

    results = {}

    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in services.items():
            try:
                response = await client.get(f"{url}/health")
                results[name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
            except:
                results[name] = {
                    "status": "unavailable",
                    "response_time_ms": None
                }

    return {
        "service": "api_gateway",
        "version": settings.APP_VERSION,
        "services": results
    }


# -----------------------------
# AUTH ROUTES
# -----------------------------
@router.post("/api/auth/register")
async def register(user: UserRegister):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{AUTH_SERVICE_URL}/register",
                json=user.dict()
            )

            if response.status_code == 201:
                return response.json()

            raise HTTPException(
                status_code=response.status_code,
                detail=response.json().get("detail", "Registration failed")
            )

    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Auth service unavailable")


@router.post("/api/auth/login")
async def login(credentials: UserLogin):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{AUTH_SERVICE_URL}/login",
                json=credentials.dict()
            )

            if response.status_code == 200:
                return response.json()

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Auth service unavailable")

# -----------------------------
# MACHINE ROUTES
# -----------------------------
# Add machine management routes
@router.post("/api/machines")
async def add_machine_proxy(machine_data: dict, user=Depends(verify_token)):
    if not user: raise HTTPException(status_code=401)
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{AUTH_SERVICE_URL}/machines", json=machine_data)
        return response.json()

@router.get("/api/machines")
async def get_machines_proxy(user=Depends(verify_token)):
    if not user: raise HTTPException(status_code=401)
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{AUTH_SERVICE_URL}/machines")
        return response.json()

# -----------------------------
# ML ROUTES
# -----------------------------
@router.post("/api/ml/predict")
async def predict(sensor_data: SensorData, user=Depends(verify_token)):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ML_SERVICE_URL}/predict",
                json=sensor_data.dict()
            )
            return response.json()
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="ML service unavailable")


@router.get("/api/ml/model/info")
async def model_info(user=Depends(verify_token)):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ML_SERVICE_URL}/model/info")
            return response.json()
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="ML service unavailable")


@router.post("/api/ml/train")
async def train_model(n_samples: int = 10000, user=Depends(verify_token)):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ML_SERVICE_URL}/train",
                json={"n_samples": n_samples, "regenerate_data": True}
            )
            return response.json()
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="ML service unavailable")


# -----------------------------
# AGENT ROUTES
# -----------------------------
@router.post("/api/agents/analyze")
async def analyze(sensor_data: SensorData, user=Depends(verify_token)):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{AGENT_SERVICE_URL}/analyze",
                json=sensor_data.dict()
            )
            return response.json()
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Agent service unavailable")


@router.post("/api/agents/analyze/summary")
async def analyze_summary(sensor_data: SensorData, user=Depends(verify_token)):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{AGENT_SERVICE_URL}/analyze/summary",
                json=sensor_data.dict()
            )
            return response.json()
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Agent service unavailable")


@router.get("/api/agents/status")
async def agents_status(user=Depends(verify_token)):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{AGENT_SERVICE_URL}/agents/status")
            return response.json()
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Agent service unavailable")
