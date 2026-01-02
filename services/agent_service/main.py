"""
Agent Service API
Microservice for AI agent orchestration
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict

from config.settings import settings
from .orchestrator import AgentOrchestrator

# Initialize FastAPI app
app = FastAPI(
    title=f"{settings.APP_NAME} - Agent Service",
    version=settings.APP_VERSION,
    description="AI Agent Orchestration Service"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator
orchestrator = AgentOrchestrator()

# Pydantic models
class SensorData(BaseModel):
    """Schema for sensor input"""
    temperature: float = Field(..., ge=0, le=150)
    vibration: float = Field(..., ge=0, le=20)
    pressure: float = Field(..., ge=0, le=300)
    rpm: float = Field(..., ge=0, le=5000)

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "service": "agent_service",
        "status": "healthy",
        "version": settings.APP_VERSION,
        "orchestrator": orchestrator.name,
        "agents": [
            "MonitoringAgent",
            "PredictionAgent",
            "AlertAgent",
            "MaintenanceAgent"
        ]
    }

@app.post("/analyze")
async def analyze_machine(sensor_data: SensorData):
    """
    Execute complete multi-agent analysis
    This is the main endpoint that orchestrates all agents
    """
    try:
        result = await orchestrator.analyze_machine_status(sensor_data.dict())
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/analyze/summary")
async def analyze_with_summary(sensor_data: SensorData):
    """Analyze and return executive summary"""
    try:
        result = await orchestrator.analyze_machine_status(sensor_data.dict())
        summary = orchestrator.generate_executive_summary(result)
        
        return {
            "decision": result,
            "executive_summary": summary
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/agents/status")
async def get_agents_status():
    """Get status of all agents"""
    return {
        "orchestrator": orchestrator.name,
        "agents": {
            "monitoring": {
                "name": orchestrator.monitoring_agent.name,
                "status": "active",
                "type": "rule-based"
            },
            "prediction": {
                "name": orchestrator.prediction_agent.name,
                "status": "active",
                "type": "ml-powered"
            },
            "alert": {
                "name": orchestrator.alert_agent.name,
                "status": "active",
                "type": "decision-making"
            },
            "maintenance": {
                "name": orchestrator.maintenance_agent.name,
                "status": "active",
                "type": "recommendation"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.AGENT_SERVICE_HOST,
        port=settings.AGENT_SERVICE_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )