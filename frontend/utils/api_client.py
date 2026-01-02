"""
API Client Utility
Handles all communication with API Gateway
"""
import requests
from typing import Dict, Optional
from config.settings import settings

class APIClient:
    """Client for interacting with PredictXAI API Gateway"""
    
    def __init__(self):
        self.base_url = f"http://{settings.GATEWAY_HOST}:{settings.GATEWAY_PORT}"
        self.token: Optional[str] = None
        self.user: Optional[Dict] = None
    
    def _get_headers(self) -> Dict:
        """Get headers with authentication token"""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    def register(self, username: str, email: str, password: str, role: str = "operator") -> Dict:
        """Register new user"""
        response = requests.post(
            f"{self.base_url}/api/auth/register",
            json={
                "username": username,
                "email": email,
                "password": password,
                "role": role
            }
        )
        return response.json() if response.status_code == 201 else {"error": response.json().get("detail")}
    
    def login(self, username: str, password: str) -> Dict:
        """Login user"""
        response = requests.post(
            f"{self.base_url}/api/auth/login",
            json={
                "username": username,
                "password": password
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            self.token = data["access_token"]
            self.user = data["user"]
            return {"success": True, "user": self.user}
        else:
            return {"success": False, "error": "Invalid credentials"}
    
    def logout(self):
        """Logout user"""
        self.token = None
        self.user = None
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return self.token is not None
    
    def get_health(self) -> Dict:
        """Check system health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.json()
        except:
            return {"status": "unavailable"}
    
    def predict_failure(self, sensor_data: Dict) -> Dict:
        """Get ML prediction"""
        try:
            response = requests.post(
                f"{self.base_url}/api/ml/predict",
                json=sensor_data,
                headers=self._get_headers(),
                timeout=30
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail")}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def analyze_machine(self, sensor_data: Dict) -> Dict:
        """Execute multi-agent analysis"""
        try:
            response = requests.post(
                f"{self.base_url}/api/agents/analyze",
                json=sensor_data,
                headers=self._get_headers(),
                timeout=60
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail")}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_model_info(self) -> Dict:
        """Get ML model information"""
        try:
            response = requests.get(
                f"{self.base_url}/api/ml/model/info",
                headers=self._get_headers(),
                timeout=10
            )
            return response.json()
        except:
            return {"model_loaded": False}
    
    def train_model(self, n_samples: int = 10000) -> Dict:
        """Trigger model training"""
        try:
            response = requests.post(
                f"{self.base_url}/api/ml/train",
                params={"n_samples": n_samples},
                headers=self._get_headers(),
                timeout=10
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_agents_status(self) -> Dict:
        """Get agent status"""
        try:
            response = requests.get(
                f"{self.base_url}/api/agents/status",
                headers=self._get_headers(),
                timeout=10
            )
            return response.json()
        except:
            return {"error": "Agent service unavailable"}