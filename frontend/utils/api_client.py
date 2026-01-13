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
      
    def add_machine(self, machine_data: Dict) -> Dict:
        """Add a new machine to the database"""
        try:
            response = requests.post(
                f"{self.base_url}/api/machines",
                json=machine_data,
                headers=self._get_headers(),
                timeout=10
            )
            if response.status_code == 201 or response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": response.json().get("detail", "Failed to add machine")}
        except Exception as e:
            return {"success": False, "error": str(e)} 
         
    def get_machines(self) -> list:
        """Fetch machine names for the dashboard dropdown"""
        if not self.token:
            return []
        try:
            # Use self.base_url to match your __init__ 
            headers = self._get_headers() 
            response = requests.get(
                f"{self.base_url}/api/machines", 
                headers=headers, 
                timeout=10
            )
            
            if response.status_code == 200:
                # Ensure we are parsing the JSON correctly
                data = response.json()
                # If the backend returns a list of objects, extract the names
                return [m['name'] for m in data]
            return []
        except Exception as e:
            # Log the error to terminal so you can see why it fails
            print(f"DEBUG: Error fetching machines: {e}")
            return []
    def add_machine(self, machine_data: dict) -> dict:
        """Sends new machine data to the backend"""
        try:
            response = requests.post(
                f"{self.base_url}/api/machines", 
                json=machine_data, 
                headers=self._get_headers(),
                timeout=10
            )
            # Check for successful creation (201) or standard success (200)
            if response.status_code in [200, 201]:
                return {"success": True, "data": response.json()}
            
            # Extract error detail if available in JSON, else use raw text
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except:
                error_detail = response.text or f"Status Code {response.status_code}"
                
            return {"success": False, "error": error_detail}
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
    # frontend/utils/api_client.py

    def get_latest_telemetry(self, machine_name: str) -> dict:
        """Fetch the most recent sensor data for a machine"""
        if not self.token:
            return {"success": False, "error": "No token"}
        try:
            response = requests.get(
                f"{self.base_url}/api/telemetry/latest/{machine_name}", 
                headers=self._get_headers(),
                timeout=10
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            return {"success": False, "error": "No data found"}
        except Exception as e:
            return {"success": False, "error": str(e)}