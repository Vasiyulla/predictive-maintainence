import requests
import time
import random

GATEWAY_URL = "http://127.0.0.1:8000"

def get_db_machines():
    """Fetch the list of registered machine names from the platform"""
    try:
        # Accesses the now-public Gateway route
        response = requests.get(f"{GATEWAY_URL}/api/machines", timeout=5)
        if response.status_code == 200:
            machines = response.json()
            # Extract names from the objects returned by Auth Service
            return [m['name'] for m in machines]
        else:
            print(f"❌ Error: Gateway returned {response.status_code}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")
    return []

def push_data():
    print("🚀 Starting Dynamic Hardware Simulator...")
    while True:
        machines = get_db_machines()
        
        if not machines:
            print("⚠️ No machines found in DB. Check Gateway and Auth services.")
            time.sleep(5)
            continue

        for machine_name in machines:
            payload = {
                "machine_name": machine_name, 
                "temperature": round(60 + random.uniform(-5, 5), 2),
                "vibration": round(3 + random.uniform(-1, 1), 2),
                "pressure": round(100 + random.uniform(-10, 10), 2),
                "rpm": round(2000 + random.uniform(-200, 200), 2)
            }
            try:
                # Posts to the newly added Gateway route
                requests.post(f"{GATEWAY_URL}/api/telemetry", json=payload, timeout=5)
                print(f"📡 Data Pushed for {machine_name}: {payload['temperature']}°C")
            except Exception as e:
                print(f"❌ Failed to push data for {machine_name}: {e}")
        
        time.sleep(3)

if __name__ == "__main__":
    push_data()