import requests
import time
import random

def push_data():
    while True:
        payload = {
            "machine_name": "Machine-001",
            "temperature": round(60 + random.uniform(-5, 5), 2),
            "vibration": round(3 + random.uniform(-1, 1), 2),
            "pressure": round(100 + random.uniform(-10, 10), 2),
            "rpm": round(2000 + random.uniform(-200, 200), 2)
        }
        try:
            requests.post("http://127.0.0.1:8000/api/telemetry", json=payload)
            print(f"📡 Data Pushed: {payload['temperature']}°C")
        except:
            print("❌ Gateway Offline")
        time.sleep(3)

if __name__ == "__main__":
    push_data()