"""
Prediction Agent
Predicts failure probability using heuristic / ML-like logic
"""

from typing import Dict
import math

class PredictionAgent:
    """
    ML-powered agent that predicts machine failure probability
    """

    def __init__(self):
        self.name = "PredictionAgent"

    async def predict_failure(self, sensor_data: Dict[str, float]) -> Dict:
        """
        Predict probability of machine failure

        Args:
            sensor_data: temperature, vibration, pressure, rpm

        Returns:
            Prediction result dictionary
        """
        try:
            temperature = sensor_data["temperature"]
            vibration = sensor_data["vibration"]
            pressure = sensor_data["pressure"]
            rpm = sensor_data["rpm"]

            # ---- Normalization (simple scaling) ----
            temp_score = min(temperature / 100, 1.0)
            vib_score = min(vibration / 10, 1.0)
            pressure_score = min(pressure / 150, 1.0)
            rpm_score = min(rpm / 3000, 1.0)

            # ---- Weighted risk score ----
            risk_score = (
                0.35 * temp_score +
                0.30 * vib_score +
                0.20 * pressure_score +
                0.15 * rpm_score
            )

            failure_probability = round(min(risk_score, 1.0), 2)

            # ---- Risk classification ----
            if failure_probability >= 0.75:
                risk_level = "HIGH"
                confidence = "HIGH"
            elif failure_probability >= 0.40:
                risk_level = "MEDIUM"
                confidence = "MEDIUM"
            else:
                risk_level = "LOW"
                confidence = "HIGH"

            return {
                "agent": self.name,
                "failure_probability": failure_probability,
                "risk_level": risk_level,
                "confidence": confidence,
                "analysis": (
                    f"Calculated risk score based on "
                    f"temperature={temperature}, vibration={vibration}, "
                    f"pressure={pressure}, rpm={rpm}"
                )
            }

        except Exception as e:
            return {
                "agent": self.name,
                "error": True,
                "message": str(e)
            }
