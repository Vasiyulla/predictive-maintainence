"""
Alert Agent
Generates alerts and priority based on monitoring & prediction
"""

from typing import Dict
from datetime import datetime

class AlertAgent:
    """
    Decision-making agent that generates alerts
    """

    def __init__(self):
        self.name = "AlertAgent"

    def generate_alert(
        self,
        monitoring_result: Dict,
        prediction_result: Dict
    ) -> Dict:
        """
        Generate alerts based on monitoring + prediction

        Args:
            monitoring_result: Output from MonitoringAgent
            prediction_result: Output from PredictionAgent

        Returns:
            Alert details
        """
        critical_count = monitoring_result.get("critical_count", 0)
        warning_count = monitoring_result.get("warning_count", 0)
        failure_prob = prediction_result.get("failure_probability", 0)

        messages = []
        priority_score = 0

        # ---- Monitoring impact ----
        priority_score += critical_count * 30
        priority_score += warning_count * 15

        if critical_count > 0:
            messages.append(
                f"{critical_count} critical sensor anomalies detected"
            )

        if warning_count > 0:
            messages.append(
                f"{warning_count} warning-level sensor issues detected"
            )

        # ---- Prediction impact ----
        priority_score += int(failure_prob * 40)

        if failure_prob >= 0.7:
            messages.append("High probability of machine failure detected")
        elif failure_prob >= 0.4:
            messages.append("Moderate risk of machine failure")

        # ---- Clamp priority score ----
        priority_score = min(priority_score, 100)

        # ---- Severity & priority levels ----
        if priority_score >= 80:
            severity = "critical"
            priority_level = "P1"
            requires_immediate_attention = True
        elif priority_score >= 60:
            severity = "high"
            priority_level = "P2"
            requires_immediate_attention = True
        elif priority_score >= 40:
            severity = "medium"
            priority_level = "P3"
            requires_immediate_attention = False
        else:
            severity = "low"
            priority_level = "P4"
            requires_immediate_attention = False

        return {
            "agent": self.name,
            "severity": severity,
            "priority_score": priority_score,
            "priority_level": priority_level,
            "messages": messages,
            "requires_immediate_attention": requires_immediate_attention,
            "alert_generated_at": datetime.utcnow().isoformat()
        }
