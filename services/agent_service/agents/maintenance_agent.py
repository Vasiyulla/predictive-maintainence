"""
Maintenance Recommendation Agent
Provides actionable maintenance recommendations
"""
from typing import Dict, List
from datetime import datetime, timedelta

class MaintenanceAgent:
    """
    Agent that generates specific maintenance recommendations
    Based on alert severity and system analysis
    """
    
    def __init__(self):
        self.name = "MaintenanceAgent"
        self.maintenance_procedures = self._load_maintenance_procedures()
    
    def _load_maintenance_procedures(self) -> Dict:
        """Load maintenance procedure knowledge base"""
        return {
            'temperature': {
                'critical': [
                    "Emergency shutdown procedure",
                    "Inspect cooling system (fans, radiators, coolant levels)",
                    "Check thermal sensors for accuracy",
                    "Examine heat exchangers for blockages",
                    "Verify ambient temperature conditions"
                ],
                'warning': [
                    "Clean cooling system components",
                    "Check coolant levels and quality",
                    "Inspect fan operation",
                    "Monitor ambient temperature"
                ],
                'preventive': [
                    "Regular cooling system maintenance",
                    "Coolant replacement schedule",
                    "Thermal sensor calibration"
                ]
            },
            'vibration': {
                'critical': [
                    "Immediate machine shutdown",
                    "Inspect bearings for wear or damage",
                    "Check shaft alignment",
                    "Examine mounting bolts and foundation",
                    "Inspect for loose components",
                    "Check balance of rotating parts"
                ],
                'warning': [
                    "Schedule vibration analysis",
                    "Lubricate bearings",
                    "Check mounting hardware",
                    "Inspect belt tensions"
                ],
                'preventive': [
                    "Regular bearing lubrication",
                    "Periodic alignment checks",
                    "Vibration monitoring program"
                ]
            },
            'pressure': {
                'critical': [
                    "Activate pressure relief systems",
                    "Emergency pressure reduction",
                    "Inspect pressure relief valves",
                    "Check for system leaks",
                    "Examine pressure sensors",
                    "Verify safety interlocks"
                ],
                'warning': [
                    "Inspect pressure control systems",
                    "Check valve operation",
                    "Test pressure sensors",
                    "Review operating parameters"
                ],
                'preventive': [
                    "Regular valve maintenance",
                    "Pressure sensor calibration",
                    "System pressure testing"
                ]
            },
            'rpm': {
                'critical': [
                    "Controlled machine shutdown",
                    "Inspect motor and drive system",
                    "Check speed control mechanisms",
                    "Examine coupling and gearbox",
                    "Verify power supply stability"
                ],
                'warning': [
                    "Inspect drive system",
                    "Check motor performance",
                    "Verify speed sensors",
                    "Review control settings"
                ],
                'preventive': [
                    "Regular motor maintenance",
                    "Drive system inspection",
                    "Speed sensor calibration"
                ]
            }
        }
    
    def generate_recommendations(
        self,
        alert: Dict,
        monitoring_result: Dict,
        prediction_result: Dict
    ) -> Dict:
        """
        Generate comprehensive maintenance recommendations
        
        Args:
            alert: Alert from AlertAgent
            monitoring_result: Analysis from MonitoringAgent
            prediction_result: Prediction from PredictionAgent
        
        Returns:
            Detailed maintenance plan
        """
        severity = alert['severity']
        priority = alert['priority_score']
        
        # Collect all recommendations
        immediate_actions = []
        scheduled_maintenance = []
        preventive_measures = []
        parts_to_inspect = []
        estimated_downtime = None
        
        # Process critical sensors
        for anomaly in monitoring_result.get('anomalies', []):
            sensor = anomaly['sensor']
            if sensor in self.maintenance_procedures:
                immediate_actions.extend(
                    self.maintenance_procedures[sensor]['critical']
                )
                parts_to_inspect.append(f"{sensor.capitalize()} related components")
        
        # Process warning sensors
        for warning in monitoring_result.get('warnings', []):
            sensor = warning['sensor']
            if sensor in self.maintenance_procedures:
                scheduled_maintenance.extend(
                    self.maintenance_procedures[sensor]['warning']
                )
        
        # Add ML-based recommendations
        if prediction_result.get('risk_level') == 'HIGH':
            immediate_actions.append("Prepare for potential machine failure")
            immediate_actions.append("Arrange backup equipment if available")
            estimated_downtime = "4-8 hours (emergency maintenance)"
            
        elif prediction_result.get('risk_level') == 'MEDIUM':
            scheduled_maintenance.append("Comprehensive system inspection within 24 hours")
            scheduled_maintenance.append("Diagnostic testing of all major components")
            estimated_downtime = "2-4 hours (planned maintenance)"
        
        # General preventive measures
        if severity in ['critical', 'warning']:
            preventive_measures = [
                "Implement increased monitoring frequency",
                "Update maintenance logs",
                "Review operational procedures",
                "Train operators on warning signs"
            ]
        
        # Determine action timeline
        if priority >= 80:
            action_timeline = "IMMEDIATE - Within 1 hour"
            operation_recommendation = "SHUTDOWN REQUIRED"
        elif priority >= 60:
            action_timeline = "URGENT - Within 4 hours"
            operation_recommendation = "REDUCED OPERATION"
        elif priority >= 40:
            action_timeline = "HIGH - Within 24 hours"
            operation_recommendation = "NORMAL WITH MONITORING"
        else:
            action_timeline = "ROUTINE - Next scheduled maintenance"
            operation_recommendation = "CONTINUE NORMAL OPERATION"
        
        # Remove duplicates
        immediate_actions = list(dict.fromkeys(immediate_actions))
        scheduled_maintenance = list(dict.fromkeys(scheduled_maintenance))
        preventive_measures = list(dict.fromkeys(preventive_measures))
        parts_to_inspect = list(dict.fromkeys(parts_to_inspect))
        
        return {
            'agent': self.name,
            'timestamp': datetime.utcnow().isoformat(),
            'action_timeline': action_timeline,
            'operation_recommendation': operation_recommendation,
            'immediate_actions': immediate_actions[:5],  # Top 5
            'scheduled_maintenance': scheduled_maintenance[:5],
            'preventive_measures': preventive_measures,
            'parts_to_inspect': parts_to_inspect,
            'estimated_downtime': estimated_downtime,
            'priority_score': priority,
            'requires_expert': priority >= 70,
            'requires_shutdown': priority >= 80
        }
    
    def generate_maintenance_report(self, recommendations: Dict) -> str:
        """Generate formatted maintenance report"""
        report = "="*70 + "\n"
        report += "MAINTENANCE RECOMMENDATION REPORT\n"
        report += "="*70 + "\n\n"
        
        report += f"Generated: {recommendations['timestamp']}\n"
        report += f"Priority: {recommendations['priority_score']}/100\n"
        report += f"Timeline: {recommendations['action_timeline']}\n"
        report += f"Operation: {recommendations['operation_recommendation']}\n\n"
        
        if recommendations['immediate_actions']:
            report += "IMMEDIATE ACTIONS REQUIRED:\n"
            for i, action in enumerate(recommendations['immediate_actions'], 1):
                report += f"  {i}. {action}\n"
            report += "\n"
        
        if recommendations['scheduled_maintenance']:
            report += "SCHEDULED MAINTENANCE:\n"
            for i, task in enumerate(recommendations['scheduled_maintenance'], 1):
                report += f"  {i}. {task}\n"
            report += "\n"
        
        if recommendations['parts_to_inspect']:
            report += "PARTS TO INSPECT:\n"
            for part in recommendations['parts_to_inspect']:
                report += f"  • {part}\n"
            report += "\n"
        
        if recommendations['preventive_measures']:
            report += "PREVENTIVE MEASURES:\n"
            for measure in recommendations['preventive_measures']:
                report += f"  • {measure}\n"
            report += "\n"
        
        if recommendations['estimated_downtime']:
            report += f"ESTIMATED DOWNTIME: {recommendations['estimated_downtime']}\n\n"
        
        if recommendations['requires_expert']:
            report += "⚠ EXPERT TECHNICIAN REQUIRED\n"
        
        if recommendations['requires_shutdown']:
            report += "🛑 MACHINE SHUTDOWN RECOMMENDED\n"
        
        report += "="*70 + "\n"
        
        return report
    
    def estimate_cost(self, recommendations: Dict) -> Dict:
        """Estimate maintenance cost (simplified)"""
        base_cost = 0
        
        # Cost based on actions
        base_cost += len(recommendations['immediate_actions']) * 500
        base_cost += len(recommendations['scheduled_maintenance']) * 200
        
        # Priority multiplier
        priority = recommendations['priority_score']
        if priority >= 80:
            multiplier = 2.0  # Emergency rates
        elif priority >= 60:
            multiplier = 1.5
        else:
            multiplier = 1.0
        
        estimated_cost = base_cost * multiplier
        
        return {
            'base_cost': base_cost,
            'priority_multiplier': multiplier,
            'estimated_cost': estimated_cost,
            'currency': 'USD',
            'includes': ['Labor', 'Standard parts', 'Inspection']
        }

if __name__ == "__main__":
    agent = MaintenanceAgent()
    
    # Test data
    alert = {
        'severity': 'critical',
        'priority_score': 85
    }
    
    monitoring = {
        'anomalies': [
            {'sensor': 'temperature', 'value': 98},
            {'sensor': 'vibration', 'value': 9.5}
        ],
        'warnings': []
    }
    
    prediction = {
        'risk_level': 'HIGH',
        'failure_probability': 0.92
    }
    
    print("\nTESTING MAINTENANCE AGENT")
    recommendations = agent.generate_recommendations(alert, monitoring, prediction)
    print(agent.generate_maintenance_report(recommendations))
    
    cost = agent.estimate_cost(recommendations)
    print(f"\nEstimated Cost: ${cost['estimated_cost']:.2f} {cost['currency']}")