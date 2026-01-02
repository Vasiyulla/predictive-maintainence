"""
Monitoring Agent
Detects anomalies in sensor readings using rule-based logic
"""
from typing import Dict, List
from config.settings import settings

class MonitoringAgent:
    """
    Rule-based agent that monitors sensor values and detects anomalies
    Uses threshold-based detection for immediate alerts
    """
    
    def __init__(self):
        self.name = "MonitoringAgent"
        self.thresholds = {
            'temperature': {
                'min': settings.TEMPERATURE_MIN,
                'max': settings.TEMPERATURE_MAX,
                'warning_max': 85,
                'critical_max': 95
            },
            'vibration': {
                'min': settings.VIBRATION_MIN,
                'max': settings.VIBRATION_MAX,
                'warning_max': 6,
                'critical_max': 8
            },
            'pressure': {
                'min': settings.PRESSURE_MIN,
                'max': settings.PRESSURE_MAX,
                'warning_max': 130,
                'critical_max': 145
            },
            'rpm': {
                'min': settings.RPM_MIN,
                'max': settings.RPM_MAX,
                'warning_max': 2600,
                'critical_max': 2900
            }
        }
    
    def analyze_sensor(self, sensor_name: str, value: float) -> Dict:
        """
        Analyze individual sensor reading
        
        Returns:
            Dictionary with status, severity, and message
        """
        if sensor_name not in self.thresholds:
            return {
                'sensor': sensor_name,
                'value': value,
                'status': 'unknown',
                'severity': 'info',
                'message': f'Unknown sensor: {sensor_name}'
            }
        
        threshold = self.thresholds[sensor_name]
        
        # Check critical range
        if value > threshold['critical_max']:
            return {
                'sensor': sensor_name,
                'value': value,
                'status': 'critical',
                'severity': 'critical',
                'message': f'{sensor_name.capitalize()} critically high: {value:.2f}'
            }
        elif value < threshold['min']:
            return {
                'sensor': sensor_name,
                'value': value,
                'status': 'critical',
                'severity': 'critical',
                'message': f'{sensor_name.capitalize()} critically low: {value:.2f}'
            }
        
        # Check warning range
        elif value > threshold['warning_max']:
            return {
                'sensor': sensor_name,
                'value': value,
                'status': 'warning',
                'severity': 'warning',
                'message': f'{sensor_name.capitalize()} elevated: {value:.2f}'
            }
        
        # Normal range
        else:
            return {
                'sensor': sensor_name,
                'value': value,
                'status': 'normal',
                'severity': 'info',
                'message': f'{sensor_name.capitalize()} within normal range'
            }
    
    def analyze_all_sensors(self, sensor_data: Dict[str, float]) -> Dict:
        """
        Analyze all sensor readings and generate report
        
        Args:
            sensor_data: Dictionary with sensor names and values
        
        Returns:
            Comprehensive analysis report
        """
        analyses = []
        anomalies = []
        warnings = []
        critical_count = 0
        warning_count = 0
        
        # Analyze each sensor
        for sensor_name, value in sensor_data.items():
            analysis = self.analyze_sensor(sensor_name, value)
            analyses.append(analysis)
            
            if analysis['severity'] == 'critical':
                critical_count += 1
                anomalies.append(analysis)
            elif analysis['severity'] == 'warning':
                warning_count += 1
                warnings.append(analysis)
        
        # Determine overall system status
        if critical_count > 0:
            overall_status = 'critical'
            overall_severity = 'critical'
        elif warning_count > 0:
            overall_status = 'warning'
            overall_severity = 'warning'
        else:
            overall_status = 'normal'
            overall_severity = 'info'
        
        # Generate summary message
        if critical_count > 0:
            summary = f'CRITICAL: {critical_count} sensor(s) in critical state'
        elif warning_count > 0:
            summary = f'WARNING: {warning_count} sensor(s) showing elevated readings'
        else:
            summary = 'All sensors operating normally'
        
        return {
            'agent': self.name,
            'overall_status': overall_status,
            'overall_severity': overall_severity,
            'summary': summary,
            'critical_count': critical_count,
            'warning_count': warning_count,
            'analyses': analyses,
            'anomalies': anomalies,
            'warnings': warnings,
            'requires_attention': critical_count > 0 or warning_count > 0
        }
    
    def get_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if analysis['critical_count'] > 0:
            recommendations.append("Immediate inspection required")
            recommendations.append("Consider emergency shutdown procedures")
            
        if analysis['warning_count'] > 0:
            recommendations.append("Schedule maintenance check within 24 hours")
            recommendations.append("Monitor sensors closely")
        
        # Specific sensor recommendations
        for anomaly in analysis['anomalies']:
            sensor = anomaly['sensor']
            if sensor == 'temperature':
                recommendations.append("Check cooling system")
            elif sensor == 'vibration':
                recommendations.append("Inspect mechanical components for wear")
            elif sensor == 'pressure':
                recommendations.append("Check pressure relief valves")
            elif sensor == 'rpm':
                recommendations.append("Inspect motor and drive system")
        
        if not recommendations:
            recommendations.append("Continue normal operations")
        
        return recommendations

if __name__ == "__main__":
    # Test monitoring agent
    agent = MonitoringAgent()
    
    test_cases = [
        {
            "name": "Normal Operation",
            "data": {"temperature": 60, "vibration": 3, "pressure": 100, "rpm": 2000}
        },
        {
            "name": "High Temperature Warning",
            "data": {"temperature": 88, "vibration": 3, "pressure": 100, "rpm": 2000}
        },
        {
            "name": "Critical Multiple Sensors",
            "data": {"temperature": 98, "vibration": 9, "pressure": 150, "rpm": 2950}
        }
    ]
    
    print("\n" + "="*60)
    print("TESTING MONITORING AGENT")
    print("="*60)
    
    for test in test_cases:
        print(f"\n{test['name']}:")
        print(f"  Input: {test['data']}")
        result = agent.analyze_all_sensors(test['data'])
        print(f"  Status: {result['overall_status'].upper()}")
        print(f"  Summary: {result['summary']}")
        if result['anomalies']:
            print(f"  Anomalies: {len(result['anomalies'])}")
        recommendations = agent.get_recommendations(result)
        print(f"  Recommendations: {', '.join(recommendations[:2])}")