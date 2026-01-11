"""
Agent Orchestrator
Coordinates all AI agents to produce comprehensive system decisions
"""
from typing import Dict
from datetime import datetime
from .agents.monitoring_agent import MonitoringAgent
from .agents.prediction_agent import PredictionAgent
from .agents.alert_agent import AlertAgent
from .agents.maintenance_agent import MaintenanceAgent
from .agents.ticketing_agent import TicketingAgent

class AgentOrchestrator:
    """
    Master orchestrator that coordinates all AI agents
    Implements the complete agentic decision-making workflow
    """
    
    def __init__(self):
        self.name = "AgentOrchestrator"
        
        # Initialize all agents
        self.monitoring_agent = MonitoringAgent()
        self.prediction_agent = PredictionAgent()
        self.alert_agent = AlertAgent()
        self.maintenance_agent = MaintenanceAgent()
        self.ticketing_agent = TicketingAgent()
        
        print(f"✓ {self.name} initialized with 5 agents")
    
    async def analyze_machine_status(
        self, 
        sensor_data: Dict[str, float], 
        machine_id: str = "MACHINE_001"
    ) -> Dict:
        """
        Execute complete multi-agent analysis workflow
        
        Workflow:
        1. MonitoringAgent analyzes sensor values
        2. PredictionAgent predicts failure probability
        3. AlertAgent generates alerts
        4. MaintenanceAgent provides recommendations
        5. TicketingAgent creates tickets for high-priority issues
        6. Orchestrator synthesizes final decision
        
        Args:
            sensor_data: Current sensor readings
            machine_id: Unique machine identifier
        
        Returns:
            Comprehensive analysis with final decision
        """
        analysis_start = datetime.utcnow()
        
        # Store results from each agent
        agent_results = {}
        
        print(f"\n{'='*60}")
        print(f"AGENT ORCHESTRATOR - ANALYSIS WORKFLOW")
        print(f"{'='*60}")
        print(f"Machine ID: {machine_id}")
        print(f"Input: {sensor_data}")
        
        # STEP 1: Monitoring Agent Analysis
        print("\n[1/5] Running Monitoring Agent...")
        monitoring_result = self.monitoring_agent.analyze_all_sensors(sensor_data)
        agent_results['monitoring'] = monitoring_result
        print(f"  ✓ Status: {monitoring_result['overall_status']}")
        print(f"  ✓ Anomalies: {monitoring_result['critical_count']}")
        
        # STEP 2: ML Prediction Agent
        print("\n[2/5] Running Prediction Agent...")
        prediction_result = await self.prediction_agent.predict_failure(sensor_data)
        agent_results['prediction'] = prediction_result
        
        if 'error' not in prediction_result:
            print(f"  ✓ Risk Level: {prediction_result['risk_level']}")
            print(f"  ✓ Failure Probability: {prediction_result['failure_probability']:.2%}")
        else:
            print(f"  ⚠ Prediction unavailable: {prediction_result.get('message')}")
            # Use fallback prediction based on monitoring
            prediction_result = self._fallback_prediction(monitoring_result)
            agent_results['prediction'] = prediction_result
        
        # STEP 3: Alert Agent
        print("\n[3/5] Running Alert Agent...")
        alert_result = self.alert_agent.generate_alert(
            monitoring_result,
            prediction_result
        )
        agent_results['alert'] = alert_result
        print(f"  ✓ Severity: {alert_result['severity']}")
        print(f"  ✓ Priority: {alert_result['priority_level']} ({alert_result['priority_score']}/100)")
        
        # STEP 4: Maintenance Agent
        print("\n[4/5] Running Maintenance Agent...")
        maintenance_result = self.maintenance_agent.generate_recommendations(
            alert_result,
            monitoring_result,
            prediction_result
        )
        agent_results['maintenance'] = maintenance_result
        print(f"  ✓ Timeline: {maintenance_result['action_timeline']}")
        print(f"  ✓ Actions: {len(maintenance_result['immediate_actions'])} immediate")
        
        # STEP 5: Ticketing Agent
        print("\n[5/5] Running Ticketing Agent...")
        ticketing_result = self.ticketing_agent.create_ticket(
            alert_result, 
            machine_id
        )
        agent_results['ticketing'] = ticketing_result
        
        if ticketing_result['status'] == 'created':
            print(f"  ✓ Ticket Created: {ticketing_result['ticket_id']}")
            print(f"  ✓ System: {ticketing_result['system']}")
        else:
            print(f"  ℹ Ticket Status: {ticketing_result['status']}")
            print(f"  ℹ Reason: {ticketing_result['reason']}")
        
        # STEP 6: Synthesize Final Decision
        print("\n[6/6] Synthesizing Final Decision...")
        final_decision = self._synthesize_decision(
            agent_results, 
            sensor_data, 
            machine_id
        )
        
        analysis_end = datetime.utcnow()
        processing_time = (analysis_end - analysis_start).total_seconds()
        
        final_decision['processing_time_seconds'] = processing_time
        final_decision['analysis_timestamp'] = analysis_end.isoformat()
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE - Processing Time: {processing_time:.2f}s")
        print(f"{'='*60}")
        
        return final_decision
    
    def _fallback_prediction(self, monitoring_result: Dict) -> Dict:
        """
        Generate fallback prediction when ML service is unavailable
        Uses rule-based logic based on monitoring results
        """
        critical_count = monitoring_result.get('critical_count', 0)
        warning_count = monitoring_result.get('warning_count', 0)
        
        if critical_count >= 2:
            failure_prob = 0.85
            risk_level = "HIGH"
        elif critical_count >= 1:
            failure_prob = 0.65
            risk_level = "MEDIUM"
        elif warning_count >= 2:
            failure_prob = 0.45
            risk_level = "MEDIUM"
        elif warning_count >= 1:
            failure_prob = 0.25
            risk_level = "LOW"
        else:
            failure_prob = 0.10
            risk_level = "LOW"
        
        return {
            'agent': 'PredictionAgent (Fallback)',
            'failure_probability': failure_prob,
            'risk_level': risk_level,
            'confidence': 'MEDIUM',
            'analysis': f'Rule-based prediction: {risk_level} risk',
            'fallback_mode': True
        }
    
    def _synthesize_decision(
        self, 
        agent_results: Dict, 
        sensor_data: Dict,
        machine_id: str
    ) -> Dict:
        """
        Synthesize final decision from all agent outputs
        This is where orchestration intelligence resides
        """
        monitoring = agent_results['monitoring']
        prediction = agent_results['prediction']
        alert = agent_results['alert']
        maintenance = agent_results['maintenance']
        ticketing = agent_results['ticketing']
        
        # Determine final system decision
        priority_score = alert['priority_score']
        
        if priority_score >= 80:
            system_decision = "EMERGENCY_SHUTDOWN"
            decision_confidence = "HIGH"
            decision_rationale = (
                "Multiple critical indicators detected. "
                "Immediate shutdown required to prevent catastrophic failure. "
                f"Ticket {ticketing.get('ticket_id', 'N/A')} has been created for emergency response."
            )
        elif priority_score >= 60:
            system_decision = "MAINTENANCE_REQUIRED"
            decision_confidence = "HIGH"
            decision_rationale = (
                "Significant risk indicators present. "
                "Urgent maintenance action required within 4 hours. "
                f"Ticket {ticketing.get('ticket_id', 'N/A')} has been assigned to maintenance team."
            )
        elif priority_score >= 40:
            system_decision = "MONITOR_CLOSELY"
            decision_confidence = "MEDIUM"
            decision_rationale = (
                "Early warning signs detected. "
                "Increased monitoring and planned maintenance recommended."
            )
        else:
            system_decision = "CONTINUE_OPERATION"
            decision_confidence = "HIGH"
            decision_rationale = (
                "System operating within acceptable parameters. "
                "Continue normal operation with routine monitoring."
            )
        
        # Build comprehensive response
        return {
            'orchestrator': self.name,
            'machine_id': machine_id,
            'system_decision': system_decision,
            'decision_confidence': decision_confidence,
            'decision_rationale': decision_rationale,
            
            # Summary metrics
            'overall_severity': alert['severity'],
            'priority_score': priority_score,
            'priority_level': alert['priority_level'],
            'failure_probability': prediction.get('failure_probability', 0),
            'risk_level': prediction.get('risk_level', 'UNKNOWN'),
            
            # Sensor data
            'sensor_data': sensor_data,
            
            # Agent outputs (summarized)
            'monitoring_summary': {
                'status': monitoring['overall_status'],
                'critical_sensors': monitoring['critical_count'],
                'warning_sensors': monitoring['warning_count'],
                'anomalies': [a['message'] for a in monitoring.get('anomalies', [])[:3]]
            },
            
            'prediction_summary': {
                'failure_probability': prediction.get('failure_probability', 0),
                'risk_level': prediction.get('risk_level', 'UNKNOWN'),
                'analysis': prediction.get('analysis', 'N/A'),
                'confidence': prediction.get('confidence', 'UNKNOWN')
            },
            
            'alert_summary': {
                'severity': alert['severity'],
                'messages': alert['messages'][:3],
                'requires_immediate_attention': alert['requires_immediate_attention']
            },
            
            'maintenance_summary': {
                'action_timeline': maintenance['action_timeline'],
                'operation_recommendation': maintenance['operation_recommendation'],
                'immediate_actions': maintenance['immediate_actions'],
                'requires_shutdown': maintenance['requires_shutdown'],
                'estimated_downtime': maintenance.get('estimated_downtime')
            },
            
            'ticketing_summary': {
                'status': ticketing['status'],
                'ticket_id': ticketing.get('ticket_id'),
                'system': ticketing.get('system'),
                'reason': ticketing.get('reason')
            },
            
            # Full agent results (for detailed view)
            'detailed_results': {
                'monitoring': monitoring,
                'prediction': prediction,
                'alert': alert,
                'maintenance': maintenance,
                'ticketing': ticketing
            }
        }
    
    def generate_executive_summary(self, decision: Dict) -> str:
        """Generate executive summary report"""
        summary = "="*70 + "\n"
        summary += "PREDICTXAI - EXECUTIVE SUMMARY\n"
        summary += "="*70 + "\n\n"
        
        summary += f"Machine ID: {decision['machine_id']}\n"
        summary += f"Analysis Time: {decision['analysis_timestamp']}\n"
        summary += f"Processing Time: {decision['processing_time_seconds']:.2f} seconds\n\n"
        
        summary += f"SYSTEM DECISION: {decision['system_decision']}\n"
        summary += f"Confidence: {decision['decision_confidence']}\n"
        summary += f"Priority: {decision['priority_level']} ({decision['priority_score']}/100)\n\n"
        
        summary += f"RISK ASSESSMENT:\n"
        summary += f"  Failure Probability: {decision['failure_probability']:.1%}\n"
        summary += f"  Risk Level: {decision['risk_level']}\n"
        summary += f"  Overall Severity: {decision['overall_severity'].upper()}\n\n"
        
        # Ticketing information
        if decision['ticketing_summary']['status'] == 'created':
            summary += f"TICKETING:\n"
            summary += f"  Status: Ticket Created\n"
            summary += f"  Ticket ID: {decision['ticketing_summary']['ticket_id']}\n"
            summary += f"  System: {decision['ticketing_summary']['system']}\n\n"
        
        summary += f"RATIONALE:\n{decision['decision_rationale']}\n\n"
        
        if decision['maintenance_summary']['immediate_actions']:
            summary += "IMMEDIATE ACTIONS:\n"
            for action in decision['maintenance_summary']['immediate_actions'][:3]:
                summary += f"  • {action}\n"
            summary += "\n"
        
        summary += "="*70 + "\n"
        
        return summary

if __name__ == "__main__":
    import asyncio
    
    orchestrator = AgentOrchestrator()
    
    test_cases = [
        {
            "name": "Normal Operation",
            "machine_id": "MACHINE_001",
            "data": {"temperature": 60, "vibration": 3, "pressure": 100, "rpm": 2000}
        },
        {
            "name": "Critical Failure Imminent",
            "machine_id": "MACHINE_002",
            "data": {"temperature": 98, "vibration": 9.5, "pressure": 148, "rpm": 2950}
        },
        {
            "name": "Medium Risk - Ticket Creation",
            "machine_id": "MACHINE_003",
            "data": {"temperature": 85, "vibration": 7.5, "pressure": 135, "rpm": 2700}
        }
    ]
    
    async def test_orchestrator():
        for test in test_cases:
            print(f"\n\n{'='*70}")
            print(f"TEST CASE: {test['name']}")
            print(f"{'='*70}")
            
            result = await orchestrator.analyze_machine_status(
                test['data'], 
                test['machine_id']
            )
            print("\n" + orchestrator.generate_executive_summary(result))
    
    asyncio.run(test_orchestrator())