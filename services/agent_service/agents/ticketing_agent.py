from datetime import datetime
import json
from pathlib import Path

class TicketingAgent:
    """
    Integrates with Jira/ServiceNow/SAP PM.
    For MVP, logs tickets to a file.
    """
    def __init__(self):
        self.name = "TicketingAgent"
        self.log_path = Path("logs/tickets.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def create_ticket(self, alert_data: dict, machine_id: str) -> dict:
        """Create a maintenance ticket if severity is high"""
        
        if alert_data['priority_level'] not in ['P1', 'P2']:
            return {"status": "skipped", "reason": "Low priority"}

        ticket_id = f"TKT-{int(datetime.utcnow().timestamp())}"
        
        ticket = {
            "ticket_id": ticket_id,
            "machine_id": machine_id,
            "created_at": datetime.utcnow().isoformat(),
            "severity": alert_data['severity'],
            "priority": alert_data['priority_level'],
            "summary": f"Predictive Alert: {alert_data['messages'][0]}",
            "status": "OPEN",
            "assigned_to": "Unassigned"
        }

        # Log to file (simulating API call)
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(ticket) + "\n")

        return {
            "status": "created", 
            "ticket_id": ticket_id, 
            "system": "Jira-Mock"
        }