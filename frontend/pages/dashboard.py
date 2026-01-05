"""
Dashboard Page - Real-time Monitoring
Enhanced dashboard with live updates and comprehensive monitoring
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import time
import json

def show_dashboard(api_client):
    """
    Display enhanced real-time monitoring dashboard
    
    Features:
    - Real-time sensor monitoring
    - Live status indicators
    - Historical trends
    - Quick actions
    - Alert notifications
    """
    
    st.title("🎯 Real-Time Dashboard")
    
    # Initialize session state for live data
    if 'live_data' not in st.session_state:
        st.session_state.live_data = []
    if 'selected_machine' not in st.session_state:
        st.session_state.selected_machine = "Machine-001"
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🟢 Active Machines",
            value="12",
            delta="2 online"
        )
    
    with col2:
        st.metric(
            label="⚠️ Warnings",
            value="3",
            delta="-1 from yesterday",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="🔴 Critical Alerts",
            value="1",
            delta="0 change"
        )
    
    with col4:
        st.metric(
            label="📊 Predictions Today",
            value="247",
            delta="+32 from yesterday"
        )
    
    st.markdown("---")
    
    # Machine selector and controls
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        machines = [f"Machine-{i:03d}" for i in range(1, 16)]
        selected_machine = st.selectbox(
            "🏭 Select Machine",
            machines,
            index=machines.index(st.session_state.selected_machine)
        )
        st.session_state.selected_machine = selected_machine
    
    with col2:
        monitoring_mode = st.radio(
            "Monitoring Mode",
            ["Real-time", "Manual", "Simulated"],
            horizontal=True
        )
    
    with col3:
        st.write("")
        st.write("")
        auto_refresh = st.checkbox("Auto-refresh", value=False)
    
    # Main dashboard layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Monitoring",
        "📈 Trends",
        "🔔 Alerts",
        "📋 Reports"
    ])
    
    with tab1:
        show_live_monitoring(api_client, selected_machine, monitoring_mode)
    
    with tab2:
        show_trends(selected_machine)
    
    with tab3:
        show_alerts()
    
    with tab4:
        show_reports()
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(5)
        st.rerun()


def show_live_monitoring(api_client, machine_id, mode):
    """Display live monitoring section"""
    
    st.subheader(f"📡 Live Sensor Data - {machine_id}")
    
    # Create two columns for sensors and status
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Sensor Readings")
        
        # Sensor input section
        sensor_col1, sensor_col2 = st.columns(2)
        
        with sensor_col1:
            if mode == "Manual":
                temperature = st.number_input(
                    "🌡️ Temperature (°C)",
                    min_value=20.0,
                    max_value=120.0,
                    value=60.0,
                    step=1.0,
                    key=f"temp_{machine_id}"
                )
                vibration = st.number_input(
                    "📳 Vibration (mm/s)",
                    min_value=0.0,
                    max_value=15.0,
                    value=3.0,
                    step=0.1,
                    key=f"vib_{machine_id}"
                )
            else:
                temperature = st.slider(
                    "🌡️ Temperature (°C)",
                    20.0, 120.0, 60.0,
                    key=f"temp_slider_{machine_id}"
                )
                vibration = st.slider(
                    "📳 Vibration (mm/s)",
                    0.0, 15.0, 3.0,
                    key=f"vib_slider_{machine_id}"
                )
        
        with sensor_col2:
            if mode == "Manual":
                pressure = st.number_input(
                    "⚡ Pressure (bar)",
                    min_value=50.0,
                    max_value=200.0,
                    value=100.0,
                    step=1.0,
                    key=f"press_{machine_id}"
                )
                rpm = st.number_input(
                    "⚙️ RPM",
                    min_value=500.0,
                    max_value=3500.0,
                    value=2000.0,
                    step=10.0,
                    key=f"rpm_{machine_id}"
                )
            else:
                pressure = st.slider(
                    "⚡ Pressure (bar)",
                    50.0, 200.0, 100.0,
                    key=f"press_slider_{machine_id}"
                )
                rpm = st.slider(
                    "⚙️ RPM",
                    500.0, 3500.0, 2000.0,
                    key=f"rpm_slider_{machine_id}"
                )
        
        # Quick status indicators
        st.markdown("#### Quick Status")
        status_col1, status_col2, status_col3, status_col4 = st.columns(4)
        
        def get_status_emoji(value, warn, crit):
            if value > crit:
                return "🔴"
            elif value > warn:
                return "🟡"
            return "🟢"
        
        status_col1.metric(
            "Temp Status",
            f"{temperature:.1f}°C",
            delta=get_status_emoji(temperature, 85, 95)
        )
        status_col2.metric(
            "Vib Status",
            f"{vibration:.1f}",
            delta=get_status_emoji(vibration, 6, 8)
        )
        status_col3.metric(
            "Press Status",
            f"{pressure:.0f}",
            delta=get_status_emoji(pressure, 130, 145)
        )
        status_col4.metric(
            "RPM Status",
            f"{rpm:.0f}",
            delta=get_status_emoji(rpm, 2600, 2900)
        )
    
    with col2:
        st.markdown("### System Status")
        
        # Status card
        st.markdown("""
        <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6;">
            <h4>🟢 OPERATIONAL</h4>
            <p><strong>Uptime:</strong> 47 days 13h</p>
            <p><strong>Last Check:</strong> 2 min ago</p>
            <p><strong>Next Maintenance:</strong> 5 days</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Quick actions
        st.markdown("### Quick Actions")
        
        if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
            run_analysis(api_client, machine_id, temperature, vibration, pressure, rpm)
        
        if st.button("📊 View History", use_container_width=True):
            st.session_state.show_history = True
        
        if st.button("🔔 Create Alert", use_container_width=True):
            st.info("Alert creation feature")
        
        if st.button("📥 Export Data", use_container_width=True):
            export_data(machine_id)
    
    # Live sensor chart
    st.markdown("---")
    st.markdown("### 📈 Real-time Sensor Visualization")
    
    # Create live chart
    fig = create_live_sensor_chart(temperature, vibration, pressure, rpm)
    st.plotly_chart(fig, use_container_width=True)


def run_analysis(api_client, machine_id, temperature, vibration, pressure, rpm):
    """Execute AI analysis with progress"""
    
    sensor_data = {
        "temperature": float(temperature),
        "vibration": float(vibration),
        "pressure": float(pressure),
        "rpm": float(rpm)
    }
    
    with st.spinner("🤖 AI Agents analyzing..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("⚙️ Monitoring Agent analyzing sensors...")
        progress_bar.progress(25)
        time.sleep(0.3)
        
        status_text.text("🧠 ML Agent calculating failure probability...")
        progress_bar.progress(50)
        time.sleep(0.3)
        
        status_text.text("🚨 Alert Agent generating recommendations...")
        progress_bar.progress(75)
        time.sleep(0.3)
        
        # Get actual results
        result = api_client.analyze_machine(sensor_data)
        
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(100)
        time.sleep(0.2)
        
        progress_bar.empty()
        status_text.empty()
    
    if result.get("success"):
        display_analysis_results(result["data"], machine_id)
    else:
        st.error(f"Analysis failed: {result.get('error')}")


def display_analysis_results(data, machine_id):
    """Display comprehensive analysis results"""
    
    st.markdown("---")
    st.markdown("## 📋 Analysis Results")
    
    # Decision banner
    decision = data.get("system_decision", "UNKNOWN")
    priority = data.get("priority_score", 0)
    
    if decision == "EMERGENCY_SHUTDOWN":
        st.error("🛑 **EMERGENCY SHUTDOWN REQUIRED**")
    elif decision == "MAINTENANCE_REQUIRED":
        st.warning("⚠️ **URGENT MAINTENANCE REQUIRED**")
    elif decision == "MONITOR_CLOSELY":
        st.info("👁️ **MONITOR CLOSELY**")
    else:
        st.success("✅ **CONTINUE NORMAL OPERATION**")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "Decision",
        decision.replace("_", " ").title(),
        delta=f"Confidence: {data.get('decision_confidence', 'N/A')}"
    )
    
    col2.metric(
        "Priority",
        f"{priority}/100",
        delta=data.get("priority_level", "N/A")
    )
    
    col3.metric(
        "Failure Risk",
        data.get("risk_level", "UNKNOWN"),
        delta=f"{data.get('failure_probability', 0):.1%}"
    )
    
    col4.metric(
        "Processing Time",
        f"{data.get('processing_time_seconds', 0):.2f}s",
        delta="Analysis time"
    )
    
    # Detailed tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Executive Summary",
        "🔍 Detailed Analysis",
        "🔧 Maintenance Plan",
        "📈 Visualizations"
    ])
    
    with tab1:
        show_executive_summary(data)
    
    with tab2:
        show_detailed_analysis(data)
    
    with tab3:
        show_maintenance_plan(data)
    
    with tab4:
        show_visualizations(data)


def show_executive_summary(data):
    """Display executive summary"""
    st.markdown("### Executive Summary")
    
    st.write(f"**Decision Rationale:** {data.get('decision_rationale', 'N/A')}")
    
    alert_summary = data.get("alert_summary", {})
    if alert_summary.get("messages"):
        st.markdown("**Key Issues:**")
        for msg in alert_summary["messages"]:
            st.write(f"• {msg}")
    
    # Risk gauge
    failure_prob = data.get("failure_probability", 0)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=failure_prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Failure Probability (%)"},
        delta={'reference': 30},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def show_detailed_analysis(data):
    """Display detailed analysis"""
    st.markdown("### Detailed Analysis")
    
    monitoring = data.get("monitoring_summary", {})
    prediction = data.get("prediction_summary", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Monitoring Analysis")
        st.metric("Overall Status", monitoring.get("status", "N/A").upper())
        st.metric("Critical Sensors", monitoring.get("critical_sensors", 0))
        st.metric("Warning Sensors", monitoring.get("warning_sensors", 0))
        
        if monitoring.get("anomalies"):
            st.markdown("**Anomalies Detected:**")
            for anomaly in monitoring["anomalies"]:
                st.error(f"🔴 {anomaly}")
    
    with col2:
        st.markdown("#### 🧠 ML Prediction")
        st.metric("Risk Level", prediction.get("risk_level", "N/A"))
        st.metric(
            "Failure Probability",
            f"{prediction.get('failure_probability', 0):.2%}"
        )
        st.metric("Confidence", prediction.get("confidence", "N/A"))
        
        st.write(f"**Analysis:** {prediction.get('analysis', 'N/A')}")


def show_maintenance_plan(data):
    """Display maintenance plan"""
    st.markdown("### Maintenance Recommendations")
    
    maintenance = data.get("maintenance_summary", {})
    
    col1, col2 = st.columns(2)
    
    col1.metric("Action Timeline", maintenance.get("action_timeline", "N/A"))
    col2.metric("Operation Status", maintenance.get("operation_recommendation", "N/A"))
    
    if maintenance.get("immediate_actions"):
        st.markdown("#### 🚨 Immediate Actions Required")
        for i, action in enumerate(maintenance["immediate_actions"], 1):
            st.write(f"{i}. {action}")
    
    if maintenance.get("estimated_downtime"):
        st.info(f"⏱️ **Estimated Downtime:** {maintenance['estimated_downtime']}")
    
    # Cost estimation (mock)
    if maintenance.get("requires_shutdown"):
        st.warning("💰 **Estimated Cost:** $5,000 - $15,000 (Emergency maintenance)")
    elif maintenance.get("immediate_actions"):
        st.info("💰 **Estimated Cost:** $1,000 - $3,000 (Preventive maintenance)")
    else:
        st.success("💰 **Estimated Cost:** $0 (No action required)")


def show_visualizations(data):
    """Display visualizations"""
    st.markdown("### Visualizations")
    
    sensor_data = data.get("sensor_data", {})
    
    # Sensor comparison
    fig = go.Figure(data=[
        go.Bar(
            x=list(sensor_data.keys()),
            y=list(sensor_data.values()),
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        )
    ])
    
    fig.update_layout(
        title="Current Sensor Readings",
        xaxis_title="Sensor",
        yaxis_title="Value",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_trends(machine_id):
    """Display historical trends"""
    st.subheader(f"📈 Historical Trends - {machine_id}")
    
    # Generate mock historical data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'temperature': [60 + (i % 10) * 2 for i in range(30)],
        'vibration': [3 + (i % 8) * 0.5 for i in range(30)],
        'pressure': [100 + (i % 12) * 3 for i in range(30)],
        'rpm': [2000 + (i % 15) * 50 for i in range(30)],
        'failure_prob': [(i % 20) * 0.03 for i in range(30)]
    })
    
    # Time range selector
    time_range = st.select_slider(
        "Time Range",
        options=["Last 7 Days", "Last 14 Days", "Last 30 Days", "Last 90 Days"],
        value="Last 30 Days"
    )
    
    # Sensor trends
    st.markdown("### Sensor Trends")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['temperature'],
        mode='lines+markers',
        name='Temperature',
        line=dict(color='#FF6B6B', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['vibration'] * 10,  # Scale for visibility
        mode='lines+markers',
        name='Vibration (×10)',
        line=dict(color='#4ECDC4', width=2)
    ))
    
    fig.update_layout(
        title="Sensor Readings Over Time",
        xaxis_title="Date",
        yaxis_title="Value",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Failure probability trend
    st.markdown("### Failure Risk Trend")
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=df['date'], y=df['failure_prob'] * 100,
        fill='tozeroy',
        mode='lines',
        name='Failure Probability',
        line=dict(color='#FF6B6B', width=2),
        fillcolor='rgba(255, 107, 107, 0.2)'
    ))
    
    fig2.add_hline(y=30, line_dash="dash", line_color="yellow", annotation_text="Medium Risk")
    fig2.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="High Risk")
    
    fig2.update_layout(
        title="Failure Probability Trend",
        xaxis_title="Date",
        yaxis_title="Probability (%)",
        height=400
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Avg Temperature", f"{df['temperature'].mean():.1f}°C")
    col2.metric("Avg Vibration", f"{df['vibration'].mean():.1f} mm/s")
    col3.metric("Avg Pressure", f"{df['pressure'].mean():.0f} bar")
    col4.metric("Avg Failure Risk", f"{df['failure_prob'].mean():.1%}")


def show_alerts():
    """Display alerts and notifications"""
    st.subheader("🔔 Alerts & Notifications")
    
    # Alert summary
    col1, col2, col3 = st.columns(3)
    
    col1.metric("🔴 Critical", "1", delta="Last hour")
    col2.metric("🟡 Warnings", "5", delta="Last 24h")
    col3.metric("🟢 Info", "12", delta="Last week")
    
    st.markdown("---")
    
    # Recent alerts
    st.markdown("### Recent Alerts")
    
    alerts = [
        {
            "time": "5 min ago",
            "machine": "Machine-003",
            "severity": "🔴 Critical",
            "message": "High vibration detected: 9.2 mm/s"
        },
        {
            "time": "1 hour ago",
            "machine": "Machine-007",
            "severity": "🟡 Warning",
            "message": "Temperature elevated: 88°C"
        },
        {
            "time": "3 hours ago",
            "machine": "Machine-012",
            "severity": "🟡 Warning",
            "message": "Pressure above threshold: 135 bar"
        },
        {
            "time": "Yesterday",
            "machine": "Machine-005",
            "severity": "🟢 Info",
            "message": "Maintenance completed successfully"
        }
    ]
    
    for alert in alerts:
        with st.expander(f"{alert['severity']} - {alert['machine']} ({alert['time']})"):
            st.write(f"**Message:** {alert['message']}")
            col1, col2, col3 = st.columns(3)
            col1.button("View Details", key=f"view_{alert['time']}")
            col2.button("Acknowledge", key=f"ack_{alert['time']}")
            col3.button("Create Ticket", key=f"ticket_{alert['time']}")


def show_reports():
    """Display reports section"""
    st.subheader("📋 Reports & Export")
    
    report_type = st.selectbox(
        "Report Type",
        ["Daily Summary", "Weekly Analysis", "Monthly Report", "Custom Report"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=7))
    
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    machines = st.multiselect(
        "Select Machines",
        [f"Machine-{i:03d}" for i in range(1, 16)],
        default=["Machine-001", "Machine-002"]
    )
    
    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            time.sleep(2)
            st.success("✅ Report generated successfully!")
            
            # Mock report data
            st.markdown("### Report Summary")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Predictions", "1,234")
            col2.metric("Critical Alerts", "8")
            col3.metric("Avg Failure Risk", "24%")
            
            # Download buttons
            st.markdown("### Download Options")
            col1, col2, col3 = st.columns(3)
            
            col1.download_button(
                "📄 Download PDF",
                data="Mock PDF data",
                file_name=f"report_{datetime.now().strftime('%Y%m%d')}.pdf"
            )
            
            col2.download_button(
                "📊 Download CSV",
                data="Mock CSV data",
                file_name=f"data_{datetime.now().strftime('%Y%m%d')}.csv"
            )
            
            col3.download_button(
                "📧 Email Report",
                data="Mock email",
                file_name="email.txt"
            )


def create_live_sensor_chart(temperature, vibration, pressure, rpm):
    """Create live sensor visualization"""
    
    fig = go.Figure()
    
    # Add sensor readings as bars
    sensors = ['Temperature', 'Vibration', 'Pressure', 'RPM']
    values = [temperature, vibration * 10, pressure, rpm / 10]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    fig.add_trace(go.Bar(
        x=sensors,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Current Sensor Readings",
        yaxis_title="Value (scaled)",
        height=300,
        showlegend=False
    )
    
    return fig


def export_data(machine_id):
    """Export machine data"""
    st.success(f"✅ Data export initiated for {machine_id}")
    st.info("Export will be available in the Downloads section")