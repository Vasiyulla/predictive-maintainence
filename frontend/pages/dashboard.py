"""
Dashboard Page - Real-time Monitoring
Full-featured Enterprise Dashboard with Real-Time Database Connection
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import time
import random

def show_dashboard(api_client):
    """
    Display enhanced real-time monitoring dashboard
    
    Features:
    - Real-time sensor monitoring (DB connected)
    - Live status indicators
    - Historical trends (Session-based)
    - Quick actions
    - Alert notifications
    """
    
    st.title("🎯 Real-Time Dashboard")
    
    # ---------------------------------------------------------
    # 1. INITIALIZE SESSION STATE
    # ---------------------------------------------------------
    # Store historical data in session for the "Trends" tab (since we don't have a history API yet)
    if 'history_data' not in st.session_state:
        st.session_state.history_data = {} # Format: {machine_id: pd.DataFrame}
        
    if 'alert_log' not in st.session_state:
        st.session_state.alert_log = []
        
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()

    # ---------------------------------------------------------
    # 2. FETCH REAL DATA
    # ---------------------------------------------------------
    try:
        # Fetch actual machine list from DB
        db_machines = api_client.get_machines()
    except Exception as e:
        st.error(f"Connection Error: {e}")
        db_machines = []

    # Calculate metrics based on REAL data
    active_count = len(db_machines) if db_machines else 0
    
    # ---------------------------------------------------------
    # 3. TOP METRICS ROW
    # ---------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🟢 Active Machines",
            value=str(active_count),
            delta="Online (DB)"
        )
    
    with col2:
        # Dynamic warning count from session alerts
        warnings = len([a for a in st.session_state.alert_log if a['severity'] == "Warning"])
        st.metric(
            label="⚠️ Warnings",
            value=str(warnings),
            delta="Session Total",
            delta_color="inverse"
        )
    
    with col3:
        # Dynamic critical count
        criticals = len([a for a in st.session_state.alert_log if a['severity'] == "Critical"])
        st.metric(
            label="🔴 Critical Alerts",
            value=str(criticals),
            delta="Session Total"
        )
    
    with col4:
        # Count total data points collected this session
        total_points = sum([len(df) for df in st.session_state.history_data.values()]) if st.session_state.history_data else 0
        st.metric(
            label="📊 Data Points",
            value=str(total_points),
            delta="Live Stream"
        )
    
    st.markdown("---")
    
    # ---------------------------------------------------------
    # 4. CONTROLS & SELECTION
    # ---------------------------------------------------------
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # DYNAMIC DROP DOWN (No hardcoded values)
        if db_machines:
            # Sync selection with session state
            if 'selected_machine' not in st.session_state or st.session_state.selected_machine not in db_machines:
                st.session_state.selected_machine = db_machines[0]
                
            selected_machine = st.selectbox(
                "🏭 Select Machine",
                options=db_machines,
                index=db_machines.index(st.session_state.selected_machine)
            )
            st.session_state.selected_machine = selected_machine
        else:
            st.warning("⚠️ No machines found. Add one in Settings.")
            selected_machine = None
    
    with col2:
        # Mode Selection
        monitoring_mode = st.radio(
            "Data Source",
            ["Real-Time (Database)", "Manual Input"],
            horizontal=True
        )
    
    with col3:
        st.write("")
        st.write("")
        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔴 Live Sync", value=False)
    
    # ---------------------------------------------------------
    # 5. DATA FETCHING LOGIC
    # ---------------------------------------------------------
    # Default values
    current_temp = 0.0
    current_vib = 0.0
    current_press = 0.0
    current_rpm = 0.0
    
    if selected_machine:
        if monitoring_mode == "Real-Time (Database)":
            # Fetch latest from DB
            if hasattr(api_client, 'get_latest_telemetry'):
                res = api_client.get_latest_telemetry(selected_machine)
                if res.get("success") and res.get("data"):
                    d = res["data"]
                    current_temp = float(d.get('temperature', 0))
                    current_vib = float(d.get('vibration', 0))
                    current_press = float(d.get('pressure', 0))
                    current_rpm = float(d.get('rpm', 0))
                    st.toast(f"📡 Updated from DB: {d.get('timestamp')}")
                else:
                    st.info("Waiting for data stream...")
            else:
                st.error("API Client missing telemetry method")
                
        # Update Session History for Trends
        if selected_machine not in st.session_state.history_data:
            st.session_state.history_data[selected_machine] = pd.DataFrame(
                columns=['timestamp', 'temperature', 'vibration', 'pressure', 'rpm']
            )
            
        # Append new data point if it's new (simple check) or just append for demo flow
        new_row = {
            'timestamp': datetime.now(),
            'temperature': current_temp,
            'vibration': current_vib,
            'pressure': current_press,
            'rpm': current_rpm
        }
        # Only append if values are non-zero (to avoid cluttering history with empty inits)
        if current_temp > 0:
            st.session_state.history_data[selected_machine] = pd.concat([
                st.session_state.history_data[selected_machine], 
                pd.DataFrame([new_row])
            ], ignore_index=True)

    # ---------------------------------------------------------
    # 6. MAIN DASHBOARD TABS
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Monitoring",
        "📈 Trends",
        "🔔 Alerts",
        "📋 Reports"
    ])
    
    with tab1:
        if selected_machine:
            show_live_monitoring(
                api_client, 
                selected_machine, 
                monitoring_mode, 
                current_temp, current_vib, current_press, current_rpm
            )
        else:
            st.info("Select a machine to view live monitoring.")
    
    with tab2:
        if selected_machine:
            show_trends(selected_machine)
        else:
            st.info("Select a machine to view trends.")
    
    with tab3:
        show_alerts()
    
    with tab4:
        show_reports(db_machines)
    
    # ---------------------------------------------------------
    # 7. AUTO-REFRESH TRIGGER
    # ---------------------------------------------------------
    if auto_refresh:
        time.sleep(2) # Refresh every 2 seconds
        st.rerun()

def get_real_time_values(api_client, machine_id):
    """Fetch the latest data pushed by the simulated hardware service"""
    res = api_client.get_latest_telemetry(machine_id)
    if res.get("success"):
        return res["data"]
    return None

def show_live_monitoring(api_client, machine_id, mode, db_temp, db_vib, db_press, db_rpm):
    """Display live monitoring section"""
    
    st.subheader(f"📡 Live Sensor Data - {machine_id}")
    
    # Create two columns for sensors and status
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Sensor Readings")
        
        # Determine values based on mode
        if mode == "Manual Input":
            # Manual Sliders
            c1, c2 = st.columns(2)
            temperature = c1.slider("🌡️ Temperature (°C)", 20.0, 120.0, 60.0, key="man_temp")
            vibration = c1.slider("📳 Vibration (mm/s)", 0.0, 15.0, 3.0, key="man_vib")
            pressure = c2.slider("⚡ Pressure (bar)", 50.0, 200.0, 100.0, key="man_press")
            rpm = c2.slider("⚙️ RPM", 500.0, 3500.0, 2000.0, key="man_rpm")
        else:
            # DB Values (Read-only display)
            data = get_real_time_values(api_client, machine_id)
            temperature = data['temperature'] if data else 60.0
            vibration = data['vibration'] if data else 3.0
            # ... fetch others ...
            st.info(f"⚡ Streaming live data from Database for {machine_id}")
        
        # Quick status indicators
        st.markdown("#### Quick Status")
        status_col1, status_col2, status_col3, status_col4 = st.columns(4)
        
        def get_status_emoji(value, warn, crit):
            if value > crit: return "🔴"
            elif value > warn: return "🟡"
            return "🟢"
        
        status_col1.metric("Temp", f"{temperature:.1f}°C", delta=get_status_emoji(temperature, 85, 95))
        status_col2.metric("Vibration", f"{vibration:.1f}", delta=get_status_emoji(vibration, 6, 8))
        status_col3.metric("Pressure", f"{pressure:.0f}", delta=get_status_emoji(pressure, 130, 145))
        status_col4.metric("RPM", f"{rpm:.0f}", delta=get_status_emoji(rpm, 2600, 2900))

        # Live sensor chart (Gauges)
        st.markdown("---")
        fig = create_live_sensor_chart(temperature, vibration, pressure, rpm)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### System Status")
        
        # Dynamic Status Card
        status_color = "#d4edda" # Green
        status_text = "NORMAL"
        if temperature > 95 or vibration > 8:
            status_color = "#f8d7da" # Red
            status_text = "CRITICAL"
        elif temperature > 85 or vibration > 6:
            status_color = "#fff3cd" # Yellow
            status_text = "WARNING"

        st.markdown(f"""
        <div style="padding: 20px; border-radius: 10px; background-color: {status_color}; color: #333;">
            <h4 style="margin:0;">● {status_text}</h4>
            <hr style="margin:10px 0;">
            <p><strong>Connection:</strong> {"Active" if mode == "Real-Time (Database)" else "Manual"}</p>
            <p><strong>Last Update:</strong> {datetime.now().strftime('%H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Quick actions
        st.markdown("### Quick Actions")
        
        if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
            run_analysis(api_client, machine_id, temperature, vibration, pressure, rpm)
        
        if st.button("🔔 Log Manual Alert", use_container_width=True):
            # Add to session alert log
            st.session_state.alert_log.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "machine": machine_id,
                "severity": "Warning",
                "message": "Manual alert triggered by operator"
            })
            st.success("Alert logged locally")


def run_analysis(api_client, machine_id, temperature, vibration, pressure, rpm):
    """Execute AI analysis with progress"""
    
    sensor_data = {
        "temperature": float(temperature),
        "vibration": float(vibration),
        "pressure": float(pressure),
        "rpm": float(rpm)
    }
    
    with st.spinner("🤖 AI Agents analyzing..."):
        # Progress bar simulation for UX
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)
        
        # Actual API Call
        result = api_client.analyze_machine(sensor_data)
        progress.empty()
    
    if result.get("success"):
        display_analysis_results(result["data"], machine_id)
        
        # Auto-log alerts based on result
        decision = result["data"].get("system_decision", "NORMAL")
        if decision != "NORMAL_OPERATION":
            st.session_state.alert_log.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "machine": machine_id,
                "severity": "Critical" if "SHUTDOWN" in decision else "Warning",
                "message": f"AI Detected: {decision}"
            })
            
    else:
        st.error(f"Analysis failed: {result.get('error')}")


def display_analysis_results(data, machine_id):
    """Display comprehensive analysis results"""
    
    st.markdown("---")
    st.markdown("## 📋 Analysis Results")
    
    # Decision banner
    decision = data.get("system_decision", "UNKNOWN")
    
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
    col1.metric("Confidence", data.get("decision_confidence", "N/A"))
    col2.metric("Priority Score", f"{data.get('priority_score', 0)}/100")
    col3.metric("Failure Risk", data.get("risk_level", "UNKNOWN"))
    col4.metric("RUL", f"{data.get('rul_days', 'N/A')} Days")
    
    # Detailed tabs
    sub_tab1, sub_tab2 = st.tabs(["📝 Details", "🔧 Maintenance"])
    
    with sub_tab1:
        st.write(f"**Rationale:** {data.get('decision_rationale', 'N/A')}")
        st.json(data.get("monitoring_summary", {}))
        
    with sub_tab2:
        maint = data.get("maintenance_summary", {})
        st.write(f"**Recommendation:** {maint.get('operation_recommendation', 'N/A')}")
        if maint.get("immediate_actions"):
            for action in maint["immediate_actions"]:
                st.write(f"- {action}")


def show_trends(machine_id):
    """Display historical trends from Session Data"""
    st.subheader(f"📈 Real-Time Trends - {machine_id}")
    
    if machine_id in st.session_state.history_data and not st.session_state.history_data[machine_id].empty:
        df = st.session_state.history_data[machine_id]
        
        # Only show last 50 points to keep chart clean
        if len(df) > 50:
            df = df.tail(50)
            
        # Sensor trends
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['temperature'], name='Temperature', line=dict(color='#FF6B6B')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['vibration'], name='Vibration', line=dict(color='#4ECDC4')))
        
        fig.update_layout(title="Live Sensor History (Session)", height=400, xaxis_title="Time", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Temp", f"{df['temperature'].mean():.1f}")
        c2.metric("Max Vib", f"{df['vibration'].max():.1f}")
        c3.metric("Data Points", len(df))
        
    else:
        st.warning("No data collected yet. Start 'Real-Time' mode and enable 'Live Sync' to build history.")


def show_alerts():
    """Display alerts and notifications"""
    st.subheader("🔔 Session Alerts")
    
    if not st.session_state.alert_log:
        st.info("No alerts generated in this session.")
        return

    # Convert log to dataframe for display
    df = pd.DataFrame(st.session_state.alert_log)
    
    # Display most recent first
    for i, row in df.iloc[::-1].iterrows():
        color = "🔴" if row['severity'] == "Critical" else "🟡" if row['severity'] == "Warning" else "🔵"
        with st.expander(f"{color} {row['severity']} - {row['machine']} ({row['time']})"):
            st.write(f"**Message:** {row['message']}")


def show_reports(machines_list):
    """Display reports section"""
    st.subheader("📋 Reports & Export")
    
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Report Type", ["Daily Summary", "Incident Report", "Full Audit Log"])
    with col2:
        st.multiselect("Select Machines", machines_list if machines_list else ["All"])
    
    if st.button("Generate Report"):
        st.success("Report generated! (Mock functionality)")
        st.download_button("Download CSV", data="timestamp,machine,status\n2024-01-01,M1,OK", file_name="report.csv")


def create_live_sensor_chart(temperature, vibration, pressure, rpm):
    """Create live sensor visualization"""
    fig = go.Figure()
    
    sensors = ['Temperature', 'Vibration', 'Pressure', 'RPM']
    # Normalize rpm for visualization
    values = [temperature, vibration * 10, pressure, rpm / 10]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    fig.add_trace(go.Bar(
        x=sensors,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition='auto',
    ))
    
    fig.update_layout(title="Current Sensor Readings (Scaled)", height=300)
    return fig