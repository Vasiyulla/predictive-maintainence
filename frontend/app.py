"""
PredictXAI - Streamlit Frontend Application
Enterprise AI Predictive Maintenance Platform
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

from utils.api_client import APIClient
from config.settings import settings

# Page configuration
st.set_page_config(
    page_title="PredictXAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .critical-alert {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .warning-alert {
        background-color: #ffa600;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .success-alert {
        background-color: #00cc66;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state - MUST BE FIRST
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.api_client = APIClient()
    st.session_state.authenticated = False
    st.session_state.current_page = "login"
    st.session_state.user = None

# Ensure API client exists
if 'api_client' not in st.session_state:
    st.session_state.api_client = APIClient()

# Ensure authenticated state exists
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Ensure current page exists
if 'current_page' not in st.session_state:
    st.session_state.current_page = "login"

# Helper functions
def show_login_page():
    """Display login/register page"""
    st.markdown('<h1 class="main-header">🤖 PredictXAI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">Enterprise AI Predictive Maintenance Platform</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login to Your Account")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if username and password:
                    with st.spinner("Authenticating..."):
                        result = st.session_state.api_client.login(username, password)
                        
                        if result.get("success"):
                            st.session_state.authenticated = True
                            st.session_state.user = result['user']
                            st.session_state.current_page = "dashboard"
                            st.success(f"Welcome, {result['user']['username']}!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(result.get("error", "Login failed"))
                else:
                    st.warning("Please enter username and password")
    
    with tab2:
        st.subheader("Create New Account")
        with st.form("register_form"):
            new_username = st.text_input("Username", key="reg_username")
            new_email = st.text_input("Email", key="reg_email")
            new_password = st.text_input("Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
            role = st.selectbox("Role", ["operator", "admin"])
            submit = st.form_submit_button("Register", use_container_width=True)
            
            if submit:
                if not all([new_username, new_email, new_password, confirm_password]):
                    st.warning("Please fill all fields")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    with st.spinner("Creating account..."):
                        result = st.session_state.api_client.register(
                            new_username, new_email, new_password, role
                        )
                        
                        if "error" not in result:
                            st.success("Account created successfully! Please login.")
                        else:
                            st.error(result["error"])

def show_dashboard():
    """Display main dashboard"""
    st.markdown('<h1 class="main-header">🎯 PredictXAI Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### User Information")
        user = st.session_state.get('user', st.session_state.api_client.user)
        if user:
            st.write(f"**Username:** {user.get('username', 'N/A')}")
            st.write(f"**Role:** {user.get('role', 'N/A').capitalize()}")
        st.markdown("---")
        
        # System health
        st.markdown("### System Health")
        health = st.session_state.api_client.get_health()
        
        if health.get("status") == "healthy":
            st.success("All Systems Operational")
        elif health.get("status") == "degraded":
            st.warning("Some Services Degraded")
        else:
            st.error("System Unavailable")
        
        if "services" in health:
            for service, info in health["services"].items():
                status = info.get("status", "unknown")
                icon = "✅" if status == "healthy" else "⚠️" if status == "degraded" else "❌"
                st.write(f"{icon} {service.capitalize()}: {status}")
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### Navigation")
        
        # Dashboard button
        if st.session_state.current_page == "dashboard":
            st.button("📊 Dashboard", disabled=True, use_container_width=True)
        elif st.button("📊 Dashboard", use_container_width=True, key="nav_dashboard"):
            st.session_state.current_page = "dashboard"
            st.rerun()
        
        # Analytics button
        if st.session_state.current_page == "analytics":
            st.button("📈 Analytics", disabled=True, use_container_width=True)
        elif st.button("📈 Analytics", use_container_width=True, key="nav_analytics"):
            st.session_state.current_page = "analytics"
            st.rerun()
        
        # Quick Analysis button
        if st.session_state.current_page == "analysis":
            st.button("🔬 Quick Analysis", disabled=True, use_container_width=True)
        elif st.button("🔬 Quick Analysis", use_container_width=True, key="nav_analysis"):
            st.session_state.current_page = "analysis"
            st.rerun()
        
        # Settings button
        if st.session_state.current_page == "settings":
            st.button("⚙️ Settings", disabled=True, use_container_width=True)
        elif st.button("⚙️ Settings", use_container_width=True, key="nav_settings"):
            st.session_state.current_page = "settings"
            st.rerun()
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.api_client.logout()
            st.session_state.authenticated = False
            st.session_state.current_page = "login"
            st.session_state.user = None
            st.success("Logged out successfully!")
            time.sleep(0.5)
            st.rerun()
    
    # Main content based on current page
    try:
        if st.session_state.current_page == "dashboard":
            try:
                from pages.dashboard import show_dashboard
                show_dashboard(st.session_state.api_client)
            except ImportError:
                st.warning("Dashboard page not found. Showing monitoring page instead.")
                show_monitoring_page()
        elif st.session_state.current_page == "analytics":
            try:
                from pages.analytics import show_analytics
                show_analytics(st.session_state.api_client)
            except ImportError:
                st.error("Analytics page not found. Please ensure pages/analytics.py exists.")
                st.info("Showing basic monitoring instead.")
                show_monitoring_page()
        elif st.session_state.current_page == "monitoring":
            show_monitoring_page()
        elif st.session_state.current_page == "analysis":
            show_analysis_page()
        elif st.session_state.current_page == "settings":
            show_settings_page()
        else:
            # Fallback to dashboard
            st.session_state.current_page = "dashboard"
            st.rerun()
    except Exception as e:
        st.error(f"Error loading page: {e}")
        st.info("Showing basic monitoring page.")
        show_monitoring_page()

def show_monitoring_page():
    """Display real-time monitoring interface"""
    st.header("Real-Time Machine Monitoring")
    
    # Machine selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        machine_id = st.selectbox(
            "Select Machine",
            ["Machine-001", "Machine-002", "Machine-003", "Machine-004"]
        )
    
    with col2:
        monitoring_mode = st.radio("Mode", ["Manual Input", "Simulated"])
    
    st.markdown("---")
    
    # Sensor inputs
    st.subheader("📊 Sensor Readings")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if monitoring_mode == "Manual Input":
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=20.0,
                max_value=120.0,
                value=60.0,
                step=1.0
            )
        else:
            temperature = st.slider("Temperature (°C)", 20.0, 120.0, 60.0)
    
    with col2:
        if monitoring_mode == "Manual Input":
            vibration = st.number_input(
                "Vibration (mm/s)",
                min_value=0.0,
                max_value=15.0,
                value=3.0,
                step=0.1
            )
        else:
            vibration = st.slider("Vibration (mm/s)", 0.0, 15.0, 3.0)
    
    with col3:
        if monitoring_mode == "Manual Input":
            pressure = st.number_input(
                "Pressure (bar)",
                min_value=50.0,
                max_value=200.0,
                value=100.0,
                step=1.0
            )
        else:
            pressure = st.slider("Pressure (bar)", 50.0, 200.0, 100.0)
    
    with col4:
        if monitoring_mode == "Manual Input":
            rpm = st.number_input(
                "RPM",
                min_value=500.0,
                max_value=3500.0,
                value=2000.0,
                step=10.0
            )
        else:
            rpm = st.slider("RPM", 500.0, 3500.0, 2000.0)
    
    # Quick status indicators
    st.markdown("### Current Status")
    col1, col2, col3, col4 = st.columns(4)
    
    def get_status_color(value, min_val, max_val, warn_val, crit_val):
        if value > crit_val:
            return "🔴"
        elif value > warn_val:
            return "🟡"
        return "🟢"
    
    col1.metric(
        "Temperature",
        f"{temperature:.1f}°C",
        delta=f"{get_status_color(temperature, 20, 100, 85, 95)} Status"
    )
    
    col2.metric(
        "Vibration",
        f"{vibration:.1f} mm/s",
        delta=f"{get_status_color(vibration, 0, 10, 6, 8)} Status"
    )
    
    col3.metric(
        "Pressure",
        f"{pressure:.0f} bar",
        delta=f"{get_status_color(pressure, 50, 150, 130, 145)} Status"
    )
    
    col4.metric(
        "RPM",
        f"{rpm:.0f}",
        delta=f"{get_status_color(rpm, 500, 3000, 2600, 2900)} Status"
    )
    
    st.markdown("---")
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
            run_analysis(temperature, vibration, pressure, rpm, machine_id)

def run_analysis(temperature, vibration, pressure, rpm, machine_id):
    """Execute AI analysis"""
    sensor_data = {
        "temperature": float(temperature),
        "vibration": float(vibration),
        "pressure": float(pressure),
        "rpm": float(rpm)
    }
    
    with st.spinner("🤖 AI Agents analyzing machine status..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate progressive analysis
        status_text.text("⚙️ Monitoring Agent analyzing sensor values...")
        progress_bar.progress(25)
        time.sleep(0.5)
        
        status_text.text("🧠 ML Prediction Agent calculating failure probability...")
        progress_bar.progress(50)
        time.sleep(0.5)
        
        status_text.text("🚨 Alert Agent generating recommendations...")
        progress_bar.progress(75)
        time.sleep(0.5)
        
        # Get actual results
        result = st.session_state.api_client.analyze_machine(sensor_data)
        
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(100)
        time.sleep(0.3)
        
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
    
    # System Decision
    decision = data.get("system_decision", "UNKNOWN")
    confidence = data.get("decision_confidence", "UNKNOWN")
    priority = data.get("priority_score", 0)
    
    if decision == "EMERGENCY_SHUTDOWN":
        st.markdown('<div class="critical-alert">🛑 EMERGENCY SHUTDOWN REQUIRED</div>', unsafe_allow_html=True)
    elif decision == "MAINTENANCE_REQUIRED":
        st.markdown('<div class="warning-alert">⚠️ URGENT MAINTENANCE REQUIRED</div>', unsafe_allow_html=True)
    elif decision == "MONITOR_CLOSELY":
        st.markdown('<div class="warning-alert">👁️ MONITOR CLOSELY</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-alert">✅ CONTINUE NORMAL OPERATION</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "System Decision",
        decision.replace("_", " ").title(),
        delta=f"Confidence: {confidence}"
    )
    
    col2.metric(
        "Priority Score",
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
    
    # Detailed results in tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Executive Summary",
        "📊 Monitoring Report",
        "🧠 ML Prediction",
        "🔧 Maintenance Plan"
    ])
    
    with tab1:
        st.subheader("Executive Summary")
        st.write(f"**Rationale:** {data.get('decision_rationale', 'N/A')}")
        
        alert_summary = data.get("alert_summary", {})
        if alert_summary.get("messages"):
            st.markdown("**Key Issues:**")
            for msg in alert_summary["messages"]:
                st.write(f"• {msg}")
    
    with tab2:
        st.subheader("Sensor Monitoring Analysis")
        monitoring = data.get("monitoring_summary", {})
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Status", monitoring.get("status", "N/A").upper())
        col2.metric("Critical Sensors", monitoring.get("critical_sensors", 0))
        col3.metric("Warning Sensors", monitoring.get("warning_sensors", 0))
        
        if monitoring.get("anomalies"):
            st.markdown("**Detected Anomalies:**")
            for anomaly in monitoring["anomalies"]:
                st.error(f"🔴 {anomaly}")
    
    with tab3:
        st.subheader("ML Prediction Analysis")
        prediction = data.get("prediction_summary", {})
        
        failure_prob = prediction.get("failure_probability", 0)
        
        # Gauge chart for failure probability
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
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write(f"**Risk Level:** {prediction.get('risk_level', 'N/A')}")
        st.write(f"**Confidence:** {prediction.get('confidence', 'N/A')}")
        st.write(f"**Analysis:** {prediction.get('analysis', 'N/A')}")
    
    with tab4:
        st.subheader("Maintenance Recommendations")
        maintenance = data.get("maintenance_summary", {})
        
        col1, col2 = st.columns(2)
        col1.metric("Action Timeline", maintenance.get("action_timeline", "N/A"))
        col2.metric("Operation Status", maintenance.get("operation_recommendation", "N/A"))
        
        if maintenance.get("immediate_actions"):
            st.markdown("**Immediate Actions Required:**")
            for i, action in enumerate(maintenance["immediate_actions"], 1):
                st.write(f"{i}. {action}")
        
        if maintenance.get("estimated_downtime"):
            st.info(f"⏱️ **Estimated Downtime:** {maintenance['estimated_downtime']}")

def show_analysis_page():
    """ML-only analysis page"""
    st.header("🔬 Machine Learning Analysis")
    st.write("Direct ML prediction without full agent orchestration")
    
    st.markdown("---")
    
    # Model info
    model_info = st.session_state.api_client.get_model_info()
    
    if model_info.get("model_loaded"):
        st.success("✅ ML Model Loaded and Ready")
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Type", model_info.get("model_type", "N/A"))
        col2.metric("Features", len(model_info.get("features", [])))
        col3.metric("Estimators", model_info.get("n_estimators", "N/A"))
    else:
        st.warning("⚠️ ML Model not loaded. Please train the model.")
    
    st.markdown("---")
    
    # Quick prediction
    st.subheader("Quick Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temp = st.number_input("Temperature (°C)", 20.0, 120.0, 60.0)
        vib = st.number_input("Vibration (mm/s)", 0.0, 15.0, 3.0)
    
    with col2:
        press = st.number_input("Pressure (bar)", 50.0, 200.0, 100.0)
        rpm_val = st.number_input("RPM", 500.0, 3500.0, 2000.0)
    
    if st.button("Predict", type="primary"):
        sensor_data = {
            "temperature": float(temp),
            "vibration": float(vib),
            "pressure": float(press),
            "rpm": float(rpm_val)
        }
        
        result = st.session_state.api_client.predict_failure(sensor_data)
        
        if result.get("success"):
            data = result["data"]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Failure Probability", f"{data['failure_probability']:.1%}")
            col2.metric("Risk Level", data['risk_level'])
            col3.metric("Predicted Failure", "Yes" if data['failure_predicted'] else "No")
        else:
            st.error(f"Prediction failed: {result.get('error')}")

def show_settings_page():
    """System settings page"""
    st.header("⚙️ System Settings")
    
    tab1, tab2 = st.tabs(["ML Model", "System Info"])
    
    with tab1:
        st.subheader("Model Management")
        
        model_info = st.session_state.api_client.get_model_info()
        
        if model_info.get("model_loaded"):
            st.success("Current model is loaded and operational")
        else:
            st.warning("No model loaded - training required")
        
        st.markdown("---")
        
        st.subheader("Train New Model")
        n_samples = st.number_input(
            "Number of training samples",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000
        )
        
        if st.button("Start Training", type="primary"):
            with st.spinner("Training in progress..."):
                result = st.session_state.api_client.train_model(n_samples)
                
                if "error" not in result:
                    st.success("Training started! This will take a few minutes.")
                    st.info("The model will be available once training completes.")
                else:
                    st.error(f"Training failed: {result['error']}")
    
    with tab2:
        st.subheader("System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Application**")
            st.write(f"Name: {settings.APP_NAME}")
            st.write(f"Version: {settings.APP_VERSION}")
            st.write(f"Environment: {settings.ENVIRONMENT}")
        
        with col2:
            st.markdown("**Configuration**")
            st.write(f"Gateway Port: {settings.GATEWAY_PORT}")
            st.write(f"ML Service Port: {settings.ML_SERVICE_PORT}")
            st.write(f"Agent Service Port: {settings.AGENT_SERVICE_PORT}")

# Main application logic
def main():
    """Main application entry point"""
    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_dashboard()

if __name__ == "__main__":
    main()