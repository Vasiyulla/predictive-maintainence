"""
Analytics Page - Advanced Data Analysis
Historical analysis, trends, and predictive insights
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def show_analytics(api_client):
    """Display advanced analytics dashboard"""
    
    st.title("📈 Advanced Analytics")
    
    # Analytics options
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🔍 Deep Dive",
        "🎯 Predictions",
        "📉 Comparisons"
    ])
    
    with tab1:
        show_overview_analytics()
    
    with tab2:
        show_deep_dive_analytics()
    
    with tab3:
        show_predictive_analytics()
    
    with tab4:
        show_comparison_analytics()


def show_overview_analytics():
    """Display overview analytics"""
    
    st.markdown("## Fleet Overview")
    
    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Total Machines", "15", delta="2 added")
    col2.metric("Operational", "12", delta="95% uptime")
    col3.metric("Avg Efficiency", "87%", delta="+3%")
    col4.metric("Total Alerts", "156", delta="-12 vs last week")
    col5.metric("Maintenance Cost", "$45K", delta="-$5K")
    
    st.markdown("---")
    
    # Fleet health distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Fleet Health Distribution")
        
        health_data = pd.DataFrame({
            'Status': ['Excellent', 'Good', 'Fair', 'Poor', 'Critical'],
            'Count': [5, 6, 2, 1, 1]
        })
        
        fig = px.pie(
            health_data,
            values='Count',
            names='Status',
            color='Status',
            color_discrete_map={
                'Excellent': '#00CC66',
                'Good': '#66CC00',
                'Fair': '#FFCC00',
                'Poor': '#FF9900',
                'Critical': '#FF3333'
            }
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Alert Distribution (Last 30 Days)")
        
        alert_data = pd.DataFrame({
            'Type': ['Critical', 'Warning', 'Info'],
            'Count': [12, 45, 99]
        })
        
        fig = px.bar(
            alert_data,
            x='Type',
            y='Count',
            color='Type',
            color_discrete_map={
                'Critical': '#FF3333',
                'Warning': '#FFCC00',
                'Info': '#3399FF'
            }
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance trends
    st.markdown("### Performance Trends (90 Days)")
    
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'Efficiency': [85 + np.random.randn() * 5 for _ in range(90)],
        'Failure_Risk': [25 + np.random.randn() * 10 for _ in range(90)],
        'Maintenance_Cost': [1500 + np.random.randn() * 500 for _ in range(90)]
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Efficiency'],
        name='Efficiency (%)',
        line=dict(color='#00CC66', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Failure_Risk'],
        name='Avg Failure Risk (%)',
        line=dict(color='#FF6B6B', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Efficiency & Risk Trends',
        yaxis=dict(title='Efficiency (%)'),
        yaxis2=dict(title='Risk (%)', overlaying='y', side='right'),
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_deep_dive_analytics():
    """Display deep dive analytics"""
    
    st.markdown("## Deep Dive Analysis")
    
    # Machine selector
    machine = st.selectbox(
        "Select Machine for Analysis",
        [f"Machine-{i:03d}" for i in range(1, 16)]
    )
    
    # Time period
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("To", value=datetime.now())
    
    st.markdown("---")
    
    # Sensor correlation
    st.markdown("### Sensor Correlation Matrix")
    
    # Generate mock correlation data
    sensors = ['Temperature', 'Vibration', 'Pressure', 'RPM']
    correlation = np.array([
        [1.0, 0.65, 0.72, 0.45],
        [0.65, 1.0, 0.58, 0.38],
        [0.72, 0.58, 1.0, 0.51],
        [0.45, 0.38, 0.51, 1.0]
    ])
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation,
        x=sensors,
        y=sensors,
        colorscale='RdYlGn',
        zmid=0,
        text=correlation,
        texttemplate='%{text:.2f}',
        textfont={"size": 12}
    ))
    
    fig.update_layout(height=400, title='Sensor Correlation Analysis')
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly detection
    st.markdown("### Anomaly Detection")
    
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    normal_data = 60 + np.random.randn(30) * 5
    anomalies_idx = [5, 12, 20]
    for idx in anomalies_idx:
        normal_data[idx] += 25
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=normal_data,
        mode='lines+markers',
        name='Temperature',
        line=dict(color='#4ECDC4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[dates[i] for i in anomalies_idx],
        y=[normal_data[i] for i in anomalies_idx],
        mode='markers',
        name='Anomalies',
        marker=dict(color='#FF3333', size=15, symbol='x')
    ))
    
    fig.add_hline(y=85, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
    
    fig.update_layout(
        title='Temperature with Anomaly Detection',
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Statistical Summary")
        
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Std Dev', 'Min', 'Max', '25th %ile', '75th %ile'],
            'Temperature': [62.3, 8.5, 45.2, 92.1, 57.8, 68.9],
            'Vibration': [3.2, 1.8, 0.5, 9.5, 2.1, 4.3]
        })
        
        st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        st.markdown("### Failure Predictions")
        
        pred_df = pd.DataFrame({
            'Risk Level': ['Low', 'Medium', 'High', 'Critical'],
            'Count': [18, 8, 3, 1],
            'Avg Probability': ['15%', '45%', '72%', '91%']
        })
        
        st.dataframe(pred_df, use_container_width=True)


def show_predictive_analytics():
    """Display predictive analytics"""
    
    st.markdown("## Predictive Analytics")
    
    # Forecasting options
    forecast_period = st.slider(
        "Forecast Period (days)",
        min_value=7,
        max_value=90,
        value=30
    )
    
    st.markdown("### Failure Risk Forecast")
    
    # Historical data
    hist_dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    hist_risk = [20 + i * 0.5 + np.random.randn() * 3 for i in range(30)]
    
    # Forecast data
    future_dates = pd.date_range(start=datetime.now() + timedelta(days=1), periods=forecast_period, freq='D')
    future_risk = [hist_risk[-1] + i * 0.7 + np.random.randn() * 5 for i in range(forecast_period)]
    
    # Confidence intervals
    upper_bound = [r + 10 for r in future_risk]
    lower_bound = [max(0, r - 10) for r in future_risk]
    
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=hist_dates,
        y=hist_risk,
        mode='lines+markers',
        name='Historical',
        line=dict(color='#4ECDC4', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_risk,
        mode='lines',
        name='Forecast',
        line=dict(color='#FF6B6B', width=2, dash='dash')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=upper_bound + lower_bound[::-1],
        fill='toself',
        fillcolor='rgba(255, 107, 107, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence',
        showlegend=True
    ))
    
    fig.add_hline(y=60, line_dash="dot", line_color="red", annotation_text="High Risk Threshold")
    
    fig.update_layout(
        title=f'Failure Risk Forecast ({forecast_period} Days)',
        xaxis_title='Date',
        yaxis_title='Failure Risk (%)',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Maintenance recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Predicted Maintenance Windows")
        
        maint_df = pd.DataFrame({
            'Date': [
                (datetime.now() + timedelta(days=5)).strftime('%Y-%m-%d'),
                (datetime.now() + timedelta(days=15)).strftime('%Y-%m-%d'),
                (datetime.now() + timedelta(days=25)).strftime('%Y-%m-%d')
            ],
            'Machine': ['Machine-003', 'Machine-007', 'Machine-012'],
            'Priority': ['High', 'Medium', 'Low'],
            'Estimated Cost': ['$3,500', '$1,200', '$800']
        })
        
        st.dataframe(maint_df, use_container_width=True)
    
    with col2:
        st.markdown("### Cost Optimization")
        
        cost_df = pd.DataFrame({
            'Strategy': ['Reactive', 'Preventive', 'Predictive'],
            'Estimated Cost': [75000, 45000, 32000],
            'Downtime (hrs)': [120, 60, 25]
        })
        
        fig = px.bar(
            cost_df,
            x='Strategy',
            y='Estimated Cost',
            color='Strategy',
            text='Estimated Cost'
        )
        
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def show_comparison_analytics():
    """Display comparison analytics"""
    
    st.markdown("## Multi-Machine Comparison")
    
    # Machine selection
    selected_machines = st.multiselect(
        "Select Machines to Compare",
        [f"Machine-{i:03d}" for i in range(1, 16)],
        default=[f"Machine-{i:03d}" for i in range(1, 6)]
    )
    
    if len(selected_machines) < 2:
        st.warning("Please select at least 2 machines to compare")
        return
    
    # Comparison metrics
    st.markdown("### Performance Comparison")
    
    # Generate comparison data
    comp_data = []
    for machine in selected_machines:
        comp_data.append({
            'Machine': machine,
            'Efficiency': np.random.randint(75, 95),
            'Failure Risk': np.random.randint(10, 50),
            'Uptime': np.random.randint(90, 99),
            'Maintenance Cost': np.random.randint(800, 3000)
        })
    
    df = pd.DataFrame(comp_data)
    
    # Radar chart
    fig = go.Figure()
    
    for _, row in df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['Efficiency'], row['Uptime'], 100-row['Failure Risk'], 
               100-(row['Maintenance Cost']/30)],
            theta=['Efficiency', 'Uptime', 'Reliability', 'Cost Efficiency'],
            fill='toself',
            name=row['Machine']
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=500,
        title='Multi-Dimensional Performance Comparison'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed comparison table
    st.markdown("### Detailed Metrics")
    
    st.dataframe(
        df.style.background_gradient(cmap='RdYlGn', subset=['Efficiency', 'Uptime'])
               .background_gradient(cmap='RdYlGn_r', subset=['Failure Risk', 'Maintenance Cost']),
        use_container_width=True
    )
    
    # Ranking
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Best Performers")
        
        best = df.nlargest(3, 'Efficiency')[['Machine', 'Efficiency', 'Uptime']]
        for idx, row in best.iterrows():
            st.success(f"🏆 {row['Machine']}: {row['Efficiency']}% efficiency, {row['Uptime']}% uptime")
    
    with col2:
        st.markdown("### Needs Attention")
        
        worst = df.nlargest(3, 'Failure Risk')[['Machine', 'Failure Risk', 'Maintenance Cost']]
        for idx, row in worst.iterrows():
            st.warning(f"⚠️ {row['Machine']}: {row['Failure Risk']}% risk, ${row['Maintenance Cost']} cost")