"""
Machine Management Page - Admin Only
Add, edit, and configure machines with custom features
"""
import streamlit as st
import json
from pathlib import Path
from datetime import datetime

# Machine storage file
MACHINES_FILE = Path("data/machines.json")

def load_machines():
    """Load machines from JSON file"""
    if MACHINES_FILE.exists():
        with open(MACHINES_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_machines(machines):
    """Save machines to JSON file"""
    MACHINES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MACHINES_FILE, 'w') as f:
        json.dump(machines, f, indent=2)

def show_machine_management(api_client):
    """Display machine management interface"""
    
    st.title("🏭 Machine Management")
    
    # Check if user is admin
    user = st.session_state.get('user', {})
    if user.get('role') != 'admin':
        st.error("⛔ Access Denied - Admin Only")
        st.info("This feature is restricted to administrators.")
        return
    
    st.success(f"✅ Admin Access: {user.get('username')}")
    
    # Load existing machines
    machines = load_machines()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["➕ Add Machine", "📋 Manage Machines", "⚙️ Configure Features"])
    
    with tab1:
        show_add_machine(machines)
    
    with tab2:
        show_manage_machines(machines)
    
    with tab3:
        show_configure_features(machines)

def show_add_machine(machines):
    """Add new machine interface"""
    
    st.subheader("➕ Add New Machine")
    
    with st.form("add_machine_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            machine_id = st.text_input(
                "Machine ID *",
                placeholder="e.g., MACH-001",
                help="Unique identifier for the machine"
            )
            
            machine_name = st.text_input(
                "Machine Name *",
                placeholder="e.g., Hydraulic Press #1"
            )
            
            location = st.text_input(
                "Location",
                placeholder="e.g., Factory Floor A, Section 3"
            )
            
            manufacturer = st.text_input(
                "Manufacturer",
                placeholder="e.g., Acme Industries"
            )
        
        with col2:
            machine_type = st.selectbox(
                "Machine Type *",
                ["Hydraulic Press", "CNC Machine", "Conveyor", "Pump", "Motor", 
                 "Compressor", "Generator", "Turbine", "Other"]
            )
            
            model = st.text_input(
                "Model Number",
                placeholder="e.g., HP-3000X"
            )
            
            install_date = st.date_input(
                "Installation Date",
                value=datetime.now()
            )
            
            status = st.selectbox(
                "Initial Status",
                ["Active", "Inactive", "Maintenance", "Testing"]
            )
        
        st.markdown("---")
        st.markdown("### 📊 Select Sensor Features")
        
        # Available features
        all_features = {
            'temperature': 'Temperature (°C)',
            'vibration': 'Vibration (mm/s)',
            'pressure': 'Pressure (bar)',
            'rpm': 'RPM',
            'current': 'Current (A)',
            'voltage': 'Voltage (V)',
            'power': 'Power (kW)',
            'torque': 'Torque (Nm)',
            'speed': 'Speed (m/s)',
            'humidity': 'Humidity (%)',
            'noise': 'Noise Level (dB)',
            'flow_rate': 'Flow Rate (L/min)'
        }
        
        st.info("Select which sensors this machine has. At least 2 required for predictions.")
        
        # Feature selection in columns
        col1, col2, col3 = st.columns(3)
        
        selected_features = {}
        
        features_list = list(all_features.items())
        for idx, (key, label) in enumerate(features_list):
            col = [col1, col2, col3][idx % 3]
            with col:
                if st.checkbox(label, key=f"feat_{key}", value=(key in ['temperature', 'vibration', 'pressure', 'rpm'])):
                    selected_features[key] = label
        
        st.markdown("---")
        st.markdown("### 📝 Additional Information")
        
        notes = st.text_area(
            "Notes",
            placeholder="Any additional information about this machine..."
        )
        
        submitted = st.form_submit_button("➕ Add Machine", type="primary", use_container_width=True)
        
        if submitted:
            # Validation
            if not machine_id or not machine_name or not machine_type:
                st.error("❌ Please fill in all required fields (*)")
            elif machine_id in machines:
                st.error(f"❌ Machine ID '{machine_id}' already exists!")
            elif len(selected_features) < 2:
                st.error("❌ Please select at least 2 sensor features for predictions")
            else:
                # Add machine
                machines[machine_id] = {
                    'id': machine_id,
                    'name': machine_name,
                    'type': machine_type,
                    'location': location,
                    'manufacturer': manufacturer,
                    'model': model,
                    'install_date': install_date.isoformat(),
                    'status': status,
                    'features': selected_features,
                    'notes': notes,
                    'created_at': datetime.now().isoformat(),
                    'created_by': st.session_state.get('user', {}).get('username', 'admin')
                }
                
                save_machines(machines)
                st.success(f"✅ Machine '{machine_name}' added successfully!")
                st.balloons()
                st.rerun()

def show_manage_machines(machines):
    """Manage existing machines"""
    
    st.subheader("📋 Existing Machines")
    
    if not machines:
        st.info("No machines configured yet. Add your first machine in the 'Add Machine' tab.")
        return
    
    st.write(f"**Total Machines:** {len(machines)}")
    
    # Filter
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search = st.text_input("🔍 Search", placeholder="Search by ID, name, or type...")
    
    with col2:
        filter_status = st.selectbox("Filter by Status", ["All", "Active", "Inactive", "Maintenance", "Testing"])
    
    # Display machines
    for machine_id, machine in machines.items():
        # Apply filters
        if search and search.lower() not in f"{machine_id} {machine['name']} {machine['type']}".lower():
            continue
        
        if filter_status != "All" and machine['status'] != filter_status:
            continue
        
        with st.expander(f"🏭 {machine['name']} ({machine_id})", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**ID:** {machine['id']}")
                st.write(f"**Type:** {machine['type']}")
                st.write(f"**Location:** {machine.get('location', 'N/A')}")
                st.write(f"**Status:** {machine['status']}")
            
            with col2:
                st.write(f"**Manufacturer:** {machine.get('manufacturer', 'N/A')}")
                st.write(f"**Model:** {machine.get('model', 'N/A')}")
                st.write(f"**Installed:** {machine.get('install_date', 'N/A')}")
                st.write(f"**Created:** {machine.get('created_at', 'N/A')[:10]}")
            
            st.markdown("**Sensors:**")
            features_str = ", ".join(machine['features'].values())
            st.write(features_str)
            
            if machine.get('notes'):
                st.markdown("**Notes:**")
                st.write(machine['notes'])
            
            # Actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(f"✏️ Edit", key=f"edit_{machine_id}"):
                    st.session_state[f'editing_{machine_id}'] = True
                    st.rerun()
            
            with col2:
                new_status = st.selectbox(
                    "Change Status",
                    ["Active", "Inactive", "Maintenance", "Testing"],
                    index=["Active", "Inactive", "Maintenance", "Testing"].index(machine['status']),
                    key=f"status_{machine_id}"
                )
                if new_status != machine['status']:
                    machine['status'] = new_status
                    save_machines(machines)
                    st.success(f"Status updated to {new_status}")
            
            with col3:
                if st.button(f"🗑️ Delete", key=f"del_{machine_id}"):
                    if st.session_state.get(f'confirm_del_{machine_id}'):
                        del machines[machine_id]
                        save_machines(machines)
                        st.success(f"Machine {machine_id} deleted")
                        st.rerun()
                    else:
                        st.session_state[f'confirm_del_{machine_id}'] = True
                        st.warning("Click again to confirm deletion")

def show_configure_features(machines):
    """Configure machine features"""
    
    st.subheader("⚙️ Configure Machine Features")
    
    if not machines:
        st.info("No machines available. Add a machine first.")
        return
    
    # Select machine
    machine_id = st.selectbox(
        "Select Machine",
        options=list(machines.keys()),
        format_func=lambda x: f"{machines[x]['name']} ({x})"
    )
    
    if machine_id:
        machine = machines[machine_id]
        
        st.markdown(f"### 🏭 {machine['name']}")
        
        st.markdown("**Current Sensors:**")
        for key, label in machine['features'].items():
            st.write(f"✓ {label}")
        
        st.markdown("---")
        
        with st.form(f"config_{machine_id}"):
            st.markdown("### Update Sensor Configuration")
            
            all_features = {
                'temperature': 'Temperature (°C)',
                'vibration': 'Vibration (mm/s)',
                'pressure': 'Pressure (bar)',
                'rpm': 'RPM',
                'current': 'Current (A)',
                'voltage': 'Voltage (V)',
                'power': 'Power (kW)',
                'torque': 'Torque (Nm)',
                'speed': 'Speed (m/s)',
                'humidity': 'Humidity (%)',
                'noise': 'Noise Level (dB)',
                'flow_rate': 'Flow Rate (L/min)'
            }
            
            col1, col2, col3 = st.columns(3)
            
            new_features = {}
            features_list = list(all_features.items())
            
            for idx, (key, label) in enumerate(features_list):
                col = [col1, col2, col3][idx % 3]
                with col:
                    if st.checkbox(label, value=(key in machine['features']), key=f"cfg_{machine_id}_{key}"):
                        new_features[key] = label
            
            if st.form_submit_button("💾 Save Configuration", type="primary", use_container_width=True):
                if len(new_features) < 2:
                    st.error("At least 2 features required")
                else:
                    machine['features'] = new_features
                    machine['updated_at'] = datetime.now().isoformat()
                    save_machines(machines)
                    st.success("✅ Configuration updated!")
                    st.rerun()

if __name__ == "__main__":
    st.write("This is a module. Use show_machine_management(api_client)")