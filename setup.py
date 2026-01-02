#!/usr/bin/env python3
"""
Complete Setup Script for PredictXAI
Handles all initialization tasks
"""
import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_command(command, description, check=True):
    """Run a command and handle errors"""
    print(f"→ {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  ✓ {description} complete")
            return True
        else:
            if result.stderr:
                print(f"  ⚠ Warning: {result.stderr[:200]}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error: {e}")
        if e.stderr:
            print(f"    {e.stderr[:200]}")
        return False

def check_python_version():
    """Check Python version"""
    print_header("CHECKING PYTHON VERSION")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("✗ Python 3.9 or higher is required")
        return False
    
    print("✓ Python version is compatible")
    return True

def create_directory_structure():
    """Create necessary directories"""
    print_header("CREATING DIRECTORY STRUCTURE")
    
    directories = [
        "data/raw",
        "data/models",
        "database",
        "logs",
        "config",
        "services/auth_service",
        "services/ml_service",
        "services/agent_service/agents",
        "services/api_gateway",
        "frontend/pages",
        "frontend/utils"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {directory}")
    
    return True

def create_init_files():
    """Create all __init__.py files"""
    print_header("CREATING PYTHON PACKAGE FILES")
    
    init_files = [
        "config/__init__.py",
        "services/__init__.py",
        "services/auth_service/__init__.py",
        "services/ml_service/__init__.py",
        "services/agent_service/__init__.py",
        "services/agent_service/agents/__init__.py",
        "services/api_gateway/__init__.py",
        "frontend/__init__.py",
        "frontend/pages/__init__.py",
        "frontend/utils/__init__.py"
    ]
    
    for init_file in init_files:
        path = Path(init_file)
        if not path.exists():
            path.touch()
            print(f"  ✓ Created: {init_file}")
        else:
            print(f"  • Exists: {init_file}")
    
    return True

def setup_virtual_environment():
    """Setup Python virtual environment"""
    print_header("SETTING UP VIRTUAL ENVIRONMENT")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("  • Virtual environment already exists")
        response = input("  Recreate? (y/n): ")
        if response.lower() == 'y':
            import shutil
            shutil.rmtree(venv_path)
            print("  ✓ Removed old virtual environment")
        else:
            return True
    
    return run_command(
        f"{sys.executable} -m venv venv",
        "Creating virtual environment"
    )

def install_dependencies():
    """Install Python dependencies"""
    print_header("INSTALLING DEPENDENCIES")
    
    # Determine pip path based on OS
    if sys.platform == "win32":
        pip_path = "venv\\Scripts\\pip"
    else:
        pip_path = "venv/bin/pip"
    
    # Upgrade pip
    run_command(
        f"{pip_path} install --upgrade pip",
        "Upgrading pip",
        check=False
    )
    
    # Install dependencies
    if Path("requirements.txt").exists():
        return run_command(
            f"{pip_path} install -r requirements.txt",
            "Installing requirements"
        )
    else:
        print("  ⚠ requirements.txt not found")
        return False

def generate_sample_data():
    """Generate sample dataset"""
    print_header("GENERATING SAMPLE DATA")
    
    response = input("Generate sample sensor data? (y/n): ")
    if response.lower() != 'y':
        print("  Skipped data generation")
        return True
    
    # Activate venv and run data generator
    if sys.platform == "win32":
        python_path = "venv\\Scripts\\python"
    else:
        python_path = "venv/bin/python"
    
    return run_command(
        f"{python_path} -m services.ml_service.data_generator",
        "Generating sample data"
    )

def train_ml_model():
    """Train the ML model"""
    print_header("TRAINING ML MODEL")
    
    # Check if data exists
    data_path = Path("data/raw/sensor_data.csv")
    if not data_path.exists():
        print("  ⚠ Dataset not found. Generate data first.")
        return False
    
    response = input("Train ML model? (y/n): ")
    if response.lower() != 'y':
        print("  Skipped model training")
        return True
    
    # Activate venv and run trainer
    if sys.platform == "win32":
        python_path = "venv\\Scripts\\python"
    else:
        python_path = "venv/bin/python"
    
    return run_command(
        f"{python_path} -m services.ml_service.trainer",
        "Training ML model"
    )

def initialize_database():
    """Initialize database and create test users"""
    print_header("INITIALIZING DATABASE")
    
    # Activate venv and run database init
    if sys.platform == "win32":
        python_path = "venv\\Scripts\\python"
    else:
        python_path = "venv/bin/python"
    
    return run_command(
        f"{python_path} init_database.py --all",
        "Initializing database and creating test users"
    )

def create_env_file():
    """Create .env file from example"""
    print_header("CREATING ENVIRONMENT FILE")
    
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if env_path.exists():
        print("  • .env file already exists")
        response = input("  Overwrite? (y/n): ")
        if response.lower() != 'y':
            return True
    
    if env_example_path.exists():
        import shutil
        shutil.copy(env_example_path, env_path)
        print("  ✓ Created .env from .env.example")
        print("  ⚠ Remember to update SECRET_KEY in production!")
        return True
    else:
        print("  ⚠ .env.example not found")
        return False

def create_startup_instructions():
    """Create instructions file"""
    print_header("CREATING STARTUP INSTRUCTIONS")
    
    instructions = """
# PredictXAI - Quick Start Instructions

## Start All Services

You need to run 5 services in separate terminals:

### Terminal 1 - Auth Service
```
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate   # Windows
python -m services.auth_service.main
```

### Terminal 2 - ML Service
```
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate   # Windows
python -m services.ml_service.main
```

### Terminal 3 - Agent Service
```
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate   # Windows
python -m services.agent_service.main
```

### Terminal 4 - API Gateway
```
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate   # Windows
python -m services.api_gateway.main
```

### Terminal 5 - Frontend
```
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate   # Windows
streamlit run frontend/app.py
```

## Access the Application

Open your browser and go to: http://localhost:8501

## Default Login Credentials

- **Admin User**
  - Username: admin
  - Password: admin123

- **Operator User**
  - Username: operator
  - Password: operator123

## Quick Test

1. Login with admin credentials
2. Select a machine
3. Enter sensor values
4. Click "Run AI Analysis"

## Troubleshooting

If services don't start:
1. Check if ports are free: 8000, 8001, 8002, 8003, 8501
2. Verify virtual environment is activated
3. Check logs in ./logs/ directory
"""
    
    with open("START_HERE.md", "w") as f:
        f.write(instructions)
    
    print("  ✓ Created START_HERE.md")
    return True

def main():
    """Main setup execution"""
    print("\n" + "="*70)
    print("  PREDICTXAI - COMPLETE SETUP")
    print("="*70)
    
    steps = [
        ("Python Version", check_python_version),
        ("Directory Structure", create_directory_structure),
        ("Package Files", create_init_files),
        ("Virtual Environment", setup_virtual_environment),
        ("Dependencies", install_dependencies),
        ("Environment File", create_env_file),
        ("Sample Data", generate_sample_data),
        ("ML Model", train_ml_model),
        ("Database", initialize_database),
        ("Instructions", create_startup_instructions)
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        try:
            results[step_name] = step_func()
        except Exception as e:
            print(f"  ✗ Error in {step_name}: {e}")
            results[step_name] = False
    
    # Summary
    print_header("SETUP SUMMARY")
    
    for step_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {step_name}")
    
    all_success = all(results.values())
    
    if all_success:
        print("\n" + "="*70)
        print("  ✓ SETUP COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Review START_HERE.md for startup instructions")
        print("  2. Update .env file with your SECRET_KEY (production)")
        print("  3. Start all 5 services (see START_HERE.md)")
        print("  4. Open http://localhost:8501 in your browser")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("  ⚠ SETUP COMPLETED WITH WARNINGS")
        print("="*70)
        print("\nSome steps failed. Please check the output above.")
        print("You may need to run certain steps manually.")
        print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)