#!/bin/bash
###############################################################################
# PredictXAI Quick Start Script
# One command to set everything up and start the system
###############################################################################

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║                    PREDICTXAI QUICK START                          ║"
echo "║           Enterprise AI Predictive Maintenance Platform            ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Check Python
echo -e "${BLUE}[1/6] Checking Python version...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.9+${NC}"
    exit 1
fi

# Step 2: Create virtual environment
echo -e "${BLUE}[2/6] Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Step 3: Install dependencies
echo -e "${BLUE}[3/6] Installing dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 4: Initialize database
echo -e "${BLUE}[4/6] Initializing database...${NC}"
if [ -f "database/predictxai.db" ]; then
    echo -e "${YELLOW}⚠ Database already exists${NC}"
else
    python init_database.py --init --create-users > /dev/null 2>&1
    echo -e "${GREEN}✓ Database initialized${NC}"
fi

# Step 5: Prepare ML model
echo -e "${BLUE}[5/6] Preparing ML model...${NC}"
if [ ! -f "data/models/predictive_model.pkl" ]; then
    if [ ! -f "data/raw/sensor_data.csv" ]; then
        echo "  Generating sample data..."
        python -m services.ml_service.data_generator > /dev/null 2>&1
        echo -e "${GREEN}  ✓ Sample data generated${NC}"
    fi
    echo "  Training ML model (this may take 1-2 minutes)..."
    python -m services.ml_service.trainer > /dev/null 2>&1
    echo -e "${GREEN}✓ ML model trained${NC}"
else
    echo -e "${GREEN}✓ ML model already exists${NC}"
fi

# Step 6: Start services
echo -e "${BLUE}[6/6] Starting all services...${NC}"
echo ""

# Create logs directory
mkdir -p logs

echo -e "${GREEN}Starting services...${NC}"
echo "  (You'll need to keep this terminal open)"
echo ""

# Start all services
python -m services.auth_service.main > logs/auth.log 2>&1 &
AUTH_PID=$!
echo -e "  ${GREEN}✓${NC} Auth Service (PID: $AUTH_PID)"
sleep 2

python -m services.ml_service.main > logs/ml.log 2>&1 &
ML_PID=$!
echo -e "  ${GREEN}✓${NC} ML Service (PID: $ML_PID)"
sleep 2

python -m services.agent_service.main > logs/agent.log 2>&1 &
AGENT_PID=$!
echo -e "  ${GREEN}✓${NC} Agent Service (PID: $AGENT_PID)"
sleep 2

python -m services.api_gateway.main > logs/gateway.log 2>&1 &
GATEWAY_PID=$!
echo -e "  ${GREEN}✓${NC} API Gateway (PID: $GATEWAY_PID)"
sleep 2

streamlit run frontend/app.py > logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo -e "  ${GREEN}✓${NC} Frontend (PID: $FRONTEND_PID)"
sleep 3

# Save PIDs for cleanup
echo "$AUTH_PID" > logs/pids.txt
echo "$ML_PID" >> logs/pids.txt
echo "$AGENT_PID" >> logs/pids.txt
echo "$GATEWAY_PID" >> logs/pids.txt
echo "$FRONTEND_PID" >> logs/pids.txt

# Final message
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                   ALL SERVICES STARTED!                            ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}✓ Frontend:${NC}      http://localhost:8501"
echo -e "${GREEN}✓ API Gateway:${NC}   http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}Default Login Credentials:${NC}"
echo "  Username: admin"
echo "  Password: admin123"
echo ""
echo -e "${BLUE}To stop all services, run:${NC}"
echo "  ./stop_all_services.sh"
echo ""
echo -e "${BLUE}To view logs:${NC}"
echo "  tail -f logs/*.log"
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  Open your browser: http://localhost:8501                          ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Open browser automatically (optional)
if command -v xdg-open &> /dev/null; then
    sleep 2
    xdg-open http://localhost:8501 &> /dev/null || true
elif command -v open &> /dev/null; then
    sleep 2
    open http://localhost:8501 &> /dev/null || true
fi

# Wait and show status
echo -e "${YELLOW}Services are running in background. Press Ctrl+C to stop monitoring.${NC}"
echo ""

# Monitor services
while true; do
    sleep 30
    
    # Check if all services are still running
    if ! ps -p $AUTH_PID > /dev/null 2>&1; then
        echo -e "${RED}✗ Auth Service stopped unexpectedly${NC}"
    fi
    if ! ps -p $ML_PID > /dev/null 2>&1; then
        echo -e "${RED}✗ ML Service stopped unexpectedly${NC}"
    fi
    if ! ps -p $AGENT_PID > /dev/null 2>&1; then
        echo -e "${RED}✗ Agent Service stopped unexpectedly${NC}"
    fi
    if ! ps -p $GATEWAY_PID > /dev/null 2>&1; then
        echo -e "${RED}✗ API Gateway stopped unexpectedly${NC}"
    fi
    if ! ps -p $FRONTEND_PID > /dev/null 2>&1; then
        echo -e "${RED}✗ Frontend stopped unexpectedly${NC}"
    fi
done