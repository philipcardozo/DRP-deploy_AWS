#!/bin/bash
# AWS Deployment Script for Regime Volatility Arbitrage Engine
# Usage: bash aws_deploy.sh

set -e

echo "════════════════════════════════════════════════════════════════"
echo "  REGIME VOLATILITY ARBITRAGE ENGINE — AWS DEPLOYMENT"
echo "════════════════════════════════════════════════════════════════"

# Configuration
INSTANCE_IP="18.220.38.215"
SSH_KEY="volterra-new.pem"
REMOTE_USER="ubuntu"
REPO_URL="https://github.com/FelipeCardozo0/Regime-Volatility-Arbitrage-Engine.git"
REMOTE_DIR="~/regime-engine"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Check SSH key
log_info "Checking SSH key…"
if [ ! -f "$SSH_KEY" ]; then
    log_error "SSH key not found: $SSH_KEY"
    exit 1
fi
chmod 400 "$SSH_KEY"
log_info "SSH key ready (chmod 400)"

# Step 2: Test connection
log_info "Testing SSH connection to $INSTANCE_IP…"
if ! ssh -i "$SSH_KEY" -o ConnectTimeout=5 "$REMOTE_USER@$INSTANCE_IP" "echo 'SSH connection OK'" > /dev/null 2>&1; then
    log_error "Cannot connect to $INSTANCE_IP. Check IP address or security group."
    exit 1
fi
log_info "SSH connection successful"

# Step 3: Clone or pull repository
log_info "Setting up repository on AWS…"
ssh -i "$SSH_KEY" "$REMOTE_USER@$INSTANCE_IP" << 'SETUP_SCRIPT'
set -e

if [ ! -d ~/regime-engine ]; then
    echo "[SETUP] Cloning repository…"
    git clone https://github.com/FelipeCardozo0/Regime-Volatility-Arbitrage-Engine.git ~/regime-engine
else
    echo "[SETUP] Repository exists, pulling latest…"
    cd ~/regime-engine
    git pull origin main
fi

echo "[SETUP] Repository ready at ~/regime-engine"
SETUP_SCRIPT

log_info "Repository synced"

# Step 4: Install Python dependencies
log_info "Installing Python dependencies…"
ssh -i "$SSH_KEY" "$REMOTE_USER@$INSTANCE_IP" << 'DEPS_SCRIPT'
set -e

cd ~/regime-engine

if ! command -v pip &> /dev/null; then
    echo "[DEPS] pip not found, installing python3-pip…"
    sudo apt update > /dev/null
    sudo apt install -y python3-pip > /dev/null
fi

echo "[DEPS] Installing packages from requirements.txt…"
pip install --user -r requirements.txt --quiet

echo "[DEPS] Dependencies installed"
DEPS_SCRIPT

log_info "Dependencies installed"

# Step 5: Set environment variables
log_info "Configuring environment…"
ssh -i "$SSH_KEY" "$REMOTE_USER@$INSTANCE_IP" << 'ENV_SCRIPT'
set -e

# Add API key to bashrc if not already present
if ! grep -q "MASSIVE_API_KEY" ~/.bashrc; then
    echo 'export MASSIVE_API_KEY="o1Jntxe01_Ahkm39ZB7nJuvhrXIP6nbf"' >> ~/.bashrc
    echo "[ENV] Added MASSIVE_API_KEY to ~/.bashrc"
fi

source ~/.bashrc
echo "[ENV] Environment configured"
ENV_SCRIPT

log_info "Environment configured"

# Step 6: Run validation suite (optional)
log_warn "Ready to run validation. This may take 10+ minutes."
read -p "Run validation suite now? (y/n) " -n 1 -r VALIDATE
echo

if [[ $VALIDATE =~ ^[Yy]$ ]]; then
    log_info "Running validation suite on AWS…"
    ssh -i "$SSH_KEY" "$REMOTE_USER@$INSTANCE_IP" << 'VALIDATE_SCRIPT'
set -e

cd ~/regime-engine
echo "[VALIDATE] Starting validation suite…"
python main.py --mode validate 2>&1 | head -100

echo "[VALIDATE] Validation complete. Check full output above."
VALIDATE_SCRIPT

    log_info "Validation suite completed"
else
    log_warn "Skipped validation suite"
fi

# Step 7: Run research mode (lightweight test)
log_info "Running research mode (lightweight test)…"
ssh -i "$SSH_KEY" "$REMOTE_USER@$INSTANCE_IP" << 'RESEARCH_SCRIPT'
set -e

cd ~/regime-engine
echo "[RESEARCH] Fetching SPY data and fitting HMM…"
timeout 300 python main.py --mode research 2>&1 | tail -50 || true

echo "[RESEARCH] Research mode complete"
RESEARCH_SCRIPT

log_info "Research mode completed"

# Step 8: Print summary
log_info "═══════════════════════════════════════════════════════════════"
log_info "DEPLOYMENT COMPLETE"
log_info "═══════════════════════════════════════════════════════════════"

cat << SUMMARY

Next Steps:

1. SSH into the instance and start paper trading:
   ssh -i "$SSH_KEY" "$REMOTE_USER@$INSTANCE_IP"
   cd ~/regime-engine
   python main.py --mode paper

2. Monitor in another terminal:
   ssh -i "$SSH_KEY" "$REMOTE_USER@$INSTANCE_IP"
   tail -f ~/regime-engine/trade_log.csv
   tail -f ~/regime-engine/heartbeat.log

3. Download logs for analysis:
   scp -i "$SSH_KEY" "$REMOTE_USER@$INSTANCE_IP:~/regime-engine/trade_log.csv" .
   scp -i "$SSH_KEY" "$REMOTE_USER@$INSTANCE_IP:~/regime-engine/fills.csv" .
   scp -i "$SSH_KEY" "$REMOTE_USER@$INSTANCE_IP:~/regime-engine/heartbeat.log" .

4. For live trading (only after 48+ hrs of paper trading):
   python main.py --mode live

5. To stop the strategy:
   Ctrl+C (graceful shutdown) or pkill -f "python main.py" (forced)

═════════════════════════════════════════════════════════════════════

Instance: $INSTANCE_IP
User: $REMOTE_USER
Directory: ~/regime-engine
API Key: o1Jntxe01_Ahkm39ZB7nJuvhrXIP6nbf (Polygon.io)

See DEPLOYMENT_GUIDE.md for full documentation.

SUMMARY

log_info "Deployment finished successfully!"
