#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo -e "${GREEN}===== AiQuant Freqtrade One-Click Setup =====${NC}"
echo ""

# 1. Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[Error] Docker not found.${NC}"
    echo "Please install Docker Desktop first: https://www.docker.com/products/docker-desktop"
    echo "For macOS: brew install --cask docker"
    exit 1
fi

# 2. Check Docker Compose
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo -e "${RED}[Error] Docker Compose not found.${NC}"
    echo "Please install Docker Compose plugin."
    exit 1
fi

echo -e "${GREEN}[OK]${NC} Docker and Docker Compose detected."
echo ""

# 3. Create directory structure
echo -e "${YELLOW}Creating project directories...${NC}"
mkdir -p "$PROJECT_ROOT/freqtrade/user_data"/{strategies,models,data,notebooks,logs}
mkdir -p "$PROJECT_ROOT/research"
mkdir -p "$PROJECT_ROOT/deploy"
# scripts directory removed; setup.sh lives at project root

touch "$PROJECT_ROOT/freqtrade/user_data/strategies/__init__.py"

echo -e "${GREEN}[OK]${NC} Directory structure created."
echo ""

# 4. Pull Freqtrade image
echo -e "${YELLOW}Pulling Freqtrade stable image...${NC}"
docker pull freqtradeorg/freqtrade:stable

echo -e "${GREEN}[OK]${NC} Freqtrade image pulled."
echo ""

# 5. Set permissions
chmod +x "$PROJECT_ROOT/setup.sh" 2>/dev/null || true

# 6. Final instructions
echo -e "${GREEN}===== Setup Complete =====${NC}"
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo "1. Configure your exchange API keys:"
echo "   Edit: $PROJECT_ROOT/freqtrade/config_ai_model.json"
echo "   - Replace 'YOUR_API_KEY' and 'YOUR_SECRET' with Binance testnet keys"
echo "   - Or keep as-is for pure dry-run backtesting"
echo ""
echo "2. Train an AI model (optional):"
echo "   cd research && pip install -r requirements.txt"
echo "   python train_classifier.py"
echo ""
echo "3. Download historical data for backtesting:"
echo "   $COMPOSE_CMD -f deploy/docker-compose.yml run --rm freqtrade \\"
echo "       download-data --pairs BTC/USDT ETH/USDT --timeframe 1h --timerange 20240101-20241231"
echo ""
echo "4. Run backtest:"
echo "   $COMPOSE_CMD -f deploy/docker-compose.yml run --rm freqtrade \\"
echo "       backtesting --strategy AIModelStrategy --timerange 20240101-20241231"
echo ""
echo "5. Start dry-run trading (Web UI at http://localhost:8080):"
echo "   $COMPOSE_CMD -f deploy/docker-compose.yml up -d"
echo ""
echo -e "${YELLOW}Security Notes:${NC}"
echo "- config_ai_model.json is gitignored to protect your API keys."
echo "- Dry run mode is ENABLED by default. Do not disable until ready for live trading."
echo ""
