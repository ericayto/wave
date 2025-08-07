# 🌊 Wave

**Local LLM-Driven Crypto Trading Bot with Paper Trading**

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-4.9+-blue.svg)](https://typescriptlang.org)
[![Status: Development](https://img.shields.io/badge/Status-Development-orange.svg)](#development-status)

> **⚠️ Development Notice**: Wave is currently in active development. Features may be incomplete, unstable, or subject to breaking changes. Use at your own risk and do not use with live trading funds.

Wave is a sophisticated, **privacy-first** crypto trading bot that runs entirely on your local machine. It combines traditional trading strategies with LLM-powered market analysis and strategy generation, all while maintaining complete control over your data and trading decisions.

![Wave Dashboard](https://via.placeholder.com/800x400/1a1a2e/0f3460?text=🌊+Wave+Dashboard)

## ✨ **Key Features**

### 🛡️ **Safety First**
- **Paper Trading Only** - No live money risk in current version
- **Local Execution** - All data stays on your machine
- **Risk Management** - Comprehensive position and loss limits
- **Kill Switch** - Emergency stop all trading instantly

### 🤖 **Intelligent Trading**
- **Multi-Strategy Engine** - Run multiple strategies simultaneously  
- **Technical Analysis** - 10+ built-in indicators (SMA, RSI, MACD, etc.)
- **Real-time Execution** - 30-second analysis cycles
- **Performance Tracking** - Win rates, P&L, Sharpe ratios

### 🎨 **Beautiful Interface**
- **Ocean-Themed UI** - Dark, modern design with wave aesthetics
- **Real-time Updates** - WebSocket-powered live dashboard
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Accessibility** - Screen reader and keyboard navigation support

### 🔧 **Developer Friendly**
- **Modern Stack** - FastAPI + React + TypeScript
- **Event-Driven** - Scalable microservices architecture
- **Extensible** - Easy to add new strategies and exchanges
- **Well-Documented** - Comprehensive API docs and code comments

## 🚀 **Quick Start**

### Prerequisites

- **Python 3.11+** - [Download Python](https://python.org/downloads/)
- **Node.js 18+** - [Download Node.js](https://nodejs.org/)
- **Git** - [Install Git](https://git-scm.com/)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/wave.git
   cd wave
   ```

2. **Set up the environment** (one command!)
   ```bash
   make setup
   ```
   This will:
   - Create Python virtual environment
   - Install all Python dependencies
   - Install Node.js dependencies
   - Set up configuration files

3. **Start Wave**
   ```bash
   make dev
   ```
   This starts both backend and frontend services.

4. **Access the application**
   - 🌐 **Trading Interface**: http://localhost:5173
   - 📊 **API Documentation**: http://localhost:8080/docs
   - ⚡ **System Status**: http://localhost:8080/status

That's it! Wave is now running with demo data and paper trading enabled.

## 📊 **Current Implementation Status**

### ✅ **M0 + M1 Complete** (Production Ready)
- **CLI Bootstrap** - Cross-platform setup and service management
- **FastAPI Backend** - Async API with WebSocket real-time updates
- **React Frontend** - Ocean-themed dashboard with live data
- **Paper Trading** - Realistic order execution with slippage and fees
- **Strategy Engine** - Multi-strategy runtime with performance tracking
- **Risk Management** - Comprehensive position and loss limits
- **Market Data** - Kraken integration with CCXT (or mock data)
- **Technical Analysis** - Complete indicator library

### 🚧 **M2 In Progress** (LLM Integration)
- **LLM Orchestration** - Multi-provider support (OpenAI, Azure, local models)
- **Strategy Generation** - Natural language to trading strategy
- **Market Analysis** - AI-powered market narrative and sentiment
- **Context Management** - Long-term memory and decision history

## 🛠️ **Available Commands**

| Command | Description |
|---------|-------------|
| `make setup` | Initial environment setup |
| `make dev` | Start development servers |
| `make stop` | Stop all services |
| `make test` | Run all tests |
| `make lint` | Lint all code |
| `make format` | Format all code |
| `make build` | Build for production |
| `make clean` | Clean generated files |

Or use the CLI directly:
```bash
./wave setup    # Set up environment
./wave start    # Start services  
./wave stop     # Stop services
```

## ⚙️ **Configuration**

Wave uses a TOML configuration file at `config/wave.toml`:

```toml
[core]
base_currency = "USD"
mode = "paper"  # Paper trading only in v0.2

[exchanges.kraken]
api_key = "env:KRAKEN_KEY"      # Optional - uses mock data if not provided
api_secret = "env:KRAKEN_SECRET"

[risk]
max_position_pct = 0.25         # Max 25% of portfolio per position
daily_loss_limit_pct = 2.0      # Max 2% daily loss
max_orders_per_hour = 6         # Rate limiting

[ui]
port = 5173
show_thinking_feed = true
```

## 🏗️ **Architecture Overview**

Wave follows a modern microservices architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI Bootstrap │───▶│   FastAPI        │───▶│   React UI      │
│   ./wave start  │    │   Backend API    │    │   Port 5173     │
└─────────────────┘    │   Port 8080      │    └─────────────────┘
                       └──────────────────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
            ┌───────▼───┐ ┌───▼────┐ ┌──▼──────┐
            │Market Data│ │Strategy│ │Risk     │
            │Service    │ │Engine  │ │Engine   │
            └───────────┘ └────────┘ └─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Paper Broker    │
                    │   SQLite Database │
                    └───────────────────┘
```

### Key Components

- **Event Bus** - Async pub/sub messaging between services
- **Market Data Service** - CCXT integration for live price feeds
- **Strategy Engine** - Multi-strategy execution with 30s cycles
- **Risk Engine** - Pre-trade validation and monitoring
- **Paper Broker** - Realistic order execution simulation
- **WebSocket Manager** - Real-time updates to frontend

## 🔒 **Security & Privacy**

Wave is designed with **privacy and security** as top priorities:

- **🏠 Local Only** - No external servers, all data stays on your machine
- **🔐 Encrypted Storage** - API keys stored in OS keychain with encryption
- **🛡️ Paper Trading** - No live money risk in current version
- **📋 Audit Logs** - Complete history of all trading decisions
- **🚫 No Tracking** - No telemetry, analytics, or data collection

## 📈 **Trading Strategies**

### Built-in Strategies

1. **SMA Crossover** (Trend Following)
   - Uses 20/50 period moving averages
   - Volume and volatility filtering
   - Dynamic stop-loss and take-profit

2. **RSI Mean Reversion**  
   - 14-period RSI with 30/70 levels
   - Time-based position exits
   - Trend context filtering

### Custom Strategies (Coming in M2)

With LLM integration, you'll be able to:
- Generate strategies from natural language descriptions
- Analyze market conditions with AI
- Optimize parameters automatically
- Create complex multi-indicator strategies

## 📊 **Supported Exchanges**

### Current
- **Kraken** - Spot trading (paper mode only)
- **Mock Exchange** - Demo data for testing

### Planned
- **Coinbase Pro** - Spot trading
- **Binance** - Spot trading
- **Additional exchanges** based on community feedback

## 🧪 **Development Status**

Wave is actively developed with a clear roadmap:

- **✅ M0** - Basic architecture and paper trading
- **✅ M1** - Strategy engine and risk management  
- **🚧 M2** - LLM integration and strategy generation
- **📅 M3** - Advanced features and live trading preparation

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/wave.git
cd wave
make setup

# Start development
make dev

# Run tests
make test

# Format code
make format
```

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ **Important Disclaimers**

- **No Financial Advice** - Wave is for educational and research purposes only
- **No Profit Guarantees** - Trading cryptocurrencies involves substantial risk
- **Development Software** - May contain bugs, use at your own risk
- **Paper Trading Only** - Current version does not support live trading

## 🆘 **Support & Documentation**

- 📚 **Documentation** - [Full documentation](docs/)
- 🐛 **Bug Reports** - [GitHub Issues](https://github.com/yourusername/wave/issues)
- 💬 **Discussions** - [GitHub Discussions](https://github.com/yourusername/wave/discussions)
- 📧 **Contact** - [Email](mailto:your-email@example.com)

## 🌟 **Star History**

If you find Wave useful, please consider giving it a star! ⭐

---

<div align="center">

**Built with ❤️ by the Wave Team**

[🌊 Visit Website](https://wave-bot.dev) • [📖 Documentation](docs/) • [🐛 Report Bug](https://github.com/yourusername/wave/issues) • [💡 Request Feature](https://github.com/yourusername/wave/issues)

</div>