# 🌊 Wave

**Local LLM-Driven Crypto Trading Bot (Paper Trading)**

## 👋 About Me

Hi, I’m **Eric Aytekin**, a student in the UK.
I built Wave as a personal learning project to explore crypto trading, AI, and full-stack development.
It’s an experimental bot — no real money involved — coded with a mix of curiosity, late nights, and the help of Claude.

Wave runs entirely on your own computer, combining traditional trading strategies with AI-powered market analysis. It’s **paper-trading only**, so you can explore ideas without risking real funds.

## ✨ Features

### Safety

* **Paper trading only** – No real funds involved.
* **Local execution** – Your data stays with you.
* **Risk management** – Position sizing and loss limits.
* **Kill switch** – Stop all trading instantly.

### Trading

* Multiple strategies at once.
* Built-in indicators (SMA, RSI, MACD, etc.).
* Fast analysis cycles (every 30 seconds).
* Performance tracking.

### Interface

* Simple, dark-themed dashboard.
* Real-time updates via WebSockets.
* Works on desktop and mobile.

### For tinkerers

* FastAPI backend + React frontend.
* Easy to add your own strategies.
* Well-commented code for learning.

## 🚀 Getting Started

**Requirements**

* Python 3.11+
* Node.js 18+
* Git

**Install & Run**

```bash
git clone https://github.com/yourusername/wave.git
cd wave
make setup
make dev
```

* Dashboard: `http://localhost:5173`
* API Docs: `http://localhost:8080/docs`

Wave will start with demo data and paper trading enabled.

## 🛠 Commands

| Command       | What it does           |
| ------------- | ---------------------- |
| `make setup`  | Install everything     |
| `make dev`    | Run backend & frontend |
| `make stop`   | Stop services          |
| `make test`   | Run tests              |
| `make format` | Format code            |
| `make clean`  | Remove generated files |

## ⚙ Configuration

Edit `config/wave.toml` to change settings like:

```toml
[core]
base_currency = "USD"
mode = "paper"

[risk]
max_position_pct = 0.25
daily_loss_limit_pct = 2.0
```

## 🔒 Privacy & Security

* Runs locally — no data sent anywhere.
* API keys stored securely.
* No tracking, telemetry, or analytics.

## 📜 License

**Personal Use License – UK Jurisdiction**
Copyright © 2025 Eric Aytekin.
All rights reserved.
You may download and use this software for personal, educational, or non-commercial purposes only.
You may **not** redistribute, modify, or use it for commercial purposes.
Full license terms are in the `LICENSE` file.

## ⚠ Disclaimer

* This is for educational use only — not financial advice.
* Crypto trading carries risk; even in paper trading, results may not reflect real markets.
