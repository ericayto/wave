# 🌊 Wave

![License](https://img.shields.io/badge/license-Personal--Use-orange)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Node](https://img.shields.io/badge/node-18%2B-green)
![Status](https://img.shields.io/badge/status-Experimental-yellow)

> **An AI-powered crypto trading bot — runs entirely on your machine.**
> Test strategies now. Real trading support coming soon.

---

![Wave Dashboard Preview](docs/preview.png)
*(Dark theme, real-time updates, fully local)*

---

## 👋 About

I’m **Eric Aytekin**, a UK-based student interested in combining AI, crypto, and full-stack development in a single project.
**Wave** is a local-first crypto bot that blends traditional technical indicators with LLM-powered market analysis.

It:

* Currently runs in paper-trading mode.
* Keeps all your data on your machine.
* Will support real trading in the future.

---

## ✨ Features

### 🛡 Safety First (for now)

* Paper trading mode — no real funds used yet.
* Local execution — your data stays with you.
* Risk management — position sizing, daily loss limits.
* Kill switch — halt everything instantly.

### 📈 Trading Power

* Run multiple strategies in parallel.
* Built-in indicators: SMA, RSI, MACD, and more.
* Rapid analysis cycles every 30 seconds.
* Track performance over time.

### 🖥 Interface You’ll Actually Use

* Minimal, dark-themed dashboard.
* Real-time updates via WebSockets.
* Works on desktop *and* mobile.

### 🧩 For Tinkerers

* **FastAPI** backend + **React** frontend.
* Drop in your own strategies easily.
* Code commented for learners, not just future-you.

---

## 🚀 Quick Start

### Requirements

* 🐍 Python 3.11+
* 🟩 Node.js 18+
* 🌀 Git

### Install & Run

```bash
git clone https://github.com/yourusername/wave.git
cd wave
make setup && make dev
```

* **Dashboard:** [http://localhost:5173](http://localhost:5173)
* **API Docs:** [http://localhost:8080/docs](http://localhost:8080/docs)

Wave starts in paper-trading mode by default.

---

## 🛠 Commands

| Command       | Action                 |
| ------------- | ---------------------- |
| `make setup`  | Install everything     |
| `make dev`    | Run backend & frontend |
| `make stop`   | Stop services          |
| `make test`   | Run tests              |
| `make format` | Format code            |
| `make clean`  | Remove generated files |

---

## ⚙ Configuration

Edit `config/wave.toml` to tweak settings:

```toml
[core]
base_currency = "USD"
mode = "paper"

[risk]
max_position_pct = 0.25
daily_loss_limit_pct = 2.0
```

---

## 🔒 Privacy & Security

* Runs locally — no data leaves your machine.
* API keys stored securely.
* No tracking, telemetry, or analytics.

---

## 📜 License

**Personal Use License – UK Jurisdiction**
Copyright © 2025 Eric Aytekin.
Use for personal, educational, or non-commercial purposes only.
See LICENSE file for full terms.

---

## ⚠ Disclaimer

This is **educational only** — not financial advice.
Crypto is volatile; paper results ≠ real results.
Real trading support is planned but not yet implemented.

