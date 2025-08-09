# 🌊 Wave

![License](https://img.shields.io/badge/license-Personal--Use-orange)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Node](https://img.shields.io/badge/node-18%2B-green)
![Status](https://img.shields.io/badge/status-Experimental-yellow)

> **An AI-powered crypto trading bot — runs locally with optional cloud AI integration.**
> Test strategies now. Real trading support coming soon.

---

![Wave Dashboard Preview](docs/preview.png)
*(Dark theme, real-time updates, local-first design)*

---

## 👋 About

I’m **Eric Aytekin**, a UK-based student interested in combining AI, crypto, and full-stack development in a single project.
**Wave** is a local-first crypto bot that blends traditional technical indicators with LLM-powered market analysis.

It:

* Currently runs in paper-trading mode.
* Executes locally, with the option to use local AI models or connect to cloud-based ones.
* Will support real trading in the future.

---

## ✨ Features

### 🛡 Safety First (for now)

* Paper trading mode — no real funds used yet.
* Local execution — your data stays with you unless you choose to use cloud-based AI models.
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

### Global Installation (Recommended)

Install Wave as a global `wave` command with auto-update functionality:

```bash
git clone https://github.com/ericayto/wave.git
cd wave
chmod +x install.sh
./install.sh
```

Once installed, you can use Wave from anywhere:

```bash
wave setup    # Set up Wave environment (first time only)
wave start    # Start Wave services
wave stop     # Stop Wave services
wave update   # Manually update to latest version
```

* **Dashboard:** [http://localhost:5173](http://localhost:5173)
* **API Docs:** [http://localhost:8080/docs](http://localhost:8080/docs)

Wave starts in paper-trading mode by default and **automatically updates itself** on startup to ensure you always have the latest features!

### Local Development Installation

For development or if you prefer not to install globally:

```bash
git clone https://github.com/ericayto/wave.git
cd wave
make setup && make dev
```

---

## 🛠 Commands

### Global Commands (After Installation)

| Command       | Action                           |
| ------------- | -------------------------------- |
| `wave setup`  | Set up Wave environment          |
| `wave start`  | Start Wave services              |
| `wave stop`   | Stop Wave services               |
| `wave update` | Manually update to latest version|

### Development Commands (Local Installation)

| Command       | Action                 |
| ------------- | ---------------------- |
| `make setup`  | Install everything     |
| `make dev`    | Run backend & frontend |
| `make stop`   | Stop services          |
| `make test`   | Run tests              |
| `make format` | Format code            |
| `make clean`  | Remove generated files |

### Installation Management

| Command            | Action                    |
| ------------------ | ------------------------- |
| `./install.sh`     | Install Wave globally     |
| `./uninstall.sh`   | Remove global installation|

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

## 🔄 Auto-Update

Wave includes intelligent auto-update functionality when installed globally:

* **Automatic Updates:** Wave checks for updates every 24 hours when you run `wave start`
* **Smart Timing:** Updates only check once per day to avoid unnecessary network requests
* **User Control:** 5-second countdown gives you time to skip updates with Ctrl+C
* **Dependency Management:** Automatically reinstalls dependencies after updates
* **Safe Updates:** Only works with git repositories to ensure integrity

### Manual Updates

You can manually update at any time:

```bash
wave update
```

### Disable Auto-Updates

To disable auto-updates temporarily, you can skip them with Ctrl+C during the 5-second countdown, or edit the `UPDATE_INTERVAL_HOURS` setting in the `wave` script.

---

## 🔒 Privacy & Security

* Runs locally, with optional use of cloud-based AI models.
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


