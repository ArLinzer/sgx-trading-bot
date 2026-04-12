# SGX Day Trading Signal Bot

> **For educational and research purposes only. Not financial advice. Never places real trades.**

A fully autonomous SGX (Singapore Exchange) day-trading signal system powered by a local LLM (Ollama). The bot scans SGX stocks, aggregates news from multiple sources, uses the LLM to generate structured trading signals, and delivers them to Telegram — all intraday, exit-before-close.

---

## Architecture

```
main.py  ←  APScheduler (SGT cron)
   │
   ├── sgx_scanner.py    →  Full SGX stock list (API + scraper fallback)
   ├── news_fetcher.py   →  Google News, BT, ST, SGX API, Yahoo Finance, Moomoo
   ├── moomoo_client.py  →  Real-time quotes, order book, kline, news
   ├── llm_analyst.py    →  Ollama LLM (Call A: stock selection, Call B: signal)
   ├── signal_engine.py  →  Filter, rank, track open signals, emit exit alerts
   └── telegram_bot.py   →  Formatted Telegram messages
```

### Daily schedule (Asia/Singapore)

| Time  | Task |
|-------|------|
| 08:30 | Fetch SGX stock list + LLM stock selection (up to 20 tickers) |
| 09:00 | Fetch news + LLM strategy → send morning signals |
| 12:00 | Re-run news fetch + LLM → midday updates |
| 14:00 | Afternoon session signals |
| 16:30 | EOD sell reminder for all open positions |
| 17:15 | Daily summary |

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Uses `zoneinfo` (stdlib) |
| Ollama | Latest | Local LLM runtime |
| Moomoo Desktop | Latest | Optional — for real-time L2 data |
| Telegram Bot | — | For signal delivery |

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo>
cd sgx-trading-bot
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install and configure Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama service
ollama serve

# Pull the LLM model (in a separate terminal)
ollama pull llama3           # ~4.7 GB, recommended
# or
ollama pull mistral          # ~4.1 GB, fallback
# or
ollama pull phi3             # ~2.3 GB, fastest / lowest RAM
```

Verify it works:
```bash
ollama run llama3 "Say hello in JSON"
```

### 3. Configure Moomoo gateway (optional but recommended)

1. Download and install [Moomoo Desktop](https://www.moomoo.com/download).
2. Log in to your account.
3. In Moomoo Desktop → Settings → OpenAPI → enable the gateway.
4. Default gateway address: `127.0.0.1:11111`.
5. Set `trade_env: "SIMULATE"` in `config.yaml` (keep this as signal-only).

If you don't have Moomoo, start the bot with `--no-moomoo`. All other data sources will still work.

### 4. Create a Telegram bot

1. Open Telegram and search for `@BotFather`.
2. Send `/newbot` and follow the prompts.
3. Copy the **bot token** (looks like `123456:ABC-DEF...`).
4. Start a conversation with your new bot, then get your **chat ID**:
   ```bash
   curl "https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates"
   ```
   Look for `"chat":{"id": 123456789}` in the response.

### 5. Edit config.yaml

```yaml
telegram:
  bot_token: "123456:ABC-DEF..."   # From BotFather
  chat_id: "123456789"             # Your user or group chat ID

moomoo:
  host: "127.0.0.1"
  port: 11111
  trade_env: "SIMULATE"            # Never change to REAL — signal-only system

llm:
  model: "llama3"                  # or mistral / phi3
  ollama_host: "http://localhost:11434"
```

---

## Running the bot

```bash
# Normal mode — starts scheduler, runs at scheduled SGT times
python main.py

# Dry-run — prints signals to stdout, no Telegram messages
python main.py --dry-run

# Single cycle test — runs stock selection + signals + summary immediately
python main.py --once --dry-run

# Without Moomoo (no desktop gateway)
python main.py --no-moomoo --dry-run

# Custom config path
python main.py --config /path/to/config.yaml
```

### Health check

While running, a status endpoint is available at:

```
GET http://localhost:8080/health
GET http://localhost:8080/signals
```

---

## Signal format (Telegram)

```
🟢 BUY SIGNAL — $D05

━━━━━━━━━━━━━━━━━━━━
Entry:       S$38.500
Target:      S$39.100  (+1.6%)
Stop Loss:   S$38.200  (-0.8%)
Confidence:  78%
Strategy:    News Catalyst
Exit by EOD: ✅ Yes

📊 Rationale: DBS reported Q3 earnings beat. Price broke above 20-day SMA on 2x avg volume.
📰 Catalyst: DBS Q3 profit jumps 12% on higher net interest income
⚡ Source:   Business Times + SGX Announcements

[Generated at 09:14 SGT]
```

---

## Signal filters

Signals are dropped if any of these conditions are not met:

| Filter | Threshold |
|--------|-----------|
| Minimum confidence | ≥ 65% |
| Volume vs 20-day avg | ≥ 1.2× (20% above avg) |
| BUY: target > entry | Required |
| BUY: stop loss < entry | Required |
| exit_before_eod | Always `true` |

Signals are ranked by `confidence × expected_return`. The top 5 per session are sent.

---

## Data sources

| Source | Data type | Rate limited |
|--------|-----------|-------------|
| `api2.sgx.com/securities` | Stock list | Cached 24h |
| `api2.sgx.com/announcements` | Corp announcements | Per ticker |
| Google News RSS | News headlines | — |
| Business Times (scrape) | News | 1 req/s per domain |
| Straits Times (scrape) | News | 1 req/s per domain |
| Yahoo Finance (scrape) | News | 1 req/s per domain |
| Moomoo OpenAPI | Quotes, L2, kline, news | SDK managed |

---

## Project structure

```
sgx-trading-bot/
├── main.py                  # Orchestrator, scheduler, health API
├── config.yaml              # API keys, thresholds, LLM config
├── requirements.txt
├── README.md
├── modules/
│   ├── __init__.py
│   ├── sgx_scanner.py       # SGX stock list fetcher + cache
│   ├── news_fetcher.py      # Multi-source async news aggregator
│   ├── moomoo_client.py     # Moomoo OpenAPI wrapper
│   ├── llm_analyst.py       # Ollama LLM engine (Call A + B)
│   ├── signal_engine.py     # Filter, rank, track, exit alerts
│   └── telegram_bot.py      # Telegram message formatter + sender
└── data/
    ├── cache/               # Daily stock list, open signals JSON
    └── logs/                # Rotating daily log files
```

---

## Disclaimer

This software is for **educational and research purposes only**.

- It does **not** place any orders or trades.
- It does **not** provide financial advice.
- Past signals do not guarantee future performance.
- Trading involves significant financial risk. Never risk money you cannot afford to lose.
- Always validate signals independently before making any investment decisions.
